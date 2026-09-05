#!/usr/bin/env python
"""E8 diagnostic: Task-Relevant Response Alignment.

Motivation (after E6 GO + E7 NO-GO, RESEARCH.md S23-24): backbone adversarial
response coupling (C_response) explains transfer DIRECTION within each
matched-head pair, but a simple scalar downstream-amplification ratio (E7)
does not explain why the ASR gap between two different detector heads on the
SAME Swin backbone doesn't scale with their C_response gap. Three "scalar
disturbance size" explanations have now failed in this project's diagnostic
chain: raw ||dF|| (E1), instantaneous surrogate-vs-target gradient cosine
(S19), and pipeline amplification ratio (E7). This tests a different kind of
quantity: not HOW MUCH the target's backbone response changes, but whether
the part that changes actually falls along directions that matter for the
TARGET's OWN detection task.

Formal quantity, per target model t, per image i, per backbone stage l:
    dF_t^l    = F_t^l(x_i + delta_i) - F_t^l(x_i)      (same as E6, unpooled)
    g_t^l     = d L_det,t(x_i) / d F_t^l               (target's OWN detection
                                                          loss gradient w.r.t.
                                                          its OWN backbone
                                                          feature, evaluated at
                                                          the CLEAN image --
                                                          "what does this
                                                          target's own task
                                                          care about here")
    P_t^l     = |<dF_t^l, g_t^l>| / (||dF_t^l|| * ||g_t^l|| + eps)

P in [0,1]: 0 = the perturbation's effect on this backbone is entirely
orthogonal to what the target's own detector cares about (should transfer
poorly regardless of ||dF||); 1 = the effect is fully aligned with the
target's task-sensitive direction (should transfer well even if ||dF|| is
modest). Aggregated across stages by pooling+concatenating exactly like E6
(mean pooling only -- RMS is NOT valid here since it discards sign, and P
needs a signed inner product).

Prediction (pre-registered before running): P(mask_rcnn_swin_t) >
P(dino_swin_l), matching ASR(68.1%) > ASR(33.6%) on the same Swin backbone --
and P should explain the cross-target ASR ordering better than ||dF|| (E1)
and better than the E7 amplification ratio.

Engineering notes (read before trusting numbers):
  - g_t is obtained via a forward hook on model.backbone that calls
    .retain_grad() on each stage output tensor, then model.loss(...).backward()
    (reusing transfer_attack.losses.detector_task_loss/build_gt_data_sample,
    already used for the mi_fgsm baseline attack -- but that has only ever
    been exercised against the surrogate faster_rcnn_r50; this is the first
    use against mask_rcnn_swin_t/dino_r50/dino_swin_l). Uses .backward() +
    .grad (never torch.autograd.grad), matching the project's established
    DINO-checkpointing-safe pattern (HANDOFF.md).
  - dino_swin_l's backbone config has with_cp=True (Swin gradient checkpointing
    ON) -- the one architecture where this pattern is least proven. Smoke-test
    with --limit 2 before any real run; a None/NaN/all-zero grad on this
    target specifically is the failure mode to watch for.
  - No new attack crafted -- reuses noise already on disk under
    results/noise/<manifest-stem>/<attack>/ (produced by run_attack.py,
    already crafted for dev_300 in this session for E6/E7).

Example (smoke test):
    python scripts/e8_task_relevant_alignment.py --manifest data/manifests/dev_50.json --limit 2
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

MATCHED_PAIRS = [
    ("dino", "dino_r50", "dino_swin_l"),
    ("mask_rcnn", "mask_rcnn_r50", "mask_rcnn_swin_t"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["osfd", "mi_fgsm"], default="osfd")
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--noise-dir", type=Path, default=None)
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--targets", nargs="+", default=["all"])
    p.add_argument("--pool-size", type=int, default=7)
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "e8_task_relevant_alignment.csv")
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    return p


def find_matching_run_log(runs_dir: Path, attack: str, manifest_stem: str) -> dict | None:
    candidates = sorted(runs_dir.glob(f"run_attack_{attack}_{manifest_stem}_*.json"))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


def pool_and_concat(stages, pool_size: int):
    """stages: tuple of (1,C,H,W) tensors (grad or no-grad) -> flattened,
    concatenated (D,) vector. Mean pooling ONLY (RMS is invalid for a signed
    inner product)."""
    import torch
    import torch.nn.functional as Fnn

    parts = [Fnn.adaptive_avg_pool2d(t, (pool_size, pool_size)).flatten() for t in stages]
    return torch.cat(parts)


def _set_training_keep_norm_frozen(model, training: bool):
    """Set model.training (recursively, via .train()/.eval()) but immediately
    re-freeze every BatchNorm/Dropout submodule back to eval-mode behavior.

    Needed because DINO's pre_decoder only populates enc_outputs_class/
    enc_outputs_coord/dn_meta (required positional args of DINOHead.loss())
    when self.training is True (mmdet/models/detectors/dino.py:191-213) --
    those are gated purely on the detector's own `self.training` flag, not on
    anything BatchNorm-related. But dino_r50/mask_rcnn_r50 use a REAL
    (non-frozen-stats) BatchNorm in their backbone (norm_cfg BN, not GN/LN
    like the Swin variants) -- naively calling model.train() would make BN
    use THIS SINGLE IMAGE's batch statistics instead of the checkpoint's
    calibrated running stats, silently computing gradients through a
    different (and wrong, batch-size-1-unstable) forward pass than the one
    actually evaluated for ASR/mAP elsewhere in this project. This function
    keeps BN (and Dropout, if any) exactly as eval() would, while flipping
    every other module's self.training flag.
    """
    import torch.nn as nn

    model.train(training)
    if training:
        for m in model.modules():
            if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
                m.eval()


def target_task_gradient_at_backbone(handle, x_clean, gt_boxes, gt_cat_ids, canvas, device):
    """Returns a tuple of per-stage gradient tensors (same shapes as
    model.backbone(...)'s output) of the target's OWN total detection loss
    w.r.t. its OWN backbone feature maps, evaluated at the CLEAN image.
    """
    import torch

    from transfer_attack.losses import build_gt_data_sample, detector_task_loss

    model = handle.model
    captured = {}

    def _hook(_module, _inp, out):
        # Some targets (mask_rcnn_r50, out_indices=(0,1,2,3) with
        # frozen_stages=1) include a frozen early stage (e.g. layer1) whose
        # output has requires_grad=False (neither the input image nor any
        # param on that path requires grad) -- retain_grad() would raise on
        # that tensor. Skip it; its contribution is handled as an all-zero
        # gradient downstream (semantically correct: no task-gradient signal
        # is available through a frozen path, so it drops out of the
        # alignment inner product for that stage rather than being guessed at).
        for t in out:
            if t.requires_grad:
                t.retain_grad()
        captured["feats"] = out

    handle_ref = model.backbone.register_forward_hook(_hook)
    try:
        gt_labels = torch.tensor(
            [handle.cat_id_to_label[int(c)] for c in gt_cat_ids], dtype=torch.long, device=device
        )
        ds = build_gt_data_sample(gt_boxes.to(device), gt_labels, canvas)
        x_norm = handle.normalize(x_clean.unsqueeze(0))
        model.zero_grad(set_to_none=True)
        _set_training_keep_norm_frozen(model, True)
        # This project only ever evaluates/attacks box detection (mAP/ASR on
        # bboxes, RESEARCH.md S2) -- never instance segmentation. Mask R-CNN
        # targets (mask_rcnn_r50/mask_rcnn_swin_t) DO carry a mask_head, whose
        # .loss() unconditionally needs gt_instances.masks (which
        # build_gt_data_sample/build_gt_index never populate, since nothing
        # else in this project uses segmentation GT). Temporarily detach
        # mask_head so RoIHead.loss() skips that branch entirely -- this is
        # the semantically correct scope (box task-loss only), not a
        # workaround for missing data.
        roi_head = getattr(model, "roi_head", None)
        saved_mask_head = None
        if roi_head is not None and getattr(roi_head, "mask_head", None) is not None:
            saved_mask_head = roi_head.mask_head
            roi_head.mask_head = None
        try:
            loss = detector_task_loss(model, x_norm, ds)
            loss.backward()
        finally:
            if saved_mask_head is not None:
                roi_head.mask_head = saved_mask_head
            _set_training_keep_norm_frozen(model, False)
        grads = tuple(t.grad.detach().clone() if t.grad is not None else torch.zeros_like(t) for t in captured["feats"])
    finally:
        handle_ref.remove()
    return grads


def compute_alignment_for_image(handle, x_clean, x_adv, gt_boxes, gt_cat_ids, canvas, pool_size, device):
    """Returns P (float) for one image, or None if this image has no valid GT
    (can't compute a task-loss gradient) or the gradient degenerates to 0."""
    import torch
    import torch.nn.functional as Fnn

    if gt_boxes.shape[0] == 0:
        return None

    model = handle.model
    with torch.no_grad():
        feats_clean = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
        feats_adv = model.backbone(handle.normalize(x_adv.unsqueeze(0)))
    delta_vec = pool_and_concat(feats_adv, pool_size) - pool_and_concat(feats_clean, pool_size)

    grads = target_task_gradient_at_backbone(handle, x_clean, gt_boxes, gt_cat_ids, canvas, device)
    grad_vec = pool_and_concat(grads, pool_size)

    denom = delta_vec.norm().item() * grad_vec.norm().item()
    if denom < 1e-12:
        return None
    p = abs(torch.dot(delta_vec, grad_vec).item()) / denom
    return p


def bootstrap_mean(values, n_draws, seed, frac=0.8):
    """Subsample WITHOUT replacement (see e6_response_coupling.py's module
    docstring for why not classic with-replacement bootstrap for this kind of
    per-image statistic aggregation -- same convention reused here)."""
    rng = random.Random(seed)
    n = len(values)
    m = max(2, round(frac * n))
    point = sum(values) / n
    means = []
    for _ in range(n_draws):
        sample = rng.sample(values, k=m)
        means.append(sum(sample) / m)
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(int(0.975 * len(means)), len(means) - 1)]
    return point, lo, hi


def main() -> None:
    args = build_arg_parser().parse_args()

    import numpy as np
    import torch

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, load_noise
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle, get_spec

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    all_image_ids = manifest["image_ids"]
    if args.limit is not None:
        all_image_ids = all_image_ids[: args.limit]

    noise_dir = args.noise_dir or (PROJECT_DIR / "results" / "noise" / args.manifest.stem / args.attack)
    if not noise_dir.exists():
        logger.error(f"noise dir not found: {noise_dir}")
        sys.exit(1)

    used_image_ids = [iid for iid in all_image_ids if (noise_dir / f"{iid}.pt").exists()]
    n_missing = len(all_image_ids) - len(used_image_ids)
    if n_missing:
        logger.warning(f"{n_missing}/{len(all_image_ids)} images had no crafted noise -- skipped")
    logger.info(f"using up to N={len(used_image_ids)} images (per-target n may be smaller, GT-dependent)")

    if args.targets == ["all"]:
        target_specs = [s for s in MODEL_REGISTRY if s.role == "target"]
    else:
        target_specs = [get_spec(m) for m in args.targets]

    img_dir = args.data_dir / "val2017"
    device = args.device
    gt_index = build_gt_index(coco, used_image_ids)

    run_log = find_matching_run_log(args.runs_dir, args.attack, args.manifest.stem)
    asr_by_model = {}
    if run_log is not None:
        for row in run_log.get("results", {}).get("rows", []):
            if row["attack"] == args.attack:
                asr_by_model[row["model_name"]] = row.get("ASR")

    detail_rows = []
    p_by_target: dict[str, list[float]] = {}

    for spec in target_specs:
        logger.info(f"=== target {spec.name} (group={spec.group}) ===")
        handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)

        p_values = []
        n_no_gt = 0
        n_degenerate = 0
        for image_id in used_image_ids:
            gt_entries = gt_index[image_id]
            if not gt_entries:
                n_no_gt += 1
                continue
            canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
            gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
            noise = load_noise(noise_dir / f"{image_id}.pt")
            x_clean = canvas_img.to(device)
            x_adv = (canvas_img + noise).clamp(0.0, 255.0).to(device)

            p = compute_alignment_for_image(handle, x_clean, x_adv, gt_boxes, gt_cat_ids, args.canvas, args.pool_size, device)
            if p is None:
                n_degenerate += 1
                continue
            p_values.append(p)

        if n_no_gt or n_degenerate:
            logger.warning(f"  {spec.name}: {n_no_gt} no-GT skipped, {n_degenerate} degenerate (0-norm) skipped")
        if not p_values:
            logger.error(f"  {spec.name}: 0 usable images -- skipping this target entirely")
            del handle
            torch.cuda.empty_cache()
            continue

        p_by_target[spec.name] = p_values
        point, lo, hi = bootstrap_mean(p_values, args.n_bootstrap, args.seed, args.bootstrap_frac)
        logger.info(
            f"  {spec.name}: n={len(p_values)} mean_P={point:.4f} 95% CI=[{lo:.4f},{hi:.4f}] "
            f"ASR={asr_by_model.get(spec.name)}"
        )
        detail_rows.append(
            {
                "target": spec.name,
                "group": spec.group or "",
                "n_images": len(p_values),
                "mean_P": point,
                "P_ci_lo": lo,
                "P_ci_hi": hi,
                "ASR": asr_by_model.get(spec.name),
            }
        )
        del handle
        torch.cuda.empty_cache()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "group", "n_images", "mean_P", "P_ci_lo", "P_ci_hi", "ASR"])
        writer.writeheader()
        writer.writerows(detail_rows)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info("=== Matched-pair comparison (R50 vs Swin, same head) ===")
    for pair_name, r50_name, swin_name in MATCHED_PAIRS:
        if r50_name not in p_by_target or swin_name not in p_by_target:
            logger.warning(f"  {pair_name}: missing {r50_name} or {swin_name} -- skipped")
            continue
        p_r50 = sum(p_by_target[r50_name]) / len(p_by_target[r50_name])
        p_swin = sum(p_by_target[swin_name]) / len(p_by_target[swin_name])
        logger.info(f"  {pair_name}: P({r50_name})={p_r50:.4f} P({swin_name})={p_swin:.4f} delta={p_r50 - p_swin:+.4f}")

    target_rows = [r for r in detail_rows if r["ASR"] is not None]
    if len(target_rows) >= 3:
        p_vec = np.array([r["mean_P"] for r in target_rows])
        asr_vec = np.array([r["ASR"] for r in target_rows])
        corr = float(np.corrcoef(p_vec, asr_vec)[0, 1])
        logger.info(f"[diagnostic] Pearson corr(mean_P, ASR) across {len(target_rows)} targets = {corr:.3f}")


if __name__ == "__main__":
    main()
