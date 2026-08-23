#!/usr/bin/env python
"""Phase N4 method pilot v1: Difficulty-Aware Object [Feature] Weighting
(internal tag stays DOB-v1). v0 weighted the per-pixel STEP SIZE and turned
out to be a NO-GO by implementation, not by hypothesis: with epsilon=5,
alpha=1, w_min=0.5, every pixel saturates to +/-epsilon within <=10 steps
regardless of W, and 100 steps is more than enough for ~86% of pixels to
saturate in EVERY variant identically (verified: final noise saturation
fraction was 0.858/0.858/0.859 across osfd/dob_easy/dob_hard) -- W only
changed how fast a pixel got to the boundary, never where it ended up, so
with enough steps to converge, all three variants converge to nearly the
same noise. v0's core design flaw, not evidence against difficulty-aware
weighting itself.

v1 instead weights the OPTIMIZATION OBJECTIVE (osfd_loss), never epsilon,
alpha, steps, or the update rule -- eliminating the saturation confound
entirely, because now DOB-EASY and DOB-HARD are optimizing genuinely
DIFFERENT loss surfaces, not the same target at different speeds:

    L_OSFD(stage) = mean_{c,p} (target[c,p] - adv[c,p])^2
    L_DOB(stage)  = sum_p W_l(p) * D_p(stage) / sum_p W_l(p)
                    where D_p(stage) = mean_c (target[c,p] - adv[c,p])^2

W_l: the same canvas-level difficulty map W(x) as v0 (built once from
surrogate clean detections, same w_j formula/clipping/max-overlap-combine),
just resized (area interpolation, proper for downsampling) to each backbone
stage's own (H_l, W_l) instead of being multiplied into the update step.
epsilon/alpha/steps/k/RRB/the sign(g_mom) update rule are all IDENTICAL to
plain OSFD -- literally craft_one_image's own update loop, unmodified; only
osfd_loss is swapped for the weighted version.

Same three variants, same GO criterion as v0 (unchanged):
  osfd vs dob_easy_v1 vs dob_hard_v1 -- ASR_EASY - ASR_OSFD >= +5 on
  dino_swin_l AND ASR_EASY > ASR_HARD clearly. If v1 still fails this, the
  difficulty-aware weighting hypothesis can be killed with much more
  confidence than after v0, since the saturation confound is gone.

Example:
    python scripts/n5b_dob_v1_pilot.py --manifest data/manifests/dev_50.json --n-images 20
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from n5_dob_pilot import compute_weight_map  # reuse v0's weight-map construction unchanged

VARIANTS = [
    {"tag": "osfd", "direction": None},
    {"tag": "dob_easy_v1", "direction": "easy"},
    {"tag": "dob_hard_v1", "direction": "hard"},
]

BETA = 1.0
W_MIN = 0.5
W_MAX = 1.5


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--w-min", type=float, default=W_MIN)
    p.add_argument("--w-max", type=float, default=W_MAX)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n5b_dob_v1_pilot_summary.csv")
    p.add_argument("--run-tags", nargs="+", default=None)
    p.add_argument(
        "--models", nargs="+", default=["faster_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"],
    )
    return p


def weighted_stage_loss(target, pred, weight_map, eps: float = 1e-8):
    """target, pred: (1,C,H,W). weight_map: (H,W). Returns
    sum_p W(p)*mean_c((pred-target)^2)[p] / sum_p W(p)."""
    diff2 = (pred - target) ** 2  # (1,C,H,W)
    d_p = diff2.mean(dim=1).squeeze(0)  # (H,W) -- mean over channels
    return (weight_map * d_p).sum() / (weight_map.sum() + eps)


def osfd_loss_dob(feats_cln, feats_adv_2groups, k: float, weight_maps):
    total = feats_cln[0].new_zeros(())
    for stage_cln, stage_adv, w_l in zip(feats_cln, feats_adv_2groups, weight_maps):
        target = k * stage_cln
        for g in range(stage_adv.shape[0]):
            total = total + weighted_stage_loss(target, stage_adv[g : g + 1], w_l)
    return total


def craft_one_image_dob_v1(handle, x_clean, gt_boxes, gt_cat_ids, cfg, direction, beta, w_min, w_max, iou_thr, score_thr, device):
    import torch
    import torch.nn.functional as F

    from transfer_attack.augment import rrb_forward
    from transfer_attack.eval_metrics import predict_canvas

    model = handle.model
    x_clean = x_clean.to(device)
    gt_boxes = gt_boxes.to(device)
    gt_cat_ids = gt_cat_ids.to(device)

    if direction is not None:
        with torch.no_grad():
            surrogate_pred = predict_canvas(handle, x_clean, x_clean.shape[-1], device=device)
        W_canvas = compute_weight_map(
            x_clean.shape[-1], gt_boxes, gt_cat_ids, surrogate_pred, iou_thr, score_thr, beta, w_min, w_max, direction, device
        )
    else:
        W_canvas = torch.ones((x_clean.shape[-2], x_clean.shape[-1]), device=device)

    with torch.no_grad():
        feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))

    # W is static per image (built once from clean surrogate detections) -- resize once
    # per stage, reuse every step. 'area' interpolation is the correct choice when
    # downsampling (800 -> ~25-200 px), avoids the aliasing bilinear can introduce here.
    weight_maps = []
    for stage in feats_cln:
        h, w = stage.shape[-2], stage.shape[-1]
        resized = F.interpolate(W_canvas.view(1, 1, *W_canvas.shape), size=(h, w), mode="area").view(h, w)
        weight_maps.append(resized)

    noise = torch.randint_like(x_clean, low=-2, high=3).float()
    g_mom = torch.zeros_like(x_clean)

    for _ in range(cfg.steps):
        noise = noise.detach().requires_grad_(True)
        x_adv = torch.clamp(x_clean + noise, 0.0, 255.0)
        aug = rrb_forward(x_adv.unsqueeze(0), gt_boxes, cfg)
        feats_adv = model.backbone(handle.normalize(aug))
        loss = osfd_loss_dob(feats_cln, feats_adv, cfg.k, weight_maps)
        loss.backward()

        with torch.no_grad():
            g = noise.grad
            g_mom = cfg.mu * g_mom + g / g.abs().mean(dim=[0, 1, 2], keepdim=True)
            # UNCHANGED update rule -- no W here, unlike v0. Only the loss above differs.
            noise = torch.clamp(noise + cfg.alpha * torch.sign(g_mom), -cfg.epsilon, cfg.epsilon)

    return noise.detach()


def craft_and_evaluate_variant(variant: dict, args, coco, image_ids, gt_index, img_dir, logger) -> list[dict]:
    from types import SimpleNamespace

    import torch
    import evaluate as evaluate_mod

    from transfer_attack.attack import AttackConfig
    from transfer_attack.data import gt_to_canvas, load_canvas_image
    from transfer_attack.io_utils import save_noise, save_run_log
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle, get_spec

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tag = variant["tag"]
    cfg = AttackConfig(attack_type="osfd", k=3.0, use_rrb=True, steps=args.steps, canvas=args.canvas)

    noise_dir = PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"n5b_{tag}"
    noise_dir.mkdir(parents=True, exist_ok=True)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=args.device, coco=coco)
    logger.info(f"[{tag}] craft: direction={variant['direction']}")

    n_crafted, n_skipped = 0, 0
    t0 = time.time()
    used_image_ids = []
    for image_id in image_ids:
        if len(used_image_ids) >= args.n_images:
            break
        gt_entries = gt_index[image_id]
        if not gt_entries:
            n_skipped += 1
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, cfg.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        noise = craft_one_image_dob_v1(
            surrogate, canvas_img, gt_boxes, gt_cat_ids, cfg,
            variant["direction"], args.beta, args.w_min, args.w_max, args.iou_thr, args.score_thr, args.device,
        )
        save_noise(noise_dir / f"{image_id}.pt", noise)
        used_image_ids.append(image_id)
        n_crafted += 1
        if n_crafted % args.log_every == 0:
            logger.info(f"[{tag}] [{n_crafted}/{args.n_images}] elapsed={time.time() - t0:.1f}s")
    craft_elapsed = time.time() - t0
    logger.info(f"[{tag}] craft finished: {n_crafted} crafted, {n_skipped} skipped -> {noise_dir}")
    del surrogate
    torch.cuda.empty_cache()

    eval_args = SimpleNamespace(
        checkpoints_dir=args.checkpoints_dir,
        canvas=args.canvas,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        noise_dir=PROJECT_DIR / "results" / "noise" / args.manifest.stem,
        predictions_dir=PROJECT_DIR / "results" / "_n5b_predictions",
        force_clean=False,
        device=args.device,
        attacks=[f"n5b_{tag}"],
    )

    specs = MODEL_REGISTRY
    if args.models:
        by_name = {s.name: s for s in MODEL_REGISTRY}
        specs = [by_name[m] for m in args.models]

    t_eval0 = time.time()
    gt_cache = evaluate_mod.build_gt_cache(coco, used_image_ids, img_dir, args.canvas)
    rows = []
    for spec in specs:
        rows.extend(evaluate_mod.evaluate_one_model(spec, eval_args, coco, used_image_ids, img_dir, gt_cache, logger))
    eval_elapsed = time.time() - t_eval0
    logger.info(f"[{tag}] eval finished {len(specs)} models in {eval_elapsed:.1f}s")

    run_log_path = save_run_log(
        args.runs_dir,
        "run_attack",
        f"osfd_n5b_{tag}_{args.manifest.stem}_n{args.n_images}",
        {
            "attack": f"n5b_{tag}",
            "manifest": str(args.manifest),
            "seed": args.seed,
            "n_images": args.n_images,
            "score_thr": args.score_thr,
            "iou_thr": args.iou_thr,
            "config": {**vars(cfg), **variant, "beta": args.beta, "w_min": args.w_min, "w_max": args.w_max},
            "results": {
                "n_images_used": len(used_image_ids),
                "n_crafted": n_crafted,
                "n_skipped": n_skipped,
                "craft_elapsed_sec": round(craft_elapsed, 1),
                "eval_elapsed_sec": round(eval_elapsed, 1),
                "rows": rows,
            },
        },
    )
    logger.info(f"[{tag}] run log written -> {run_log_path}")
    return [r for r in rows if r["attack"] == f"n5b_{tag}"]


def main() -> None:
    args = build_arg_parser().parse_args()

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"

    run_variants = [v for v in VARIANTS if not args.run_tags or v["tag"] in args.run_tags]

    all_rows: dict[str, list[dict]] = {}
    for variant in run_variants:
        all_rows[variant["tag"]] = craft_and_evaluate_variant(variant, args, coco, image_ids, gt_index, img_dir, logger)

    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    import csv

    fieldnames = ["model_name", "group", "ASR_osfd", "ASR_dob_easy_v1", "ASR_dob_hard_v1",
                  "mAP_drop_osfd", "mAP_drop_dob_easy_v1", "mAP_drop_dob_hard_v1",
                  "delta_easy_vs_osfd", "delta_easy_vs_hard"]
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for tag in ("osfd", "dob_easy_v1", "dob_hard_v1"):
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
            row[f"mAP_drop_{tag}"] = r.get("mAP_drop_pct")
        if row.get("ASR_osfd") is not None and row.get("ASR_dob_easy_v1") is not None:
            row["delta_easy_vs_osfd"] = row["ASR_dob_easy_v1"] - row["ASR_osfd"]
        if row.get("ASR_dob_hard_v1") is not None and row.get("ASR_dob_easy_v1") is not None:
            row["delta_easy_vs_hard"] = row["ASR_dob_easy_v1"] - row["ASR_dob_hard_v1"]
        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info("=== N5b DOB-v1 pilot: ASR (%) osfd -> dob_easy_v1 -> dob_hard_v1 ===")
    for r in comparison:
        logger.info(
            f"{r['model_name']:20s} {r['group'] or '-':4s} "
            f"osfd={r.get('ASR_osfd', float('nan')):.1f} easy={r.get('ASR_dob_easy_v1', float('nan')):.1f} "
            f"hard={r.get('ASR_dob_hard_v1', float('nan')):.1f}  "
            f"(easy-osfd={r.get('delta_easy_vs_osfd', float('nan')):+.1f}, "
            f"easy-hard={r.get('delta_easy_vs_hard', float('nan')):+.1f})"
        )


if __name__ == "__main__":
    main()
