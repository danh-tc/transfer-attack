#!/usr/bin/env python
"""N6-B mechanism-proof diagnostic (post held-out val_100 MIXED result --
DINO gain confirmed, mask_rcnn_swin_t gain not statistically significant at
N=98 -- see RESEARCH.md N6-B held-out section). Tests the actual mechanism
claim directly instead of just re-measuring ASR: does the path-averaged
surrogate gradient point in a direction that better matches each TARGET
model's own true feature-distortion ascent direction than the instantaneous
(local) surrogate gradient does? If yes, path-averaging finds a MORE
TRANSFERABLE direction (the story N6-B needs). If the alignment gain is ~0
even on dino_swin_l (where the ASR gain is robust and confirmed 3x), the ASR
gain has some other explanation (bigger effective step / less saturation,
not "a better direction") -- an important result either way, not just a
confirmation-seeking diagnostic.

No recrafting: reuses already-crafted noise_local / noise_path (the FINAL,
step-100 delta) from an existing n6b_path_pilot.py run on the surrogate
(default: dev_300, N=296, the richest already-crafted set). For each image
and each target model:

  g_local^sur  = grad_xi [osfd_loss on SURROGATE backbone] at
                 x_clean + noise_local + xi (xi=0), ONE fresh RRB draw --
                 matches the real per-step crafting recipe (n6b_path_pilot.py
                 craft_paired_local_path's m=M / local term).
  g_path^sur   = mean_{m=1..M=3} grad_xi [osfd_loss on SURROGATE backbone] at
                 x_clean + lambda_m * noise_path + xi (xi=0), fresh RRB draw
                 per m -- same recipe as n6b_path_pilot.py's g_path,t.
  g_target_at(x) = grad_xi [osfd_loss on the TARGET's OWN backbone] at x+xi
                 (xi=0), NO RRB -- isolates "does this direction increase
                 distortion in the target's raw feature space", independent
                 of the surrogate-side augmentation-robustness trick (RRB is
                 specifically a surrogate-side transfer mechanism, not part
                 of what we mean by "the target's true gradient").

Metric per (image, target): cos(g_local^sur, g_target_at(x_clean+noise_local))
vs cos(g_path^sur, g_target_at(x_clean+noise_path)) -- each surrogate
trajectory compared against the target's true gradient AT THE STATE THAT
TRAJECTORY ACTUALLY REACHED (both trajectories diverge after step 1; this is
the same "evaluate at your own current state" convention n6b_path_pilot.py's
own per-step self-diagnostic already used). Averaged per target, with paired
image-cluster bootstrap 95% CI on the delta (same convention as
n6b_path_pilot.py / N4).

The surrogate's g_local/g_path do NOT depend on the target, so they are
computed ONCE per image (cached across all --targets in one process run) --
only the target-side gradient is recomputed per target model.

GO criterion (pre-registered, decided before running): mean_cos_path -
mean_cos_local >= +0.03 with 95% CI not crossing 0, on dino_swin_l and/or
mask_rcnn_swin_t, counts as direct mechanistic evidence for "path finds a
more transferable direction". A near-zero or CI-crossing-0 delta on
dino_swin_l specifically (where the ASR gain is the strongest and most
robust finding of the whole project) would mean the alignment story does NOT
explain the ASR gain, and the honest conclusion is "some other mechanism
(effective step size / saturation dynamics) drives the DINO gain" -- this
must be reported as a real finding, not suppressed or re-run with a
different threshold.

Example:
    python scripts/n6b_alignment_diagnostic.py --manifest data/manifests/dev_300.json \
        --n-images 50 --targets yolox_l mask_rcnn_swin_t dino_swin_l deformable_detr
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

M_LAMBDA = 3  # matches N6-B v0's pre-registered M -- do not sweep here either


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_300.json")
    p.add_argument(
        "--noise-dir", type=Path, default=None,
        help="default: results/noise/<manifest-stem>/n6b_{osfd_local,path_m3}/ (from an n6b_path_pilot.py run)",
    )
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-images", type=int, default=50)
    p.add_argument("--targets", nargs="+", default=["yolox_l", "mask_rcnn_swin_t", "dino_swin_l", "deformable_detr"])
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n6b_alignment_diagnostic.csv")
    p.add_argument("--log-every", type=int, default=10)
    return p


def paired_bootstrap_mean_delta(per_image_local: list[float], per_image_path: list[float], n_draws: int, seed: int):
    """per_image_local/path: parallel lists, one cosine value per image
    (image-cluster unit). Returns (delta_point, ci_low, ci_high, same_side_frac)."""
    rng = random.Random(seed)
    n = len(per_image_local)
    idx = list(range(n))
    delta_point = sum(per_image_path) / n - sum(per_image_local) / n

    deltas = []
    for _ in range(n_draws):
        sample = rng.choices(idx, k=n)
        s_local = sum(per_image_local[i] for i in sample) / n
        s_path = sum(per_image_path[i] for i in sample) / n
        deltas.append(s_path - s_local)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]
    if delta_point >= 0:
        same_side = sum(1 for d in deltas if d >= 0) / len(deltas)
    else:
        same_side = sum(1 for d in deltas if d < 0) / len(deltas)
    return delta_point, lo, hi, same_side


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch
    import torch.nn.functional as F

    from transfer_attack.attack import AttackConfig
    from transfer_attack.augment import rrb_forward
    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, load_noise
    from transfer_attack.losses import osfd_loss
    from transfer_attack.models import get_spec, build_model_handle

    logger = get_logger()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"

    noise_dir = args.noise_dir or (PROJECT_DIR / "results" / "noise" / args.manifest.stem)
    noise_local_dir = noise_dir / "n6b_osfd_local"
    noise_path_dir = noise_dir / "n6b_path_m3"
    if not noise_local_dir.exists() or not noise_path_dir.exists():
        logger.error(
            f"expected crafted noise under {noise_local_dir} and {noise_path_dir} "
            f"(run scripts/n6b_path_pilot.py --manifest {args.manifest} first)"
        )
        sys.exit(1)

    cfg = AttackConfig(attack_type="osfd", k=3.0, use_rrb=True, steps=100, canvas=args.canvas)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=device, coco=coco)

    # ---- Pass 1: collect per-image (x_clean, gt_boxes, noise_local, noise_path,
    # g_local^sur, g_path^sur) -- surrogate-side only, independent of target.
    used_image_ids = []
    x_clean_by_id = {}
    g_local_sur_by_id = {}
    g_path_sur_by_id = {}
    x_local_by_id = {}
    x_path_by_id = {}

    t0 = time.time()
    n_used, n_skipped = 0, 0
    for image_id in image_ids:
        if len(used_image_ids) >= args.n_images:
            break
        gt_entries = gt_index[image_id]
        if not gt_entries:
            n_skipped += 1
            continue
        local_path = noise_local_dir / f"{image_id}.pt"
        path_path = noise_path_dir / f"{image_id}.pt"
        if not local_path.exists() or not path_path.exists():
            n_skipped += 1
            continue

        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, cfg.canvas)
        gt_boxes, _ = gt_to_canvas(gt_entries, scale)
        x_clean = canvas_img.to(device)
        gt_boxes = gt_boxes.to(device)

        noise_local = load_noise(local_path).to(device)
        noise_path = load_noise(path_path).to(device)
        x_local = torch.clamp(x_clean + noise_local, 0.0, 255.0)
        x_path = torch.clamp(x_clean + noise_path, 0.0, 255.0)

        with torch.no_grad():
            feats_cln_sur = surrogate.model.backbone(surrogate.normalize(x_clean.unsqueeze(0)))

        random.seed(args.seed + image_id)
        torch.manual_seed(args.seed + image_id)

        xi = torch.zeros_like(x_clean, requires_grad=True)
        x_eval = torch.clamp(x_local + xi, 0.0, 255.0)
        aug = rrb_forward(x_eval.unsqueeze(0), gt_boxes, cfg)
        feats_adv = surrogate.model.backbone(surrogate.normalize(aug))
        loss_local = osfd_loss(feats_cln_sur, feats_adv, cfg.k)
        loss_local.backward()
        g_local = xi.grad.detach().clone()

        grads_path = []
        for m in range(1, M_LAMBDA + 1):
            lam = m / M_LAMBDA
            xi2 = torch.zeros_like(x_clean, requires_grad=True)
            x_lam = torch.clamp(x_clean + lam * noise_path + xi2, 0.0, 255.0)
            aug2 = rrb_forward(x_lam.unsqueeze(0), gt_boxes, cfg)
            feats_adv2 = surrogate.model.backbone(surrogate.normalize(aug2))
            loss_m = osfd_loss(feats_cln_sur, feats_adv2, cfg.k)
            loss_m.backward()
            grads_path.append(xi2.grad.detach().clone())
        g_path = torch.stack(grads_path, dim=0).mean(dim=0)

        used_image_ids.append(image_id)
        x_clean_by_id[image_id] = x_clean
        g_local_sur_by_id[image_id] = g_local.detach()
        g_path_sur_by_id[image_id] = g_path.detach()
        x_local_by_id[image_id] = x_local.detach()
        x_path_by_id[image_id] = x_path.detach()
        n_used += 1
        if n_used % args.log_every == 0:
            logger.info(f"[surrogate pass] {n_used}/{args.n_images} elapsed={time.time() - t0:.1f}s")

    logger.info(f"surrogate pass done: {n_used} used, {n_skipped} skipped, elapsed={time.time() - t0:.1f}s")
    del surrogate
    torch.cuda.empty_cache()

    # ---- Pass 2: per target model, compute g_target at x_local / x_path (no RRB).
    import csv

    all_rows = []
    for target_name in args.targets:
        target_spec = get_spec(target_name)
        target = build_model_handle(target_spec, args.checkpoints_dir, device=device, coco=coco)
        logger.info(f"=== target: {target_name} (group={target_spec.group}) ===")

        cos_local_list = []
        cos_path_list = []
        t1 = time.time()
        for i, image_id in enumerate(used_image_ids, start=1):
            x_clean = x_clean_by_id[image_id]
            with torch.no_grad():
                feats_cln_t = target.model.backbone(target.normalize(x_clean.unsqueeze(0)))

            def target_grad(x_state):
                xi = torch.zeros_like(x_clean, requires_grad=True)
                x_eval = torch.clamp(x_state + xi, 0.0, 255.0)
                feats_adv_t = target.model.backbone(target.normalize(x_eval.unsqueeze(0)))
                loss_t = osfd_loss(feats_cln_t, feats_adv_t, cfg.k)
                loss_t.backward()
                return xi.grad.detach().clone()

            g_target_local = target_grad(x_local_by_id[image_id])
            g_target_path = target_grad(x_path_by_id[image_id])

            cos_local = F.cosine_similarity(
                g_local_sur_by_id[image_id].flatten().unsqueeze(0), g_target_local.flatten().unsqueeze(0)
            ).item()
            cos_path = F.cosine_similarity(
                g_path_sur_by_id[image_id].flatten().unsqueeze(0), g_target_path.flatten().unsqueeze(0)
            ).item()
            cos_local_list.append(cos_local)
            cos_path_list.append(cos_path)

            if i % args.log_every == 0:
                logger.info(f"  [{target_name}] {i}/{len(used_image_ids)} elapsed={time.time() - t1:.1f}s")

        delta, ci_lo, ci_hi, same_side = paired_bootstrap_mean_delta(
            cos_local_list, cos_path_list, args.bootstrap_draws, args.seed
        )
        mean_local = sum(cos_local_list) / len(cos_local_list)
        mean_path = sum(cos_path_list) / len(cos_path_list)
        logger.info(
            f"[{target_name}] mean_cos_local={mean_local:.4f} mean_cos_path={mean_path:.4f} "
            f"delta={delta:+.4f} 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] same_side_frac={same_side:.3f}"
        )
        all_rows.append(
            {
                "target": target_name,
                "group": target_spec.group or "",
                "n_images": len(used_image_ids),
                "mean_cos_local": mean_local,
                "mean_cos_path": mean_path,
                "delta_path_vs_local": delta,
                "bootstrap_ci_lo": ci_lo,
                "bootstrap_ci_hi": ci_hi,
                "bootstrap_same_side_frac": same_side,
            }
        )

        del target
        torch.cuda.empty_cache()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target", "group", "n_images", "mean_cos_local", "mean_cos_path",
                "delta_path_vs_local", "bootstrap_ci_lo", "bootstrap_ci_hi", "bootstrap_same_side_frac",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info(f"wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
