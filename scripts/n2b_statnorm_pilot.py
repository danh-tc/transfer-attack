#!/usr/bin/env python
"""Phase N2-B pilot: does removing per-channel spatial mean/std from backbone
features before computing OSFD's suppress/amplify loss push the attack
toward structural feature changes that survive an architecture change better?

Motivation (careful phrasing -- E1 is motivation, not proof of causality):
OSFD may exploit feature magnitude and channel-statistic patterns that are
specific to the surrogate representation; removing affine/statistical
information before feature distortion may force the attack toward structural
feature changes that survive architecture changes better. E1 found the
surrogate (BatchNorm ResNet-50) damages differently than the Swin targets
(LayerNorm) -- that difference is the MOTIVATION for trying this, not
evidence it's caused by BN-vs-LN specifically.

Mechanism: standardize every backbone stage's feature map per (image,
channel) over spatial dims before feeding it to the UNCHANGED osfd_loss --

    F_hat[c,h,w] = (F[c,h,w] - mean_{h,w}(F[c])) / (std_{h,w}(F[c]) + eps)

Applied identically to the clean reference and the RRB-augmented adv
features, every step. k, RRB, epsilon/alpha/steps all unchanged -- the ONLY
difference from plain OSFD is whether osfd_loss sees raw or standardized
features.

Two variants this round (HYBRID parked -- only run if STAT-NORM shows signal):
  osfd      -- baseline, raw features (current OSFD exactly).
  statnorm  -- per-channel spatially-standardized features.

GO criterion (decided before running): >= +5 ASR points on dino_swin_l OR
mask_rcnn_swin_t, gain on >= 2/3 hard targets, surrogate not collapsing by
more than 10 points. The pattern that matters most is NOT white-box: if
surrogate ASR drops slightly while dino/mask ASR rises clearly, that is the
signature of reduced surrogate-specific representation dependence.

Example:
    python scripts/n2b_statnorm_pilot.py --manifest data/manifests/dev_50.json --n-images 20
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

VARIANTS = [
    {"tag": "osfd", "standardize": False},
    {"tag": "statnorm", "standardize": True},
]


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
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n2b_statnorm_pilot_summary.csv")
    p.add_argument("--run-tags", nargs="+", default=None)
    p.add_argument(
        "--models", nargs="+", default=None,
        help="default: surrogate + the 3 hard targets (faster_rcnn_r50 yolox_l mask_rcnn_swin_t dino_swin_l)",
    )
    return p


def standardize_feat(f, eps: float = 1e-5):
    """f: (N,C,H,W). Standardize each (n,c) map independently over H,W."""
    mu = f.mean(dim=(-2, -1), keepdim=True)
    sigma = f.std(dim=(-2, -1), keepdim=True)
    return (f - mu) / (sigma + eps)


def craft_one_image_statnorm(handle, x_clean, gt_boxes, cfg, standardize: bool, device: str):
    import torch

    from transfer_attack.augment import rrb_forward
    from transfer_attack.losses import osfd_loss

    model = handle.model
    x_clean = x_clean.to(device)
    gt_boxes = gt_boxes.to(device)

    with torch.no_grad():
        feats_cln_raw = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
        feats_cln = tuple(standardize_feat(f) for f in feats_cln_raw) if standardize else feats_cln_raw

    noise = torch.randint_like(x_clean, low=-2, high=3).float()
    g_mom = torch.zeros_like(x_clean)

    for _ in range(cfg.steps):
        noise = noise.detach().requires_grad_(True)
        x_adv = torch.clamp(x_clean + noise, 0.0, 255.0)
        aug = rrb_forward(x_adv.unsqueeze(0), gt_boxes, cfg)
        feats_adv_raw = model.backbone(handle.normalize(aug))
        feats_adv = tuple(standardize_feat(f) for f in feats_adv_raw) if standardize else feats_adv_raw

        loss = osfd_loss(feats_cln, feats_adv, cfg.k)
        loss.backward()

        with torch.no_grad():
            g = noise.grad
            g_mom = cfg.mu * g_mom + g / g.abs().mean(dim=[0, 1, 2], keepdim=True)
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

    noise_dir = PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"n2b_{tag}"
    noise_dir.mkdir(parents=True, exist_ok=True)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=args.device, coco=coco)
    logger.info(f"[{tag}] craft: standardize={variant['standardize']}")

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
        gt_boxes, _ = gt_to_canvas(gt_entries, scale)
        noise = craft_one_image_statnorm(surrogate, canvas_img, gt_boxes, cfg, variant["standardize"], args.device)
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
        predictions_dir=PROJECT_DIR / "results" / "_n2b_predictions",
        force_clean=False,
        device=args.device,
        attacks=[f"n2b_{tag}"],
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
        f"osfd_n2b_{tag}_{args.manifest.stem}_n{args.n_images}",
        {
            "attack": f"n2b_{tag}",
            "manifest": str(args.manifest),
            "seed": args.seed,
            "n_images": args.n_images,
            "score_thr": args.score_thr,
            "iou_thr": args.iou_thr,
            "config": {**vars(cfg), **variant},
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
    return [r for r in rows if r["attack"] == f"n2b_{tag}"]


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

    if args.models is None:
        args.models = ["faster_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"]

    run_variants = [v for v in VARIANTS if not args.run_tags or v["tag"] in args.run_tags]

    all_rows: dict[str, list[dict]] = {}
    for variant in run_variants:
        all_rows[variant["tag"]] = craft_and_evaluate_variant(variant, args, coco, image_ids, gt_index, img_dir, logger)

    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    import csv

    fieldnames = ["model_name", "group", "ASR_osfd", "ASR_statnorm", "mAP_drop_osfd", "mAP_drop_statnorm", "delta_ASR"]
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for tag in ("osfd", "statnorm"):
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
            row[f"mAP_drop_{tag}"] = r.get("mAP_drop_pct")
        if row.get("ASR_osfd") is not None and row.get("ASR_statnorm") is not None:
            row["delta_ASR"] = row["ASR_statnorm"] - row["ASR_osfd"]
        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info("=== N2-B pilot: ASR (%) osfd -> statnorm ===")
    for r in comparison:
        logger.info(
            f"{r['model_name']:20s} {r['group'] or '-':4s} "
            f"osfd={r.get('ASR_osfd', float('nan')):.1f} statnorm={r.get('ASR_statnorm', float('nan')):.1f}  "
            f"(delta={r.get('delta_ASR', float('nan')):+.1f})"
        )


if __name__ == "__main__":
    main()
