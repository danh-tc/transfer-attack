#!/usr/bin/env python
"""Craft adversarial noise for one attack (osfd | mi_fgsm) against the
Faster R-CNN R50 surrogate, over a manifest of COCO val2017 image ids.

Example:
    python scripts/craft.py --attack osfd --limit 5 --steps 5 --out-dir results/smoke_noise/osfd
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["osfd", "mi_fgsm"], required=True)
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_300.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=None, help="default: constants.STEPS (200)")
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--mu", type=float, default=None)
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--theta", type=float, default=None)
    p.add_argument("--l-s", type=int, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--s-max", type=float, default=None)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--canvas", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="only process the first N images (smoke tests)")
    p.add_argument("--log-every", type=int, default=20)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

    from transfer_attack.attack import AttackConfig, craft_one_image
    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, save_noise
    from transfer_attack.models import build_model_handle, get_spec

    logger = get_logger()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    spec = get_spec("faster_rcnn_r50")
    handle = build_model_handle(spec, args.checkpoints_dir, device=args.device, coco=coco)
    logger.info(f"surrogate {spec.name} loaded on {args.device}")

    cfg_kwargs = {}
    for field in ("epsilon", "alpha", "steps", "mu", "k", "theta", "l_s", "rho", "s_max", "sigma", "canvas"):
        val = getattr(args, field)
        if val is not None:
            cfg_kwargs[field] = val
    cfg = AttackConfig(attack_type=args.attack, **cfg_kwargs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "config_used.json", "w") as f:
        json.dump({"attack": args.attack, **vars(cfg)}, f, indent=2)

    gt_index = build_gt_index(coco, image_ids)

    img_dir = args.data_dir / "val2017"
    losses_csv_path = args.out_dir / "losses.csv"
    n_done, n_skipped = 0, 0
    t0 = time.time()

    with open(losses_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "n_gt_boxes", "loss_first", "loss_last", "loss_min", "loss_max"])

        for i, image_id in enumerate(image_ids):
            gt_entries = gt_index[image_id]
            if not gt_entries:
                logger.warning(f"image_id={image_id}: 0 valid GT boxes after filtering -- skipping")
                n_skipped += 1
                continue

            canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, cfg.canvas)
            gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)

            noise, step_losses = craft_one_image(handle, canvas_img, gt_boxes, gt_cat_ids, cfg, device=args.device)
            save_noise(args.out_dir / f"{image_id}.pt", noise)
            writer.writerow([image_id, len(gt_entries), step_losses[0], step_losses[-1], min(step_losses), max(step_losses)])
            n_done += 1

            if (i + 1) % args.log_every == 0 or (i + 1) == len(image_ids):
                elapsed = time.time() - t0
                logger.info(
                    f"[{i + 1}/{len(image_ids)}] done={n_done} skipped={n_skipped} "
                    f"elapsed={elapsed:.1f}s last_image_id={image_id} "
                    f"loss[0]={step_losses[0]:.4f} loss[-1]={step_losses[-1]:.4f}"
                )

    logger.info(f"finished: {n_done} crafted, {n_skipped} skipped -> {args.out_dir}")


if __name__ == "__main__":
    main()
