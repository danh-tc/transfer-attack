#!/usr/bin/env python
"""End-to-end for ONE attack: craft adversarial noise against the surrogate,
then immediately evaluate it (clean vs adversarial) across all 7 registered
models -- one command, one combined run-log JSON (full config + full metrics)
written to runs/ per invocation.

Run once per attack to get one comparable log each:
    python scripts/run_attack.py --attack osfd --manifest data/manifests/dev_50.json --steps 100
    python scripts/run_attack.py --attack mi_fgsm --manifest data/manifests/dev_50.json --steps 100
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["osfd", "mi_fgsm"], required=True)
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_300.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--noise-dir", type=Path, default=None, help="default: results/noise/<manifest-stem>/<attack>/")
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
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="only process the first N images (smoke tests)")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch
    import evaluate as evaluate_mod

    from transfer_attack.attack import AttackConfig, craft_one_image
    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, save_noise, save_run_log
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle, get_spec

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

    noise_root = args.noise_dir.parent if args.noise_dir else PROJECT_DIR / "results" / "noise" / args.manifest.stem
    attack_noise_dir = args.noise_dir or (noise_root / args.attack)

    # ---- Phase 1: craft (surrogate faster_rcnn_r50, same as craft.py) ----
    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=args.device, coco=coco)
    logger.info(f"[craft] surrogate {surrogate_spec.name} loaded on {args.device}")

    cfg_kwargs = {}
    for field in ("epsilon", "alpha", "steps", "mu", "k", "theta", "l_s", "rho", "s_max", "sigma"):
        val = getattr(args, field)
        if val is not None:
            cfg_kwargs[field] = val
    cfg = AttackConfig(attack_type=args.attack, canvas=args.canvas, **cfg_kwargs)

    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"
    attack_noise_dir.mkdir(parents=True, exist_ok=True)

    n_crafted, n_skipped = 0, 0
    t_craft0 = time.time()
    step_losses = [0.0]
    for i, image_id in enumerate(image_ids):
        gt_entries = gt_index[image_id]
        if not gt_entries:
            n_skipped += 1
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, cfg.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        noise, step_losses = craft_one_image(surrogate, canvas_img, gt_boxes, gt_cat_ids, cfg, device=args.device)
        save_noise(attack_noise_dir / f"{image_id}.pt", noise)
        n_crafted += 1
        if (i + 1) % args.log_every == 0 or (i + 1) == len(image_ids):
            logger.info(
                f"[craft] [{i + 1}/{len(image_ids)}] crafted={n_crafted} skipped={n_skipped} "
                f"elapsed={time.time() - t_craft0:.1f}s loss[-1]={step_losses[-1]:.4f}"
            )
    craft_elapsed = time.time() - t_craft0
    logger.info(f"[craft] finished: {n_crafted} crafted, {n_skipped} skipped -> {attack_noise_dir}")

    del surrogate
    torch.cuda.empty_cache()

    # ---- Phase 2: evaluate (reuses evaluate.py's per-model clean/adv/ASR/mAP logic) ----
    run_id = time.strftime("%Y%m%dT%H%M%S")
    eval_args = SimpleNamespace(
        checkpoints_dir=args.checkpoints_dir,
        canvas=args.canvas,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        noise_dir=noise_root,
        predictions_dir=PROJECT_DIR / "results" / "_run_attack_predictions" / f"{args.attack}_{run_id}",
        force_clean=False,
        device=args.device,
        attacks=[args.attack],
    )

    t_eval0 = time.time()
    gt_cache = evaluate_mod.build_gt_cache(coco, image_ids, img_dir, args.canvas)
    rows = []
    for spec in MODEL_REGISTRY:
        rows.extend(evaluate_mod.evaluate_one_model(spec, eval_args, coco, image_ids, img_dir, gt_cache, logger))
    eval_elapsed = time.time() - t_eval0
    logger.info(f"[evaluate] finished {len(MODEL_REGISTRY)} models in {eval_elapsed:.1f}s")

    run_log_path = save_run_log(
        args.runs_dir,
        "run_attack",
        f"{args.attack}_{args.manifest.stem}",
        {
            "attack": args.attack,
            "manifest": str(args.manifest.relative_to(PROJECT_DIR)) if args.manifest.is_absolute() else str(args.manifest),
            "seed": args.seed,
            "limit": args.limit,
            "score_thr": args.score_thr,
            "iou_thr": args.iou_thr,
            "config": vars(cfg),
            "results": {
                "n_images_in_manifest": len(image_ids),
                "n_crafted": n_crafted,
                "n_skipped": n_skipped,
                "craft_elapsed_sec": round(craft_elapsed, 1),
                "eval_elapsed_sec": round(eval_elapsed, 1),
                "rows": rows,
            },
        },
    )
    logger.info(f"run log written -> {run_log_path}")


if __name__ == "__main__":
    main()
