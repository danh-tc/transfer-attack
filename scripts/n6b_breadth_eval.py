#!/usr/bin/env python
"""N6-B breadth test: evaluate ADDITIONAL target models against ALREADY-CRAFTED
noise (no recrafting -- noise is surrogate-only, independent of which targets
get evaluated against it). Frozen config: reuses whatever noise an existing
n6b_path_pilot.py run already produced under
results/noise/<manifest-stem>/n6b_{osfd_local,path_m3}/.

Added specifically to test whether N6-B's outsized dino_swin_l gain is a
DINO-decoder-specific effect or a Swin-backbone effect: dino_r50 (same DINO
decoder as dino_swin_l, ResNet-50 backbone instead of Swin) was added to
MODEL_REGISTRY for exactly this. Since it reuses dev_300's already-crafted
noise (N=296, the same run behind the CONFIRMED dino_swin_l +6.2/+6.5 result),
this breadth check comes for free at N=296 -- no craft cost at all, only
inference.

Example:
    python scripts/n6b_breadth_eval.py --manifest data/manifests/dev_300.json \
        --models dino_r50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_300.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", required=True, help="model names to evaluate, e.g. dino_r50")
    p.add_argument("--predictions-dir", type=Path, default=PROJECT_DIR / "results" / "_n6b_predictions")
    p.add_argument("--noise-dir", type=Path, default=None, help="default: results/noise/<manifest-stem>")
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n6b_breadth_eval.csv")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    import evaluate as evaluate_mod
    from n6b_path_pilot import paired_bootstrap_asr_delta

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import load_coco, load_manifest
    from transfer_attack.eval_metrics import compute_asr_for_image
    from transfer_attack.io_utils import get_logger, load_predictions
    from transfer_attack.models import get_spec

    from mmdet.utils import register_all_modules

    logger = get_logger()
    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    img_dir = args.data_dir / "val2017"
    noise_dir = args.noise_dir or (PROJECT_DIR / "results" / "noise" / args.manifest.stem)

    local_dir = noise_dir / "n6b_osfd_local"
    path_dir = noise_dir / "n6b_path_m3"
    used_image_ids = sorted(
        {int(p.stem) for p in local_dir.glob("*.pt")} & {int(p.stem) for p in path_dir.glob("*.pt")}
        & set(image_ids)
    )
    logger.info(f"reusing {len(used_image_ids)} already-crafted images from {noise_dir}")

    gt_cache = evaluate_mod.build_gt_cache(coco, used_image_ids, img_dir, args.canvas)
    _, gt_boxes_by_image_id, gt_cat_ids_by_image_id = gt_cache

    specs = [get_spec(m) for m in args.models]
    eval_args = SimpleNamespace(
        checkpoints_dir=args.checkpoints_dir,
        canvas=args.canvas,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        noise_dir=noise_dir,
        predictions_dir=args.predictions_dir,
        force_clean=False,
        device=args.device,
        attacks=["n6b_osfd_local", "n6b_path_m3"],
    )

    rows_by_tag = {"n6b_osfd_local": [], "n6b_path_m3": []}
    for spec in specs:
        rows = evaluate_mod.evaluate_one_model(spec, eval_args, coco, used_image_ids, img_dir, gt_cache, logger)
        for r in rows:
            if r["attack"] in rows_by_tag:
                rows_by_tag[r["attack"]].append(r)

    import csv

    comparison = []
    for spec in specs:
        clean_preds = load_predictions(args.predictions_dir / "clean" / f"{spec.name}.json")
        preds_local = load_predictions(args.predictions_dir / "n6b_osfd_local" / f"{spec.name}.json")
        preds_path = load_predictions(args.predictions_dir / "n6b_path_m3" / f"{spec.name}.json")
        per_image = []
        for image_id in used_image_ids:
            if image_id not in preds_local or image_id not in preds_path:
                continue
            gt_boxes = gt_boxes_by_image_id[image_id]
            gt_cat_ids = gt_cat_ids_by_image_id[image_id]
            evaded_l, cc = compute_asr_for_image(clean_preds[image_id], preds_local[image_id], gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr)
            evaded_p, _ = compute_asr_for_image(clean_preds[image_id], preds_path[image_id], gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr)
            per_image.append((evaded_l, evaded_p, cc))
        delta, ci_lo, ci_hi, same_side = paired_bootstrap_asr_delta(per_image, args.bootstrap_draws, args.seed)

        row_local = next(r for r in rows_by_tag["n6b_osfd_local"] if r["model_name"] == spec.name)
        row_path = next(r for r in rows_by_tag["n6b_path_m3"] if r["model_name"] == spec.name)
        comparison.append(
            {
                "model_name": spec.name,
                "group": spec.group or "",
                "n_images": len(used_image_ids),
                "ASR_osfd_local": row_local["ASR"],
                "ASR_path_m3": row_path["ASR"],
                "delta_path_vs_local": delta,
                "bootstrap_ci_lo": ci_lo,
                "bootstrap_ci_hi": ci_hi,
                "bootstrap_same_side_frac": same_side,
            }
        )
        logger.info(
            f"{spec.name:16s} local={row_local['ASR']:.1f} path={row_path['ASR']:.1f} "
            f"delta={delta:+.1f} 95% CI=[{ci_lo:+.1f}, {ci_hi:+.1f}] same_side={same_side:.3f}"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name", "group", "n_images", "ASR_osfd_local", "ASR_path_m3",
                "delta_path_vs_local", "bootstrap_ci_lo", "bootstrap_ci_hi", "bootstrap_same_side_frac",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
