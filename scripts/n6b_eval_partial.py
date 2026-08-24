#!/usr/bin/env python
"""Early-peek evaluation for an in-progress N6-B v0 dev_300 craft run: reuses
whatever noise n6b_path_pilot.py has ALREADY written to disk (both
n6b_osfd_local and n6b_path_m3 must exist for an image_id to be included),
without touching the still-running craft process. Same eval + paired
bootstrap logic as n6b_path_pilot.py's tail end, factored out standalone so
it can run concurrently on a fresh set of model handles (safe: craft only
ever APPENDS new noise files for images beyond what's already on disk, never
rewrites existing ones, and this script only reads).

Purely a look-ahead sanity check -- NOT a substitute for the full dev_300 run
(different, smaller image count -> different sample composition, so absolute
numbers here should not be quoted as the confirmed dev_300 result).

Example:
    python scripts/n6b_eval_partial.py --manifest data/manifests/dev_300.json
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

sys.path.insert(0, str(SCRIPTS_DIR))
from n6b_path_pilot import paired_bootstrap_asr_delta  # noqa: E402


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
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n6b_path_pilot_partial_peek.csv")
    p.add_argument(
        "--models", nargs="+",
        default=["faster_rcnn_r50", "fcos_r50", "deformable_detr", "yolov3_d53", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"],
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch
    import evaluate as evaluate_mod

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import load_coco, load_manifest
    from transfer_attack.eval_metrics import compute_asr_for_image
    from transfer_attack.io_utils import get_logger, load_predictions
    from transfer_attack.models import MODEL_REGISTRY

    logger = get_logger()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    manifest_image_ids = manifest["image_ids"]
    img_dir = args.data_dir / "val2017"

    noise_root = PROJECT_DIR / "results" / "noise" / args.manifest.stem
    local_dir = noise_root / "n6b_osfd_local"
    path_dir = noise_root / "n6b_path_m3"
    local_ids = {int(p.stem) for p in local_dir.glob("*.pt")}
    path_ids = {int(p.stem) for p in path_dir.glob("*.pt")}
    used_image_ids = [i for i in manifest_image_ids if i in local_ids and i in path_ids]
    logger.info(f"found {len(used_image_ids)} images with BOTH osfd_local and path_m3 noise on disk so far")
    if not used_image_ids:
        logger.error("no images ready yet -- craft hasn't written any complete pairs")
        return

    from types import SimpleNamespace

    predictions_dir = PROJECT_DIR / "results" / "_n6b_partial_predictions"
    specs = MODEL_REGISTRY
    if args.models:
        by_name = {s.name: s for s in MODEL_REGISTRY}
        specs = [by_name[m] for m in args.models]
    gt_cache = evaluate_mod.build_gt_cache(coco, used_image_ids, img_dir, args.canvas)
    _, gt_boxes_by_image_id, gt_cat_ids_by_image_id = gt_cache

    all_rows = {}
    for tag in ("osfd_local", "path_m3"):
        eval_args = SimpleNamespace(
            checkpoints_dir=args.checkpoints_dir,
            canvas=args.canvas,
            score_thr=args.score_thr,
            iou_thr=args.iou_thr,
            noise_dir=noise_root,
            predictions_dir=predictions_dir,
            force_clean=False,
            device=args.device,
            attacks=[f"n6b_{tag}"],
        )
        rows = []
        for spec in specs:
            rows.extend(evaluate_mod.evaluate_one_model(spec, eval_args, coco, used_image_ids, img_dir, gt_cache, logger))
            torch.cuda.empty_cache()
        all_rows[tag] = [r for r in rows if r["attack"] == f"n6b_{tag}"]

    bootstrap_by_model = {}
    for spec in specs:
        clean_preds = load_predictions(predictions_dir / "clean" / f"{spec.name}.json")
        preds_local = load_predictions(predictions_dir / "n6b_osfd_local" / f"{spec.name}.json")
        preds_path = load_predictions(predictions_dir / "n6b_path_m3" / f"{spec.name}.json")
        per_image = []
        for image_id in used_image_ids:
            if image_id not in preds_local or image_id not in preds_path:
                continue
            gt_boxes = gt_boxes_by_image_id[image_id]
            gt_cat_ids = gt_cat_ids_by_image_id[image_id]
            evaded_l, cc = compute_asr_for_image(clean_preds[image_id], preds_local[image_id], gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr)
            evaded_p, _ = compute_asr_for_image(clean_preds[image_id], preds_path[image_id], gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr)
            per_image.append((evaded_l, evaded_p, cc))
        delta_point, ci_lo, ci_hi, same_side = paired_bootstrap_asr_delta(per_image, args.bootstrap_draws, args.seed)
        bootstrap_by_model[spec.name] = {"delta_point": delta_point, "ci_lo": ci_lo, "ci_hi": ci_hi, "same_side_frac": same_side}

    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    import csv

    fieldnames = ["model_name", "group", "n_images", "ASR_osfd_local", "ASR_path_m3",
                  "mAP_drop_osfd_local", "mAP_drop_path_m3", "delta_path_vs_local",
                  "bootstrap_ci_lo", "bootstrap_ci_hi", "bootstrap_same_side_frac"]
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group, "n_images": len(used_image_ids)}
        for tag in ("osfd_local", "path_m3"):
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
            row[f"mAP_drop_{tag}"] = r.get("mAP_drop_pct")
        if row.get("ASR_osfd_local") is not None and row.get("ASR_path_m3") is not None:
            row["delta_path_vs_local"] = row["ASR_path_m3"] - row["ASR_osfd_local"]
        bs = bootstrap_by_model.get(model_name, {})
        row["bootstrap_ci_lo"] = bs.get("ci_lo")
        row["bootstrap_ci_hi"] = bs.get("ci_hi")
        row["bootstrap_same_side_frac"] = bs.get("same_side_frac")
        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info(f"=== N6-B partial peek ({len(used_image_ids)} images) : ASR (%) osfd_local -> path_m3 ===")
    for r in comparison:
        logger.info(
            f"{r['model_name']:20s} {r['group'] or '-':4s} "
            f"local={r.get('ASR_osfd_local', float('nan')):.1f} path={r.get('ASR_path_m3', float('nan')):.1f} "
            f"(path-local={r.get('delta_path_vs_local', float('nan')):+.1f}, "
            f"95% CI=[{r.get('bootstrap_ci_lo', float('nan')):+.1f}, {r.get('bootstrap_ci_hi', float('nan')):+.1f}], "
            f"same_side_frac={r.get('bootstrap_same_side_frac', float('nan')):.3f})"
        )


if __name__ == "__main__":
    main()
