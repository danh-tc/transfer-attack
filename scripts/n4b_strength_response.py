#!/usr/bin/env python
"""N4b diagnostic: N4 found clean_confidence/iou_quality/object_area predict
which objects evade dino_swin_l (weak-detection objects evade more easily),
but that only answers "which objects are easy to evade at the FULL crafted
budget" -- not "where would EXTRA perturbation budget do the most good".
Those are different questions: an easy object may already be saturated
(evades at low perturbation strength, more budget wasted on it), while a hard
object may only start yielding near full strength (more budget there could
still move the needle).

This scales the ALREADY-CRAFTED OSFD noise (no recrafting) to {0.5, 0.75,
1.0} x its original magnitude and re-evaluates evasion on dino_swin_l per
object, split into "easy" (below-median clean_confidence -- the strongest
N4 predictor) vs "hard" (above-median) groups. Reading key:

  - easy group evasion rate already high/saturated at 0.5x, flat after ->
    hard-object emphasis is the better use of extra budget (easy objects
    have no more headroom to gain).
  - easy group's rate keeps climbing steeply through 1.0x while hard group
    stays flat -> weak-object emphasis is supported instead.
  - both climb similarly -> no clear allocation direction from this signal.

Example:
    python scripts/n4b_strength_response.py --object-csv results/n4_object_level_diagnostic.csv --target dino_swin_l
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

SCALES = [0.5, 0.75, 1.0]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["osfd", "mi_fgsm"], default="osfd")
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--noise-dir", type=Path, default=None)
    p.add_argument("--object-csv", type=Path, default=PROJECT_DIR / "results" / "n4_object_level_diagnostic.csv")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--target", type=str, default="dino_swin_l")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n4b_strength_response.csv")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.eval_metrics import greedy_match, predict_canvas
    from transfer_attack.io_utils import get_logger, load_noise
    from transfer_attack.models import build_model_handle, get_spec

    logger = get_logger()

    with open(args.object_csv) as f:
        obj_rows = list(csv.DictReader(f))
    for r in obj_rows:
        r["image_id"] = int(r["image_id"])
        r["gt_idx"] = int(r["gt_idx"])
        r["clean_confidence"] = float(r["clean_confidence"])

    confidences = sorted(r["clean_confidence"] for r in obj_rows)
    median_conf = statistics.median(confidences)
    for r in obj_rows:
        r["difficulty_group"] = "hard" if r["clean_confidence"] >= median_conf else "easy"
    logger.info(
        f"median clean_confidence = {median_conf:.4f}; "
        f"easy(n={sum(1 for r in obj_rows if r['difficulty_group']=='easy')}) "
        f"hard(n={sum(1 for r in obj_rows if r['difficulty_group']=='hard')})"
    )

    by_image: dict[int, list[dict]] = {}
    for r in obj_rows:
        by_image.setdefault(r["image_id"], []).append(r)

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    gt_index = build_gt_index(coco, manifest["image_ids"])
    img_dir = args.data_dir / "val2017"
    device = args.device

    noise_dir = args.noise_dir or (PROJECT_DIR / "results" / "noise" / args.manifest.stem / args.attack)

    target_spec = get_spec(args.target)
    target_handle = build_model_handle(target_spec, args.checkpoints_dir, device=device, coco=coco)

    # scale -> group -> [evaded flags]
    results = {s: {"easy": [], "hard": []} for s in SCALES}

    for image_id, rows in by_image.items():
        noise_path = noise_dir / f"{image_id}.pt"
        if not noise_path.exists():
            continue
        gt_entries = gt_index[image_id]
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        noise = load_noise(noise_path)
        x_clean = canvas_img.to(device)

        clean_pred = predict_canvas(target_handle, x_clean, args.canvas, device=device)
        clean_match = greedy_match(
            clean_pred["bboxes"], clean_pred["scores"], clean_pred["labels"],
            gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr,
        )

        row_by_gt_idx = {r["gt_idx"]: r for r in rows}

        for s in SCALES:
            x_adv = (canvas_img + s * noise).clamp(0.0, 255.0).to(device)
            adv_pred = predict_canvas(target_handle, x_adv, args.canvas, device=device)
            adv_match = greedy_match(
                adv_pred["bboxes"], adv_pred["scores"], adv_pred["labels"],
                gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr,
            )
            for g_idx, matched_p in enumerate(clean_match):
                if matched_p is None or g_idx not in row_by_gt_idx:
                    continue
                evaded = int(adv_match[g_idx] is None)
                group = row_by_gt_idx[g_idx]["difficulty_group"]
                results[s][group].append(evaded)

    out_rows = []
    logger.info("=== evasion rate by difficulty group x perturbation scale ===")
    for s in SCALES:
        for group in ("easy", "hard"):
            vals = results[s][group]
            rate = sum(vals) / len(vals) if vals else float("nan")
            out_rows.append({"scale": s, "group": group, "n": len(vals), "evasion_rate": rate})
            logger.info(f"  scale={s:.2f} group={group:5s} n={len(vals):3d} evasion_rate={rate:.3f}")

    for group in ("easy", "hard"):
        r05 = next(r["evasion_rate"] for r in out_rows if r["scale"] == 0.5 and r["group"] == group)
        r10 = next(r["evasion_rate"] for r in out_rows if r["scale"] == 1.0 and r["group"] == group)
        slope = (r10 - r05) / 0.5
        logger.info(f"  [{group}] marginal slope (rate@1.0 - rate@0.5)/0.5 = {slope:+.3f}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scale", "group", "n", "evasion_rate"])
        writer.writeheader()
        writer.writerows(out_rows)
    logger.info(f"wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
