#!/usr/bin/env python
"""Full evaluation sweep: surrogate + 6 targets x {clean, osfd, mi_fgsm} ->
ASR + mAP-drop, over a manifest of COCO val2017 image ids.

Three-pass structure:
  1. Clean pass (once per model, cached to results/predictions/clean/<model>.json).
  2. Per-attack pass: load crafted noise, predict on x_clean+noise per model.
  3. Metrics pass: mAP (pycocotools, original-image space) + ASR (canvas-space,
     using the cached clean predictions as the "should have been detected" set).

Example:
    python scripts/evaluate.py --limit 5 --attacks osfd mi_fgsm --out results/smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_300.json")
    p.add_argument("--attacks", nargs="+", default=["osfd", "mi_fgsm"], choices=["osfd", "mi_fgsm"])
    p.add_argument("--noise-dir", type=Path, default=PROJECT_DIR / "results" / "noise")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--predictions-dir", type=Path, default=PROJECT_DIR / "results" / "predictions")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "results" / "metrics_summary.csv")
    p.add_argument("--out-json", type=Path, default=PROJECT_DIR / "results" / "metrics_summary.json")
    p.add_argument("--force-clean", action="store_true", help="recompute cached clean predictions")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    return p


def build_gt_cache(coco, image_ids, img_dir: Path, canvas: int):
    """Per image: (scale, canvas-space GT boxes[M,4], raw COCO cat_ids[M])."""
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image

    gt_index = build_gt_index(coco, image_ids)
    scale_by_image_id: dict = {}
    gt_boxes_by_image_id: dict = {}
    gt_cat_ids_by_image_id: dict = {}
    for image_id in image_ids:
        _, scale, _, _ = load_canvas_image(img_dir, coco, image_id, canvas)
        boxes, cat_ids = gt_to_canvas(gt_index[image_id], scale)
        scale_by_image_id[image_id] = scale
        gt_boxes_by_image_id[image_id] = boxes
        gt_cat_ids_by_image_id[image_id] = cat_ids
    return scale_by_image_id, gt_boxes_by_image_id, gt_cat_ids_by_image_id


def get_clean_predictions(handle, spec, image_ids, coco, img_dir, canvas, predictions_dir, device, force_clean, logger):
    from transfer_attack.data import load_canvas_image
    from transfer_attack.eval_metrics import predict_canvas
    from transfer_attack.io_utils import load_predictions, save_predictions

    clean_path = predictions_dir / "clean" / f"{spec.name}.json"
    preds = {}
    if clean_path.exists() and not force_clean:
        preds = load_predictions(clean_path)

    # Cache is keyed only by model name, not by manifest/limit -- a cache built
    # for a smaller image set (e.g. a smoke test) would otherwise silently miss
    # ids for a later, larger run. Recompute + merge only what's missing.
    missing = [i for i in image_ids if i not in preds]
    if missing:
        for image_id in missing:
            canvas_img, _, _, _ = load_canvas_image(img_dir, coco, image_id, canvas)
            preds[image_id] = predict_canvas(handle, canvas_img, canvas, device=device)
        save_predictions(clean_path, preds)
        logger.info(f"  clean predictions: {len(missing)} computed + cached, {len(preds) - len(missing)} from cache")
    else:
        logger.info(f"  clean predictions loaded from cache ({len(image_ids)} images)")
    return preds


def get_adv_predictions(handle, spec, attack, image_ids, coco, img_dir, canvas, noise_dir, predictions_dir, device, logger):
    from transfer_attack.data import load_canvas_image
    from transfer_attack.eval_metrics import predict_canvas
    from transfer_attack.io_utils import load_noise, load_predictions, save_predictions

    pred_path = predictions_dir / attack / f"{spec.name}.json"
    preds = {}
    if pred_path.exists():
        preds = load_predictions(pred_path)

    # As with clean predictions: cache is keyed only by model+attack, not by
    # manifest/limit. Only recompute ids this run actually needs and that
    # aren't already cached; ids with no crafted noise stay permanently absent
    # (craft.py skipped them, e.g. 0 valid GT boxes) rather than being retried.
    attack_noise_dir = noise_dir / attack
    to_compute = [i for i in image_ids if i not in preds]
    n_missing = 0
    n_computed = 0
    for image_id in to_compute:
        noise_path = attack_noise_dir / f"{image_id}.pt"
        if not noise_path.exists():
            n_missing += 1
            continue
        canvas_img, _, _, _ = load_canvas_image(img_dir, coco, image_id, canvas)
        x_adv = (canvas_img + load_noise(noise_path)).clamp(0.0, 255.0)
        preds[image_id] = predict_canvas(handle, x_adv, canvas, device=device)
        n_computed += 1
    if n_missing:
        logger.warning(f"  [{attack}] {n_missing} images had no crafted noise -- skipped")
    if n_computed:
        save_predictions(pred_path, preds)
    logger.info(
        f"  [{attack}] adversarial predictions: {n_computed} computed + cached, "
        f"{len(preds) - n_computed} from cache ({len(preds)} total)"
    )
    return preds


def compute_asr(clean_preds, adv_preds, gt_boxes_by_image_id, gt_cat_ids_by_image_id, image_ids, iou_thr, score_thr):
    from transfer_attack.eval_metrics import compute_asr_for_image

    evaded_total, clean_correct_total = 0, 0
    for image_id in image_ids:
        gt_boxes = gt_boxes_by_image_id[image_id]
        if gt_boxes.shape[0] == 0:
            continue
        evaded, clean_correct = compute_asr_for_image(
            clean_preds[image_id], adv_preds[image_id], gt_boxes, gt_cat_ids_by_image_id[image_id], iou_thr, score_thr
        )
        evaded_total += evaded
        clean_correct_total += clean_correct
    asr = 100.0 * evaded_total / clean_correct_total if clean_correct_total > 0 else float("nan")
    return asr, evaded_total, clean_correct_total


def to_identity_coco_results(preds_by_image_id: dict, scale_by_image_id: dict) -> list[dict]:
    """preds already carry raw COCO category ids as "labels" (see
    predict_canvas), so the label-remapping step is the identity map."""
    from transfer_attack.eval_metrics import to_coco_results

    identity_map = {int(label): int(label) for pred in preds_by_image_id.values() for label in pred["labels"].tolist()}
    return to_coco_results(preds_by_image_id, identity_map, scale_by_image_id)


def make_row(attack: str, spec, ap: float, ap50: float, asr) -> dict:
    return {
        "attack": attack,
        "model_name": spec.name,
        "group": spec.group or "",
        "mAP_clean": None,
        "mAP_adv": ap,
        "mAP_drop_pct": None,
        "AP50_clean": None,
        "AP50_adv": ap50,
        "ASR": asr if asr is not None else "",
    }


def evaluate_one_model(spec, args, coco, image_ids, img_dir, gt_cache, logger) -> list[dict]:
    from transfer_attack.eval_metrics import compute_coco_map
    from transfer_attack.models import build_model_handle

    scale_by_image_id, gt_boxes_by_image_id, gt_cat_ids_by_image_id = gt_cache
    logger.info(f"=== model: {spec.name} (role={spec.role}, group={spec.group}) ===")
    handle = build_model_handle(spec, args.checkpoints_dir, device=args.device, coco=coco)

    clean_preds = get_clean_predictions(
        handle, spec, image_ids, coco, img_dir, args.canvas, args.predictions_dir, args.device, args.force_clean, logger
    )
    # clean_preds is a cache accumulated across manifests sharing predictions_dir
    # (e.g. dev_50/dev_300/val_100 all writing to the same clean/<model>.json) --
    # scale_by_image_id only covers THIS run's image_ids, so scope to those before
    # building coco results (mirrors the common_ids filtering below for adv_preds).
    clean_metrics = compute_coco_map(
        coco, to_identity_coco_results({i: clean_preds[i] for i in image_ids}, scale_by_image_id), image_ids
    )
    rows = [make_row("clean", spec, clean_metrics["AP"], clean_metrics["AP50"], asr=None)]

    for attack in args.attacks:
        adv_preds = get_adv_predictions(
            handle, spec, attack, image_ids, coco, img_dir, args.canvas, args.noise_dir, args.predictions_dir,
            args.device, logger,
        )
        # common_ids excludes any image craft.py skipped (0 valid GT boxes) --
        # recompute the clean baseline over the SAME subset so mAP_drop_pct
        # compares like-for-like rather than mixing a full-manifest clean
        # baseline with a possibly-smaller adversarial set.
        common_ids = [i for i in image_ids if i in adv_preds]
        clean_common_metrics = compute_coco_map(
            coco, to_identity_coco_results({i: clean_preds[i] for i in common_ids}, scale_by_image_id), common_ids
        )
        adv_metrics = compute_coco_map(
            coco, to_identity_coco_results({i: adv_preds[i] for i in common_ids}, scale_by_image_id), common_ids
        )
        asr, evaded_total, clean_correct_total = compute_asr(
            clean_preds, adv_preds, gt_boxes_by_image_id, gt_cat_ids_by_image_id, common_ids, args.iou_thr, args.score_thr
        )

        row = make_row(attack, spec, adv_metrics["AP"], adv_metrics["AP50"], asr=asr)
        row["mAP_clean"] = clean_common_metrics["AP"]
        row["AP50_clean"] = clean_common_metrics["AP50"]
        row["mAP_drop_pct"] = (
            100.0 * (clean_common_metrics["AP"] - adv_metrics["AP"]) / clean_common_metrics["AP"]
            if clean_common_metrics["AP"] > 0
            else float("nan")
        )
        rows.append(row)
        logger.info(
            f"  [{attack}] AP={adv_metrics['AP']:.4f} (clean={clean_common_metrics['AP']:.4f}, "
            f"drop={row['mAP_drop_pct']:.1f}%) ASR={asr:.1f}% ({evaded_total}/{clean_correct_total})"
        )
    return rows


def write_outputs(rows: list[dict], out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["attack", "model_name", "group", "mAP_clean", "mAP_adv", "mAP_drop_pct", "AP50_clean", "AP50_adv", "ASR"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    args = build_arg_parser().parse_args()

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, save_run_log
    from transfer_attack.models import MODEL_REGISTRY

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    image_ids = load_manifest(args.manifest)["image_ids"]
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    img_dir = args.data_dir / "val2017"
    gt_cache = build_gt_cache(coco, image_ids, img_dir, args.canvas)

    t0 = time.time()
    rows = []
    for spec in MODEL_REGISTRY:
        rows.extend(evaluate_one_model(spec, args, coco, image_ids, img_dir, gt_cache, logger))
    elapsed_total = time.time() - t0

    write_outputs(rows, args.out, args.out_json)
    logger.info(f"wrote {args.out} and {args.out_json}")

    run_log_path = save_run_log(
        PROJECT_DIR / "runs",
        "evaluate",
        f"{'-'.join(args.attacks)}_{args.manifest.stem}",
        {
            "manifest": str(args.manifest),
            "attacks": args.attacks,
            "noise_dir": str(args.noise_dir),
            "checkpoints_dir": str(args.checkpoints_dir),
            "canvas": args.canvas,
            "score_thr": args.score_thr,
            "iou_thr": args.iou_thr,
            "device": args.device,
            "limit": args.limit,
            "results": {
                "n_images_in_manifest": len(image_ids),
                "elapsed_sec": round(elapsed_total, 1),
                "rows": rows,
            },
        },
    )
    logger.info(f"run log written -> {run_log_path}")


if __name__ == "__main__":
    main()
