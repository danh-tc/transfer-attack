#!/usr/bin/env python
"""E2c diagnostic: for GT boxes that mask_rcnn_swin_t's RPN still locates on
BOTH the clean and the adv image (IoU>=0.5 on each side, independently --
i.e. exactly the "recall survives" subset E2b measured at 81.9%), does the
RoI head's classification confidence for the correct class still collapse?

This is the last link E2/E2b left open: E2 (fixed clean proposals) showed the
RoI head's raw output barely moves (cos_dist 0.078) but that measurement was
blind to proposal-identity changes by construction. E2b showed proposal
IDENTITY changes a lot (Jaccard 0.24) but GT COVERAGE only drops modestly
(14.1%, recall 95.3%->81.9%) -- so ~82% of GT objects still have a
legitimately-localized (IoU>=0.5) proposal on the adv image. If mAP still
craters 72% despite that, the loss must be happening at classification: the
correctly-localized adv-side proposal gets scored wrong. This script checks
that directly, using each side's OWN proposals (not fixed to clean, unlike
E2) -- on the SAME already-crafted OSFD noise E1/E2/E2b used (no recrafting).

For every GT box matched (IoU>=0.5) by a proposal on BOTH the clean and the
adv proposal set:
  1. correct-class score drop: softmax(cls_score)[gt_label], clean -> adv.
  2. misclassification: does argmax(cls_score) (background is a valid
     argmax outcome, counts as misclassified) change identity vs gt_label.
  3. fraction whose correct-class score falls below 0.3 -- this project's
     evaluation score_thr (transfer_attack/constants.py::DEFAULT_SCORE_THR),
     i.e. would this exact detection actually disappear from the ASR/mAP
     computation in evaluate.py.

Example:
    python scripts/e2c_roi_classification_collapse.py --attack osfd --manifest data/manifests/dev_50.json
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["osfd", "mi_fgsm"], default="osfd")
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--noise-dir", type=Path, default=None, help="default: results/noise/<manifest-stem>/<attack>/")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--model", type=str, default="mask_rcnn_swin_t")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--score-thr", type=float, default=0.3, help="matches constants.DEFAULT_SCORE_THR")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "e2c_roi_classification_collapse.csv")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch
    import torch.nn.functional as F

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.eval_metrics import box_iou
    from transfer_attack.io_utils import get_logger, load_noise
    from transfer_attack.models import build_model_handle, get_spec

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    noise_dir = args.noise_dir or (PROJECT_DIR / "results" / "noise" / args.manifest.stem / args.attack)
    if not noise_dir.exists():
        logger.error(f"noise dir not found: {noise_dir} -- craft it first (see scripts/craft.py)")
        sys.exit(1)

    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"
    device = args.device

    spec = get_spec(args.model)
    handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)
    model = handle.model

    from mmdet.structures import DetDataSample
    from mmdet.structures.bbox import bbox2roi

    ds = DetDataSample()
    ds.set_metainfo(
        dict(
            img_shape=(args.canvas, args.canvas),
            ori_shape=(args.canvas, args.canvas),
            scale_factor=(1.0, 1.0),
            batch_input_shape=(args.canvas, args.canvas),
        )
    )

    n_images = 0
    n_missing_noise = 0
    n_gt_total = 0
    scores_clean, scores_adv = [], []
    misclassified_clean, misclassified_adv = [], []

    for image_id in image_ids:
        noise_path = noise_dir / f"{image_id}.pt"
        if not noise_path.exists():
            n_missing_noise += 1
            continue

        gt_entries = gt_index[image_id]
        if not gt_entries:
            n_images += 1
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        gt_labels = torch.tensor(
            [handle.cat_id_to_label[int(c)] for c in gt_cat_ids.tolist()], dtype=torch.long
        )
        n_gt_total += gt_boxes.shape[0]

        noise = load_noise(noise_path)
        x_clean = canvas_img.to(device)
        x_adv = (canvas_img + noise).clamp(0.0, 255.0).to(device)

        with torch.no_grad():
            feats_c = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
            feats_a = model.backbone(handle.normalize(x_adv.unsqueeze(0)))
            neck_c = model.neck(feats_c)
            neck_a = model.neck(feats_a)
            props_c = model.rpn_head.predict(neck_c, [ds], rescale=False)[0].bboxes.cpu()
            props_a = model.rpn_head.predict(neck_a, [ds], rescale=False)[0].bboxes.cpu()

            iou_gt_c = box_iou(gt_boxes, props_c)  # (G, Nc)
            iou_gt_a = box_iou(gt_boxes, props_a)  # (G, Na)
            best_iou_c, best_idx_c = iou_gt_c.max(dim=1)
            best_iou_a, best_idx_a = iou_gt_a.max(dim=1)
            both_matched = (best_iou_c >= args.iou_thr) & (best_iou_a >= args.iou_thr)
            if not bool(both_matched.any()):
                n_images += 1
                continue

            gt_rows = both_matched.nonzero(as_tuple=True)[0]
            matched_props_c = props_c[best_idx_c[gt_rows]].to(device)
            matched_props_a = props_a[best_idx_a[gt_rows]].to(device)

            rois_c = bbox2roi([matched_props_c])
            rois_a = bbox2roi([matched_props_a])
            cls_score_c = model.roi_head._bbox_forward(neck_c, rois_c)["cls_score"]
            cls_score_a = model.roi_head._bbox_forward(neck_a, rois_a)["cls_score"]

            probs_c = F.softmax(cls_score_c, dim=-1).cpu()
            probs_a = F.softmax(cls_score_a, dim=-1).cpu()
            matched_labels = gt_labels[gt_rows]

            correct_c = probs_c[torch.arange(len(gt_rows)), matched_labels]
            correct_a = probs_a[torch.arange(len(gt_rows)), matched_labels]
            argmax_c = probs_c.argmax(dim=1)
            argmax_a = probs_a.argmax(dim=1)

            scores_clean.extend(correct_c.tolist())
            scores_adv.extend(correct_a.tolist())
            misclassified_clean.extend((argmax_c != matched_labels).tolist())
            misclassified_adv.extend((argmax_a != matched_labels).tolist())

        n_images += 1

    if n_missing_noise:
        logger.warning(f"{n_missing_noise} images had no crafted noise under {noise_dir} -- skipped")

    n_both = len(scores_clean)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    mean_score_clean = mean(scores_clean)
    mean_score_adv = mean(scores_adv)
    frac_misclass_clean = mean(misclassified_clean)
    frac_misclass_adv = mean(misclassified_adv)
    frac_below_clean = mean([s < args.score_thr for s in scores_clean])
    frac_below_adv = mean([s < args.score_thr for s in scores_adv])

    summary = {
        "model_name": spec.name,
        "n_images": n_images,
        "n_gt_total": n_gt_total,
        "n_gt_matched_both_sides": n_both,
        "matched_both_coverage_pct": 100.0 * n_both / n_gt_total if n_gt_total else float("nan"),
        "mean_correct_class_score_clean": mean_score_clean,
        "mean_correct_class_score_adv": mean_score_adv,
        "score_drop_pct": (
            100.0 * (mean_score_clean - mean_score_adv) / mean_score_clean if mean_score_clean > 0 else float("nan")
        ),
        "frac_misclassified_clean": frac_misclass_clean,
        "frac_misclassified_adv": frac_misclass_adv,
        "frac_below_score_thr_clean": frac_below_clean,
        "frac_below_score_thr_adv": frac_below_adv,
    }

    logger.info(f"=== {spec.name}: RoI classification on each side's OWN GT-matched proposals ===")
    for k, v in summary.items():
        logger.info(f"  {k:32s} = {v}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    logger.info(f"wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
