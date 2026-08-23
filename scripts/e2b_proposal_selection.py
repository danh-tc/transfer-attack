#!/usr/bin/env python
"""E2b diagnostic: does OSFD change WHICH regions mask_rcnn_swin_t's RPN
proposes, not just the continuous score/bbox-delta values at each region?

E2 found RPN raw score/bbox distortion is small (cos_dist ~0.05) and RoI-head
output at FIXED clean-derived proposals is also small (~0.08), yet the real
black-box mAP drop for this model is 72.2%. The candidate explanation: NMS +
top-K proposal selection is a DISCRETE threshold operation, so a small
continuous RPN score shift can still flip which boxes survive as proposals --
damage that a "same proposals, compare content" measurement (like E2's RoI
checkpoint) is blind to by construction.

This script tests that directly by comparing the ACTUAL proposal sets RPN
selects on x_clean vs x_adv (not fixing proposals to the clean set), on the
same already-crafted OSFD noise E1/E2 used (no recrafting):

  1. Proposal overlap: greedy IoU>=0.5 match between the clean and adv
     top-K proposal sets -> recall (fraction of clean proposals that survive
     in the adv set) and Jaccard.
  2. Proposal displacement: for every clean proposal, its best-IoU match in
     the adv set REGARDLESS of threshold (captures gradual box drift even
     when a proposal isn't lost outright).
  3. GT object coverage: fraction of GT boxes with an IoU>=0.5 proposal,
     computed separately for the clean and adv proposal sets -- this is
     "recall" in the classic RPN sense, and is the number that would most
     directly explain a downstream mAP collapse if it drops sharply on adv.

Example:
    python scripts/e2b_proposal_selection.py --attack osfd --manifest data/manifests/dev_50.json
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
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "e2b_proposal_selection.csv")
    return p


def greedy_class_agnostic_match(iou: "torch.Tensor", iou_thr: float) -> int:
    """iou: (N_clean, N_adv), clean proposals assumed already score-sorted
    (RPN's predict() returns proposals sorted desc by score after NMS).
    Returns the number of matched pairs under standard greedy one-to-one
    assignment (each clean proposal claims its best still-unclaimed adv
    proposal if IoU >= iou_thr)."""
    import torch

    n_clean, n_adv = iou.shape
    if n_clean == 0 or n_adv == 0:
        return 0
    adv_taken = torch.zeros(n_adv, dtype=torch.bool, device=iou.device)
    n_matched = 0
    for i in range(n_clean):
        row = iou[i].clone()
        row[adv_taken] = -1.0
        best_j = int(torch.argmax(row).item())
        if row[best_j] >= iou_thr:
            adv_taken[best_j] = True
            n_matched += 1
    return n_matched


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

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
    n_clean_props, n_adv_props = [], []
    recalls_clean_in_adv, jaccards, best_match_ious = [], [], []
    gt_recalls_clean, gt_recalls_adv = [], []

    for image_id in image_ids:
        noise_path = noise_dir / f"{image_id}.pt"
        if not noise_path.exists():
            n_missing_noise += 1
            continue

        gt_entries = gt_index[image_id]
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
        gt_boxes, _ = gt_to_canvas(gt_entries, scale)
        noise = load_noise(noise_path)
        x_clean = canvas_img.to(device)
        x_adv = (canvas_img + noise).clamp(0.0, 255.0).to(device)

        with torch.no_grad():
            feats_c = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
            feats_a = model.backbone(handle.normalize(x_adv.unsqueeze(0)))
            neck_c = model.neck(feats_c)
            neck_a = model.neck(feats_a)
            props_c = model.rpn_head.predict(neck_c, [ds], rescale=False)[0].bboxes
            props_a = model.rpn_head.predict(neck_a, [ds], rescale=False)[0].bboxes

        n_clean_props.append(props_c.shape[0])
        n_adv_props.append(props_a.shape[0])

        iou_ca = box_iou(props_c.cpu(), props_a.cpu())
        n_matched = greedy_class_agnostic_match(iou_ca, args.iou_thr)
        n_c, n_a = props_c.shape[0], props_a.shape[0]
        recalls_clean_in_adv.append(n_matched / n_c if n_c > 0 else float("nan"))
        jaccards.append(n_matched / (n_c + n_a - n_matched) if (n_c + n_a - n_matched) > 0 else float("nan"))
        if n_c > 0 and n_a > 0:
            best_match_ious.append(float(iou_ca.max(dim=1).values.mean()))

        if gt_boxes.shape[0] > 0:
            iou_gt_c = box_iou(gt_boxes, props_c.cpu())
            iou_gt_a = box_iou(gt_boxes, props_a.cpu())
            gt_recalls_clean.append(float((iou_gt_c.max(dim=1).values >= args.iou_thr).float().mean()))
            gt_recalls_adv.append(float((iou_gt_a.max(dim=1).values >= args.iou_thr).float().mean()))

        n_images += 1

    if n_missing_noise:
        logger.warning(f"{n_missing_noise} images had no crafted noise under {noise_dir} -- skipped")

    def mean(xs):
        xs = [x for x in xs if x == x]  # drop nan
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "model_name": spec.name,
        "n_images": n_images,
        "mean_n_proposals_clean": mean(n_clean_props),
        "mean_n_proposals_adv": mean(n_adv_props),
        "mean_recall_clean_in_adv": mean(recalls_clean_in_adv),
        "mean_jaccard": mean(jaccards),
        "mean_best_match_iou": mean(best_match_ious),
        "mean_gt_recall_clean": mean(gt_recalls_clean),
        "mean_gt_recall_adv": mean(gt_recalls_adv),
        "gt_recall_drop_pct": (
            100.0 * (mean(gt_recalls_clean) - mean(gt_recalls_adv)) / mean(gt_recalls_clean)
            if mean(gt_recalls_clean) > 0
            else float("nan")
        ),
    }

    logger.info(f"=== {spec.name}: proposal-selection stability (clean vs adv RPN output) ===")
    for k, v in summary.items():
        logger.info(f"  {k:28s} = {v}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    logger.info(f"wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
