#!/usr/bin/env python
"""N4 diagnostic (Phase N3 continuation, reframed to the right unit of
analysis): E5 found no simple PER-IMAGE property predicts DINO transfer
(|r|<=0.20 after removing the binary-success confound). E2b/E2c already
showed the real failure mechanism operates at the object/proposal level, not
the whole-image level -- so this repeats the same question one level down:
per clean-correct GT OBJECT, does any narrow, causally-plausible property
predict whether THAT object evades detection on dino_swin_l?

Unit of analysis: one row per GT box that was correctly detected on the CLEAN
image (the same "clean-correct" set ASR/E5 use), y = evaded (1) or still
detected (0) on the ADV image. No recrafting -- reuses the already-crafted
OSFD noise.

Narrow property set (causal plausibility, not a fishing expedition):
  - object_area          : GT box area, canvas-space.
  - clean_confidence      : score of the matched clean prediction.
  - iou_quality           : IoU of that matched clean prediction with the GT box.
  - pert_energy_object    : mean |noise| (L1, over channels+pixels) inside the GT box.
  - pert_energy_ring      : same, in a dilated "vicinal ring" around the box
                             (dilated box minus the box itself) -- OSFD's own
                             stated mechanism (Eq. 2, paper) is to amplify
                             exactly this vicinal region, so this is the most
                             directly mechanism-relevant property here.
  - surrogate_local_cos_dist : 1-cosine_similarity(F_clean, F_adv) restricted
                             to the box's spatial footprint in the surrogate's
                             FINEST backbone stage (stage 0, coarse stride
                             inferred from the actual feature map size).
  - gt_cat_id (class)     : stratification/control only -- NOT tested as a
                             mechanism, just reported descriptively.

Statistics: point-estimate AUC (Mann-Whitney U, no sklearn dependency) of
each continuous property alone predicting y, PLUS an IMAGE-CLUSTER bootstrap
(resample IMAGES with replacement, not individual objects -- objects from the
same image are correlated, e.g. all evaded or all not) to get a 95% CI and
the fraction of bootstrap draws keeping the same side of 0.5 as the point
estimate. GO signal = at least one property with a clearly non-0.5 AUC that
holds sign/direction robustly across the bootstrap; otherwise close this
branch too (not just this one property) rather than adding more features.

Example:
    python scripts/n4_object_level_diagnostic.py --attack osfd --manifest data/manifests/dev_50.json --target dino_swin_l
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

PROPERTIES = [
    "object_area", "clean_confidence", "iou_quality",
    "pert_energy_object", "pert_energy_ring", "surrogate_local_cos_dist",
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["osfd", "mi_fgsm"], default="osfd")
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--noise-dir", type=Path, default=None)
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--target", type=str, default="dino_swin_l")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--ring-expand-frac", type=float, default=0.25, help="dilate box by this fraction of its own w/h on each side")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n4_object_level_diagnostic.csv")
    p.add_argument("--out-summary-csv", type=Path, default=PROJECT_DIR / "results" / "n4_object_level_summary.csv")
    return p


def auc_mann_whitney(scores, labels):
    """scores: list[float], labels: list[0/1]. Returns AUC (prob a random
    positive scores higher than a random negative), via rank-sum -- no
    sklearn dependency, and handles ties by average rank."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = sum(ranks[i] for i in range(n) if labels[i] == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)

    import torch
    import torch.nn.functional as F

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.eval_metrics import box_iou, greedy_match, predict_canvas
    from transfer_attack.io_utils import get_logger, load_noise
    from transfer_attack.models import build_model_handle, get_spec

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]

    noise_dir = args.noise_dir or (PROJECT_DIR / "results" / "noise" / args.manifest.stem / args.attack)
    if not noise_dir.exists():
        logger.error(f"noise dir not found: {noise_dir}")
        sys.exit(1)

    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"
    device = args.device

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate_handle = build_model_handle(surrogate_spec, args.checkpoints_dir, device=device, coco=coco)
    target_spec = get_spec(args.target)
    target_handle = build_model_handle(target_spec, args.checkpoints_dir, device=device, coco=coco)

    object_rows = []
    for image_id in image_ids:
        noise_path = noise_dir / f"{image_id}.pt"
        if not noise_path.exists():
            continue
        gt_entries = gt_index[image_id]
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        if gt_boxes.shape[0] == 0:
            continue
        noise = load_noise(noise_path)
        x_clean = canvas_img.to(device)
        x_adv = (canvas_img + noise).clamp(0.0, 255.0).to(device)

        clean_pred = predict_canvas(target_handle, x_clean, args.canvas, device=device)
        adv_pred = predict_canvas(target_handle, x_adv, args.canvas, device=device)
        clean_match = greedy_match(
            clean_pred["bboxes"], clean_pred["scores"], clean_pred["labels"],
            gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr,
        )
        adv_match = greedy_match(
            adv_pred["bboxes"], adv_pred["scores"], adv_pred["labels"],
            gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr,
        )
        iou_clean = box_iou(clean_pred["bboxes"], gt_boxes)  # (P, G)

        with torch.no_grad():
            feats_cln = surrogate_handle.model.backbone(surrogate_handle.normalize(x_clean.unsqueeze(0)))
            feats_adv = surrogate_handle.model.backbone(surrogate_handle.normalize(x_adv.unsqueeze(0)))
        stage0_cln, stage0_adv = feats_cln[0][0], feats_adv[0][0]  # (C, h, w)
        fh, fw = stage0_cln.shape[-2], stage0_cln.shape[-1]
        stride_y, stride_x = args.canvas / fh, args.canvas / fw

        noise_cpu = noise  # (3, canvas, canvas)

        for g_idx, matched_p in enumerate(clean_match):
            if matched_p is None:
                continue  # not clean-correct -- no attack opportunity for this object
            evaded = int(adv_match[g_idx] is None)

            x1, y1, x2, y2 = gt_boxes[g_idx].tolist()
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            clean_conf = float(clean_pred["scores"][matched_p].item())
            iou_q = float(iou_clean[matched_p, g_idx].item())

            xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
            xi2, yi2 = min(args.canvas, int(round(x2))), min(args.canvas, int(round(y2)))
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            box_region = noise_cpu[:, yi1:yi2, xi1:xi2]
            pert_energy_object = float(box_region.abs().mean().item())

            w, h = xi2 - xi1, yi2 - yi1
            pad_x, pad_y = int(round(args.ring_expand_frac * w)), int(round(args.ring_expand_frac * h))
            rx1, ry1 = max(0, xi1 - pad_x), max(0, yi1 - pad_y)
            rx2, ry2 = min(args.canvas, xi2 + pad_x), min(args.canvas, yi2 + pad_y)
            ring_full = noise_cpu[:, ry1:ry2, rx1:rx2].abs()
            ring_mask = torch.ones((ring_full.shape[-2], ring_full.shape[-1]), dtype=torch.bool)
            iy1, iy2 = yi1 - ry1, yi2 - ry1
            ix1, ix2 = xi1 - rx1, xi2 - rx1
            ring_mask[iy1:iy2, ix1:ix2] = False
            if ring_mask.any():
                pert_energy_ring = float(ring_full.mean(dim=0)[ring_mask].mean().item())
            else:
                pert_energy_ring = float("nan")

            fx1, fy1 = max(0, int(x1 / stride_x)), max(0, int(y1 / stride_y))
            fx2, fy2 = min(fw, int(-(-x2 // stride_x))), min(fh, int(-(-y2 // stride_y)))
            fx2, fy2 = max(fx2, fx1 + 1), max(fy2, fy1 + 1)
            region_cln = stage0_cln[:, fy1:fy2, fx1:fx2].flatten().unsqueeze(0)
            region_adv = stage0_adv[:, fy1:fy2, fx1:fx2].flatten().unsqueeze(0)
            local_cos_dist = float((1.0 - F.cosine_similarity(region_cln, region_adv)).item())

            object_rows.append(
                {
                    "image_id": image_id,
                    "gt_idx": g_idx,
                    "gt_cat_id": int(gt_cat_ids[g_idx].item()),
                    "evaded": evaded,
                    "object_area": area,
                    "clean_confidence": clean_conf,
                    "iou_quality": iou_q,
                    "pert_energy_object": pert_energy_object,
                    "pert_energy_ring": pert_energy_ring,
                    "surrogate_local_cos_dist": local_cos_dist,
                }
            )

    n_evaded = sum(r["evaded"] for r in object_rows)
    logger.info(f"n objects (clean-correct on {args.target}): {len(object_rows)}, evaded={n_evaded}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(object_rows[0].keys()) if object_rows else [])
        writer.writeheader()
        writer.writerows(object_rows)
    logger.info(f"wrote per-object -> {args.out_csv}")

    # ---- image-cluster bootstrap AUC per property ----
    by_image: dict[int, list[dict]] = {}
    for r in object_rows:
        by_image.setdefault(r["image_id"], []).append(r)
    image_ids_with_objects = list(by_image.keys())

    summary = []
    for prop in PROPERTIES:
        valid_rows = [r for r in object_rows if r[prop] == r[prop]]  # drop NaN
        scores = [r[prop] for r in valid_rows]
        labels = [r["evaded"] for r in valid_rows]
        point_auc = auc_mann_whitney(scores, labels)

        boot_aucs = []
        for _ in range(args.n_bootstrap):
            sampled_images = [random.choice(image_ids_with_objects) for _ in range(len(image_ids_with_objects))]
            boot_rows = [row for img in sampled_images for row in by_image[img] if row[prop] == row[prop]]
            b_scores = [row[prop] for row in boot_rows]
            b_labels = [row["evaded"] for row in boot_rows]
            auc_b = auc_mann_whitney(b_scores, b_labels)
            if auc_b == auc_b:  # not NaN
                boot_aucs.append(auc_b)
        boot_aucs.sort()
        n_b = len(boot_aucs)
        ci_lo = boot_aucs[int(0.025 * n_b)] if n_b else float("nan")
        ci_hi = boot_aucs[int(0.975 * n_b) - 1] if n_b else float("nan")
        same_side = sum(1 for a in boot_aucs if (a > 0.5) == (point_auc > 0.5)) / n_b if n_b else float("nan")

        summary.append(
            {
                "property": prop,
                "n": len(valid_rows),
                "point_AUC": point_auc,
                "boot_CI_lo": ci_lo,
                "boot_CI_hi": ci_hi,
                "frac_boot_same_side_of_0.5": same_side,
            }
        )
        logger.info(
            f"  {prop:28s} n={len(valid_rows):4d} AUC={point_auc:.3f} "
            f"[{ci_lo:.3f}, {ci_hi:.3f}]  same_side_frac={same_side:.3f}"
        )

    with open(args.out_summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["property", "n", "point_AUC", "boot_CI_lo", "boot_CI_hi", "frac_boot_same_side_of_0.5"])
        writer.writeheader()
        writer.writerows(summary)
    logger.info(f"wrote summary -> {args.out_summary_csv}")

    # ---- class: descriptive stratification only, not a tested mechanism ----
    by_class: dict[int, list[int]] = {}
    for r in object_rows:
        by_class.setdefault(r["gt_cat_id"], []).append(r["evaded"])
    logger.info("=== gt_cat_id (class) -- descriptive only, n>=5 ===")
    for cat_id, evs in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        if len(evs) >= 5:
            logger.info(f"  cat_id={cat_id:3d} n={len(evs):3d} evasion_rate={sum(evs)/len(evs):.3f}")


if __name__ == "__main__":
    main()
