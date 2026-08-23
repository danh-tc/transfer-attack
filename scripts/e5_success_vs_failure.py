#!/usr/bin/env python
"""E5 diagnostic (Phase N3): what distinguishes per-image OSFD perturbations
that DO transfer to dino_swin_l from those that DON'T, on the SAME already-
crafted noise (no recrafting)?

Rationale: after a long chain of NO-GO results at the AGGREGATE/AVERAGE level
(E1 feature damage, E3b instantaneous gradient, MVC/RCG consistency, E4/E4b
spectral bands, N2-B stat-norm), dino_swin_l's own baseline ASR (~25-30% on
dev_50) means most images fail to transfer but a meaningful minority succeed
-- a natural within-dataset controlled comparison (same surrogate, same
attack, same budget) that per-image property analysis can exploit, instead of
inventing another method and looking at the average ASR shift again.

Per-image "success" definition: at least one of the GT boxes correctly
detected on the CLEAN dino_swin_l prediction (the same "clean-correct" set
ASR is normally computed over) evades detection on the ADV prediction -- i.e.
this image contributed >=1 evaded box to the pooled ASR numerator. Images
with zero clean-correct boxes are excluded (no attack opportunity to succeed
or fail on).

Properties compared between S (>=1 evaded) and F (0 evaded), all computed
WITHOUT recrafting, reusing artifacts already on disk from E1/E4:
  - object size/count: mean GT box area (canvas-space), number of GT boxes.
  - surrogate craft loss: final osfd_loss value (from craft.py's losses.csv
    if present, else recomputed once, no_grad, from the existing noise).
  - backbone distortion on the SURROGATE: cos_dist(F(x_clean), F(x_adv)),
    same metric as E1, computed on faster_rcnn_r50 (cheap, no target-model
    coupling).
  - perturbation stats: L1 mean, L2 norm, saturation ratio (|noise|>=0.99*eps).
  - spectral energy ratio: low/mid/high fraction of noise energy (E4's bands).
  - spatial concentration: fraction of perturbation L1 energy falling inside
    the union of GT boxes vs background.

For each property, report mean(S) vs mean(F) and a simple Mann-Whitney-free
signal: Cohen's d style standardized mean difference (cheap, no scipy dep).

Example:
    python scripts/e5_success_vs_failure.py --attack osfd --manifest data/manifests/dev_50.json --target dino_swin_l
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
    p.add_argument("--target", type=str, default="dino_swin_l")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "e5_success_vs_failure.csv")
    p.add_argument("--out-summary-csv", type=Path, default=PROJECT_DIR / "results" / "e5_success_vs_failure_summary.csv")
    return p


def decompose_bands(noise, boundaries=(1 / 3, 2 / 3)):
    import torch

    C, H, W = noise.shape
    device = noise.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = H / 2.0, W / 2.0
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = (cy**2 + cx**2) ** 0.5
    r_norm = r / max_r
    b1, b2 = boundaries
    masks = {
        "low": (r_norm <= b1).float(),
        "mid": ((r_norm > b1) & (r_norm <= b2)).float(),
        "high": (r_norm > b2).float(),
    }
    energies = {}
    total = (noise**2).sum().item() + 1e-12
    for name, mask in masks.items():
        e = 0.0
        for c in range(C):
            F = torch.fft.fftshift(torch.fft.fft2(noise[c]))
            back = torch.fft.ifft2(torch.fft.ifftshift(F * mask)).real
            e += (back**2).sum().item()
        energies[name] = e / total
    return energies


def cohens_d(xs_s, xs_f):
    import statistics

    if len(xs_s) < 2 or len(xs_f) < 2:
        return float("nan")
    mean_s, mean_f = statistics.mean(xs_s), statistics.mean(xs_f)
    var_s, var_f = statistics.variance(xs_s), statistics.variance(xs_f)
    n_s, n_f = len(xs_s), len(xs_f)
    pooled_std = (((n_s - 1) * var_s + (n_f - 1) * var_f) / (n_s + n_f - 2)) ** 0.5
    return (mean_s - mean_f) / pooled_std if pooled_std > 1e-12 else float("nan")


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

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
        logger.error(f"noise dir not found: {noise_dir} -- craft it first (see scripts/craft.py)")
        sys.exit(1)

    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"
    device = args.device

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate_handle = build_model_handle(surrogate_spec, args.checkpoints_dir, device=device, coco=coco)

    target_spec = get_spec(args.target)
    target_handle = build_model_handle(target_spec, args.checkpoints_dir, device=device, coco=coco)

    rows = []
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

        # ---- per-image success/failure on the target, via the same
        # clean-correct / evaded definition used for the pooled ASR ----
        clean_pred = predict_canvas(target_handle, x_clean, args.canvas, device=device)
        adv_pred = predict_canvas(target_handle, x_adv, args.canvas, device=device)
        clean_match = greedy_match(
            clean_pred["bboxes"], clean_pred["scores"], clean_pred["labels"],
            gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr,
        )
        clean_correct_idx = [i for i, m in enumerate(clean_match) if m is not None]
        if not clean_correct_idx:
            continue  # no attack opportunity on this target for this image
        adv_match = greedy_match(
            adv_pred["bboxes"], adv_pred["scores"], adv_pred["labels"],
            gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr,
        )
        n_evaded = sum(1 for i in clean_correct_idx if adv_match[i] is None)
        success = n_evaded > 0

        # ---- properties, no recrafting ----
        n_gt = gt_boxes.shape[0]
        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
        mean_gt_area = float(areas.mean().item())

        with torch.no_grad():
            feats_cln = surrogate_handle.model.backbone(surrogate_handle.normalize(x_clean.unsqueeze(0)))
            feats_adv = surrogate_handle.model.backbone(surrogate_handle.normalize(x_adv.unsqueeze(0)))
        import torch.nn.functional as F

        cos_dists = []
        for fc, fa in zip(feats_cln, feats_adv):
            fcf, faf = fc.flatten().unsqueeze(0), fa.flatten().unsqueeze(0)
            cos_dists.append(1.0 - F.cosine_similarity(fcf, faf).item())
        mean_cos_dist = sum(cos_dists) / len(cos_dists)

        noise_cpu = noise
        l1_mean = float(noise_cpu.abs().mean().item())
        l2_norm = float(noise_cpu.norm().item())
        eps_est = float(noise_cpu.abs().max().item())
        saturation_ratio = float((noise_cpu.abs() >= 0.99 * eps_est).float().mean().item()) if eps_est > 0 else float("nan")

        band_energy = decompose_bands(noise_cpu.to(device), (1 / 3, 2 / 3))

        # spatial concentration: L1 energy inside union of GT boxes vs whole canvas
        gt_mask = torch.zeros((args.canvas, args.canvas), dtype=torch.bool)
        for box in gt_boxes.tolist():
            x1, y1, x2, y2 = [max(0, min(args.canvas, int(round(v)))) for v in box]
            gt_mask[y1:y2, x1:x2] = True
        l1_map = noise_cpu.abs().sum(dim=0)  # (H,W)
        total_l1 = l1_map.sum().item() + 1e-12
        inside_l1 = l1_map[gt_mask].sum().item() if gt_mask.any() else 0.0
        inside_frac_energy = inside_l1 / total_l1
        inside_frac_area = float(gt_mask.float().mean().item())
        concentration_ratio = inside_frac_energy / (inside_frac_area + 1e-12)  # >1 => energy concentrated inside GT vs uniform baseline

        rows.append(
            {
                "image_id": image_id,
                "success": int(success),
                "n_evaded": n_evaded,
                "n_clean_correct": len(clean_correct_idx),
                "n_gt": n_gt,
                "mean_gt_area": mean_gt_area,
                "surrogate_cos_dist": mean_cos_dist,
                "noise_l1_mean": l1_mean,
                "noise_l2_norm": l2_norm,
                "saturation_ratio": saturation_ratio,
                "band_low": band_energy["low"],
                "band_mid": band_energy["mid"],
                "band_high": band_energy["high"],
                "gt_energy_concentration_ratio": concentration_ratio,
            }
        )

    logger.info(f"n images with attack opportunity on {args.target}: {len(rows)} (success={sum(r['success'] for r in rows)})")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote per-image -> {args.out_csv}")

    properties = [
        "n_gt", "mean_gt_area", "surrogate_cos_dist", "noise_l1_mean", "noise_l2_norm",
        "saturation_ratio", "band_low", "band_mid", "band_high", "gt_energy_concentration_ratio",
    ]
    s_rows = [r for r in rows if r["success"] == 1]
    f_rows = [r for r in rows if r["success"] == 0]
    logger.info(f"=== S (n={len(s_rows)}) vs F (n={len(f_rows)}) on {args.target} ===")
    summary = []
    for prop in properties:
        xs_s = [r[prop] for r in s_rows]
        xs_f = [r[prop] for r in f_rows]
        mean_s = sum(xs_s) / len(xs_s) if xs_s else float("nan")
        mean_f = sum(xs_f) / len(xs_f) if xs_f else float("nan")
        d = cohens_d(xs_s, xs_f)
        summary.append({"property": prop, "mean_S": mean_s, "mean_F": mean_f, "cohens_d": d})
        logger.info(f"  {prop:30s} mean_S={mean_s:.4f}  mean_F={mean_f:.4f}  cohens_d={d:.3f}")

    with open(args.out_summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["property", "mean_S", "mean_F", "cohens_d"])
        writer.writeheader()
        writer.writerows(summary)
    logger.info(f"wrote summary -> {args.out_summary_csv}")


if __name__ == "__main__":
    main()
