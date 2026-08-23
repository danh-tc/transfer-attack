#!/usr/bin/env python
"""E4 diagnostic (Phase N groundwork): does the ALREADY-CRAFTED OSFD
perturbation have a spectral structure where different frequency bands carry
disproportionate transfer power to different target families?

Motivation: after Phase M closed (consistency/agreement and "more stochastic
views" both ruled out as productive axes -- see RESEARCH.md SS9-10), a
literature scan of classification-domain CNN<->ViT transfer work flagged
frequency-domain / spectral constraints as a mechanism not yet explored for
object detection, and NOT overlapping with anything tried in E1-E3b or Phase
M. Before designing any spectral-aware attack, check cheaply whether there is
ANY signal at all.

Method: radially decompose each already-crafted OSFD noise tensor (2D FFT per
channel, masked by normalized radial frequency distance from DC) into three
disjoint bands -- low/mid/high, each reconstructed back to a real spatial
perturbation via inverse FFT -- so low+mid+high == the original noise (exact
up to float precision; verified in a self-check at startup). Then, WITHOUT
any recrafting, inject each band alone (clamp(x_clean + band, 0, 255)) and
the full noise (as a sanity reproduction of the known OSFD baseline numbers)
against the surrogate + 3 hard targets, reusing the existing eval pipeline
(predict_canvas / greedy IoU match / COCO mAP).

Reading key (decided before running):
  - If one band (e.g. low/mid) retains most of the DINO/Mask-Swin/YOLOX ASR
    while another band (e.g. high) mostly only hurts the surrogate/CNN ->
    positive mechanistic clue for a shared spectral vulnerability -> worth
    designing a spectral-aware attack next.
  - If no band correlates cleanly with B/C transfer (e.g. all bands roughly
    proportional to their energy share, no band punches above its weight) ->
    kill the spectral direction here, cheaply, without touching a single
    training loop.

Example:
    python scripts/e4_spectral_decomposition.py --attack osfd --manifest data/manifests/dev_50.json
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

BAND_NAMES = ["low", "mid", "high"]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["osfd", "mi_fgsm"], default="osfd")
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--noise-dir", type=Path, default=None, help="default: results/noise/<manifest-stem>/<attack>/")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument(
        "--models", nargs="+",
        default=["faster_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"],
        help="model name(s) from MODEL_REGISTRY",
    )
    p.add_argument("--band-boundaries", type=float, nargs=2, default=[1 / 3, 2 / 3], help="normalized radial cutoffs (low<=b1, mid<=b2, high>b2)")
    p.add_argument(
        "--skip-normalized", action="store_true",
        help="only evaluate raw (non-rescaled) bands, skip the L-inf-normalized comparison (E4b)",
    )
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "e4_spectral_decomposition.csv")
    return p


def decompose_bands(noise, boundaries: tuple[float, float]) -> dict[str, "torch.Tensor"]:
    """noise: (3,H,W) real. Returns {'low','mid','high'}: each (3,H,W) real,
    summing back to ~noise (radial masks in frequency space are symmetric
    under (u,v)->(-u,-v) since they depend only on |freq|, so they preserve
    the conjugate symmetry of a real signal's spectrum -> real reconstruction,
    negligible imaginary residual)."""
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

    bands = {name: torch.zeros_like(noise) for name in masks}
    for c in range(C):
        F = torch.fft.fftshift(torch.fft.fft2(noise[c]))
        for name, mask in masks.items():
            back = torch.fft.ifft2(torch.fft.ifftshift(F * mask)).real
            bands[name][c] = back
    return bands


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

    from transfer_attack.constants import COCO_ANN_FILE, EPSILON
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.eval_metrics import compute_asr_for_image, compute_coco_map, predict_canvas, to_coco_results
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

    # ---- precompute per-image band decomposition once, reused across all models ----
    per_image = {}  # image_id -> {"x_clean":..., "gt_boxes":..., "gt_cat_ids":..., "bands": {...}, "energy": {...}}
    recon_errs = []
    band_energy_frac = {b: [] for b in BAND_NAMES}
    n_missing_noise = 0
    for image_id in image_ids:
        noise_path = noise_dir / f"{image_id}.pt"
        if not noise_path.exists():
            n_missing_noise += 1
            continue
        gt_entries = gt_index[image_id]
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        noise = load_noise(noise_path).to(device)
        bands = decompose_bands(noise, tuple(args.band_boundaries))

        recon = sum(bands.values())
        recon_errs.append((recon - noise).abs().max().item())
        total_energy = (noise**2).sum().item() + 1e-12
        for b in BAND_NAMES:
            band_energy_frac[b].append((bands[b] ** 2).sum().item() / total_energy)

        # E4b: rescale each band to the SAME L-inf budget as the original attack
        # (constants.EPSILON) -- separates "this band matters" from "this band
        # just has less raw amplitude because it holds less energy".
        bands_norm = {}
        for name, band in bands.items():
            peak = band.abs().max().item()
            band_scale = (EPSILON / peak) if peak > 1e-8 else 0.0
            bands_norm[name] = torch.clamp(band * band_scale, -EPSILON, EPSILON)

        per_image[image_id] = {
            "x_clean": canvas_img.to(device),
            "scale": scale,
            "gt_boxes": gt_boxes,
            "gt_cat_ids": gt_cat_ids,
            "bands": bands,
            "bands_norm": bands_norm,
            "noise": noise,
        }
    if n_missing_noise:
        logger.warning(f"{n_missing_noise} images had no crafted noise under {noise_dir} -- skipped")
    logger.info(
        f"reconstruction check (low+mid+high vs original noise): max abs err = {max(recon_errs):.6f} "
        f"(should be ~0, float residual only)"
    )
    for b in BAND_NAMES:
        vals = band_energy_frac[b]
        logger.info(f"band '{b}': mean energy fraction of total noise = {sum(vals)/len(vals):.4f}")

    variants = ["full"] + BAND_NAMES
    if not args.skip_normalized:
        variants += [f"{b}_norm" for b in BAND_NAMES]
    rows = []
    for model_name in args.models:
        spec = get_spec(model_name)
        logger.info(f"=== {spec.name} (group={spec.group}) ===")
        handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)

        clean_preds = {}
        for image_id, d in per_image.items():
            clean_preds[image_id] = predict_canvas(handle, d["x_clean"], args.canvas, device=device)

        scale_by_image_id = {i: per_image[i]["scale"] for i in per_image}
        clean_map = compute_coco_map(
            coco, to_coco_results(clean_preds, {int(l): int(l) for p in clean_preds.values() for l in p["labels"].tolist()}, scale_by_image_id),
            list(per_image.keys()),
        )

        for variant in variants:
            adv_preds = {}
            for image_id, d in per_image.items():
                if variant == "full":
                    pert = d["noise"]
                elif variant.endswith("_norm"):
                    pert = d["bands_norm"][variant[: -len("_norm")]]
                else:
                    pert = d["bands"][variant]
                x_adv = (d["x_clean"] + pert).clamp(0.0, 255.0)
                adv_preds[image_id] = predict_canvas(handle, x_adv, args.canvas, device=device)

            adv_map = compute_coco_map(
                coco,
                to_coco_results(adv_preds, {int(l): int(l) for p in adv_preds.values() for l in p["labels"].tolist()}, scale_by_image_id),
                list(per_image.keys()),
            )

            evaded_total, clean_correct_total = 0, 0
            for image_id, d in per_image.items():
                if d["gt_boxes"].shape[0] == 0:
                    continue
                evaded, clean_correct = compute_asr_for_image(
                    clean_preds[image_id], adv_preds[image_id], d["gt_boxes"], d["gt_cat_ids"], args.iou_thr, args.score_thr
                )
                evaded_total += evaded
                clean_correct_total += clean_correct
            asr = 100.0 * evaded_total / clean_correct_total if clean_correct_total > 0 else float("nan")
            map_drop = 100.0 * (clean_map["AP"] - adv_map["AP"]) / clean_map["AP"] if clean_map["AP"] > 0 else float("nan")

            if variant == "full":
                mean_energy_frac = 1.0
            elif variant.endswith("_norm"):
                mean_energy_frac = float("nan")  # rescaled to fixed L-inf budget -- energy fraction no longer meaningful
            else:
                mean_energy_frac = sum(band_energy_frac[variant]) / len(band_energy_frac[variant])
            rows.append(
                {
                    "model_name": spec.name,
                    "group": spec.group or "",
                    "variant": variant,
                    "mean_energy_frac": mean_energy_frac,
                    "mAP_clean": clean_map["AP"],
                    "mAP_adv": adv_map["AP"],
                    "mAP_drop_pct": map_drop,
                    "ASR": asr,
                    "n_evaded": evaded_total,
                    "n_clean_correct": clean_correct_total,
                }
            )
            logger.info(
                f"  [{variant:5s}] energy_frac={mean_energy_frac:.3f} ASR={asr:.1f}% "
                f"({evaded_total}/{clean_correct_total}) mAP_drop={map_drop:.1f}%"
            )

        del handle
        torch.cuda.empty_cache()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_name", "group", "variant", "mean_energy_frac", "mAP_clean", "mAP_adv",
                        "mAP_drop_pct", "ASR", "n_evaded", "n_clean_correct"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
