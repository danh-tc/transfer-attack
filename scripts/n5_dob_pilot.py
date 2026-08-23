#!/usr/bin/env python
"""Phase N4 method pilot: Difficulty-Aware Object Budgeting v0 (DOB), the
first designed method after N4/N4b's positive findings -- NOT a "redistribute
total L1 budget" scheme (Linf has no conserved total to redistribute); more
precisely a difficulty-aware SPATIAL UPDATE-RATE map. osfd_loss is completely
UNCHANGED -- the only difference from plain OSFD is the per-pixel step size:

    delta_{t+1} = clip(delta_t + alpha * W(x) [x] sign(g_mom), -eps, eps)

W(x): built ONCE per image (before crafting starts) from the SURROGATE's own
clean detections matched to GT (white-box, no target-model access needed --
realistic threat model). Per matched GT object j: confidence c_j (0 if the
surrogate doesn't confidently detect it at all), difficulty d_j = 1 - c_j.
Per-object weight centered on the image's own mean difficulty (not a global
constant, so this adapts per image):

    w_j = clip(1 + sign * beta * (d_j - mean_d), w_min, w_max)

sign=+1 for DOB-EASY (low-confidence/easy-to-evade objects get the LARGER
per-pixel step -- what N4b's marginal-response finding supports), sign=-1 for
DOB-HARD (the opposite, included specifically so the pilot can tell whether
N4b's causal direction is right, not just assumed). Background (no object)
and pixels outside any GT box keep w=1 (unchanged from plain OSFD). Overlap
between objects' boxes takes max(w_i, w_j), not sum -- avoids runaway
amplification, per the explicit design call.

epsilon/alpha/steps/k/RRB all unchanged from plain OSFD -- W only changes
which pixels REACH the Linf boundary faster, never lets any pixel exceed it
(the clip(-eps, eps) projection still runs every step).

Three variants, same budget:
  osfd      -- W = 1 everywhere (current baseline, unchanged).
  dob_easy  -- W built with sign=+1 (favor low-confidence/easy objects).
  dob_hard  -- W built with sign=-1 (favor high-confidence/hard objects).

GO criterion (decided before running):
  ASR_DOB-EASY - ASR_OSFD >= +5 on dino_swin_l, AND
  ASR_DOB-EASY > ASR_DOB-HARD clearly.
If only white-box/surrogate improves, or EASY ~= HARD, the difficulty-aware
allocation hypothesis is weak.

Example:
    python scripts/n5_dob_pilot.py --manifest data/manifests/dev_50.json --n-images 20
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

VARIANTS = [
    {"tag": "osfd", "direction": None},
    {"tag": "dob_easy", "direction": "easy"},
    {"tag": "dob_hard", "direction": "hard"},
]

BETA = 1.0
W_MIN = 0.5
W_MAX = 1.5


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--w-min", type=float, default=W_MIN)
    p.add_argument("--w-max", type=float, default=W_MAX)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n5_dob_pilot_summary.csv")
    p.add_argument("--run-tags", nargs="+", default=None)
    p.add_argument(
        "--models", nargs="+", default=["faster_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"],
    )
    return p


def compute_weight_map(canvas_size, gt_boxes, gt_cat_ids, surrogate_pred, iou_thr, score_thr, beta, w_min, w_max, direction, device):
    import torch

    from transfer_attack.eval_metrics import greedy_match

    W = torch.ones((canvas_size, canvas_size), device=device)
    if direction is None or gt_boxes.shape[0] == 0:
        return W

    match = greedy_match(
        surrogate_pred["bboxes"], surrogate_pred["scores"], surrogate_pred["labels"],
        gt_boxes.cpu(), gt_cat_ids.cpu(), iou_thr, score_thr,
    )
    difficulties = []
    for g_idx, m in enumerate(match):
        conf = float(surrogate_pred["scores"][m].item()) if m is not None else 0.0
        difficulties.append(1.0 - conf)
    mean_d = sum(difficulties) / len(difficulties)
    sign = 1.0 if direction == "easy" else -1.0

    for box, d in zip(gt_boxes.tolist(), difficulties):
        w = 1.0 + sign * beta * (d - mean_d)
        w = min(max(w, w_min), w_max)
        x1, y1, x2, y2 = [max(0, min(canvas_size, int(round(v)))) for v in box]
        if x2 <= x1 or y2 <= y1:
            continue
        region = W[y1:y2, x1:x2]
        W[y1:y2, x1:x2] = torch.maximum(region, torch.full_like(region, w))
    return W


def craft_one_image_dob(handle, x_clean, gt_boxes, gt_cat_ids, cfg, direction, beta, w_min, w_max, iou_thr, score_thr, device):
    import torch

    from transfer_attack.augment import rrb_forward
    from transfer_attack.eval_metrics import predict_canvas
    from transfer_attack.losses import osfd_loss

    model = handle.model
    x_clean = x_clean.to(device)
    gt_boxes = gt_boxes.to(device)
    gt_cat_ids = gt_cat_ids.to(device)

    if direction is not None:
        with torch.no_grad():
            surrogate_pred = predict_canvas(handle, x_clean, x_clean.shape[-1], device=device)
        W = compute_weight_map(
            x_clean.shape[-1], gt_boxes, gt_cat_ids, surrogate_pred, iou_thr, score_thr, beta, w_min, w_max, direction, device
        )
    else:
        W = torch.ones((x_clean.shape[-2], x_clean.shape[-1]), device=device)
    W = W.unsqueeze(0)  # (1,H,W) broadcasts over the 3 channels

    with torch.no_grad():
        feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))

    noise = torch.randint_like(x_clean, low=-2, high=3).float()
    g_mom = torch.zeros_like(x_clean)

    for _ in range(cfg.steps):
        noise = noise.detach().requires_grad_(True)
        x_adv = torch.clamp(x_clean + noise, 0.0, 255.0)
        aug = rrb_forward(x_adv.unsqueeze(0), gt_boxes, cfg)
        feats_adv = model.backbone(handle.normalize(aug))
        loss = osfd_loss(feats_cln, feats_adv, cfg.k)
        loss.backward()

        with torch.no_grad():
            g = noise.grad
            g_mom = cfg.mu * g_mom + g / g.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise = torch.clamp(noise + cfg.alpha * W * torch.sign(g_mom), -cfg.epsilon, cfg.epsilon)

    return noise.detach()


def craft_and_evaluate_variant(variant: dict, args, coco, image_ids, gt_index, img_dir, logger) -> list[dict]:
    from types import SimpleNamespace

    import torch
    import evaluate as evaluate_mod

    from transfer_attack.attack import AttackConfig
    from transfer_attack.data import gt_to_canvas, load_canvas_image
    from transfer_attack.io_utils import save_noise, save_run_log
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle, get_spec

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tag = variant["tag"]
    cfg = AttackConfig(attack_type="osfd", k=3.0, use_rrb=True, steps=args.steps, canvas=args.canvas)

    noise_dir = PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"n5_{tag}"
    noise_dir.mkdir(parents=True, exist_ok=True)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=args.device, coco=coco)
    logger.info(f"[{tag}] craft: direction={variant['direction']}")

    n_crafted, n_skipped = 0, 0
    t0 = time.time()
    used_image_ids = []
    for image_id in image_ids:
        if len(used_image_ids) >= args.n_images:
            break
        gt_entries = gt_index[image_id]
        if not gt_entries:
            n_skipped += 1
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, cfg.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        noise = craft_one_image_dob(
            surrogate, canvas_img, gt_boxes, gt_cat_ids, cfg,
            variant["direction"], args.beta, args.w_min, args.w_max, args.iou_thr, args.score_thr, args.device,
        )
        save_noise(noise_dir / f"{image_id}.pt", noise)
        used_image_ids.append(image_id)
        n_crafted += 1
        if n_crafted % args.log_every == 0:
            logger.info(f"[{tag}] [{n_crafted}/{args.n_images}] elapsed={time.time() - t0:.1f}s")
    craft_elapsed = time.time() - t0
    logger.info(f"[{tag}] craft finished: {n_crafted} crafted, {n_skipped} skipped -> {noise_dir}")
    del surrogate
    torch.cuda.empty_cache()

    eval_args = SimpleNamespace(
        checkpoints_dir=args.checkpoints_dir,
        canvas=args.canvas,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        noise_dir=PROJECT_DIR / "results" / "noise" / args.manifest.stem,
        predictions_dir=PROJECT_DIR / "results" / "_n5_predictions",
        force_clean=False,
        device=args.device,
        attacks=[f"n5_{tag}"],
    )

    specs = MODEL_REGISTRY
    if args.models:
        by_name = {s.name: s for s in MODEL_REGISTRY}
        specs = [by_name[m] for m in args.models]

    t_eval0 = time.time()
    gt_cache = evaluate_mod.build_gt_cache(coco, used_image_ids, img_dir, args.canvas)
    rows = []
    for spec in specs:
        rows.extend(evaluate_mod.evaluate_one_model(spec, eval_args, coco, used_image_ids, img_dir, gt_cache, logger))
    eval_elapsed = time.time() - t_eval0
    logger.info(f"[{tag}] eval finished {len(specs)} models in {eval_elapsed:.1f}s")

    run_log_path = save_run_log(
        args.runs_dir,
        "run_attack",
        f"osfd_n5_{tag}_{args.manifest.stem}_n{args.n_images}",
        {
            "attack": f"n5_{tag}",
            "manifest": str(args.manifest),
            "seed": args.seed,
            "n_images": args.n_images,
            "score_thr": args.score_thr,
            "iou_thr": args.iou_thr,
            "config": {**vars(cfg), **variant, "beta": args.beta, "w_min": args.w_min, "w_max": args.w_max},
            "results": {
                "n_images_used": len(used_image_ids),
                "n_crafted": n_crafted,
                "n_skipped": n_skipped,
                "craft_elapsed_sec": round(craft_elapsed, 1),
                "eval_elapsed_sec": round(eval_elapsed, 1),
                "rows": rows,
            },
        },
    )
    logger.info(f"[{tag}] run log written -> {run_log_path}")
    return [r for r in rows if r["attack"] == f"n5_{tag}"]


def main() -> None:
    args = build_arg_parser().parse_args()

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"

    run_variants = [v for v in VARIANTS if not args.run_tags or v["tag"] in args.run_tags]

    all_rows: dict[str, list[dict]] = {}
    for variant in run_variants:
        all_rows[variant["tag"]] = craft_and_evaluate_variant(variant, args, coco, image_ids, gt_index, img_dir, logger)

    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    import csv

    fieldnames = ["model_name", "group", "ASR_osfd", "ASR_dob_easy", "ASR_dob_hard",
                  "mAP_drop_osfd", "mAP_drop_dob_easy", "mAP_drop_dob_hard",
                  "delta_easy_vs_osfd", "delta_easy_vs_hard"]
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for tag in ("osfd", "dob_easy", "dob_hard"):
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
            row[f"mAP_drop_{tag}"] = r.get("mAP_drop_pct")
        if row.get("ASR_osfd") is not None and row.get("ASR_dob_easy") is not None:
            row["delta_easy_vs_osfd"] = row["ASR_dob_easy"] - row["ASR_osfd"]
        if row.get("ASR_dob_hard") is not None and row.get("ASR_dob_easy") is not None:
            row["delta_easy_vs_hard"] = row["ASR_dob_easy"] - row["ASR_dob_hard"]
        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info("=== N5 DOB pilot: ASR (%) osfd -> dob_easy -> dob_hard ===")
    for r in comparison:
        logger.info(
            f"{r['model_name']:20s} {r['group'] or '-':4s} "
            f"osfd={r.get('ASR_osfd', float('nan')):.1f} easy={r.get('ASR_dob_easy', float('nan')):.1f} "
            f"hard={r.get('ASR_dob_hard', float('nan')):.1f}  "
            f"(easy-osfd={r.get('delta_easy_vs_osfd', float('nan')):+.1f}, "
            f"easy-hard={r.get('delta_easy_vs_hard', float('nan')):+.1f})"
        )


if __name__ == "__main__":
    main()
