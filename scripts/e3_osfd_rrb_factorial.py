#!/usr/bin/env python
"""E3: 2x2 factorial isolating OSFD's two independent ingredients --
amplification factor k (k=1 is a plain feature-MSE loss, "NRDM" per the
paper's own framing; k=3 is full OSFD suppress/amplify) and the RRB input
augmentation (on/off) -- to answer, without recrafting anything from E1/E2:

  1. How much does RRB alone contribute to transfer (holding k fixed)?
  2. Does k=3 amplification have standalone benefit over k=1 (holding RRB fixed)?
  3. Is there an interaction -- does k=3 only help WHEN combined with RRB?

Everything except {k, use_rrb} is held fixed at the current dev config: 100
steps, dev_50 manifest, epsilon/alpha/mu defaults (see constants.py). This is
the same manifest/step-count as the OSFD/MI-FGSM baseline and the E1/E2/E2b/E2c
runs, so results are directly comparable to those numbers (e.g. `k3_rrb` here
IS the existing OSFD baseline, just recrafted with a distinct noise dir).

Craft+eval all 4 variants (~40 min total: ~4 x (~8 min craft + ~2 min eval)),
write one standard run_attack run-log per variant to runs/ (so
gen_experiment_log.py's normal EXPERIMENTS.md regeneration picks them up),
then print + save a focused 2x2 comparison table (deltas answering the 3
questions above), with special attention to groups B/C and dino_swin_l --
not just the cross-model mean, per the point of this experiment.

Example:
    python scripts/e3_osfd_rrb_factorial.py --manifest data/manifests/dev_50.json --steps 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

VARIANTS = [
    {"tag": "k1_norrb", "k": 1.0, "use_rrb": False},
    {"tag": "k1_rrb", "k": 1.0, "use_rrb": True},
    {"tag": "k3_norrb", "k": 3.0, "use_rrb": False},
    {"tag": "k3_rrb", "k": 3.0, "use_rrb": True},  # = baseline OSFD (k=3, RRB on)
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "e3_osfd_rrb_factorial_summary.csv")
    return p


def craft_and_evaluate_variant(variant: dict, args, coco, image_ids, gt_index, img_dir, logger) -> list[dict]:
    import random
    from types import SimpleNamespace

    import torch
    import evaluate as evaluate_mod

    from transfer_attack.attack import AttackConfig, craft_one_image
    from transfer_attack.data import gt_to_canvas, load_canvas_image
    from transfer_attack.io_utils import save_noise, save_run_log
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle, get_spec

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tag = variant["tag"]
    cfg = AttackConfig(
        attack_type="osfd", k=variant["k"], use_rrb=variant["use_rrb"], steps=args.steps, canvas=args.canvas
    )

    noise_dir = PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"e3_{tag}"
    noise_dir.mkdir(parents=True, exist_ok=True)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=args.device, coco=coco)
    logger.info(f"[{tag}] craft: k={cfg.k} use_rrb={cfg.use_rrb} steps={cfg.steps}")

    n_crafted, n_skipped = 0, 0
    t0 = time.time()
    for i, image_id in enumerate(image_ids):
        gt_entries = gt_index[image_id]
        if not gt_entries:
            n_skipped += 1
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, cfg.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        noise, step_losses = craft_one_image(surrogate, canvas_img, gt_boxes, gt_cat_ids, cfg, device=args.device)
        save_noise(noise_dir / f"{image_id}.pt", noise)
        n_crafted += 1
        if (i + 1) % args.log_every == 0 or (i + 1) == len(image_ids):
            logger.info(
                f"[{tag}] [{i + 1}/{len(image_ids)}] crafted={n_crafted} skipped={n_skipped} "
                f"elapsed={time.time() - t0:.1f}s loss[-1]={step_losses[-1]:.4f}"
            )
    craft_elapsed = time.time() - t0
    logger.info(f"[{tag}] craft finished: {n_crafted} crafted, {n_skipped} skipped -> {noise_dir}")
    del surrogate
    torch.cuda.empty_cache()

    # predictions_dir is SHARED (not per-variant) so the clean/ subdir is computed once
    # and cached across all 4 variants -- only the adv predictions differ per variant,
    # and those are already namespaced by attack=f"e3_{tag}" so they can't collide.
    eval_args = SimpleNamespace(
        checkpoints_dir=args.checkpoints_dir,
        canvas=args.canvas,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        noise_dir=PROJECT_DIR / "results" / "noise" / args.manifest.stem,
        predictions_dir=PROJECT_DIR / "results" / "_e3_predictions",
        force_clean=False,
        device=args.device,
        attacks=[f"e3_{tag}"],
    )

    t_eval0 = time.time()
    gt_cache = evaluate_mod.build_gt_cache(coco, image_ids, img_dir, args.canvas)
    rows = []
    for spec in MODEL_REGISTRY:
        rows.extend(evaluate_mod.evaluate_one_model(spec, eval_args, coco, image_ids, img_dir, gt_cache, logger))
    eval_elapsed = time.time() - t_eval0
    logger.info(f"[{tag}] eval finished {len(MODEL_REGISTRY)} models in {eval_elapsed:.1f}s")

    run_log_path = save_run_log(
        args.runs_dir,
        "run_attack",
        f"osfd_e3_{tag}_{args.manifest.stem}",
        {
            "attack": f"e3_{tag}",
            "manifest": str(args.manifest),
            "seed": args.seed,
            "limit": args.limit,
            "score_thr": args.score_thr,
            "iou_thr": args.iou_thr,
            "config": vars(cfg),
            "results": {
                "n_images_in_manifest": len(image_ids),
                "n_crafted": n_crafted,
                "n_skipped": n_skipped,
                "craft_elapsed_sec": round(craft_elapsed, 1),
                "eval_elapsed_sec": round(eval_elapsed, 1),
                "rows": rows,
            },
        },
    )
    logger.info(f"[{tag}] run log written -> {run_log_path}")
    return rows


def build_comparison(all_rows: dict[str, list[dict]]) -> tuple[list[dict], list[str]]:
    """all_rows: tag -> list of {model_name, group, mAP_drop_pct, ASR, ...} rows
    (attack rows only, i.e. row['attack'] != 'clean')."""
    tags = [v["tag"] for v in VARIANTS]
    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for metric in ("ASR", "mAP_drop_pct"):
            for tag in tags:
                row[f"{metric}_{tag}"] = per_tag.get(tag, {}).get(metric)
            v = row
            k1_norrb, k1_rrb = v.get(f"{metric}_k1_norrb"), v.get(f"{metric}_k1_rrb")
            k3_norrb, k3_rrb = v.get(f"{metric}_k3_norrb"), v.get(f"{metric}_k3_rrb")
            if None not in (k1_norrb, k1_rrb, k3_norrb, k3_rrb):
                rrb_effect_k1 = k1_rrb - k1_norrb
                rrb_effect_k3 = k3_rrb - k3_norrb
                k_effect_norrb = k3_norrb - k1_norrb
                k_effect_rrb = k3_rrb - k1_rrb
                interaction = (k3_rrb - k1_norrb) - (rrb_effect_k1 + k_effect_norrb)
                row[f"{metric}_rrb_effect_at_k1"] = rrb_effect_k1
                row[f"{metric}_rrb_effect_at_k3"] = rrb_effect_k3
                row[f"{metric}_k_effect_at_norrb"] = k_effect_norrb
                row[f"{metric}_k_effect_at_rrb"] = k_effect_rrb
                row[f"{metric}_interaction"] = interaction
        comparison.append(row)

    fieldnames = ["model_name", "group"]
    for metric in ("ASR", "mAP_drop_pct"):
        fieldnames += [f"{metric}_{t}" for t in tags]
        fieldnames += [
            f"{metric}_rrb_effect_at_k1", f"{metric}_rrb_effect_at_k3",
            f"{metric}_k_effect_at_norrb", f"{metric}_k_effect_at_rrb", f"{metric}_interaction",
        ]
    return comparison, fieldnames


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
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"

    all_rows: dict[str, list[dict]] = {}
    for variant in VARIANTS:
        rows = craft_and_evaluate_variant(variant, args, coco, image_ids, gt_index, img_dir, logger)
        attack_key = f"e3_{variant['tag']}"
        all_rows[variant["tag"]] = [r for r in rows if r["attack"] == attack_key]

    comparison, fieldnames = build_comparison(all_rows)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    import csv

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote comparison -> {args.out_csv}")

    logger.info("=== E3 factorial summary (ASR, %) ===")
    logger.info(f"{'model':20s} {'grp':4s} {'k1_norrb':>9s} {'k1_rrb':>9s} {'k3_norrb':>9s} {'k3_rrb':>9s}")
    for r in comparison:
        logger.info(
            f"{r['model_name']:20s} {r['group']:4s} "
            f"{r.get('ASR_k1_norrb', float('nan')):9.1f} {r.get('ASR_k1_rrb', float('nan')):9.1f} "
            f"{r.get('ASR_k3_norrb', float('nan')):9.1f} {r.get('ASR_k3_rrb', float('nan')):9.1f}"
        )


if __name__ == "__main__":
    main()
