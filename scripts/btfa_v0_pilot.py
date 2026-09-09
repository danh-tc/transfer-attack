#!/usr/bin/env python
"""BTFA v0 pilot -- Boundary Transition Field Attack, WHITE-BOX SANITY GATE.

Proposed after DBTA (RESEARCH.md Sec 30): DBTA's dense per-POINT boundary
sampling (~24 discrete O/E/Bn triplets/object) passed white-box sanity
(unlike TGA's 3-scalar pooling, which collapsed even white-box) and roughly
tied OSFD-noRRB on 2/3 hard targets, but adding RRB widened rather than
closed the gap -- OSFD's global feature-map MSE benefits far more from RRB's
multi-view averaging than DBTA's spatially-sparse (perimeter-only) objective.
BTFA replaces point sampling with a continuous FIELD: a signed-distance
function around each GT box defines a boundary "band," and the loss ascends
disruption of the feature map's directional derivative (along the field's
normal) at every pixel in that band -- the dense limit of DBTA's per-point
finite differences. See transfer_attack/btfa.py for the full construction.

This pilot is scoped exactly like DBTA's: RRB OFF only (BTFA doesn't support
RRB yet -- see btfa.py's docstring), white-box sanity checked BEFORE any
hard-target read. Threshold here is an ABSOLUTE floor (ASR_btfa on the
surrogate >= WB_MIN_ASR), not a delta vs OSFD, per the pre-registered plan.

4 arms, RRB OFF, same budget (epsilon/alpha/steps):
  mi_fgsm    -- detector task loss (paper's baseline), reference.
  osfd_norrb -- OSFD backbone-feature MSE (Eq. 2), k=3, no RRB.
  dbta_norrb -- DBTA dense per-point boundary transition (Sec 30's best-so-far
                E10-derived candidate), re-crafted fresh in this same run for
                a directly comparable baseline (not read from Sec 30's numbers
                -- minor GPU-nondeterminism run-to-run variance is expected).
  btfa_norrb -- dense boundary-transition FIELD (this experiment's candidate).

Models: same 6 as E10/TGA/DBTA (DEFAULT_MODELS).

Verdict (pre-registered before running):
  Phase 1 (white-box gate) -- FAIL if btfa_norrb's ASR on faster_rcnn_r50 <
    WB_MIN_ASR (80%). This is an absolute floor, not a delta: TGA's failure
    was catastrophic collapse (far below OSFD/MI-FGSM); DBTA passed cleanly
    at ~90%. If BTFA falls back into TGA's regime, close the field-based
    formulation without touching hard targets.
  Phase 2 (hard targets, only read if Phase 1 passes) -- compare BOTH vs
    osfd_norrb (primary, same +-5/2-of-3 convention as TGA/DBTA's pilots) and
    vs dbta_norrb (secondary -- specifically interesting on mask_rcnn_swin_t,
    DBTA's weakest point, to test whether a dense field succeeds where
    DBTA's point sampling didn't). No steps/n_points/lambda tuning regardless
    of outcome -- same no-rescue discipline as every prior candidate.

Example:
    python scripts/btfa_v0_pilot.py --manifest data/manifests/dev_50.json --n-images 20
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
    {"tag": "osfd_norrb", "attack_type": "osfd", "use_rrb": False},
    {"tag": "dbta_norrb", "attack_type": "dbta", "use_rrb": False, "baseline_tag": "osfd_norrb"},
    {"tag": "btfa_norrb", "attack_type": "btfa", "use_rrb": False, "baseline_tag": "osfd_norrb"},
    {"tag": "mi_fgsm", "attack_type": "mi_fgsm", "use_rrb": False},
]
WHITEBOX_MODEL = "faster_rcnn_r50"
DEFAULT_MODELS = ["faster_rcnn_r50", "dino_r50", "mask_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"]
HARD_TARGETS = ["yolox_l", "mask_rcnn_swin_t", "dino_swin_l"]

WB_MIN_ASR = 80.0         # absolute ASR floor on the white-box surrogate for btfa_norrb -- FAIL below this
GO_DELTA_THR = 5.0        # ASR points, btfa - baseline, on a hard target -> counts as a "win"
GO_MIN_WINS = 2
COLLAPSE_THR = -20.0


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
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "btfa_v0_pilot_summary.csv")
    p.add_argument("--run-tags", nargs="+", default=None, help="subset of VARIANTS tags to craft+eval (default: all 4)")
    p.add_argument("--models", nargs="+", default=None, help=f"subset of models to evaluate (default: {DEFAULT_MODELS})")
    return p


def craft_and_evaluate_variant(variant: dict, args, coco, image_ids, gt_index, img_dir, specs, logger, run_id: str) -> list[dict]:
    from types import SimpleNamespace

    import torch
    import evaluate as evaluate_mod

    from transfer_attack.attack import AttackConfig, craft_one_image
    from transfer_attack.data import gt_to_canvas, load_canvas_image
    from transfer_attack.io_utils import save_noise, save_run_log
    from transfer_attack.models import build_model_handle, get_spec

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tag = variant["tag"]
    cfg = AttackConfig(
        attack_type=variant["attack_type"],
        use_rrb=variant["use_rrb"],
        steps=args.steps,
        canvas=args.canvas,
    )

    noise_dir = PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"btfa_v0_{tag}"
    noise_dir.mkdir(parents=True, exist_ok=True)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=args.device, coco=coco)
    logger.info(f"[{tag}] craft: attack_type={variant['attack_type']} use_rrb={variant['use_rrb']}")

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
        if gt_boxes.shape[0] == 0:
            n_skipped += 1
            continue
        try:
            noise, step_losses = craft_one_image(surrogate, canvas_img, gt_boxes, gt_cat_ids, cfg, device=args.device)
        except ValueError as e:
            logger.warning(f"[{tag}] image_id={image_id}: skipping ({e})")
            n_skipped += 1
            continue
        save_noise(noise_dir / f"{image_id}.pt", noise)
        used_image_ids.append(image_id)
        n_crafted += 1
        if n_crafted % args.log_every == 0:
            logger.info(f"[{tag}] [{n_crafted}/{args.n_images}] elapsed={time.time() - t0:.1f}s loss[-1]={step_losses[-1]:.4f}")
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
        # Timestamped per script invocation -- evaluate.py's adversarial-prediction
        # cache is keyed only by (model, attack tag), not noise content; a fixed
        # path here bit us for real during DBTA's RRB fix (RESEARCH.md Sec 30) --
        # a smoke test after fixing the crafting code silently reused predictions
        # cached from the pre-fix noise because the tag+dir were unchanged.
        predictions_dir=PROJECT_DIR / "results" / "_btfa_v0_pilot_predictions" / run_id,
        force_clean=False,
        device=args.device,
        attacks=[f"btfa_v0_{tag}"],
    )

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
        f"btfa_v0_{tag}_{args.manifest.stem}_n{args.n_images}",
        {
            "attack": f"btfa_v0_{tag}",
            "manifest": str(args.manifest),
            "seed": args.seed,
            "n_images": args.n_images,
            "score_thr": args.score_thr,
            "iou_thr": args.iou_thr,
            "config": {**vars(cfg), **variant},
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
    return [r for r in rows if r["attack"] == f"btfa_v0_{tag}"]


def main() -> None:
    args = build_arg_parser().parse_args()

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger
    from transfer_attack.models import MODEL_REGISTRY

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"

    model_names = args.models or DEFAULT_MODELS
    by_name = {s.name: s for s in MODEL_REGISTRY}
    specs = [by_name[m] for m in model_names]

    run_variants = [v for v in VARIANTS if not args.run_tags or v["tag"] in args.run_tags]
    run_id = time.strftime("%Y%m%dT%H%M%S")

    all_rows: dict[str, list[dict]] = {}
    for variant in run_variants:
        all_rows[variant["tag"]] = craft_and_evaluate_variant(variant, args, coco, image_ids, gt_index, img_dir, specs, logger, run_id)

    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    import csv

    run_tags = [v["tag"] for v in run_variants]
    fieldnames = (
        ["model_name", "group"]
        + [f"ASR_{t}" for t in run_tags]
        + [f"mAP_drop_{t}" for t in run_tags]
        + (["delta_btfa_vs_osfd_ASR", "delta_btfa_vs_dbta_ASR"] if "btfa_norrb" in run_tags else [])
    )
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for tag in run_tags:
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
            row[f"mAP_drop_{tag}"] = r.get("mAP_drop_pct")
        if "btfa_norrb" in run_tags:
            if row.get("ASR_btfa_norrb") is not None and row.get("ASR_osfd_norrb") is not None:
                row["delta_btfa_vs_osfd_ASR"] = row["ASR_btfa_norrb"] - row["ASR_osfd_norrb"]
            if row.get("ASR_btfa_norrb") is not None and row.get("ASR_dbta_norrb") is not None:
                row["delta_btfa_vs_dbta_ASR"] = row["ASR_btfa_norrb"] - row["ASR_dbta_norrb"]
        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info(f"=== BTFA v0 pilot: ASR (%) per arm ({', '.join(run_tags)}) ===")
    for r in comparison:
        parts = [f"{tag}={r.get(f'ASR_{tag}', float('nan')):.1f}" for tag in run_tags]
        extra = ""
        if "btfa_norrb" in run_tags:
            extra = f"  (btfa-osfd={r.get('delta_btfa_vs_osfd_ASR', float('nan')):+.1f}, btfa-dbta={r.get('delta_btfa_vs_dbta_ASR', float('nan')):+.1f})"
        logger.info(f"{r['model_name']:20s} {r['group'] or '-':4s} " + " ".join(parts) + extra)

    if "btfa_norrb" not in run_tags:
        logger.info("=== btfa_norrb not in this run -- no verdict to compute ===")
        return

    by_model_name = {r["model_name"]: r for r in comparison}
    wb_row = by_model_name.get(WHITEBOX_MODEL)
    wb_asr = wb_row.get("ASR_btfa_norrb") if wb_row else None

    if wb_asr is None:
        logger.info(f"=== INCONCLUSIVE -- white-box model {WHITEBOX_MODEL!r} not evaluated. ===")
        return

    logger.info(f"=== Phase 1 (white-box gate): btfa_norrb ASR on {WHITEBOX_MODEL} = {wb_asr:.1f} (need >= {WB_MIN_ASR}) ===")
    if wb_asr < WB_MIN_ASR:
        logger.info(
            f"PHASE 1 FAIL -- btfa_norrb collapses white-box (< {WB_MIN_ASR}), same failure class as TGA "
            f"(RESEARCH.md Sec 29). Close the field-based formulation; do not tune band width/kernel to rescue it."
        )
        return

    logger.info("PHASE 1 PASS -- checking Phase 2 (hard-target deltas vs osfd_norrb and dbta_norrb) next.")
    hard_deltas_osfd = {}
    hard_deltas_dbta = {}
    for m in HARD_TARGETS:
        r = by_model_name.get(m)
        if r is None:
            continue
        if r.get("delta_btfa_vs_osfd_ASR") is not None:
            hard_deltas_osfd[m] = r["delta_btfa_vs_osfd_ASR"]
        if r.get("delta_btfa_vs_dbta_ASR") is not None:
            hard_deltas_dbta[m] = r["delta_btfa_vs_dbta_ASR"]

    n_wins_osfd = sum(1 for d in hard_deltas_osfd.values() if d >= GO_DELTA_THR)
    n_wins_dbta = sum(1 for d in hard_deltas_dbta.values() if d >= GO_DELTA_THR)
    logger.info(f"btfa vs osfd_norrb: wins on {n_wins_osfd}/{len(hard_deltas_osfd)} hard targets: {hard_deltas_osfd}")
    logger.info(f"btfa vs dbta_norrb: wins on {n_wins_dbta}/{len(hard_deltas_dbta)} hard targets: {hard_deltas_dbta}")

    ms_key = "mask_rcnn_swin_t"
    if ms_key in hard_deltas_dbta and hard_deltas_dbta[ms_key] >= GO_DELTA_THR:
        logger.info(
            f"NOTABLE: btfa beats dbta_norrb on {ms_key} by {hard_deltas_dbta[ms_key]:+.1f} -- DBTA's weakest "
            f"point (Sec 30). Supports the hypothesis that a dense field succeeds where point sampling didn't."
        )

    if len(hard_deltas_osfd) < len(HARD_TARGETS):
        logger.info("PHASE 2 INCONCLUSIVE -- not all hard targets evaluated vs osfd_norrb.")
    elif n_wins_osfd >= GO_MIN_WINS:
        losers = {m: d for m, d in hard_deltas_osfd.items() if d < GO_DELTA_THR}
        collapsed = {m: d for m, d in losers.items() if d <= COLLAPSE_THR}
        if not collapsed:
            logger.info(f"STRONG GO vs osfd_norrb -- beats it by >={GO_DELTA_THR} on {n_wins_osfd}/3 hard targets, no collapse on the rest.")
        else:
            logger.info(f"NO-GO (collapse guard failed) vs osfd_norrb -- wins on {n_wins_osfd}/3 but catastrophic drop on {collapsed}.")
    elif hard_deltas_osfd.get("dino_swin_l", -1e9) >= GO_DELTA_THR:
        logger.info(f"WEAK GO vs osfd_norrb -- beats it on dino_swin_l ({hard_deltas_osfd['dino_swin_l']:+.1f}) but not >=2/3 overall.")
    else:
        logger.info(
            "NO-GO vs osfd_norrb -- btfa does not beat it by >={:.0f} on >={:d}/3 hard targets. Per pre-registered "
            "discipline, close the E10-derived attack-objective line here (no width sweep / kernel tuning / stage "
            "reweighting).".format(GO_DELTA_THR, GO_MIN_WINS)
        )


if __name__ == "__main__":
    main()
