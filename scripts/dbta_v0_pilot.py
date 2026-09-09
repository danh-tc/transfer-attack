#!/usr/bin/env python
"""DBTA v0 pilot -- Dense Boundary-Transition Attack, WHITE-BOX SANITY CHECK.

Proposed after TGA (RESEARCH.md Sec 29) tested 3 variants (rank all-stage,
rank last-stage-only, magnitude-based) that were ALL NO-GO, losing to
OSFD-noRRB on every model including the white-box surrogate itself (-22 to
-39 ASR points) -- diagnosed as the root cause being SIGNAL SPARSITY: pooling
O/E/Bn to one (C,) mean vector per object leaves only 3 scalar relations to
optimize, regardless of loss shape. DBTA keeps the same region margins
(shrink_frac/expand_frac) but samples a DENSE field of points along each GT
box's perimeter (transfer_attack/dbta.py::boundary_contour_points) and
disrupts the per-point interior->edge->near-background transition vector at
every one of them via bilinear grid_sample -- far more gradient-carrying
terms per object than TGA's 3 scalars.

This pilot deliberately did NOT jump straight to a hard-target transfer
comparison. Per the plan: first confirm DBTA is even a COMPETITIVE white-box
objective (same surrogate used to craft) before asking whether it transfers
better than OSFD. TGA's failure was not really about transfer -- it lost on
the surrogate it was crafted against. If DBTA repeated that pattern, the
right conclusion would be that E10's relational invariant should be accepted
as a diagnostic/mechanism finding, not forced into a direct attack objective.

**RRB-off round 1 result (N=20/100-step, kept for reference)**: WHITE-BOX
SANITY PASS -- dbta_norrb only -5.3 ASR points vs osfd_norrb on the surrogate
(vs TGA's -22 to -39 on every model), and roughly TIED with osfd_norrb on 2/3
hard targets (yolox_l -2.0, dino_swin_l +0.0), still behind on
mask_rcnn_swin_t (-11.3). Not yet a GO, but a clean pass at the sanity stage
that TGA never reached -- confirmed the "signal sparsity" diagnosis: denser
per-point gradient terms (~24/object) made this a competitive ascent
objective where TGA's 3-scalar summary wasn't.

**Round 2 (this version)**: since DBTA is no longer white-box-bottlenecked,
add RRB -- the single biggest transfer driver in this project's own history
(E3, RESEARCH.md Sec 8: RRB alone adds +13 to +31 ASR points on group B/C
targets). New arms `osfd` (RRB on, the real full-strength baseline) and
`dbta_rrb` (RRB on) let us ask the fair comparison: does DBTA+RRB close or
invert the gap to OSFD+RRB on the hard targets, especially mask_rcnn_swin_t
(DBTA's weakest point in round 1)?

5 arms, same budget (epsilon/alpha/steps) otherwise:
  mi_fgsm    -- detector task loss (paper's baseline), reference, RRB off.
  osfd_norrb -- OSFD backbone-feature MSE (Eq. 2), k=3, RRB off (round 1 baseline).
  dbta_norrb -- dense boundary-transition MSE, RRB off (round 1 candidate).
  osfd       -- OSFD, k=3, RRB ON (project's real default/strongest baseline).
  dbta_rrb   -- dense boundary-transition MSE, RRB ON (round 2 candidate).

Models: same 6 as E10/TGA (DEFAULT_MODELS).

Verdict (pre-registered before running round 1; round 2 reuses the same
hard-target win convention against EACH variant's own matching baseline --
dbta_norrb vs osfd_norrb, dbta_rrb vs osfd):
  WHITE-BOX SANITY FAIL -- a dbta_* variant's ASR on faster_rcnn_r50 (the
    surrogate it was crafted against) is more than WB_COLLAPSE_THR points
    below its baseline's. Same failure signature as ALL 3 TGA variants (-22
    to -39 points) -- if it repeats, close that variant's line; do not tune
    n_points/margins/RRB hyperparameters to rescue it.
  WHITE-BOX SANITY PASS -- otherwise. Only then do the hard-target deltas
    (yolox_l, mask_rcnn_swin_t, dino_swin_l vs the matching baseline,
    +-5/2-of-3 convention) become informative.

Example:
    python scripts/dbta_v0_pilot.py --manifest data/manifests/dev_50.json --n-images 20 --run-tags osfd dbta_rrb
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
    {"tag": "mi_fgsm", "attack_type": "mi_fgsm", "use_rrb": False},
    {"tag": "osfd", "attack_type": "osfd", "use_rrb": True},
    {"tag": "dbta_rrb", "attack_type": "dbta", "use_rrb": True, "baseline_tag": "osfd"},
]
DEFAULT_BASELINE_TAG = "osfd_norrb"
WHITEBOX_MODEL = "faster_rcnn_r50"

DEFAULT_MODELS = ["faster_rcnn_r50", "dino_r50", "mask_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"]
HARD_TARGETS = ["yolox_l", "mask_rcnn_swin_t", "dino_swin_l"]

WB_COLLAPSE_THR = -20.0   # ASR points, dbta - osfd on the white-box surrogate -- below this = sanity FAIL
GO_DELTA_THR = 5.0        # ASR points, dbta - osfd on a hard target -> counts as a "win" (only read if sanity PASS)
GO_MIN_WINS = 2
COLLAPSE_THR = -20.0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--n-points", type=int, default=24, help="DBTA boundary contour points per object")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "dbta_v0_pilot_summary.csv")
    p.add_argument("--run-tags", nargs="+", default=None, help="subset of VARIANTS tags to craft+eval (default: all 3)")
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
        dbta_n_points=args.n_points,
        steps=args.steps,
        canvas=args.canvas,
    )

    noise_dir = PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"dbta_v0_{tag}"
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
        # Timestamped per script invocation (NOT a fixed path) -- evaluate.py's
        # adversarial-prediction cache is keyed only by (model, attack tag), not
        # by noise content, so reusing a fixed dir across separate invocations
        # that changed the crafting code for the SAME tag would silently serve
        # STALE predictions from the old (possibly buggy) noise. Hit this for
        # real once already (dbta_rrb's naive-fixed-point bug fix, see
        # RESEARCH.md) -- a smoke test after the fix reused predictions cached
        # from the pre-fix run because the tag+dir were unchanged.
        predictions_dir=PROJECT_DIR / "results" / "_dbta_v0_pilot_predictions" / run_id,
        force_clean=False,
        device=args.device,
        attacks=[f"dbta_v0_{tag}"],
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
        f"dbta_v0_{tag}_{args.manifest.stem}_n{args.n_images}",
        {
            "attack": f"dbta_v0_{tag}",
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
    return [r for r in rows if r["attack"] == f"dbta_v0_{tag}"]


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
    dbta_tags = [v["tag"] for v in run_variants if v["attack_type"] == "dbta"]
    variant_by_tag = {v["tag"]: v for v in run_variants}
    baseline_of = {t: variant_by_tag[t].get("baseline_tag", DEFAULT_BASELINE_TAG) for t in dbta_tags}

    fieldnames = (
        ["model_name", "group"]
        + [f"ASR_{t}" for t in run_tags]
        + [f"mAP_drop_{t}" for t in run_tags]
        + [f"delta_{t}_vs_{baseline_of[t]}_ASR" for t in dbta_tags]
    )
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for tag in run_tags:
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
            row[f"mAP_drop_{tag}"] = r.get("mAP_drop_pct")
        for dbta_tag in dbta_tags:
            baseline_tag = baseline_of[dbta_tag]
            if row.get(f"ASR_{baseline_tag}") is not None and row.get(f"ASR_{dbta_tag}") is not None:
                row[f"delta_{dbta_tag}_vs_{baseline_tag}_ASR"] = row[f"ASR_{dbta_tag}"] - row[f"ASR_{baseline_tag}"]
        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info(f"=== DBTA v0 pilot: ASR (%) per arm ({', '.join(run_tags)}) ===")
    for r in comparison:
        parts = [f"{tag}={r.get(f'ASR_{tag}', float('nan')):.1f}" for tag in run_tags]
        deltas = [
            f"{tag}-{baseline_of[tag]}={r.get(f'delta_{tag}_vs_{baseline_of[tag]}_ASR', float('nan')):+.1f}"
            for tag in dbta_tags
        ]
        logger.info(f"{r['model_name']:20s} {r['group'] or '-':4s} " + " ".join(parts) + "  (" + ", ".join(deltas) + ")")

    # ---- Verdict: white-box sanity FIRST, per the pre-registered plan ----
    by_model_name = {r["model_name"]: r for r in comparison}
    for dbta_tag in dbta_tags:
        baseline_tag = baseline_of[dbta_tag]
        delta_key = f"delta_{dbta_tag}_vs_{baseline_tag}_ASR"
        wb_row = by_model_name.get(WHITEBOX_MODEL)
        wb_delta = wb_row.get(delta_key) if wb_row else None

        if wb_delta is None:
            logger.info(f"=== [{dbta_tag}] INCONCLUSIVE -- white-box model {WHITEBOX_MODEL!r} not evaluated. ===")
            continue

        logger.info(f"=== [{dbta_tag}] white-box sanity vs {baseline_tag}: {WHITEBOX_MODEL} delta={wb_delta:+.1f} (fail if <= {WB_COLLAPSE_THR}) ===")
        if wb_delta <= WB_COLLAPSE_THR:
            logger.info(
                f"[{dbta_tag}] WHITE-BOX SANITY FAIL -- same failure signature as all 3 TGA variants "
                f"(RESEARCH.md Sec 29). Recommend closing this line too; accept E10 as a diagnostic-only finding."
            )
            continue

        logger.info(f"[{dbta_tag}] WHITE-BOX SANITY PASS -- checking hard-target transfer deltas vs {baseline_tag} next.")
        hard_deltas = {}
        for m in HARD_TARGETS:
            r = by_model_name.get(m)
            if r is not None and r.get(delta_key) is not None:
                hard_deltas[m] = r[delta_key]

        n_wins = sum(1 for d in hard_deltas.values() if d >= GO_DELTA_THR)
        logger.info(f"[{dbta_tag}] wins (delta>=+{GO_DELTA_THR}) on {n_wins}/{len(hard_deltas)} hard targets: {hard_deltas}")

        if len(hard_deltas) < len(HARD_TARGETS):
            logger.info(f"[{dbta_tag}] hard-target read INCONCLUSIVE -- not all hard targets evaluated.")
        elif n_wins >= GO_MIN_WINS:
            losers = {m: d for m, d in hard_deltas.items() if d < GO_DELTA_THR}
            collapsed = {m: d for m, d in losers.items() if d <= COLLAPSE_THR}
            if not collapsed:
                logger.info(f"[{dbta_tag}] STRONG GO -- beats {baseline_tag} by >={GO_DELTA_THR} on {n_wins}/3 hard targets, no collapse on the rest.")
            else:
                logger.info(f"[{dbta_tag}] NO-GO (collapse guard failed) -- wins on {n_wins}/3 but catastrophic drop on {collapsed}.")
        elif hard_deltas.get("dino_swin_l", -1e9) >= GO_DELTA_THR:
            logger.info(f"[{dbta_tag}] WEAK GO -- beats {baseline_tag} on dino_swin_l ({hard_deltas['dino_swin_l']:+.1f}) but not >=2/3 overall.")
        else:
            logger.info(
                f"[{dbta_tag}] sanity PASS but hard-target transfer vs {baseline_tag} NOT YET winning -- may warrant "
                f"a further pilot (more steps / tuned margins) before a final GO/NO-GO call."
            )


if __name__ == "__main__":
    main()
