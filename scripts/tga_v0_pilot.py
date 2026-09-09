#!/usr/bin/env python
"""TGA v0 pilot -- Semantic Transition Geometry Attack, rank-only, RRB OFF.

First test of whether E10's confirmed cross-architecture relational
invariant (RESEARCH.md Sec 27-28: a shared O<->E / O<->nearBG / E<->nearBG /
E<->farBG / nearBG<->farBG signature, strongest on the two boundary-adjacent
relations O<->E and E<->nearBG) can be turned into an attack OBJECTIVE, not
just a diagnostic. TGA does not suppress object features or amplify vicinal
features (OSFD's Eq. 2 mechanism) -- it maximizes inversion of the relative
ORDERING between d(O,E), d(O,Bn), d(E,Bn), read off each clean image rather
than hard-coded (transfer_attack/losses.py::tga_rank_loss). v0 scope,
deliberately narrow (per the method-derivation plan, not to be expanded
before reading this pilot's result):
  - RRB OFF for every arm (isolates whether the relational objective ITSELF
    has transferability, before any augmentation is mixed in).
  - O/E/Bn only, no far-background (E10 dev_300: O<->farBG was the one
    relation that fell out of the shared-set as N scaled -- see
    transfer_attack/losses.py's TGA module docstring).
  - Rank-only loss, no L_geom magnitude term (softplus rank-inversion sum of
    the 3 pairs among {OE, OBn, EBn} -- see tga_rank_loss).

First run of this pilot averaged the rank loss over ALL backbone stages and
came back a clean NO-GO everywhere, including a -29.8 ASR point loss on the
WHITE-BOX surrogate itself vs OSFD-noRRB -- not just a transfer failure, a
weak white-box objective. `tga_lastonly_norrb` below is the one pre-
registered fallback (not a retune) tried before closing TGA-rank entirely:
restrict to each model's LAST backbone stage, matching the only stage E10
actually confirmed the O<->E/E<->nearBG invariance at.

4 arms, same budget (epsilon/alpha/steps, k=3 default where relevant), RRB
off for all so the only independent variable is the objective itself:
  osfd_norrb         -- OSFD backbone-feature MSE (Eq. 2), default k=3, no RRB.
  tga_norrb          -- TGA relational rank loss, all stages averaged (NO-GO).
  tga_lastonly_norrb -- TGA relational rank loss, last stage only (fallback check).
  mi_fgsm            -- detector task loss (paper's other baseline), reference
                        only, not part of the GO/NO-GO decision below.

Models: same 6 as E10 (DEFAULT_MODELS) -- 3 "hard" cross-architecture targets
(primary: yolox_l, mask_rcnn_swin_t, dino_swin_l) plus the surrogate and its
2 R50-matched-pair targets (secondary: check for non-collapse, not for gain).

GO criterion (pre-registered here, before running -- do not retune after
seeing results):
  Strong GO -- TGA beats OSFD-noRRB by >= +5 ASR points on >= 2/3 of
               {yolox_l, mask_rcnn_swin_t, dino_swin_l}, AND does not
               "catastrophically" collapse on the third (defined here, since
               no exact number was pre-specified beyond "not catastrophic",
               as: TGA ASR not more than 20 points below OSFD-noRRB's ASR
               on that remaining hard target).
  Weak GO   -- TGA clearly beats OSFD-noRRB on dino_swin_l specifically, but
               mixed (neither a clear win nor the Strong-GO collapse bar) on
               the other two hard targets. Keep as a candidate, do not scale.
  NO-GO     -- neither of the above (TGA does not beat OSFD-noRRB on >=2/3 of
               the hard targets, or fails the collapse guard where it does
               win on 2/3). Close TGA v0; do not tune temperature/shrink/
               expand-frac to rescue it (same discipline as every prior
               NO-GO in this project -- MVC, RCG, N2-B, DOB).
Secondary (surrogate + R50 targets): reported for reference, NOT part of the
GO/NO-GO decision (project's stated goal is closing the B/C gap, not adding
ASR on near-ceiling R50 targets).

Example:
    python scripts/tga_v0_pilot.py --manifest data/manifests/dev_50.json --n-images 20
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
    {"tag": "tga_norrb", "attack_type": "tga", "use_rrb": False},
    # Pre-registered fallback (RESEARCH.md method-derivation note + tga_loss's
    # docstring caveat), NOT a post-hoc retune: restrict to each model's LAST
    # backbone stage only, matching E10's actually-confirmed scope, instead of
    # averaging over every stage (untested territory -- see tga_v0_norrb's
    # NO-GO result, first run of this pilot). Also NO-GO, identical pattern.
    {"tag": "tga_lastonly_norrb", "attack_type": "tga", "use_rrb": False, "tga_last_stage_only": True},
    # Structurally different objective, tried after BOTH rank variants above
    # came back a clean NO-GO (including a -22 to -30 ASR point loss on the
    # WHITE-BOX surrogate) -- diagnosed as tga_rank_loss's softplus having a
    # vanishing gradient for any relation pair already well-separated in the
    # clean image. L_geom (see tga_geom_loss) has no such vanishing-gradient
    # failure mode by construction. All-stage average (same default as
    # tga_norrb) so this is an apples-to-apples swap of ONLY the objective.
    {"tag": "tga_geom_norrb", "attack_type": "tga", "use_rrb": False, "tga_objective": "geom"},
    {"tag": "mi_fgsm", "attack_type": "mi_fgsm", "use_rrb": False},
]
BASELINE_TAG = "osfd_norrb"

DEFAULT_MODELS = ["faster_rcnn_r50", "dino_r50", "mask_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"]
HARD_TARGETS = ["yolox_l", "mask_rcnn_swin_t", "dino_swin_l"]

GO_DELTA_THR = 5.0        # ASR points, TGA - OSFD-noRRB, on a hard target -> counts as a "win"
GO_MIN_WINS = 2           # need wins on >= 2 of 3 hard targets for Strong GO
COLLAPSE_THR = -20.0      # ASR points -- below this on the non-winning hard target = "catastrophic"


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
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "tga_v0_pilot_summary.csv")
    p.add_argument("--run-tags", nargs="+", default=None, help="subset of VARIANTS tags to craft+eval (default: all 3)")
    p.add_argument("--models", nargs="+", default=None, help=f"subset of models to evaluate (default: {DEFAULT_MODELS})")
    return p


def craft_and_evaluate_variant(variant: dict, args, coco, image_ids, gt_index, img_dir, specs, logger) -> list[dict]:
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
        tga_last_stage_only=variant.get("tga_last_stage_only", False),
        tga_objective=variant.get("tga_objective", "rank"),
        tga_geom_lambda=variant.get("tga_geom_lambda", 1.0),
        steps=args.steps,
        canvas=args.canvas,
    )

    noise_dir = PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"tga_v0_{tag}"
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
            # TGA-specific: an image can have GT boxes but none clear the
            # region-coverage threshold at any backbone stage (see attack.py) --
            # skip it like any other unusable image rather than crash the run.
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
        predictions_dir=PROJECT_DIR / "results" / "_tga_v0_pilot_predictions",
        force_clean=False,
        device=args.device,
        attacks=[f"tga_v0_{tag}"],
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
        f"tga_v0_{tag}_{args.manifest.stem}_n{args.n_images}",
        {
            "attack": f"tga_v0_{tag}",
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
    return [r for r in rows if r["attack"] == f"tga_v0_{tag}"]


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

    all_rows: dict[str, list[dict]] = {}
    for variant in run_variants:
        all_rows[variant["tag"]] = craft_and_evaluate_variant(variant, args, coco, image_ids, gt_index, img_dir, specs, logger)

    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    import csv

    run_tags = [v["tag"] for v in run_variants]
    tga_tags = [v["tag"] for v in run_variants if v["attack_type"] == "tga"]

    fieldnames = (
        ["model_name", "group"]
        + [f"ASR_{t}" for t in run_tags]
        + [f"mAP_drop_{t}" for t in run_tags]
        + [f"delta_{t}_vs_osfd_ASR" for t in tga_tags]
    )
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for tag in run_tags:
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
            row[f"mAP_drop_{tag}"] = r.get("mAP_drop_pct")
        for tga_tag in tga_tags:
            if row.get(f"ASR_{BASELINE_TAG}") is not None and row.get(f"ASR_{tga_tag}") is not None:
                row[f"delta_{tga_tag}_vs_osfd_ASR"] = row[f"ASR_{tga_tag}"] - row[f"ASR_{BASELINE_TAG}"]
        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info(f"=== TGA v0 pilot: ASR (%) per arm ({', '.join(run_tags)}) ===")
    for r in comparison:
        parts = [f"{tag}={r.get(f'ASR_{tag}', float('nan')):.1f}" for tag in run_tags]
        deltas = [f"{tag}-osfd={r.get(f'delta_{tag}_vs_osfd_ASR', float('nan')):+.1f}" for tag in tga_tags]
        logger.info(f"{r['model_name']:20s} {r['group'] or '-':4s} " + " ".join(parts) + "  (" + ", ".join(deltas) + ")")

    # ---- GO / Weak-GO / NO-GO verdict per TGA variant, per the pre-registered criteria above ----
    by_model_name = {r["model_name"]: r for r in comparison}
    for tga_tag in tga_tags:
        delta_key = f"delta_{tga_tag}_vs_osfd_ASR"
        hard_deltas = {}
        for m in HARD_TARGETS:
            r = by_model_name.get(m)
            if r is not None and r.get(delta_key) is not None:
                hard_deltas[m] = r[delta_key]

        n_wins = sum(1 for d in hard_deltas.values() if d >= GO_DELTA_THR)
        logger.info(f"=== [{tga_tag}] wins (delta>=+{GO_DELTA_THR}) on {n_wins}/{len(hard_deltas)} hard targets: {hard_deltas} ===")

        if len(hard_deltas) < len(HARD_TARGETS):
            logger.info(f"[{tga_tag}] INCONCLUSIVE -- not all hard targets evaluated (check --models / crafting skips above).")
        elif n_wins >= GO_MIN_WINS:
            losers = {m: d for m, d in hard_deltas.items() if d < GO_DELTA_THR}
            collapsed = {m: d for m, d in losers.items() if d <= COLLAPSE_THR}
            if not collapsed:
                logger.info(f"[{tga_tag}] STRONG GO -- beats OSFD-noRRB by >={GO_DELTA_THR} on {n_wins}/3 hard targets, no collapse on the rest.")
            else:
                logger.info(f"[{tga_tag}] NO-GO (collapse guard failed) -- wins on {n_wins}/3 but catastrophic drop on {collapsed}.")
        elif hard_deltas.get("dino_swin_l", -1e9) >= GO_DELTA_THR:
            logger.info(f"[{tga_tag}] WEAK GO -- beats OSFD-noRRB on dino_swin_l ({hard_deltas['dino_swin_l']:+.1f}) but not >=2/3 overall. Keep as candidate, do not scale.")
        else:
            logger.info(f"[{tga_tag}] NO-GO -- does not beat OSFD-noRRB by >={GO_DELTA_THR} on >={GO_MIN_WINS}/3 hard targets, and not even on dino_swin_l alone.")


if __name__ == "__main__":
    main()
