#!/usr/bin/env python
"""E11 -- Adversarial Equivariance Gap (RESEARCH.md Sec 32): existence +
relevance test for a candidate attack principle BEFORE building it, following
the project's usual cheap-diagnostic-before-method discipline (E10 before
TGA/DBTA/BTFA).

Hypothesis under test: this project's two independently-discovered transfer
drivers -- RRB (E3, E10) and path-averaged gradient (N6-B) -- work NOT by
disrupting a specific feature value, but by making the adversarial backbone
feature field violate a simple cross-architecture transformation-equivariance
law more severely. If true, both interventions should INCREASE a measured
"equivariance gap" Q specifically on the hard targets (CSP/Swin) where they
are known to increase ASR.

Q_m(x, delta) = E_tau [ ||F_m(tau(x+delta)) - W_tau F_m(x+delta)|| / ||F_m(x+delta)|| ]

tau ranges over a small pre-registered transform set (transfer_attack.
equivariance.TRANSFORM_SET: rotate +-5 deg, scale 0.9/1.1). See
transfer_attack/equivariance.py for the full construction (a single
resolution-agnostic affine warp used to define both tau(x) at pixel
resolution and W_tau F(x) at feature resolution).

This is a diagnostic, NOT a new attack: no new loss is crafted here. Noise
for the 4 non-clean arms is produced by the EXISTING crafting code, unchanged:
  osfd_rrb   -- transfer_attack.attack.craft_one_image(attack_type="osfd", use_rrb=True),
                literally the same algorithm as scripts/n6b_path_pilot.py's
                "osfd_local" -- crafted via craft_paired_local_path so it is
                RNG-paired with path_m3 (see that module for why: removes
                "path got luckier RRB draws" as a confound).
  path_m3    -- scripts/n6b_path_pilot.py's craft_paired_local_path, M=3
                clean->current path-averaged OSFD gradient (N6-B, pre-
                registered, not swept).
  osfd_norrb -- craft_one_image(attack_type="osfd", use_rrb=False) -- same
                objective as osfd_rrb with augmentation off (E3's control arm).
  mi_fgsm    -- craft_one_image(attack_type="mi_fgsm") -- task-loss baseline,
                logged for context, not used in the pre-registered verdict.
Noise for this run's own images is cached under
results/noise/<manifest-stem>/e11_<tag>/ (recraft is required on a fresh
checkout -- results/ is gitignored, see HANDOFF.md).

Pre-registered verdict (decided before running, RESEARCH.md Sec 32 Sec.11):
  RRB test   -- on the 3 hard targets {yolox_l, mask_rcnn_swin_t, dino_swin_l},
                sign(Q(osfd_rrb)-Q(osfd_norrb)) == sign(ASR(osfd_rrb)-ASR(osfd_norrb))
                on 3/3.
  Path test  -- on dino_swin_l (primary; mask_rcnn_swin_t flagged if it also
                matches, not required), sign(Q(path_m3)-Q(osfd_rrb)) ==
                sign(ASR(path_m3)-ASR(osfd_rrb)).
  STRONG GO  -- both tests pass -> proceed to design CEFA.
  NO-GO      -- either test fails (in particular: RRB/path raise ASR a lot but
                Q is flat or moves the wrong way) -> close immediately, do NOT
                change the transform set or re-weight stages to rescue.

Example:
    python scripts/e11_equivariance_gap.py --manifest data/manifests/dev_50.json --n-images 20
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

CRAFT_TAGS = ["osfd_rrb", "osfd_norrb", "mi_fgsm", "path_m3"]
ALL_TAGS = ["clean"] + CRAFT_TAGS
DEFAULT_MODELS = ["faster_rcnn_r50", "dino_r50", "mask_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"]
HARD_TARGETS = ["yolox_l", "mask_rcnn_swin_t", "dino_swin_l"]
WHITEBOX_MODEL = "faster_rcnn_r50"
M_LAMBDA = 3  # pre-registered N6-B path width, unchanged, no sweep


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
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "e11_equivariance_gap_summary.csv")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return p


# ---------------------------------------------------------------------------
# Phase 0: craft (reuses existing crafting code unchanged -- see module docstring)
# ---------------------------------------------------------------------------

def craft_all_variants(args, coco, image_ids, gt_index, img_dir, logger):
    import torch

    from n6b_path_pilot import craft_paired_local_path

    from transfer_attack.attack import AttackConfig, craft_one_image
    from transfer_attack.data import gt_to_canvas, load_canvas_image
    from transfer_attack.io_utils import save_noise
    from transfer_attack.models import build_model_handle, get_spec

    cfg_rrb = AttackConfig(attack_type="osfd", k=3.0, use_rrb=True, steps=args.steps, canvas=args.canvas)
    cfg_norrb = AttackConfig(attack_type="osfd", k=3.0, use_rrb=False, steps=args.steps, canvas=args.canvas)
    cfg_mi = AttackConfig(attack_type="mi_fgsm", steps=args.steps, canvas=args.canvas)

    noise_dirs = {tag: PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"e11_{tag}" for tag in CRAFT_TAGS}
    for d in noise_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=args.device, coco=coco)
    logger.info(f"[craft] surrogate {surrogate_spec.name} loaded on {args.device}")

    used_image_ids: list[int] = []
    n_crafted, n_skipped = 0, 0
    t0 = time.time()
    for image_id in image_ids:
        if len(used_image_ids) >= args.n_images:
            break
        gt_entries = gt_index[image_id]
        if not gt_entries:
            n_skipped += 1
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        if gt_boxes.shape[0] == 0:
            n_skipped += 1
            continue
        x_clean = canvas_img.to(args.device)
        gt_boxes_dev = gt_boxes.to(args.device)

        # RNG-paired (osfd_rrb, path_m3), per n6b_path_pilot's own convention.
        random.seed(args.seed + image_id)
        torch.manual_seed(args.seed + image_id)
        noise_rrb, noise_path, _diag = craft_paired_local_path(surrogate, x_clean, gt_boxes_dev, cfg_rrb, args.device, M_LAMBDA)
        save_noise(noise_dirs["osfd_rrb"] / f"{image_id}.pt", noise_rrb)
        save_noise(noise_dirs["path_m3"] / f"{image_id}.pt", noise_path)

        # osfd_norrb and mi_fgsm are independent single-trajectory crafts --
        # same per-image seeding convention as run_attack.py/btfa_v0_pilot.py
        # (seeded once per image, not paired with the above).
        random.seed(args.seed + image_id)
        torch.manual_seed(args.seed + image_id)
        noise_norrb, _ = craft_one_image(surrogate, x_clean, gt_boxes_dev, gt_cat_ids, cfg_norrb, device=args.device)
        save_noise(noise_dirs["osfd_norrb"] / f"{image_id}.pt", noise_norrb)

        random.seed(args.seed + image_id)
        torch.manual_seed(args.seed + image_id)
        noise_mi, _ = craft_one_image(surrogate, x_clean, gt_boxes_dev, gt_cat_ids, cfg_mi, device=args.device)
        save_noise(noise_dirs["mi_fgsm"] / f"{image_id}.pt", noise_mi)

        used_image_ids.append(image_id)
        n_crafted += 1
        if n_crafted % args.log_every == 0:
            logger.info(f"[craft] [{n_crafted}/{args.n_images}] elapsed={time.time() - t0:.1f}s")
    craft_elapsed = time.time() - t0
    logger.info(f"[craft] finished: {n_crafted} crafted, {n_skipped} skipped, elapsed={craft_elapsed:.1f}s")
    del surrogate
    torch.cuda.empty_cache()
    return used_image_ids, noise_dirs, craft_elapsed, n_crafted, n_skipped


# ---------------------------------------------------------------------------
# Phase 1: ASR/mAP eval (reuses evaluate.py's per-model logic, unchanged)
# ---------------------------------------------------------------------------

def evaluate_asr(args, coco, used_image_ids, img_dir, specs, logger, run_id: str):
    from types import SimpleNamespace

    import evaluate as evaluate_mod

    eval_args = SimpleNamespace(
        checkpoints_dir=args.checkpoints_dir,
        canvas=args.canvas,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        noise_dir=PROJECT_DIR / "results" / "noise" / args.manifest.stem,
        predictions_dir=PROJECT_DIR / "results" / "_e11_predictions" / run_id,
        force_clean=False,
        device=args.device,
        attacks=[f"e11_{tag}" for tag in CRAFT_TAGS],
    )
    gt_cache = evaluate_mod.build_gt_cache(coco, used_image_ids, img_dir, args.canvas)
    rows_by_model: dict[str, dict[str, dict]] = {}
    for spec in specs:
        rows = evaluate_mod.evaluate_one_model(spec, eval_args, coco, used_image_ids, img_dir, gt_cache, logger)
        by_tag = {}
        for r in rows:
            tag = r["attack"]
            if tag == "clean":
                by_tag["clean"] = r
            else:
                by_tag[tag[len("e11_"):]] = r
        rows_by_model[spec.name] = by_tag
    return rows_by_model


# ---------------------------------------------------------------------------
# Phase 2: equivariance gap Q (no craft, no grad -- pure diagnostic forward passes)
# ---------------------------------------------------------------------------

def compute_equivariance_table(args, coco, used_image_ids, img_dir, specs, noise_dirs, logger):
    import torch

    from transfer_attack.data import load_canvas_image
    from transfer_attack.equivariance import TRANSFORM_SET, compute_Q_for_image
    from transfer_attack.io_utils import load_noise
    from transfer_attack.models import build_model_handle

    canvas_cache = {}
    for image_id in used_image_ids:
        canvas_img, _, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
        canvas_cache[image_id] = canvas_img

    results: dict[str, dict[str, dict]] = {}
    t0 = time.time()
    for spec in specs:
        handle = build_model_handle(spec, args.checkpoints_dir, device=args.device, coco=coco)
        model = handle.model
        model.eval()

        per_tag_all_stage = {tag: [] for tag in ALL_TAGS}
        per_tag_last_stage = {tag: [] for tag in ALL_TAGS}
        for image_id in used_image_ids:
            canvas_img = canvas_cache[image_id].to(args.device)
            for tag in ALL_TAGS:
                if tag == "clean":
                    x = canvas_img
                else:
                    noise_path = noise_dirs[tag] / f"{image_id}.pt"
                    if not noise_path.exists():
                        continue
                    noise = load_noise(noise_path, device=args.device)
                    x = torch.clamp(canvas_img + noise, 0.0, 255.0)
                out = compute_Q_for_image(model, handle.normalize, x, TRANSFORM_SET)
                per_stage = out["per_stage"]
                per_tag_all_stage[tag].append(sum(per_stage) / len(per_stage))
                per_tag_last_stage[tag].append(per_stage[-1])

        results[spec.name] = {
            tag: {
                "Q_all_stage": (sum(v) / len(v)) if v else float("nan"),
                "Q_last_stage": (sum(per_tag_last_stage[tag]) / len(per_tag_last_stage[tag])) if per_tag_last_stage[tag] else float("nan"),
                "n": len(v),
            }
            for tag, v in per_tag_all_stage.items()
        }
        del handle
        torch.cuda.empty_cache()
        logger.info(
            f"[Q] {spec.name}: " + " ".join(f"{t}={results[spec.name][t]['Q_all_stage']:.4f}" for t in ALL_TAGS)
            + f" (elapsed={time.time() - t0:.1f}s)"
        )
    return results


# ---------------------------------------------------------------------------
# Phase 3: verdict
# ---------------------------------------------------------------------------

def sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, save_run_log
    from transfer_attack.models import MODEL_REGISTRY

    logger = get_logger()
    random.seed(args.seed)

    import torch

    torch.manual_seed(args.seed)

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"

    by_name = {s.name: s for s in MODEL_REGISTRY}
    specs = [by_name[m] for m in args.models]

    run_id = time.strftime("%Y%m%dT%H%M%S")

    used_image_ids, noise_dirs, craft_elapsed, n_crafted, n_skipped = craft_all_variants(
        args, coco, image_ids, gt_index, img_dir, logger
    )

    asr_rows = evaluate_asr(args, coco, used_image_ids, img_dir, specs, logger, run_id)
    q_table = compute_equivariance_table(args, coco, used_image_ids, img_dir, specs, noise_dirs, logger)

    # ---- Build comparison table ----
    import csv

    fieldnames = (
        ["model_name", "group"]
        + [f"ASR_{t}" for t in ALL_TAGS if t != "clean"]
        + [f"Q_{t}" for t in ALL_TAGS]
        + [f"Qlast_{t}" for t in ALL_TAGS]
        + ["delta_Q_RRB", "delta_ASR_RRB", "RRB_sign_match", "delta_Q_path", "delta_ASR_path", "path_sign_match"]
    )
    comparison = []
    for spec in specs:
        row = {"model_name": spec.name, "group": spec.group or ""}
        for t in ALL_TAGS:
            if t != "clean":
                row[f"ASR_{t}"] = asr_rows.get(spec.name, {}).get(t, {}).get("ASR")
            row[f"Q_{t}"] = q_table.get(spec.name, {}).get(t, {}).get("Q_all_stage")
            row[f"Qlast_{t}"] = q_table.get(spec.name, {}).get(t, {}).get("Q_last_stage")

        q_rrb, q_norrb, q_path = row["Q_osfd_rrb"], row["Q_osfd_norrb"], row["Q_path_m3"]
        asr_rrb, asr_norrb, asr_path = row["ASR_osfd_rrb"], row["ASR_osfd_norrb"], row["ASR_path_m3"]

        if None not in (q_rrb, q_norrb):
            row["delta_Q_RRB"] = q_rrb - q_norrb
        if None not in (asr_rrb, asr_norrb):
            row["delta_ASR_RRB"] = asr_rrb - asr_norrb
        if "delta_Q_RRB" in row and "delta_ASR_RRB" in row:
            row["RRB_sign_match"] = sign(row["delta_Q_RRB"]) == sign(row["delta_ASR_RRB"])

        if None not in (q_path, q_rrb):
            row["delta_Q_path"] = q_path - q_rrb
        if None not in (asr_path, asr_rrb):
            row["delta_ASR_path"] = asr_path - asr_rrb
        if "delta_Q_path" in row and "delta_ASR_path" in row:
            row["path_sign_match"] = sign(row["delta_Q_path"]) == sign(row["delta_ASR_path"])

        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    by_model_name = {r["model_name"]: r for r in comparison}

    logger.info("=== E11: per-model ASR / equivariance-gap Q (all-stage mean) ===")
    for r in comparison:
        logger.info(
            f"{r['model_name']:20s} {r['group'] or '-':4s} "
            f"ASR[norrb={r.get('ASR_osfd_norrb', float('nan')):.1f} "
            f"rrb={r.get('ASR_osfd_rrb', float('nan')):.1f} "
            f"path={r.get('ASR_path_m3', float('nan')):.1f} "
            f"mi={r.get('ASR_mi_fgsm', float('nan')):.1f}]  "
            f"Q[clean={r.get('Q_clean', float('nan')):.4f} "
            f"norrb={r.get('Q_osfd_norrb', float('nan')):.4f} "
            f"rrb={r.get('Q_osfd_rrb', float('nan')):.4f} "
            f"path={r.get('Q_path_m3', float('nan')):.4f}]  "
            f"dQ_RRB={r.get('delta_Q_RRB', float('nan')):+.4f}(match={r.get('RRB_sign_match')}) "
            f"dQ_path={r.get('delta_Q_path', float('nan')):+.4f}(match={r.get('path_sign_match')})"
        )

    # ---- Pre-registered verdict ----
    rrb_matches = {m: by_model_name[m].get("RRB_sign_match") for m in HARD_TARGETS if m in by_model_name}
    n_rrb_match = sum(1 for v in rrb_matches.values() if v is True)
    n_rrb_total = sum(1 for v in rrb_matches.values() if v is not None)
    rrb_go = n_rrb_total == len(HARD_TARGETS) and n_rrb_match == len(HARD_TARGETS)

    path_dino = by_model_name.get("dino_swin_l", {}).get("path_sign_match")
    path_mask = by_model_name.get("mask_rcnn_swin_t", {}).get("path_sign_match")
    path_go = path_dino is True

    logger.info("=== E11 pre-registered verdict ===")
    logger.info(f"RRB test: sign(dQ)==sign(dASR) on {n_rrb_match}/{len(HARD_TARGETS)} hard targets: {rrb_matches} -> {'PASS' if rrb_go else 'FAIL'} (need 3/3)")
    logger.info(f"Path test: sign(dQ)==sign(dASR) on dino_swin_l -> {path_dino} -> {'PASS' if path_go else 'FAIL'} (mask_rcnn_swin_t as-is: {path_mask}, not required)")

    if rrb_go and path_go:
        logger.info(
            "STRONG GO -- both RRB and path-averaging interventions increase the equivariance gap Q in the same "
            "direction they increase ASR, on the pre-registered hard targets. Per RESEARCH.md Sec 32: proceed to "
            "design CEFA (attack the equivariance residual directly as the crafting objective)."
        )
    else:
        logger.info(
            "NO-GO per pre-registered criteria -- at least one of the RRB/path tests failed to support the "
            "equivariance-failure hypothesis. Per project discipline, do NOT change the transform set, add more "
            "transforms, or reweight stages to rescue this result -- close the CEFA line here and report which "
            "test(s) failed and on which model(s)."
        )

    run_log_path = save_run_log(
        args.runs_dir,
        "e11_equivariance_gap",
        f"{args.manifest.stem}_n{args.n_images}",
        {
            "manifest": str(args.manifest),
            "seed": args.seed,
            "n_images": args.n_images,
            "steps": args.steps,
            "models": args.models,
            "results": {
                "n_images_used": len(used_image_ids),
                "n_crafted": n_crafted,
                "n_skipped": n_skipped,
                "craft_elapsed_sec": round(craft_elapsed, 1),
                "comparison": comparison,
                "verdict": {
                    "rrb_matches": rrb_matches,
                    "rrb_go": rrb_go,
                    "path_dino_match": path_dino,
                    "path_mask_match": path_mask,
                    "strong_go": bool(rrb_go and path_go),
                },
            },
        },
    )
    logger.info(f"run log written -> {run_log_path}")


if __name__ == "__main__":
    main()
