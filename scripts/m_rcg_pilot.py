#!/usr/bin/env python
"""Phase M pilot: RRB-resampling Consistency Gate (RCG) -- Candidate 2 (TCR),
narrowed after the novelty scan found "trajectory/temporal gradient
consistency" (across STEPS) heavily covered by MI-FGSM/NI-FGSM/VMI-FGSM/
Direction-Tuning/Decaying-Steps/Staircase-Sign. RCG instead isolates variance
from RRB's per-call random augmentation draw AT A SINGLE FIXED noise state --
not variance from the noise state drifting over time (that's the crowded
part) -- a narrower niche none of those papers occupy.

Three variants, same budget (k=3, RRB on, epsilon/alpha/steps unchanged):
  osfd      -- baseline, K=1 draw/step, no gate (current OSFD exactly).
  rcg_avg   -- K=3 independent RRB draws/step at the SAME noise state,
               gradient = mean of the 3 (an unbiased MC estimate of the
               expected RRB-augmented gradient). No gate. This is the
               "just more augmentation samples" control the M1 constraint
               explicitly rules out as a novel mechanism on its own --
               required here specifically to attribute any gain correctly.
  rcg_gate  -- same K=3 draws, PLUS a per-pixel gate = |mean_i sign(g_i)| in
               [0,1] (1 = all 3 draws agree on sign, ~0 = they disagree)
               multiplying the applied step: noise += alpha*sign(g_mom)*gate.
               No lambda to tune (gate is naturally bounded [0,1] -- avoids
               the scale-tuning pitfall the MVC pilot hit with its additive
               loss penalty).

Diagnostics logged per image (averaged into the final CSV), per your request:
  mean_gate       -- avg gate value across all pixels x all 100 steps. If
                      this sits near 1.0 throughout, the gate is barely
                      selective and rcg_gate ~= rcg_avg is expected trivially
                      (not evidence either way about the mechanism).
  frac_saturated  -- fraction of final-noise pixels at |noise| >= 0.99*eps.
                      Lets us tell "consistency genuinely helps" apart from
                      "the gate just shrinks effective step size and weakens
                      the attack" if rcg_gate's ASR drops.

GO/NO-GO reading (same bar as the MVC pilot, decided before running):
  rcg_gate > rcg_avg > osfd, esp. on YOLOX-L / mask_rcnn_swin_t / dino_swin_l,
  by >= 3-5 ASR points on >= 2/3 of those hard targets, without white-box
  (surrogate) collapse -> worth pursuing. Anything less -> NO-GO, same
  discipline as MVC (no post-hoc retuning to rescue one target).

Example:
    python scripts/m_rcg_pilot.py --manifest data/manifests/dev_50.json --n-images 20
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
    {"tag": "osfd", "k_draws": 1, "use_gate": False},
    {"tag": "rcg_avg", "k_draws": 3, "use_gate": False},
    {"tag": "rcg_gate", "k_draws": 3, "use_gate": True},
]


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
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "m_rcg_pilot_summary.csv")
    p.add_argument("--run-tags", nargs="+", default=None, help="subset of VARIANTS tags to craft+eval (default: all 3)")
    p.add_argument("--models", nargs="+", default=None, help="subset of MODEL_REGISTRY names to evaluate (default: all 7)")
    return p


def craft_one_image_rcg(handle, x_clean, gt_boxes, cfg, k_draws: int, use_gate: bool, device: str):
    import torch

    from transfer_attack.augment import rrb_forward
    from transfer_attack.losses import osfd_loss

    model = handle.model
    x_clean = x_clean.to(device)
    gt_boxes = gt_boxes.to(device)

    with torch.no_grad():
        feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))

    noise = torch.randint_like(x_clean, low=-2, high=3).float()
    g_mom = torch.zeros_like(x_clean)
    gate_sum = 0.0

    for _ in range(cfg.steps):
        noise = noise.detach().requires_grad_(True)

        grads = []
        for _ in range(k_draws):
            # Recompute x_adv fresh per draw (cheap: one clamp) so each draw's
            # graph is fully independent -- avoids retain_graph bookkeeping
            # around a shared x_adv node feeding 3 separate backward() calls.
            x_adv_i = torch.clamp(x_clean + noise, 0.0, 255.0)
            aug = rrb_forward(x_adv_i.unsqueeze(0), gt_boxes, cfg)
            feats_adv = model.backbone(handle.normalize(aug))
            loss = osfd_loss(feats_cln, feats_adv, cfg.k)
            (grad_i,) = torch.autograd.grad(loss, noise)
            grads.append(grad_i.detach())

        g_avg = sum(grads) / k_draws

        if use_gate:
            sign_sum = sum(torch.sign(g) for g in grads)
            gate = sign_sum.abs() / k_draws  # per-pixel agreement in [0,1]
        else:
            gate = torch.ones_like(g_avg)
        gate_sum += gate.mean().item()

        with torch.no_grad():
            g_mom = cfg.mu * g_mom + g_avg / g_avg.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise = torch.clamp(noise + cfg.alpha * torch.sign(g_mom) * gate, -cfg.epsilon, cfg.epsilon)

    noise = noise.detach()
    mean_gate = gate_sum / cfg.steps
    frac_saturated = (noise.abs() >= 0.99 * cfg.epsilon).float().mean().item()
    return noise, {"mean_gate": mean_gate, "frac_saturated": frac_saturated}


def craft_and_evaluate_variant(variant: dict, args, coco, image_ids, gt_index, img_dir, logger) -> tuple[list[dict], dict]:
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

    noise_dir = PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"m_{tag}"
    noise_dir.mkdir(parents=True, exist_ok=True)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=args.device, coco=coco)
    logger.info(f"[{tag}] craft: k_draws={variant['k_draws']} use_gate={variant['use_gate']}")

    n_crafted, n_skipped = 0, 0
    t0 = time.time()
    used_image_ids = []
    mean_gates, frac_sats = [], []
    for image_id in image_ids:
        if len(used_image_ids) >= args.n_images:
            break
        gt_entries = gt_index[image_id]
        if not gt_entries:
            n_skipped += 1
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, cfg.canvas)
        gt_boxes, _ = gt_to_canvas(gt_entries, scale)
        noise, diag = craft_one_image_rcg(surrogate, canvas_img, gt_boxes, cfg, variant["k_draws"], variant["use_gate"], args.device)
        save_noise(noise_dir / f"{image_id}.pt", noise)
        used_image_ids.append(image_id)
        mean_gates.append(diag["mean_gate"])
        frac_sats.append(diag["frac_saturated"])
        n_crafted += 1
        if n_crafted % args.log_every == 0:
            logger.info(
                f"[{tag}] [{n_crafted}/{args.n_images}] elapsed={time.time() - t0:.1f}s "
                f"mean_gate={diag['mean_gate']:.3f} frac_saturated={diag['frac_saturated']:.3f}"
            )
    craft_elapsed = time.time() - t0
    avg_mean_gate = sum(mean_gates) / len(mean_gates) if mean_gates else float("nan")
    avg_frac_sat = sum(frac_sats) / len(frac_sats) if frac_sats else float("nan")
    logger.info(
        f"[{tag}] craft finished: {n_crafted} crafted, {n_skipped} skipped -> {noise_dir}  "
        f"(avg mean_gate={avg_mean_gate:.3f}, avg frac_saturated={avg_frac_sat:.3f})"
    )
    del surrogate
    torch.cuda.empty_cache()

    eval_args = SimpleNamespace(
        checkpoints_dir=args.checkpoints_dir,
        canvas=args.canvas,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        noise_dir=PROJECT_DIR / "results" / "noise" / args.manifest.stem,
        predictions_dir=PROJECT_DIR / "results" / "_m_pilot_predictions",
        force_clean=False,
        device=args.device,
        attacks=[f"m_{tag}"],
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
        f"osfd_rcg_{tag}_{args.manifest.stem}_n{args.n_images}",
        {
            "attack": f"m_{tag}",
            "manifest": str(args.manifest),
            "seed": args.seed,
            "n_images": args.n_images,
            "score_thr": args.score_thr,
            "iou_thr": args.iou_thr,
            "config": {**vars(cfg), **variant},
            "diagnostics": {"avg_mean_gate": avg_mean_gate, "avg_frac_saturated": avg_frac_sat},
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
    return [r for r in rows if r["attack"] == f"m_{tag}"], {"mean_gate": avg_mean_gate, "frac_saturated": avg_frac_sat}


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
    all_diag: dict[str, dict] = {}
    for variant in run_variants:
        rows, diag = craft_and_evaluate_variant(variant, args, coco, image_ids, gt_index, img_dir, logger)
        all_rows[variant["tag"]] = rows
        all_diag[variant["tag"]] = diag

    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    import csv

    fieldnames = ["model_name", "group", "ASR_osfd", "ASR_rcg_avg", "ASR_rcg_gate",
                  "mAP_drop_osfd", "mAP_drop_rcg_avg", "mAP_drop_rcg_gate",
                  "delta_avg_vs_osfd_ASR", "delta_gate_vs_avg_ASR"]
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for tag in ("osfd", "rcg_avg", "rcg_gate"):
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
            row[f"mAP_drop_{tag}"] = r.get("mAP_drop_pct")
        if None not in (row["ASR_osfd"], row["ASR_rcg_avg"], row["ASR_rcg_gate"]):
            row["delta_avg_vs_osfd_ASR"] = row["ASR_rcg_avg"] - row["ASR_osfd"]
            row["delta_gate_vs_avg_ASR"] = row["ASR_rcg_gate"] - row["ASR_rcg_avg"]
        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info(f"=== diagnostics (mean_gate, frac_saturated) per variant: {all_diag} ===")
    logger.info("=== RCG pilot: ASR (%) osfd -> rcg_avg -> rcg_gate ===")
    for r in comparison:
        logger.info(
            f"{r['model_name']:20s} {r['group'] or '-':4s} "
            f"osfd={r.get('ASR_osfd', float('nan')):.1f} avg={r.get('ASR_rcg_avg', float('nan')):.1f} "
            f"gate={r.get('ASR_rcg_gate', float('nan')):.1f}  "
            f"(avg-osfd={r.get('delta_avg_vs_osfd_ASR', float('nan')):+.1f}, "
            f"gate-avg={r.get('delta_gate_vs_avg_ASR', float('nan')):+.1f})"
        )


if __name__ == "__main__":
    main()
