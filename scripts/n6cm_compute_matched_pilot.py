#!/usr/bin/env python
"""Compute-matched control pilot for N6-B: the strongest remaining reviewer
objection to path-averaged OSFD (path_m3, M=3, clean->current -- see
n6b_path_pilot.py) is "the ASR gain is just from 3x more forward/backward
passes per step (Monte-Carlo variance reduction), not from path structure
specifically". This pilot answers that directly by adding a THIRD trajectory,
`rrb_avg_k3`, that costs the SAME 3 forward/backward passes per step as
path_m3 but averages over K=3 independently-sampled RRB views at the
CURRENT noise state (no path/lambda-scaling) -- i.e. the mechanism this
project already tested as RCG-AVG (Phase M, results/m_rcg_pilot_summary.csv)
and as N6-A's own `rrb_avg_k3` (results/n6a_gcr_pilot_summary.csv). Those two
existing datapoints do NOT agree with each other on DINO gain (+11.0 vs
+3.0 at N=20, dev_50) despite nominally identical config -- traced to being
separate script implementations, not lockstep-paired against path_m3 or
against each other, so their disagreement is not decisive. This pilot fixes
that by crafting all THREE trajectories in true 3-way lockstep.

Three trajectories per image, noise_local / noise_avg / noise_path, updated
independently each step but sharing RNG discipline:
    osfd_local  1 draw/step  at lambda=1 (current noise)      -- reference,
                                                                  same as this
                                                                  project's
                                                                  standard OSFD
    rrb_avg_k3  K=3 draws/step, independently sampled RRB views, all at
                lambda=1 (current noise) -- gradient = mean of the 3
    path_m3     M=3 draws/step at lambda_m=m/3 (m=1,2,3) of current noise,
                SAME single augmentation draw shared across all 3 lambdas
                (per n6b_path_pilot.py's design) -- gradient = mean of the 3

RNG pairing (stronger than either prior pilot alone -- see module comparison
below): ONE rng_snapshot captured per step. It is restored immediately before
osfd_local's single draw, again before rrb_avg_k3's FIRST of its 3 sequential
draws (draws 2-3 then advance the RNG naturally, exactly as
n6a_gcr_pilot.py's draw_k_grads does -- that is what "K independently-sampled
views" means and cannot be paired away without destroying the mechanism being
tested), and again before EACH of path_m3's 3 lambda-draws (all 3 lambdas see
the IDENTICAL single augmentation instance, per n6b_path_pilot.py's own
design -- path varies lambda, not augmentation sampling). Net effect: at
every step, osfd_local's draw, rrb_avg_k3's first draw, and ALL THREE of
path_m3's draws use the EXACT SAME augmentation realization -- the only
things that can differ are (a) which noise state is being augmented
(diverges after step 1, as expected) and (b) rrb_avg_k3's 2nd/3rd draws,
which are additional independent stochastic samples by construction (removing
those would remove the mechanism itself). This is strictly tighter pairing
than n6a_gcr_pilot.py achieved for its own osfd_k1 (which was only
seed-matched at step 1, not lockstep-restored every step -- see this
project's HANDOFF.md discussion) and strictly tighter than comparing across
the two separate pre-existing scripts (rcg_avg vs rrb_avg_k3), which is
exactly why those two disagreed.

PRIMARY quantity (pre-registered before running, do not change after seeing
results): delta_path_vs_avg = ASR(path_m3) - ASR(rrb_avg_k3), with paired
image-cluster bootstrap 95% CI (same convention as n6b_path_pilot.py /
N4), on dino_swin_l specifically (the target this project's method targets
most directly). osfd_local is reported as a reference point only (both other
trajectories' own gain over it), not as the primary comparison -- the
question this pilot exists to answer is path-structure-vs-naive-averaging,
not path-vs-nothing (already answered by n6b_path_pilot.py).

GO criterion (pre-registered, decided before running, per user instruction):
delta_path_vs_avg on dino_swin_l >= +3, ideally +3 to +5, with bootstrap CI
not crossing 0 treated as "worth confirming at larger N"; a value near 0
(or CI crossing 0 well within the pilot's own noise floor) means "stop --
part of path_m3's gain may be Monte-Carlo variance reduction from extra
compute, not path structure specifically" -- to be stated plainly, not
rescued by re-tuning M/K/lambda schedule after the fact. mask_rcnn_swin_t and
yolox_l are reported for context, not gating (per the user's stated focus on
DINO for this specific control).

Example:
    python scripts/n6cm_compute_matched_pilot.py --manifest data/manifests/dev_50.json --n-images 20
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

M_LAMBDA = 3   # pre-registered, matches n6b_path_pilot.py, no sweep
K_DRAWS = 3    # pre-registered, matches n6a_gcr_pilot.py / RCG-AVG, no sweep


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
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n6cm_compute_matched_pilot_summary.csv")
    p.add_argument(
        "--models", nargs="+", default=["faster_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"],
    )
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    return p


def capture_rng(device: str):
    import torch

    cuda_state = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
    return random.getstate(), torch.get_rng_state(), cuda_state


def restore_rng(state, device: str) -> None:
    import torch

    py_state, cpu_state, cuda_state = state
    random.setstate(py_state)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def craft_paired_three_way(handle, x_clean, gt_boxes, cfg, device, m_lambda: int, k_draws: int):
    """Crafts osfd_local, rrb_avg_k3, and path_m3 in 3-way lockstep for one
    image (see module docstring for the exact RNG-sharing discipline).
    Returns (noise_local, noise_avg, noise_path)."""
    import torch

    from transfer_attack.augment import rrb_forward
    from transfer_attack.losses import osfd_loss

    model = handle.model
    with torch.no_grad():
        feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))

    noise_local = torch.randint_like(x_clean, low=-2, high=3).float()
    noise_avg = noise_local.clone()
    noise_path = noise_local.clone()
    g_mom_local = torch.zeros_like(x_clean)
    g_mom_avg = torch.zeros_like(x_clean)
    g_mom_path = torch.zeros_like(x_clean)

    for _ in range(cfg.steps):
        noise_local = noise_local.detach().requires_grad_(True)
        noise_avg = noise_avg.detach().requires_grad_(True)
        noise_path = noise_path.detach().requires_grad_(True)

        rng_snapshot = capture_rng(device)

        # --- osfd_local: 1 draw at lambda=1 ---
        restore_rng(rng_snapshot, device)
        x_adv_local = torch.clamp(x_clean + noise_local, 0.0, 255.0)
        aug_local = rrb_forward(x_adv_local.unsqueeze(0), gt_boxes, cfg)
        feats_adv_local = model.backbone(handle.normalize(aug_local))
        loss_local = osfd_loss(feats_cln, feats_adv_local, cfg.k)
        (g_local,) = torch.autograd.grad(loss_local, noise_local)

        # --- rrb_avg_k3: K=3 independently-sampled views at lambda=1 ---
        restore_rng(rng_snapshot, device)
        grads_avg = []
        for _ in range(k_draws):
            x_adv_avg = torch.clamp(x_clean + noise_avg, 0.0, 255.0)
            aug = rrb_forward(x_adv_avg.unsqueeze(0), gt_boxes, cfg)
            feats_adv = model.backbone(handle.normalize(aug))
            loss = osfd_loss(feats_cln, feats_adv, cfg.k)
            (g_i,) = torch.autograd.grad(loss, noise_avg)
            grads_avg.append(g_i.detach())
        g_avg = torch.stack(grads_avg, dim=0).mean(dim=0)

        # --- path_m3: M=3 draws at lambda_m, SAME single augmentation shared ---
        grads_path = []
        for m in range(1, m_lambda + 1):
            lam = m / m_lambda
            restore_rng(rng_snapshot, device)
            x_lam = torch.clamp(x_clean + lam * noise_path, 0.0, 255.0)
            aug = rrb_forward(x_lam.unsqueeze(0), gt_boxes, cfg)
            feats_adv = model.backbone(handle.normalize(aug))
            loss = osfd_loss(feats_cln, feats_adv, cfg.k)
            (g_m,) = torch.autograd.grad(loss, noise_path)
            grads_path.append(g_m.detach())
        g_path = torch.stack(grads_path, dim=0).mean(dim=0)

        with torch.no_grad():
            g_mom_local = cfg.mu * g_mom_local + g_local / g_local.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise_local = torch.clamp(noise_local + cfg.alpha * torch.sign(g_mom_local), -cfg.epsilon, cfg.epsilon)

            g_mom_avg = cfg.mu * g_mom_avg + g_avg / g_avg.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise_avg = torch.clamp(noise_avg + cfg.alpha * torch.sign(g_mom_avg), -cfg.epsilon, cfg.epsilon)

            g_mom_path = cfg.mu * g_mom_path + g_path / g_path.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise_path = torch.clamp(noise_path + cfg.alpha * torch.sign(g_mom_path), -cfg.epsilon, cfg.epsilon)

    return noise_local.detach(), noise_avg.detach(), noise_path.detach()


def paired_bootstrap_asr_delta(per_image, n_draws: int, seed: int):
    """per_image: list of (evaded_a, evaded_b, n_clean_correct) tuples, one
    per image. Returns (delta_point=ASR_b-ASR_a, ci_low, ci_high, same_side_frac)."""
    rng = random.Random(seed)
    n = len(per_image)
    tot_a = sum(p[0] for p in per_image)
    tot_b = sum(p[1] for p in per_image)
    tot_cc = sum(p[2] for p in per_image)
    if tot_cc == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    asr_a = 100.0 * tot_a / tot_cc
    asr_b = 100.0 * tot_b / tot_cc
    delta_point = asr_b - asr_a

    deltas = []
    for _ in range(n_draws):
        sample = rng.choices(per_image, k=n)
        s_a = sum(p[0] for p in sample)
        s_b = sum(p[1] for p in sample)
        s_cc = sum(p[2] for p in sample)
        if s_cc == 0:
            continue
        deltas.append(100.0 * s_b / s_cc - 100.0 * s_a / s_cc)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]
    if delta_point >= 0:
        same_side = sum(1 for d in deltas if d >= 0) / len(deltas)
    else:
        same_side = sum(1 for d in deltas if d < 0) / len(deltas)
    return delta_point, lo, hi, same_side


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch
    import evaluate as evaluate_mod

    from transfer_attack.attack import AttackConfig
    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, save_noise, save_run_log
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle, get_spec

    logger = get_logger()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"
    device = args.device

    cfg = AttackConfig(attack_type="osfd", k=3.0, use_rrb=True, steps=args.steps, canvas=args.canvas)

    tags = ("osfd_local", "rrb_avg_k3", "path_m3")
    noise_dirs = {tag: PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"n6cm_{tag}" for tag in tags}
    for d in noise_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=device, coco=coco)

    used_image_ids = []
    t0 = time.time()
    n_crafted, n_skipped = 0, 0
    for image_id in image_ids:
        if len(used_image_ids) >= args.n_images:
            break
        gt_entries = gt_index[image_id]
        if not gt_entries:
            n_skipped += 1
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, cfg.canvas)
        gt_boxes, _ = gt_to_canvas(gt_entries, scale)
        x_clean = canvas_img.to(device)
        gt_boxes = gt_boxes.to(device)

        random.seed(args.seed + image_id)
        torch.manual_seed(args.seed + image_id)
        noise_local, noise_avg, noise_path = craft_paired_three_way(
            surrogate, x_clean, gt_boxes, cfg, device, M_LAMBDA, K_DRAWS
        )

        save_noise(noise_dirs["osfd_local"] / f"{image_id}.pt", noise_local)
        save_noise(noise_dirs["rrb_avg_k3"] / f"{image_id}.pt", noise_avg)
        save_noise(noise_dirs["path_m3"] / f"{image_id}.pt", noise_path)
        used_image_ids.append(image_id)
        n_crafted += 1
        if n_crafted % args.log_every == 0:
            logger.info(f"[{n_crafted}/{args.n_images}] elapsed={time.time() - t0:.1f}s")
    craft_elapsed = time.time() - t0
    logger.info(f"craft finished: {n_crafted} crafted, {n_skipped} skipped.")
    del surrogate
    torch.cuda.empty_cache()

    from types import SimpleNamespace

    predictions_dir = PROJECT_DIR / "results" / "_n6cm_predictions"
    by_name = {s.name: s for s in MODEL_REGISTRY}
    specs = [by_name[m] for m in args.models]
    gt_cache = evaluate_mod.build_gt_cache(coco, used_image_ids, img_dir, args.canvas)

    all_rows = {}
    for tag in tags:
        eval_args = SimpleNamespace(
            checkpoints_dir=args.checkpoints_dir,
            canvas=args.canvas,
            score_thr=args.score_thr,
            iou_thr=args.iou_thr,
            noise_dir=PROJECT_DIR / "results" / "noise" / args.manifest.stem,
            predictions_dir=predictions_dir,
            force_clean=False,
            device=args.device,
            attacks=[f"n6cm_{tag}"],
        )
        rows = []
        for spec in specs:
            rows.extend(evaluate_mod.evaluate_one_model(spec, eval_args, coco, used_image_ids, img_dir, gt_cache, logger))
        all_rows[tag] = [r for r in rows if r["attack"] == f"n6cm_{tag}"]

        run_log_path = save_run_log(
            args.runs_dir,
            "run_attack",
            f"osfd_n6cm_{tag}_{args.manifest.stem}_n{args.n_images}",
            {
                "attack": f"n6cm_{tag}",
                "manifest": str(args.manifest),
                "seed": args.seed,
                "n_images": args.n_images,
                "config": {**vars(cfg), "m_lambda": M_LAMBDA, "k_draws": K_DRAWS},
                "results": {
                    "n_images_used": len(used_image_ids),
                    "n_crafted": n_crafted,
                    "n_skipped": n_skipped,
                    "craft_elapsed_sec": round(craft_elapsed, 1),
                    "rows": all_rows[tag],
                },
            },
        )
        logger.info(f"[{tag}] run log written -> {run_log_path}")

    # ---- Paired image-cluster bootstrap on PRIMARY quantity: path_m3 - rrb_avg_k3 ----
    from transfer_attack.eval_metrics import compute_asr_for_image
    from transfer_attack.io_utils import load_predictions

    _, gt_boxes_by_image_id, gt_cat_ids_by_image_id = gt_cache
    bootstrap_by_model = {}
    for spec in specs:
        clean_preds = load_predictions(predictions_dir / "clean" / f"{spec.name}.json")
        preds_local = load_predictions(predictions_dir / "n6cm_osfd_local" / f"{spec.name}.json")
        preds_avg = load_predictions(predictions_dir / "n6cm_rrb_avg_k3" / f"{spec.name}.json")
        preds_path = load_predictions(predictions_dir / "n6cm_path_m3" / f"{spec.name}.json")
        per_image = []
        for image_id in used_image_ids:
            if image_id not in preds_avg or image_id not in preds_path:
                continue
            gt_boxes = gt_boxes_by_image_id[image_id]
            gt_cat_ids = gt_cat_ids_by_image_id[image_id]
            evaded_avg, cc = compute_asr_for_image(clean_preds[image_id], preds_avg[image_id], gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr)
            evaded_path, cc2 = compute_asr_for_image(clean_preds[image_id], preds_path[image_id], gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr)
            per_image.append((evaded_avg, evaded_path, cc))
        delta_point, ci_lo, ci_hi, same_side = paired_bootstrap_asr_delta(per_image, args.bootstrap_draws, args.seed)
        bootstrap_by_model[spec.name] = {
            "delta_path_vs_avg": delta_point, "ci_lo": ci_lo, "ci_hi": ci_hi, "same_side_frac": same_side,
        }

    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    import csv

    fieldnames = ["model_name", "group", "ASR_osfd_local", "ASR_rrb_avg_k3", "ASR_path_m3",
                  "delta_path_vs_avg", "delta_path_vs_local", "delta_avg_vs_local",
                  "bootstrap_ci_lo", "bootstrap_ci_hi", "bootstrap_same_side_frac"]
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for tag in tags:
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
        if row.get("ASR_path_m3") is not None and row.get("ASR_rrb_avg_k3") is not None:
            row["delta_path_vs_avg"] = row["ASR_path_m3"] - row["ASR_rrb_avg_k3"]
        if row.get("ASR_path_m3") is not None and row.get("ASR_osfd_local") is not None:
            row["delta_path_vs_local"] = row["ASR_path_m3"] - row["ASR_osfd_local"]
        if row.get("ASR_rrb_avg_k3") is not None and row.get("ASR_osfd_local") is not None:
            row["delta_avg_vs_local"] = row["ASR_rrb_avg_k3"] - row["ASR_osfd_local"]
        bs = bootstrap_by_model.get(model_name, {})
        row["bootstrap_ci_lo"] = bs.get("ci_lo")
        row["bootstrap_ci_hi"] = bs.get("ci_hi")
        row["bootstrap_same_side_frac"] = bs.get("same_side_frac")
        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info("=== N6-CM compute-matched pilot: ASR (%) osfd_local / rrb_avg_k3 / path_m3 ===")
    for r in comparison:
        logger.info(
            f"{r['model_name']:20s} {r['group'] or '-':4s} "
            f"local={r.get('ASR_osfd_local', float('nan')):.1f} avg={r.get('ASR_rrb_avg_k3', float('nan')):.1f} "
            f"path={r.get('ASR_path_m3', float('nan')):.1f}  "
            f"(path-avg={r.get('delta_path_vs_avg', float('nan')):+.1f}, "
            f"95% CI=[{r.get('bootstrap_ci_lo', float('nan')):+.1f}, {r.get('bootstrap_ci_hi', float('nan')):+.1f}], "
            f"same_side_frac={r.get('bootstrap_same_side_frac', float('nan')):.3f})"
        )


if __name__ == "__main__":
    main()
