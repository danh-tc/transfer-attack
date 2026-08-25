#!/usr/bin/env python
"""N6-B novelty control (after alignment mechanism-proof was NO-GO as primary
mechanism, see RESEARCH.md §19). Answers the remaining novelty risk directly:
does N6-B win because path-averaging is a GENERICALLY good trick for
transferable OD attacks (any objective), or because it has a SPECIAL
interaction with OSFD's object-aware backbone-feature-distortion objective?

4 variants, all on the surrogate (faster_rcnn_r50), same epsilon/alpha/steps/
mu/M=3/clean->current path -- only the per-step ascent loss + whether RRB is
applied differs:

    det_local   detector task loss (mmdet .loss(), classification+box terms)
                -- SAME loss as this project's `mi_fgsm` baseline
                (transfer_attack.attack, attack_type="mi_fgsm") -- instantaneous
                gradient, NO RRB (mi_fgsm has never used RRB in this project).
    det_path    same task loss, path-averaged gradient (M=3, clean->current),
                NO RRB.
    osfd_local  OSFD backbone-feature-distortion loss, instantaneous gradient,
                WITH RRB -- algebraically identical to n6b_path_pilot.py's
                osfd_local (same function, imported directly, not
                reimplemented).
    osfd_path   OSFD loss, path-averaged gradient (M=3), WITH RRB -- identical
                to n6b_path_pilot.py's path_m3.

IMPORTANT ASYMMETRY, stated explicitly (do not read this as a bug): det_* has
no RRB and osfd_* does. This is NOT a compute-identical ablation of "path vs
no-path holding everything else fixed" -- it is an OBJECTIVE control: det_*
uses this project's own pre-existing task-loss attack (mi_fgsm) exactly as
already defined, unmodified, with the SAME path-averaging operator bolted on
in the same place OSFD's is. RRB is specific to OSFD's own design (the paper
motivates it as feature-loss-specific augmentation-robustness); inventing a
novel "RRB'd task loss" variant would not be a control for anything already
in this project and would conflate two changes at once. Lockstep RNG pairing
(shared draw between local/path at each step) is applied within osfd_local/
osfd_path (there IS augmentation randomness to pair, exactly as in
n6b_path_pilot.py) and is a no-op within det_local/det_path (no stochastic
augmentation exists there to pair -- the only randomness, noise init and
image seed, is already matched by the existing per-image seeding
convention).

Primary quantity (per target model): the PATH GAIN under each objective,
    G_det  = ASR(det_path)  - ASR(det_local)
    G_osfd = ASR(osfd_path) - ASR(osfd_local)
and the question this control exists to answer:
    G_osfd > G_det  (especially on dino_swin_l)
    => path-averaging is NOT generically sufficient; its benefit is amplified
       specifically by OSFD's object-aware feature-distortion objective
       (novelty story strengthens).
    G_det ~= G_osfd
    => path-averaging helps generically regardless of objective; the honest
       framing becomes "trajectory-integrated gradients for transferable OD"
       (closer to MIG/MuMoDIG's own framing, needs direct comparison to
       HIFA-style baselines) rather than an OSFD-specific mechanism.

No GO/NO-GO threshold is pre-registered here (unlike prior N6 pilots) because
this is explicitly a diagnostic/positioning experiment, not a method decision
gate -- the RESULT ITSELF (which regime we're in) is what determines the next
research question, not a pass/fail bar. Report both G_det and G_osfd with
bootstrap CIs on the interaction delta (G_osfd - G_det) per target, and read
the number honestly either way.

Example:
    python scripts/n6b_novelty_control.py --manifest data/manifests/dev_50.json --n-images 20
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

M_LAMBDA = 3  # matches N6-B v0 -- same M throughout, not swept


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
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n6b_novelty_control_summary.csv")
    p.add_argument("--targets", nargs="+", default=["yolox_l", "mask_rcnn_swin_t", "dino_swin_l"])
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


def craft_paired_local_path_det(handle, x_clean, gt_boxes, gt_cat_ids, cfg, device, m_lambda: int):
    """det_local / det_path in lockstep, mirroring n6b_path_pilot.py's
    craft_paired_local_path but for the detector task loss, NO RRB (see
    module docstring for why). No stochastic augmentation exists here, so the
    RNG-snapshot dance is kept only for structural symmetry / in case a
    future variant adds one -- it is a no-op today."""
    import torch

    from transfer_attack.losses import build_gt_data_sample, detector_task_loss

    model = handle.model
    gt_labels = torch.tensor(
        [handle.cat_id_to_label[int(c)] for c in gt_cat_ids], dtype=torch.long, device=device
    )
    data_sample = build_gt_data_sample(gt_boxes, gt_labels, cfg.canvas)

    noise_local = torch.randint_like(x_clean, low=-2, high=3).float()
    noise_path = noise_local.clone()
    g_mom_local = torch.zeros_like(x_clean)
    g_mom_path = torch.zeros_like(x_clean)

    for _ in range(cfg.steps):
        noise_local = noise_local.detach().requires_grad_(True)
        noise_path = noise_path.detach().requires_grad_(True)

        rng_snapshot = capture_rng(device)

        restore_rng(rng_snapshot, device)
        x_adv_local = torch.clamp(x_clean + noise_local, 0.0, 255.0)
        x_norm_local = handle.normalize(x_adv_local.unsqueeze(0))
        loss_local = detector_task_loss(model, x_norm_local, data_sample)
        (g_local,) = torch.autograd.grad(loss_local, noise_local)

        grads_path = []
        for m in range(1, m_lambda + 1):
            lam = m / m_lambda
            restore_rng(rng_snapshot, device)
            x_lam = torch.clamp(x_clean + lam * noise_path, 0.0, 255.0)
            x_norm_lam = handle.normalize(x_lam.unsqueeze(0))
            loss_m = detector_task_loss(model, x_norm_lam, data_sample)
            (grad_m,) = torch.autograd.grad(loss_m, noise_path)
            grads_path.append(grad_m.detach())
        g_path = torch.stack(grads_path, dim=0).mean(dim=0)

        with torch.no_grad():
            g_mom_local = cfg.mu * g_mom_local + g_local / g_local.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise_local = torch.clamp(noise_local + cfg.alpha * torch.sign(g_mom_local), -cfg.epsilon, cfg.epsilon)

            g_mom_path = cfg.mu * g_mom_path + g_path / g_path.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise_path = torch.clamp(noise_path + cfg.alpha * torch.sign(g_mom_path), -cfg.epsilon, cfg.epsilon)

    return noise_local.detach(), noise_path.detach()


def paired_bootstrap_interaction(per_image, n_draws: int, seed: int):
    """per_image: list of dicts with keys
    {det_local, det_path, osfd_local, osfd_path, clean_correct} (evaded
    counts + clean_correct count per image, image-cluster unit).
    Returns dict of {variant: (asr_point)} plus G_det, G_osfd,
    interaction delta point/CI/same_side_frac."""
    rng = random.Random(seed)
    n = len(per_image)

    def asrs(sample):
        tot_cc = sum(p["clean_correct"] for p in sample)
        out = {}
        for v in ("det_local", "det_path", "osfd_local", "osfd_path"):
            tot = sum(p[v] for p in sample)
            out[v] = 100.0 * tot / tot_cc if tot_cc > 0 else float("nan")
        return out

    point = asrs(per_image)
    g_det_point = point["det_path"] - point["det_local"]
    g_osfd_point = point["osfd_path"] - point["osfd_local"]
    interaction_point = g_osfd_point - g_det_point

    deltas = []
    for _ in range(n_draws):
        sample = rng.choices(per_image, k=n)
        a = asrs(sample)
        g_det = a["det_path"] - a["det_local"]
        g_osfd = a["osfd_path"] - a["osfd_local"]
        deltas.append(g_osfd - g_det)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]
    if interaction_point >= 0:
        same_side = sum(1 for d in deltas if d >= 0) / len(deltas)
    else:
        same_side = sum(1 for d in deltas if d < 0) / len(deltas)
    return point, g_det_point, g_osfd_point, interaction_point, lo, hi, same_side


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch
    import evaluate as evaluate_mod

    from transfer_attack.attack import AttackConfig
    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, save_noise, save_run_log
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle, get_spec

    from n6b_path_pilot import craft_paired_local_path

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

    variants = ("det_local", "det_path", "osfd_local", "osfd_path")
    noise_dirs = {v: PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"n6bctl_{v}" for v in variants}
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
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        x_clean = canvas_img.to(device)
        gt_boxes = gt_boxes.to(device)

        random.seed(args.seed + image_id)
        torch.manual_seed(args.seed + image_id)
        noise_osfd_local, noise_osfd_path, _diag = craft_paired_local_path(
            surrogate, x_clean, gt_boxes, cfg, device, M_LAMBDA
        )

        random.seed(args.seed + image_id)
        torch.manual_seed(args.seed + image_id)
        noise_det_local, noise_det_path = craft_paired_local_path_det(
            surrogate, x_clean, gt_boxes, gt_cat_ids, cfg, device, M_LAMBDA
        )

        save_noise(noise_dirs["osfd_local"] / f"{image_id}.pt", noise_osfd_local)
        save_noise(noise_dirs["osfd_path"] / f"{image_id}.pt", noise_osfd_path)
        save_noise(noise_dirs["det_local"] / f"{image_id}.pt", noise_det_local)
        save_noise(noise_dirs["det_path"] / f"{image_id}.pt", noise_det_path)
        used_image_ids.append(image_id)
        n_crafted += 1
        if n_crafted % args.log_every == 0:
            logger.info(f"[craft] {n_crafted}/{args.n_images} elapsed={time.time() - t0:.1f}s")
    craft_elapsed = time.time() - t0
    logger.info(f"craft finished: {n_crafted} crafted, {n_skipped} skipped, elapsed={craft_elapsed:.1f}s")
    del surrogate
    torch.cuda.empty_cache()

    from types import SimpleNamespace

    predictions_dir = PROJECT_DIR / "results" / "_n6b_novelty_control_predictions"
    specs = [get_spec("faster_rcnn_r50")] + [get_spec(t) for t in args.targets]
    gt_cache = evaluate_mod.build_gt_cache(coco, used_image_ids, img_dir, args.canvas)

    all_rows = {}
    for v in variants:
        eval_args = SimpleNamespace(
            checkpoints_dir=args.checkpoints_dir,
            canvas=args.canvas,
            score_thr=args.score_thr,
            iou_thr=args.iou_thr,
            noise_dir=PROJECT_DIR / "results" / "noise" / args.manifest.stem,
            predictions_dir=predictions_dir,
            force_clean=False,
            device=args.device,
            attacks=[f"n6bctl_{v}"],
        )
        rows = []
        for spec in specs:
            rows.extend(evaluate_mod.evaluate_one_model(spec, eval_args, coco, used_image_ids, img_dir, gt_cache, logger))
        all_rows[v] = [r for r in rows if r["attack"] == f"n6bctl_{v}"]

        run_log_path = save_run_log(
            args.runs_dir,
            "run_attack",
            f"n6bctl_{v}_{args.manifest.stem}_n{args.n_images}",
            {
                "attack": f"n6bctl_{v}",
                "manifest": str(args.manifest),
                "seed": args.seed,
                "n_images": args.n_images,
                "config": {**vars(cfg), "m_lambda": M_LAMBDA, "variant": v, "uses_rrb": v.startswith("osfd")},
                "results": {
                    "n_images_used": len(used_image_ids),
                    "n_crafted": n_crafted,
                    "n_skipped": n_skipped,
                    "craft_elapsed_sec": round(craft_elapsed, 1),
                    "rows": all_rows[v],
                },
            },
        )
        logger.info(f"[{v}] run log written -> {run_log_path}")

    from transfer_attack.eval_metrics import compute_asr_for_image
    from transfer_attack.io_utils import load_predictions

    _, gt_boxes_by_image_id, gt_cat_ids_by_image_id = gt_cache
    import csv

    fieldnames = [
        "model_name", "group",
        "ASR_det_local", "ASR_det_path", "G_det",
        "ASR_osfd_local", "ASR_osfd_path", "G_osfd",
        "interaction_G_osfd_minus_G_det", "bootstrap_ci_lo", "bootstrap_ci_hi", "bootstrap_same_side_frac",
    ]
    comparison = []
    for spec in specs:
        clean_preds = load_predictions(predictions_dir / "clean" / f"{spec.name}.json")
        preds_by_variant = {
            v: load_predictions(predictions_dir / f"n6bctl_{v}" / f"{spec.name}.json") for v in variants
        }
        per_image = []
        for image_id in used_image_ids:
            if not all(image_id in preds_by_variant[v] for v in variants):
                continue
            gt_boxes = gt_boxes_by_image_id[image_id]
            gt_cat_ids_ = gt_cat_ids_by_image_id[image_id]
            row = {}
            cc_vals = []
            for v in variants:
                evaded, cc = compute_asr_for_image(
                    clean_preds[image_id], preds_by_variant[v][image_id], gt_boxes, gt_cat_ids_, args.iou_thr, args.score_thr
                )
                row[v] = evaded
                cc_vals.append(cc)
            row["clean_correct"] = cc_vals[0]
            per_image.append(row)

        point, g_det, g_osfd, interaction, ci_lo, ci_hi, same_side = paired_bootstrap_interaction(
            per_image, args.bootstrap_draws, args.seed
        )
        group = next((s.group for s in specs if s.name == spec.name), "") or ""
        comparison.append(
            {
                "model_name": spec.name,
                "group": group,
                "ASR_det_local": point["det_local"],
                "ASR_det_path": point["det_path"],
                "G_det": g_det,
                "ASR_osfd_local": point["osfd_local"],
                "ASR_osfd_path": point["osfd_path"],
                "G_osfd": g_osfd,
                "interaction_G_osfd_minus_G_det": interaction,
                "bootstrap_ci_lo": ci_lo,
                "bootstrap_ci_hi": ci_hi,
                "bootstrap_same_side_frac": same_side,
            }
        )
        logger.info(
            f"{spec.name:20s} G_det={g_det:+.1f} G_osfd={g_osfd:+.1f} "
            f"interaction={interaction:+.1f} 95% CI=[{ci_lo:+.1f}, {ci_hi:+.1f}] same_side={same_side:.3f}"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
