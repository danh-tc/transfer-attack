#!/usr/bin/env python
"""Phase N6-B v0 pilot: clean->current-state path-averaged OSFD gradient
(PRE-REGISTERED M=3, no sweep), vs standard instantaneous OSFD (osfd_local),
same budget (epsilon, alpha, steps, momentum, k, RRB) -- the only thing that
differs between the two trajectories is how each step's ascent gradient is
computed. Follow-up to the N6-B0 diagnostic (post-hoc, no recraft;
`results/n6b0_path_gradient_diagnostic.csv`), which found g_path meaningfully
different from g_local (cos=0.699, not a no-op) and higher one-step
incremental evasion from the path direction on 2/3 hard targets
(mask_rcnn_swin_t +3.1, dino_swin_l +4.0; yolox_l tied +0.0).

Mechanism (per step t, current per-trajectory noise delta_t, x = clean image):
    lambda_m = m/M,  m = 1..M,  M = 3   (PRE-REGISTERED -- do not sweep if
                                          this pilot fails; that is a new pilot)
    g_m        = grad_delta[ L_OSFD(x + lambda_m * delta_t) ]
    g_path,t   = mean_m g_m                       (right-Riemann path average)
    m_t        = mu * m_{t-1} + g_path,t / mean(|g_path,t|)
    delta_{t+1} = clip(delta_t + alpha * sign(m_t), -epsilon, epsilon)
`osfd_local` uses the identical update rule with only the m=M term (lambda=1,
i.e. x+delta_t exactly) -- algebraically the standard OSFD attack already
used as baseline throughout this project (same as n6a_gcr_pilot.py's
`osfd_k1`).

Why clean->current path, NOT a local window around x+delta_t (x+delta_t+xi):
a window formulation tests a *different* hypothesis -- local-neighborhood
smoothing / flatness -- which overlaps VMI-FGSM and the flatness-based
transfer literature this project deliberately parked (RESEARCH.md N2-C).
N6-B0 diagnosed and got a GO signal for the clean->current path specifically;
this pilot tests exactly that, not a nearby variant.

Pairing (same rigor as n6a_gcr_pilot.py's AVG/CR pairing): `osfd_local` and
`osfd_path_m3` are crafted in LOCKSTEP per image, sharing one RNG snapshot per
step -- restored before osfd_local's single rrb_forward draw AND before each
of the M path draws -- so every trajectory sees the SAME random rotation/
resize/blur PARAMETERS at that step (each trajectory's own current noise,
which naturally diverges after step 1 -- that divergence is real and
expected; only the augmentation *sampling* is paired, removing "path got
luckier draws than local" as a confound).

Diagnostics logged at steps {1,25,50,75,100} (mean over images), computed
"for free" from the path trajectory's own M draws at ITS OWN current state
(no extra forward/backward): the m=M draw at that step already equals the
instantaneous gradient at that same point, so:
    cos_path_local   = cos(g_path,t , g_m=M)             at that step
    sign_disagree    = P[sign(g_path,t) != sign(g_m=M)]  at that step
If cosine drifts back toward 1 over steps, the path effect is front-loaded
only; if it stays ~0.7-0.8 throughout, that is a much stronger mechanism
claim than a one-shot diagnostic can support alone.

GO criterion (pre-registered, decided before running): ASR(path_m3) -
ASR(local) >= +3 on >= 2/3 of {yolox_l, mask_rcnn_swin_t, dino_swin_l}, AND no
target drops by more than 3 points. A DINO gain >= +5 is flagged as
especially notable but is not itself a separate gate.

Example:
    python scripts/n6b_path_pilot.py --manifest data/manifests/dev_50.json --n-images 20
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

M_LAMBDA = 3  # pre-registered, no sweep
DIAG_STEPS = (1, 25, 50, 75, 100)


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
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n6b_path_pilot_summary.csv")
    p.add_argument(
        "--models", nargs="+",
        default=["faster_rcnn_r50", "fcos_r50", "deformable_detr", "yolov3_d53", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"],
    )
    p.add_argument("--bootstrap-draws", type=int, default=2000, help="image-cluster paired bootstrap draws for ASR(path)-ASR(local) CI")
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


def craft_paired_local_path(handle, x_clean, gt_boxes, cfg, device, m_lambda: int, diag_steps=DIAG_STEPS):
    """Crafts osfd_local and osfd_path_m3 in lockstep for one image (see
    module docstring). Returns (noise_local, noise_path, diag_by_step) where
    diag_by_step: {step: {"cos_path_local": float, "sign_disagree": float}}."""
    import torch

    from transfer_attack.augment import rrb_forward
    from transfer_attack.losses import osfd_loss

    model = handle.model
    with torch.no_grad():
        feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))

    noise_local = torch.randint_like(x_clean, low=-2, high=3).float()
    noise_path = noise_local.clone()
    g_mom_local = torch.zeros_like(x_clean)
    g_mom_path = torch.zeros_like(x_clean)

    diag_by_step: dict[int, dict] = {}

    for step in range(1, cfg.steps + 1):
        noise_local = noise_local.detach().requires_grad_(True)
        noise_path = noise_path.detach().requires_grad_(True)

        rng_snapshot = capture_rng(device)

        restore_rng(rng_snapshot, device)
        x_adv_local = torch.clamp(x_clean + noise_local, 0.0, 255.0)
        aug_local = rrb_forward(x_adv_local.unsqueeze(0), gt_boxes, cfg)
        feats_adv_local = model.backbone(handle.normalize(aug_local))
        loss_local = osfd_loss(feats_cln, feats_adv_local, cfg.k)
        (g_local,) = torch.autograd.grad(loss_local, noise_local)

        grads_path = []
        for m in range(1, m_lambda + 1):
            lam = m / m_lambda
            restore_rng(rng_snapshot, device)
            x_lam = torch.clamp(x_clean + lam * noise_path, 0.0, 255.0)
            aug = rrb_forward(x_lam.unsqueeze(0), gt_boxes, cfg)
            feats_adv = model.backbone(handle.normalize(aug))
            loss = osfd_loss(feats_cln, feats_adv, cfg.k)
            (grad_m,) = torch.autograd.grad(loss, noise_path)
            grads_path.append(grad_m.detach())
        g_path = torch.stack(grads_path, dim=0).mean(dim=0)

        if step in diag_steps:
            g_instant_at_path_state = grads_path[-1]  # lambda=1 term -- "free" same-point comparator
            cos = float(
                torch.nn.functional.cosine_similarity(
                    g_path.flatten().unsqueeze(0), g_instant_at_path_state.flatten().unsqueeze(0)
                ).item()
            )
            sign_disagree = float((torch.sign(g_path) != torch.sign(g_instant_at_path_state)).float().mean().item())
            diag_by_step[step] = {"cos_path_local": cos, "sign_disagree": sign_disagree}

        with torch.no_grad():
            g_mom_local = cfg.mu * g_mom_local + g_local / g_local.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise_local = torch.clamp(noise_local + cfg.alpha * torch.sign(g_mom_local), -cfg.epsilon, cfg.epsilon)

            g_mom_path = cfg.mu * g_mom_path + g_path / g_path.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise_path = torch.clamp(noise_path + cfg.alpha * torch.sign(g_mom_path), -cfg.epsilon, cfg.epsilon)

    return noise_local.detach(), noise_path.detach(), diag_by_step


def paired_bootstrap_asr_delta(per_image, n_draws: int, seed: int):
    """per_image: list of (evaded_local, evaded_path, n_clean_correct) tuples,
    one per image (image-cluster unit, matching N4's bootstrap convention --
    resample IMAGES with replacement, not individual GT boxes/objects).
    Returns (delta_point, ci_low, ci_high, same_side_frac)."""
    rng = random.Random(seed)
    n = len(per_image)
    tot_local = sum(p[0] for p in per_image)
    tot_path = sum(p[1] for p in per_image)
    tot_cc = sum(p[2] for p in per_image)
    if tot_cc == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    asr_local = 100.0 * tot_local / tot_cc
    asr_path = 100.0 * tot_path / tot_cc
    delta_point = asr_path - asr_local

    deltas = []
    for _ in range(n_draws):
        sample = rng.choices(per_image, k=n)
        s_local = sum(p[0] for p in sample)
        s_path = sum(p[1] for p in sample)
        s_cc = sum(p[2] for p in sample)
        if s_cc == 0:
            continue
        deltas.append(100.0 * s_path / s_cc - 100.0 * s_local / s_cc)
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

    noise_dirs = {tag: PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"n6b_{tag}" for tag in ("osfd_local", "path_m3")}
    for d in noise_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=device, coco=coco)

    used_image_ids = []
    diag_by_step_all: dict[int, list[dict]] = {s: [] for s in DIAG_STEPS if s <= args.steps}
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
        noise_local, noise_path, diag_by_step = craft_paired_local_path(
            surrogate, x_clean, gt_boxes, cfg, device, M_LAMBDA
        )

        save_noise(noise_dirs["osfd_local"] / f"{image_id}.pt", noise_local)
        save_noise(noise_dirs["path_m3"] / f"{image_id}.pt", noise_path)
        used_image_ids.append(image_id)
        for s, d in diag_by_step.items():
            diag_by_step_all[s].append(d)
        n_crafted += 1
        if n_crafted % args.log_every == 0:
            last_step = max(diag_by_step_all.keys())
            last = diag_by_step_all[last_step][-1] if diag_by_step_all[last_step] else {}
            logger.info(
                f"[{n_crafted}/{args.n_images}] elapsed={time.time() - t0:.1f}s "
                f"cos@{last_step}={last.get('cos_path_local', float('nan')):.3f} "
                f"sign_disagree@{last_step}={last.get('sign_disagree', float('nan')):.3f}"
            )
    craft_elapsed = time.time() - t0
    diag_summary = {
        s: {
            "cos_path_local": sum(d["cos_path_local"] for d in lst) / len(lst),
            "sign_disagree": sum(d["sign_disagree"] for d in lst) / len(lst),
        }
        for s, lst in diag_by_step_all.items()
        if lst
    }
    logger.info(f"craft finished: {n_crafted} crafted, {n_skipped} skipped.")
    logger.info(f"diagnostics by step: {diag_summary}")
    del surrogate
    torch.cuda.empty_cache()

    from types import SimpleNamespace

    predictions_dir = PROJECT_DIR / "results" / "_n6b_predictions"
    specs = MODEL_REGISTRY
    if args.models:
        by_name = {s.name: s for s in MODEL_REGISTRY}
        specs = [by_name[m] for m in args.models]
    gt_cache = evaluate_mod.build_gt_cache(coco, used_image_ids, img_dir, args.canvas)

    all_rows = {}
    for tag in ("osfd_local", "path_m3"):
        eval_args = SimpleNamespace(
            checkpoints_dir=args.checkpoints_dir,
            canvas=args.canvas,
            score_thr=args.score_thr,
            iou_thr=args.iou_thr,
            noise_dir=PROJECT_DIR / "results" / "noise" / args.manifest.stem,
            predictions_dir=predictions_dir,
            force_clean=False,
            device=args.device,
            attacks=[f"n6b_{tag}"],
        )
        rows = []
        for spec in specs:
            rows.extend(evaluate_mod.evaluate_one_model(spec, eval_args, coco, used_image_ids, img_dir, gt_cache, logger))
        all_rows[tag] = [r for r in rows if r["attack"] == f"n6b_{tag}"]

        run_log_path = save_run_log(
            args.runs_dir,
            "run_attack",
            f"osfd_n6b_{tag}_{args.manifest.stem}_n{args.n_images}",
            {
                "attack": f"n6b_{tag}",
                "manifest": str(args.manifest),
                "seed": args.seed,
                "n_images": args.n_images,
                "config": {**vars(cfg), "m_lambda": M_LAMBDA},
                "diagnostics_by_step": diag_summary if tag == "path_m3" else None,
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

    # ---- Paired image-cluster bootstrap on ASR(path)-ASR(local), reusing the
    # prediction cache evaluate_one_model just populated (no re-inference).
    from transfer_attack.eval_metrics import compute_asr_for_image
    from transfer_attack.io_utils import load_predictions

    _, gt_boxes_by_image_id, gt_cat_ids_by_image_id = gt_cache
    bootstrap_by_model = {}
    for spec in specs:
        clean_preds = load_predictions(predictions_dir / "clean" / f"{spec.name}.json")
        preds_local = load_predictions(predictions_dir / "n6b_osfd_local" / f"{spec.name}.json")
        preds_path = load_predictions(predictions_dir / "n6b_path_m3" / f"{spec.name}.json")
        per_image = []
        for image_id in used_image_ids:
            if image_id not in preds_local or image_id not in preds_path:
                continue
            gt_boxes = gt_boxes_by_image_id[image_id]
            gt_cat_ids = gt_cat_ids_by_image_id[image_id]
            evaded_l, cc = compute_asr_for_image(clean_preds[image_id], preds_local[image_id], gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr)
            evaded_p, cc2 = compute_asr_for_image(clean_preds[image_id], preds_path[image_id], gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr)
            per_image.append((evaded_l, evaded_p, cc))
        delta_point, ci_lo, ci_hi, same_side = paired_bootstrap_asr_delta(per_image, args.bootstrap_draws, args.seed)
        bootstrap_by_model[spec.name] = {
            "delta_point": delta_point, "ci_lo": ci_lo, "ci_hi": ci_hi, "same_side_frac": same_side,
        }

    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    import csv

    fieldnames = ["model_name", "group", "ASR_osfd_local", "ASR_path_m3",
                  "mAP_drop_osfd_local", "mAP_drop_path_m3", "delta_path_vs_local",
                  "bootstrap_ci_lo", "bootstrap_ci_hi", "bootstrap_same_side_frac"]
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for tag in ("osfd_local", "path_m3"):
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
            row[f"mAP_drop_{tag}"] = r.get("mAP_drop_pct")
        if row.get("ASR_osfd_local") is not None and row.get("ASR_path_m3") is not None:
            row["delta_path_vs_local"] = row["ASR_path_m3"] - row["ASR_osfd_local"]
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

    logger.info(f"=== N6-B v0 pilot diagnostics by step: {diag_summary} ===")
    logger.info("=== N6-B v0 pilot: ASR (%) osfd_local -> path_m3, with 95% bootstrap CI on the delta ===")
    for r in comparison:
        logger.info(
            f"{r['model_name']:20s} {r['group'] or '-':4s} "
            f"local={r.get('ASR_osfd_local', float('nan')):.1f} path={r.get('ASR_path_m3', float('nan')):.1f} "
            f"(path-local={r.get('delta_path_vs_local', float('nan')):+.1f}, "
            f"95% CI=[{r.get('bootstrap_ci_lo', float('nan')):+.1f}, {r.get('bootstrap_ci_hi', float('nan')):+.1f}], "
            f"same_side_frac={r.get('bootstrap_same_side_frac', float('nan')):.3f})"
        )


if __name__ == "__main__":
    main()
