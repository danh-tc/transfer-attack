#!/usr/bin/env python
"""Phase N6-B0: cheap diagnostic (NOT a full attack/pilot) -- does a
PATH-AVERAGED OSFD gradient differ meaningfully from the ORDINARY instantaneous
gradient, and if so, does stepping in the path direction damage black-box
targets more than stepping in the local direction? Motivated by MIG (Ma et
al., ICCV 2023) and MuMoDIG (AAAI 2025): integrated/path gradients have been
reported more CNN<->ViT-consistent than instantaneous ones for classification
transfer. This asks the OD-specific, feature-loss-specific version of that
question, and does so BEFORE committing to a full craft-loop redesign --
exactly the discipline N6-A's full pilot skipped (code-then-discover-the-
operator-barely-matters). No recraft: reuses the already-crafted
`n6a_osfd_k1` noise (K=1 draw/step, RRB on, epsilon=5, 100 steps -- plain
OSFD) as delta_t, the "current adversarial state" the path is drawn towards.

Definitions (per pixel/channel, delta_t = existing crafted noise, x = clean
image, cfg.k/theta/l_s/rho/s_max/sigma unchanged from standard OSFD):
    lambda_m = m/M,  m = 1..M                          (path from near-0 to 1)
    g_m      = grad_delta[ L_OSFD(x + lambda_m * delta) ]  at delta=delta_t
    g_local  = g_M   (lambda_M=1 -> x+delta = x+delta_t, the ordinary
                       instantaneous OSFD gradient at the crafted state --
                       algebraically identical to what the normal attack loop
                       computes each step, no separate call needed)
    g_path   = mean_m g_m                                (right-Riemann-sum
                       path average over (0,1], includes the g_local term)
All M draws share ONE captured RNG snapshot (restored before each lambda's
rrb_forward call), so every lambda point sees the SAME random rotation/
resize/blur parameters -- isolates the effect of lambda-scaling the input
from RRB's own sampling randomness (same lockstep trick as
n6a_gcr_pilot.py's AVG/CR pairing).

Two measures, in order of how much they can tell us:
  1. cos(g_path, g_local) and elementwise sign-agreement -- cheap, answers
     "does path-averaging even produce a materially different gradient?"
     before spending anything on measure 2. If cos~1, path-averaging is a
     no-op here and there is nothing to pilot (kills N6-B the same clean way
     N6-A's correction_ratio~0 would have, had it been checked first).
  2. One-step controlled black-box probe: from x_t = clamp(x+delta_t,0,255),
     take ONE additional step of size cfg.alpha in sign(g_local) vs
     sign(g_path), and compare incremental evasion (using the SAME
     clean-correct reference set and ASR definition as the rest of the
     project, eval_metrics.compute_asr_for_image) against x_t alone, on
     yolox_l / mask_rcnn_swin_t / dino_swin_l (+ the surrogate itself as a
     white-box sanity check).

     Deliberate deviation from the real attack budget: this probe step is
     clamped only to valid pixel range [0,255], NOT re-clamped to the
     epsilon=5 L-inf ball around x_clean. Reason: N4b already measured ~86%
     of pixels in a 100-step/epsilon=5 OSFD noise are saturated at the
     boundary; re-imposing the epsilon clamp here would make local and path
     probes collapse to the near-identical saturated result regardless of
     which direction is "better" (exactly the confound that sank DOB-v0).
     This is a deliberate, narrow relaxation for isolating direction quality
     only -- not a claim that an attack could actually use this budget.

GO signal (checked, not pre-registered as a strict gate since this is a
diagnostic feeding a branch decision, not a method pilot): path must first
be shown to differ from local at all (cos(g_path,g_local) meaningfully <1);
only if that holds does the incremental-evasion comparison mean anything.
If path direction gives consistently higher incremental evasion than local on
>=2/3 of {yolox_l, mask_rcnn_swin_t, dino_swin_l} -> proceed to an N6-B
method pilot. Otherwise close N6-B the same disciplined way N6-A closed.

Example:
    python scripts/n6b0_path_gradient_diagnostic.py --manifest data/manifests/dev_50.json --num-lambda 10
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

NOISE_TAG = "n6a_osfd_k1"  # existing crafted noise reused as delta_t, see module docstring


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument(
        "--noise-dir", type=Path, default=PROJECT_DIR / "results" / "noise" / "dev_50" / NOISE_TAG,
        help="directory of existing crafted noise reused as delta_t (default: N6-A's osfd_k1 noise)",
    )
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--num-lambda", type=int, default=10, help="M, number of path-integration points in (0,1]")
    p.add_argument("--alpha", type=float, default=1.0, help="one-step probe size (matches AttackConfig.alpha)")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n6b0_path_gradient_diagnostic.csv")
    p.add_argument("--targets", nargs="+", default=["yolox_l", "mask_rcnn_swin_t", "dino_swin_l"])
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


def compute_path_gradients(surrogate, x_clean, delta_t, gt_boxes, cfg, num_lambda: int, device: str):
    """Returns (g_local, g_path, diagnostics dict) for one image. See module
    docstring for definitions; all M lambda draws share one RNG snapshot."""
    import torch

    from transfer_attack.augment import rrb_forward
    from transfer_attack.losses import osfd_loss

    model = surrogate.model
    with torch.no_grad():
        feats_cln = model.backbone(surrogate.normalize(x_clean.unsqueeze(0)))

    rng_snapshot = capture_rng(device)
    grads = []
    for m in range(1, num_lambda + 1):
        lam = m / num_lambda
        restore_rng(rng_snapshot, device)
        delta = delta_t.clone().detach().requires_grad_(True)
        x_lam = torch.clamp(x_clean + lam * delta, 0.0, 255.0)
        aug = rrb_forward(x_lam.unsqueeze(0), gt_boxes, cfg)
        feats_adv = model.backbone(surrogate.normalize(aug))
        loss = osfd_loss(feats_cln, feats_adv, cfg.k)
        (grad_m,) = torch.autograd.grad(loss, delta)
        grads.append(grad_m.detach())

    g_local = grads[-1]  # lambda_M = 1 -> x + delta_t, the ordinary instantaneous gradient
    g_path = torch.stack(grads, dim=0).mean(dim=0)

    cos = float(
        torch.nn.functional.cosine_similarity(g_path.flatten().unsqueeze(0), g_local.flatten().unsqueeze(0)).item()
    )
    sign_agree = float((torch.sign(g_path) == torch.sign(g_local)).float().mean().item())
    return g_local, g_path, {"cos_path_local": cos, "sign_agree_ratio": sign_agree}


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

    from transfer_attack.attack import AttackConfig
    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_manifest, load_coco
    from transfer_attack.eval_metrics import compute_asr_for_image, predict_canvas
    from transfer_attack.io_utils import get_logger, load_noise, save_noise
    from transfer_attack.models import build_model_handle, get_spec

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

    cfg = AttackConfig(attack_type="osfd", k=3.0, use_rrb=True, alpha=args.alpha, canvas=args.canvas)

    probe_noise_dirs = {
        "local": PROJECT_DIR / "results" / "noise" / args.manifest.stem / "n6b0_probe_local",
        "path": PROJECT_DIR / "results" / "noise" / args.manifest.stem / "n6b0_probe_path",
    }
    for d in probe_noise_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: surrogate only in memory -- compute path/local gradients,
    # save the resulting one-step probe deltas to disk, then free the model
    # before loading targets (same memory-safety pattern as n6a_gcr_pilot.py).
    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=device, coco=coco)

    used_image_ids = []
    per_image_diag = []
    for image_id in image_ids:
        if len(used_image_ids) >= args.n_images:
            break
        noise_path = args.noise_dir / f"{image_id}.pt"
        if not noise_path.exists():
            continue
        gt_entries = gt_index[image_id]
        if not gt_entries:
            continue

        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, cfg.canvas)
        gt_boxes, _ = gt_to_canvas(gt_entries, scale)
        x_clean = canvas_img.to(device)
        gt_boxes = gt_boxes.to(device)
        delta_t = load_noise(noise_path, device=device)

        g_local, g_path, diag = compute_path_gradients(surrogate, x_clean, delta_t, gt_boxes, cfg, args.num_lambda, device)

        with torch.no_grad():
            x_t = torch.clamp(x_clean + delta_t, 0.0, 255.0)
            x_local = torch.clamp(x_t + cfg.alpha * torch.sign(g_local), 0.0, 255.0)
            x_path = torch.clamp(x_t + cfg.alpha * torch.sign(g_path), 0.0, 255.0)
            delta_local = x_local - x_clean
            delta_path = x_path - x_clean

        save_noise(probe_noise_dirs["local"] / f"{image_id}.pt", delta_local)
        save_noise(probe_noise_dirs["path"] / f"{image_id}.pt", delta_path)

        diag["image_id"] = image_id
        per_image_diag.append(diag)
        used_image_ids.append(image_id)

    avg_cos = sum(d["cos_path_local"] for d in per_image_diag) / len(per_image_diag)
    avg_sign_agree = sum(d["sign_agree_ratio"] for d in per_image_diag) / len(per_image_diag)
    logger.info(
        f"[phase 1] {len(used_image_ids)} images -- avg cos(g_path,g_local)={avg_cos:.6f}, "
        f"avg sign_agree_ratio={avg_sign_agree:.4f}"
    )

    del surrogate
    torch.cuda.empty_cache()

    # ---- Phase 2: one model at a time -- clean / x_t (baseline, = delta_t
    # alone) / x_local / x_path predictions, incremental evasion vs x_t.
    model_names = ["faster_rcnn_r50"] + list(args.targets)
    rows = []
    for model_name in model_names:
        spec = get_spec(model_name)
        handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)

        n_evaded_t = n_evaded_local = n_evaded_path = n_clean_correct = 0
        for image_id in used_image_ids:
            gt_entries = gt_index[image_id]
            canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, cfg.canvas)
            gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
            x_clean = canvas_img.to(device)

            delta_t = load_noise(args.noise_dir / f"{image_id}.pt", device=device)
            delta_local = load_noise(probe_noise_dirs["local"] / f"{image_id}.pt", device=device)
            delta_path = load_noise(probe_noise_dirs["path"] / f"{image_id}.pt", device=device)

            x_t = torch.clamp(x_clean + delta_t, 0.0, 255.0)
            x_local = torch.clamp(x_clean + delta_local, 0.0, 255.0)
            x_path = torch.clamp(x_clean + delta_path, 0.0, 255.0)

            clean_pred = predict_canvas(handle, x_clean, args.canvas, device=device)
            pred_t = predict_canvas(handle, x_t, args.canvas, device=device)
            pred_local = predict_canvas(handle, x_local, args.canvas, device=device)
            pred_path = predict_canvas(handle, x_path, args.canvas, device=device)

            e_t, n_cc = compute_asr_for_image(clean_pred, pred_t, gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr)
            e_local, _ = compute_asr_for_image(clean_pred, pred_local, gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr)
            e_path, _ = compute_asr_for_image(clean_pred, pred_path, gt_boxes, gt_cat_ids, args.iou_thr, args.score_thr)

            n_evaded_t += e_t
            n_evaded_local += e_local
            n_evaded_path += e_path
            n_clean_correct += n_cc

        asr_t = 100.0 * n_evaded_t / n_clean_correct if n_clean_correct else float("nan")
        asr_local = 100.0 * n_evaded_local / n_clean_correct if n_clean_correct else float("nan")
        asr_path = 100.0 * n_evaded_path / n_clean_correct if n_clean_correct else float("nan")
        row = {
            "model_name": model_name,
            "group": spec.group or "",
            "n_clean_correct": n_clean_correct,
            "ASR_t": round(asr_t, 2),
            "ASR_local": round(asr_local, 2),
            "ASR_path": round(asr_path, 2),
            "incr_local": round(asr_local - asr_t, 2),
            "incr_path": round(asr_path - asr_t, 2),
            "path_minus_local": round(asr_path - asr_local, 2),
        }
        rows.append(row)
        logger.info(
            f"[{model_name:20s}] ASR_t={asr_t:.1f} ASR_local={asr_local:.1f} (incr={row['incr_local']:+.1f}) "
            f"ASR_path={asr_path:.1f} (incr={row['incr_path']:+.1f})  path-local={row['path_minus_local']:+.1f}"
        )

        del handle
        torch.cuda.empty_cache()

    import csv

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote -> {args.out_csv}")
    logger.info(
        f"=== N6-B0 summary: avg cos(g_path,g_local)={avg_cos:.6f} (1.0 = path-averaging is a no-op) ==="
    )


if __name__ == "__main__":
    main()
