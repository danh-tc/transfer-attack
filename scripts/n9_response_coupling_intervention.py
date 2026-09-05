#!/usr/bin/env python
"""E9 pilot: Response-Coupling Intervention.

After E6 (GO -- C_response predicts transfer under matched-head controls,
RESEARCH.md S23), E7 (NO-GO -- scalar pipeline amplification, S24), and E8
(NO-GO -- raw task-gradient alignment, S25) all failed to explain the
RESIDUAL transfer gap, this moves from explanation-by-correlation to
intervention: if we craft a perturbation specifically biased to increase
C_response, does transfer (ASR/mAP-drop) increase on UNSEEN targets too,
especially the hard cross-backbone ones?

Fundamental constraint (why this can't just optimize C_response directly):
C_response requires a forward pass through the actual TARGET model (S23) --
unavailable during a real black-box crafting loop, which only ever touches
the surrogate faster_rcnn_r50 throughout this project. So the "coupling"
side must be a SURROGATE-ONLY proxy, evaluated for its effect on C_response
and ASR only AFTER crafting (measurement, not the optimization objective).

Proxy chosen (confirmed with user before implementing): low-frequency
spectral bias, reviving S11/E4's explicitly-parked-not-falsified direction
("spectral-*constrained* optimization... OPEN, not tried because it's more
expensive than the raw-noise decomposition E4 did") with NEW motivation from
E6 (shared low-frequency structure between CNN and Transformer backbones is
plausibly what response coupling picks up on; E4 itself already found low
band retains disproportionate ASR share vs its energy share). Concretely:
after every I-FGSM+MI step, project the accumulated delta through a 2D FFT
radial low-pass mask, keeping only the SAME "low" band E4 already defined
(radius_frac <= 1/3 of max radial frequency) -- reusing an existing,
already-used cutoff rather than inventing a new free hyperparameter.

Two trajectories per image, crafted in LOCKSTEP (same RNG snapshot per step,
so both see identical RRB augmentation draws -- same convention as
n6b_path_pilot.py's craft_paired_local_path, isolating "spectral projection"
as the ONLY structural difference):
  - delta_base:     standard OSFD (osfd_local), i.e. this project's usual baseline.
  - delta_coupling: same OSFD gradient ascent, but delta is re-projected onto
                    the low-frequency band after each step's clamp.

Compute-matched by construction (same steps, same gradient computation for
both trajectories; the extra cost is one FFT+IFFT per step for the coupling
trajectory only, negligible next to the backbone forward/backward).

Pre-registered GO/NO-GO (decided before running, per user's message):
  GO:    C_response(coupling) > C_response(base) on held-out/unseen targets
         AND ASR/mAP-drop(coupling) > (base) in the same direction,
         AND the effect is stronger on cross-backbone (Swin) targets than
         same-family (CNN) targets.
  NO-GO: C_response goes up but ASR doesn't (or vice versa), or gains are
         confined to same-family targets only. Per this project's
         discipline: no metric-chasing to rescue a NO-GO.

Example (pilot):
    python scripts/n9_response_coupling_intervention.py --manifest data/manifests/dev_50.json --limit 20
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

LOW_FREQ_RADIUS = 1 / 3  # reuse E4's own "low" band boundary verbatim -- no new free hyperparameter


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--targets", nargs="+", default=["all"])
    p.add_argument("--low-freq-radius", type=float, default=LOW_FREQ_RADIUS)
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-frac", type=float, default=0.8)
    p.add_argument("--pool-size", type=int, default=7, help="passed through to e6's extraction for C_response")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n9_response_coupling_summary.csv")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=10)
    return p


def build_low_pass_mask(h: int, w: int, radius_frac: float, device: str):
    import torch

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = h / 2.0, w / 2.0
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = (cy**2 + cx**2) ** 0.5
    return (r / max_r <= radius_frac).float()


def low_pass_project(delta, mask):
    """delta: (3,H,W). Radially low-pass filter each channel via 2D FFT,
    reusing the exact mask-then-inverse-FFT recipe from
    scripts/e4_spectral_decomposition.py::decompose_bands (verified there to
    reconstruct losslessly up to float precision when bands are recombined;
    here we only keep the low band)."""
    import torch

    out = torch.zeros_like(delta)
    for c in range(delta.shape[0]):
        f = torch.fft.fftshift(torch.fft.fft2(delta[c]))
        out[c] = torch.fft.ifft2(torch.fft.ifftshift(f * mask)).real
    return out


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


def craft_paired_base_coupling(handle, x_clean, gt_boxes, cfg, device, low_pass_mask):
    """Crafts osfd (base) and osfd+spectral-lowpass (coupling) in lockstep for
    one image. Returns (noise_base, noise_coupling), both detached."""
    import torch

    from transfer_attack.augment import rrb_forward
    from transfer_attack.losses import osfd_loss

    model = handle.model
    with torch.no_grad():
        feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))

    noise_base = torch.randint_like(x_clean, low=-2, high=3).float()
    noise_coupling = noise_base.clone()
    g_mom_base = torch.zeros_like(x_clean)
    g_mom_coupling = torch.zeros_like(x_clean)

    for _ in range(cfg.steps):
        noise_base = noise_base.detach().requires_grad_(True)
        noise_coupling = noise_coupling.detach().requires_grad_(True)

        rng_snapshot = capture_rng(device)

        restore_rng(rng_snapshot, device)
        x_adv_base = torch.clamp(x_clean + noise_base, 0.0, 255.0)
        aug_base = rrb_forward(x_adv_base.unsqueeze(0), gt_boxes, cfg) if cfg.use_rrb else x_adv_base.unsqueeze(0)
        feats_adv_base = model.backbone(handle.normalize(aug_base))
        loss_base = osfd_loss(feats_cln, feats_adv_base, cfg.k)
        loss_base.backward()
        g_base = noise_base.grad

        restore_rng(rng_snapshot, device)
        x_adv_coupling = torch.clamp(x_clean + noise_coupling, 0.0, 255.0)
        aug_coupling = rrb_forward(x_adv_coupling.unsqueeze(0), gt_boxes, cfg) if cfg.use_rrb else x_adv_coupling.unsqueeze(0)
        feats_adv_coupling = model.backbone(handle.normalize(aug_coupling))
        loss_coupling = osfd_loss(feats_cln, feats_adv_coupling, cfg.k)
        loss_coupling.backward()
        g_coupling = noise_coupling.grad

        with torch.no_grad():
            g_mom_base = cfg.mu * g_mom_base + g_base / g_base.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise_base = torch.clamp(noise_base + cfg.alpha * torch.sign(g_mom_base), -cfg.epsilon, cfg.epsilon)

            g_mom_coupling = cfg.mu * g_mom_coupling + g_coupling / g_coupling.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise_coupling_raw = torch.clamp(
                noise_coupling + cfg.alpha * torch.sign(g_mom_coupling), -cfg.epsilon, cfg.epsilon
            )
            # The ONLY structural difference from base: project onto the
            # low-frequency band after every step, then reclip (the inverse
            # FFT of a masked spectrum can push a pixel marginally outside
            # +-epsilon due to Gibbs-type ringing at the mask boundary).
            noise_coupling = torch.clamp(low_pass_project(noise_coupling_raw, low_pass_mask), -cfg.epsilon, cfg.epsilon)

    return noise_base.detach(), noise_coupling.detach()


def paired_bootstrap_delta(per_image_base, per_image_coupling, per_image_cc, n_draws, seed, frac=0.8):
    """Subsample images WITHOUT replacement (see e6_response_coupling.py's
    module docstring for why not classic bootstrap for this kind of ratio
    statistic under resampling -- same convention reused here for ASR
    deltas, which are also a ratio of sums, not a plain mean)."""
    rng = random.Random(seed)
    n = len(per_image_base)
    m = max(2, round(frac * n))
    tot_base, tot_coupling, tot_cc = sum(per_image_base), sum(per_image_coupling), sum(per_image_cc)
    if tot_cc == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    asr_base = 100.0 * tot_base / tot_cc
    asr_coupling = 100.0 * tot_coupling / tot_cc
    point = asr_coupling - asr_base
    deltas = []
    idx_all = list(range(n))
    for _ in range(n_draws):
        idx = rng.sample(idx_all, k=m)
        s_base = sum(per_image_base[i] for i in idx)
        s_coupling = sum(per_image_coupling[i] for i in idx)
        s_cc = sum(per_image_cc[i] for i in idx)
        if s_cc == 0:
            continue
        deltas.append(100.0 * s_coupling / s_cc - 100.0 * s_base / s_cc)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]
    same_side = sum(1 for d in deltas if (d >= 0) == (point >= 0)) / len(deltas)
    return asr_base, asr_coupling, point, (lo, hi, same_side)


def main() -> None:
    args = build_arg_parser().parse_args()

    import numpy as np
    import torch

    import e6_response_coupling as e6
    import evaluate as evaluate_mod
    from transfer_attack.attack import AttackConfig
    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.eval_metrics import compute_asr_for_image
    from transfer_attack.io_utils import get_logger, save_noise
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle, get_spec

    logger = get_logger()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    if args.targets == ["all"]:
        target_specs = [s for s in MODEL_REGISTRY if s.role == "target"]
    else:
        target_specs = [get_spec(m) for m in args.targets]

    img_dir = args.data_dir / "val2017"
    device = args.device
    cfg = AttackConfig(attack_type="osfd", steps=args.steps, canvas=args.canvas)
    gt_index = build_gt_index(coco, image_ids)

    noise_root = PROJECT_DIR / "results" / "noise" / args.manifest.stem
    base_dir = noise_root / "n9_base"
    coupling_dir = noise_root / "n9_coupling"
    base_dir.mkdir(parents=True, exist_ok=True)
    coupling_dir.mkdir(parents=True, exist_ok=True)

    low_pass_mask = build_low_pass_mask(args.canvas, args.canvas, args.low_freq_radius, device)

    # ---- Phase 1: paired craft on the surrogate ----
    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=device, coco=coco)
    logger.info(f"[craft] surrogate loaded, low_freq_radius={args.low_freq_radius}")

    used_image_ids = []
    t0 = time.time()
    n_skipped = 0
    for i, image_id in enumerate(image_ids):
        gt_entries = gt_index[image_id]
        if not gt_entries:
            n_skipped += 1
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        x_clean = canvas_img.to(device)
        gt_boxes = gt_boxes.to(device)

        noise_base, noise_coupling = craft_paired_base_coupling(surrogate, x_clean, gt_boxes, cfg, device, low_pass_mask)
        save_noise(base_dir / f"{image_id}.pt", noise_base)
        save_noise(coupling_dir / f"{image_id}.pt", noise_coupling)
        used_image_ids.append(image_id)

        if (i + 1) % args.log_every == 0:
            logger.info(f"[craft] {i + 1}/{len(image_ids)} elapsed={time.time() - t0:.1f}s")

    logger.info(f"[craft] done: {len(used_image_ids)} crafted, {n_skipped} skipped, elapsed={time.time() - t0:.1f}s")
    del surrogate
    torch.cuda.empty_cache()

    # ---- Phase 2: ASR/mAP eval on both noise sets, all targets ----
    gt_cache = evaluate_mod.build_gt_cache(coco, used_image_ids, img_dir, args.canvas)
    eval_rows = {}
    for variant, noise_dir in [("base", base_dir), ("coupling", coupling_dir)]:
        eval_args = SimpleNamespace(
            checkpoints_dir=args.checkpoints_dir,
            canvas=args.canvas,
            score_thr=args.score_thr,
            iou_thr=args.iou_thr,
            noise_dir=noise_root,
            predictions_dir=PROJECT_DIR / "results" / f"_n9_predictions_{variant}",
            force_clean=False,
            device=device,
            attacks=[f"n9_{variant}"],
        )
        rows = []
        for spec in [surrogate_spec] + target_specs:
            rows.extend(
                evaluate_mod.evaluate_one_model(spec, eval_args, coco, used_image_ids, img_dir, gt_cache, logger)
            )
        eval_rows[variant] = {r["model_name"]: r for r in rows if r["attack"] == f"n9_{variant}"}
        logger.info(f"[eval] {variant} done")

    # ---- Phase 3: per-image ASR (evaded/clean-correct) for paired bootstrap ----
    # gt_cat_ids from gt_to_canvas are RAW COCO category ids; predict_canvas's
    # output labels are ALSO remapped back to raw COCO cat ids (see its
    # docstring) specifically so predictions from different models are
    # directly comparable against the same GT space -- no cat_id_to_label
    # mapping needed or wanted here (that would double-remap and break
    # greedy_match). Pattern verified against scripts/e5_success_vs_failure.py.
    from transfer_attack.eval_metrics import predict_canvas
    from transfer_attack.io_utils import load_noise as _load_noise

    per_model_per_image = {}
    for spec in [surrogate_spec] + target_specs:
        handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)
        base_evaded, coupling_evaded, clean_correct = [], [], []
        for image_id in used_image_ids:
            gt_entries = gt_index[image_id]
            canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
            gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)  # CPU tensors, matches predict_canvas's CPU output

            x_clean = canvas_img.to(device)
            clean_pred = predict_canvas(handle, x_clean, args.canvas, device=device)

            noise_b = _load_noise(base_dir / f"{image_id}.pt")
            x_adv_b = torch.clamp(canvas_img.to(device) + noise_b.to(device), 0.0, 255.0)
            adv_pred_b = predict_canvas(handle, x_adv_b, args.canvas, device=device)
            n_b_evaded, n_cc = compute_asr_for_image(
                clean_pred, adv_pred_b, gt_boxes, gt_cat_ids, iou_thr=args.iou_thr, score_thr=args.score_thr,
            )

            noise_c = _load_noise(coupling_dir / f"{image_id}.pt")
            x_adv_c = torch.clamp(canvas_img.to(device) + noise_c.to(device), 0.0, 255.0)
            adv_pred_c = predict_canvas(handle, x_adv_c, args.canvas, device=device)
            n_c_evaded, _ = compute_asr_for_image(
                clean_pred, adv_pred_c, gt_boxes, gt_cat_ids, iou_thr=args.iou_thr, score_thr=args.score_thr,
            )

            base_evaded.append(n_b_evaded)
            coupling_evaded.append(n_c_evaded)
            clean_correct.append(n_cc)
        per_model_per_image[spec.name] = (base_evaded, coupling_evaded, clean_correct)
        del handle
        torch.cuda.empty_cache()

    # ---- Phase 4: C_response measurement (reuse e6_response_coupling.py) ----
    c_response = {}
    for variant, noise_dir in [("base", base_dir), ("coupling", coupling_dir)]:
        sur_handle = build_model_handle(surrogate_spec, args.checkpoints_dir, device=device, coco=coco)
        sur_clean, sur_delta = e6.extract_pooled_response(
            sur_handle, used_image_ids, noise_dir, img_dir, coco, args.canvas, args.pool_size, device, "mean"
        )
        k_sur_delta = e6.gram_from_vectors(sur_delta)
        del sur_handle
        torch.cuda.empty_cache()
        for spec in target_specs:
            handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)
            _, delta_mat = e6.extract_pooled_response(
                handle, used_image_ids, noise_dir, img_dir, coco, args.canvas, args.pool_size, device, "mean"
            )
            k_target_delta = e6.gram_from_vectors(delta_mat)
            c = e6.linear_cka_from_gram(k_sur_delta, k_target_delta)
            c_response.setdefault(spec.name, {})[variant] = c
            del handle
            torch.cuda.empty_cache()

    # ---- Report ----
    detail_rows = []
    logger.info("=== E9 results: base vs coupling ===")
    for spec in target_specs:
        name = spec.name
        b_evaded, c_evaded, cc = per_model_per_image[name]
        asr_base, asr_coupling, delta_point, ci = paired_bootstrap_delta(
            b_evaded, c_evaded, cc, args.n_bootstrap, args.seed, args.bootstrap_frac
        )
        lo, hi, same_side = ci
        c_b = c_response.get(name, {}).get("base")
        c_c = c_response.get(name, {}).get("coupling")
        delta_c = (c_c - c_b) if (c_b is not None and c_c is not None) else None
        logger.info(
            f"  {name} (group={spec.group}): ASR base={asr_base:.1f} coupling={asr_coupling:.1f} "
            f"delta_ASR={delta_point:+.2f} 95% CI=[{lo:+.2f},{hi:+.2f}] same_side={same_side:.3f} | "
            f"C_response base={c_b:.4f} coupling={c_c:.4f} delta_C={delta_c:+.4f}"
        )
        detail_rows.append(
            {
                "target": name,
                "group": spec.group or "",
                "n_images": len(cc),
                "ASR_base": asr_base,
                "ASR_coupling": asr_coupling,
                "delta_ASR": delta_point,
                "delta_ASR_ci_lo": lo,
                "delta_ASR_ci_hi": hi,
                "C_response_base": c_b,
                "C_response_coupling": c_c,
                "delta_C_response": delta_c,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target", "group", "n_images", "ASR_base", "ASR_coupling", "delta_ASR",
                "delta_ASR_ci_lo", "delta_ASR_ci_hi", "C_response_base", "C_response_coupling", "delta_C_response",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)
    logger.info(f"wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
