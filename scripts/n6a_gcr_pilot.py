#!/usr/bin/env python
"""Phase N6-A v0 pilot: pixel-wise gradient-conflict-resolution ("gradient
surgery") across K=3 RRB views, vs plain K=3 averaging (RCG-AVG) and a K=1
reference. osfd_loss, epsilon, alpha, steps, momentum are all UNCHANGED --
the only thing that differs between rrb_avg_k3 and rrb_cr_k3 is how the K=3
per-step gradients are combined into one update direction.

Mechanism (per pixel p, gradient treated as a 3-vector over RGB channels):
    m(p)     = mean_i g_i(p)                              (local consensus)
    g_i'(p)  = g_i(p) - [g_i(p)^T m(p) / (|m(p)|^2+eta)] m(p)   if g_i(p)^T m(p) < 0
             = g_i(p)                                            otherwise
    g_CR(p)  = mean_i g_i'(p)
Only the component of a view's gradient that is DESTRUCTIVE to the local
consensus gets projected out -- unlike RCG's gate (which shrinks the WHOLE
per-pixel step by an agreement score), this keeps whatever part of a
conflicting view's gradient is orthogonal/non-conflicting. No lambda: eta is
pure numerical safety (division guard), not a tunable strength knob.

Pairing requirement (explicit, not left to incidental RNG-sequence luck):
rrb_avg_k3 and rrb_cr_k3 are crafted in LOCKSTEP per image -- two parallel
noise trajectories, sharing an RNG snapshot/restore around each step's K=3
rrb_forward draws so both trajectories see the SAME random rotation/resize/
blur PARAMETERS at that step (applied to each trajectory's own current
x_adv, which naturally diverges in content after step 1 -- that divergence
is real and expected, only the random augmentation *parameters* are paired,
removing "AVG got luckier draws than CR" as a confound).

Diagnostics logged per image (see module-level GCR_DIAGNOSTIC_KEYS):
  conflict_pixel_ratio -- P[exists i: g_i(p)^T m(p) < 0], mean over steps.
  correction_ratio      -- ||g_CR - g_AVG||_2 / ||g_AVG||_2 (on the CR
                            trajectory's own K views), mean over steps.
  cos_cr_avg             -- cosine(g_CR, g_AVG_of_same_views), mean over steps.
Reading key: conflict_pixel_ratio ~ 0 -> projection barely activates
(implementation issue, not a real test of the hypothesis). High conflict +
material correction_ratio but no ASR gain -> the "destructive RRB conflict
limits transfer" hypothesis is falsified with real confidence, not just
"unclear".

GO criterion (pre-registered, decided before running): CR-AVG ASR delta >=
+3 on >= 2/3 of {yolox_l, mask_rcnn_swin_t, dino_swin_l}, AND no target drops
by more than 3 points. No DINO-specific +5 requirement -- the hypothesis is
about RRB transfer broadly, not DINO alone.

Example:
    python scripts/n6a_gcr_pilot.py --manifest data/manifests/dev_50.json --n-images 20
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

K_DRAWS = 3
ETA = 1e-8


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
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n6a_gcr_pilot_summary.csv")
    p.add_argument(
        "--models", nargs="+", default=["faster_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"],
    )
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


def draw_k_grads(model, handle, x_clean, noise, gt_boxes, feats_cln, cfg, k_draws):
    """Returns list of k_draws gradient tensors (3,H,W), each from an
    independent rrb_forward draw at the CURRENT noise state."""
    import torch

    from transfer_attack.augment import rrb_forward
    from transfer_attack.losses import osfd_loss

    grads = []
    for _ in range(k_draws):
        x_adv_i = torch.clamp(x_clean + noise, 0.0, 255.0)
        aug = rrb_forward(x_adv_i.unsqueeze(0), gt_boxes, cfg)
        feats_adv = model.backbone(handle.normalize(aug))
        loss = osfd_loss(feats_cln, feats_adv, cfg.k)
        (grad_i,) = torch.autograd.grad(loss, noise)
        grads.append(grad_i.detach())
    return grads


def resolve_conflict(grads_stack, eta: float = ETA):
    """grads_stack: (K,3,H,W). Returns (g_CR, g_AVG, diagnostics dict) where
    diagnostics are scalars for THIS step.

    Uses LEAVE-ONE-OUT consensus m_{-i} = (sum_{j!=i} g_j) / (K-1), NOT the
    self-inclusive global mean. Self-inclusive consensus is biased toward
    agreeing with g_i itself (g_i is one of the K terms averaged into m),
    which can hide a real pairwise conflict: e.g. g_1=(1,1), g_2=(-1,1) ->
    self-inclusive m=(0,1) gives g_1^T m = g_2^T m = 1 > 0 for BOTH, so
    neither is flagged despite being fully opposed on the x-axis. Leave-one-
    out fixes this by testing each view against consensus of the OTHERS only
    (PCGrad-style projection against other views, not against a
    self-contaminated blend)."""
    import torch

    K = grads_stack.shape[0]
    m = grads_stack.mean(dim=0)  # (3,H,W) -- self-inclusive mean, kept only as g_AVG baseline
    m_loo = (K * m.unsqueeze(0) - grads_stack) / (K - 1)  # (K,3,H,W) -- leave-one-out consensus per view

    norm_m_loo2 = (m_loo**2).sum(dim=1, keepdim=True)  # (K,1,H,W)
    dot = (grads_stack * m_loo).sum(dim=1, keepdim=True)  # (K,1,H,W)
    conflict_mask = (dot < 0).float()  # (K,1,H,W)
    proj_coeff = dot / (norm_m_loo2 + eta)  # (K,1,H,W)
    correction = conflict_mask * proj_coeff * m_loo  # (K,3,H,W)
    g_prime = grads_stack - correction
    g_cr = g_prime.mean(dim=0)
    g_avg = m  # plain arithmetic mean, the RRB-AVG baseline for comparison

    conflict_pixel_ratio = float(conflict_mask.squeeze(1).any(dim=0).float().mean().item())
    conflict_view_ratio = float(conflict_mask.mean().item())  # fraction of (i,p) pairs, not just pixels
    correction_ratio = float((g_cr - g_avg).norm().item() / (g_avg.norm().item() + 1e-8))
    cos_cr_avg = float(
        torch.nn.functional.cosine_similarity(g_cr.flatten().unsqueeze(0), g_avg.flatten().unsqueeze(0)).item()
    )
    return g_cr, g_avg, {
        "conflict_pixel_ratio": conflict_pixel_ratio,
        "conflict_view_ratio": conflict_view_ratio,
        "correction_ratio": correction_ratio,
        "cos_cr_avg": cos_cr_avg,
    }


def craft_paired_avg_cr(handle, x_clean, gt_boxes, cfg, device):
    """Crafts rrb_avg_k3 and rrb_cr_k3 in lockstep for one image, sharing RNG
    state around each step's K=3 draws (see module docstring). Returns
    (noise_avg, noise_cr, diagnostics averaged over steps)."""
    import torch

    model = handle.model
    x_clean = x_clean.to(device)
    gt_boxes = gt_boxes.to(device)

    with torch.no_grad():
        feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))

    noise_avg = torch.randint_like(x_clean, low=-2, high=3).float()
    noise_cr = noise_avg.clone()
    g_mom_avg = torch.zeros_like(x_clean)
    g_mom_cr = torch.zeros_like(x_clean)

    diag_sums = {"conflict_pixel_ratio": 0.0, "conflict_view_ratio": 0.0, "correction_ratio": 0.0, "cos_cr_avg": 0.0}

    for _ in range(cfg.steps):
        noise_avg = noise_avg.detach().requires_grad_(True)
        noise_cr = noise_cr.detach().requires_grad_(True)

        rng_snapshot = capture_rng(device)
        grads_avg_list = draw_k_grads(model, handle, x_clean, noise_avg, gt_boxes, feats_cln, cfg, K_DRAWS)
        restore_rng(rng_snapshot, device)
        grads_cr_list = draw_k_grads(model, handle, x_clean, noise_cr, gt_boxes, feats_cln, cfg, K_DRAWS)

        g_avg_traj = torch.stack(grads_avg_list, dim=0).mean(dim=0)  # plain average, drives the AVG trajectory
        grads_cr_stack = torch.stack(grads_cr_list, dim=0)
        g_cr, g_avg_of_cr_views, step_diag = resolve_conflict(grads_cr_stack)
        for k, v in step_diag.items():
            diag_sums[k] += v

        with torch.no_grad():
            g_mom_avg = cfg.mu * g_mom_avg + g_avg_traj / g_avg_traj.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise_avg = torch.clamp(noise_avg + cfg.alpha * torch.sign(g_mom_avg), -cfg.epsilon, cfg.epsilon)

            g_mom_cr = cfg.mu * g_mom_cr + g_cr / g_cr.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise_cr = torch.clamp(noise_cr + cfg.alpha * torch.sign(g_mom_cr), -cfg.epsilon, cfg.epsilon)

    diag_means = {k: v / cfg.steps for k, v in diag_sums.items()}
    return noise_avg.detach(), noise_cr.detach(), diag_means


def craft_one_image_k1(handle, x_clean, gt_boxes, cfg, device):
    """Plain OSFD, K=1 draw/step -- reference only, not paired with anything."""
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
    for _ in range(cfg.steps):
        noise = noise.detach().requires_grad_(True)
        x_adv = torch.clamp(x_clean + noise, 0.0, 255.0)
        aug = rrb_forward(x_adv.unsqueeze(0), gt_boxes, cfg)
        feats_adv = model.backbone(handle.normalize(aug))
        loss = osfd_loss(feats_cln, feats_adv, cfg.k)
        loss.backward()
        with torch.no_grad():
            g = noise.grad
            g_mom = cfg.mu * g_mom + g / g.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise = torch.clamp(noise + cfg.alpha * torch.sign(g_mom), -cfg.epsilon, cfg.epsilon)
    return noise.detach()


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

    noise_dirs = {tag: PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"n6a_{tag}" for tag in ("osfd_k1", "rrb_avg_k3", "rrb_cr_k3")}
    for d in noise_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=device, coco=coco)

    used_image_ids = []
    diag_all = {"conflict_pixel_ratio": [], "conflict_view_ratio": [], "correction_ratio": [], "cos_cr_avg": []}
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

        random.seed(args.seed + image_id)
        torch.manual_seed(args.seed + image_id)
        noise_k1 = craft_one_image_k1(surrogate, canvas_img, gt_boxes, cfg, device)

        random.seed(args.seed + image_id)
        torch.manual_seed(args.seed + image_id)
        noise_avg, noise_cr, diag = craft_paired_avg_cr(surrogate, canvas_img, gt_boxes, cfg, device)

        save_noise(noise_dirs["osfd_k1"] / f"{image_id}.pt", noise_k1)
        save_noise(noise_dirs["rrb_avg_k3"] / f"{image_id}.pt", noise_avg)
        save_noise(noise_dirs["rrb_cr_k3"] / f"{image_id}.pt", noise_cr)
        used_image_ids.append(image_id)
        for k in diag_all:
            diag_all[k].append(diag[k])
        n_crafted += 1
        if n_crafted % args.log_every == 0:
            logger.info(
                f"[{n_crafted}/{args.n_images}] elapsed={time.time() - t0:.1f}s "
                f"pixel_ratio={diag['conflict_pixel_ratio']:.3f} view_ratio={diag['conflict_view_ratio']:.3f} "
                f"correction_ratio={diag['correction_ratio']:.3f} "
                f"cos={diag['cos_cr_avg']:.3f}"
            )
    craft_elapsed = time.time() - t0
    avg_diag = {k: sum(v) / len(v) for k, v in diag_all.items()}
    logger.info(f"craft finished: {n_crafted} crafted, {n_skipped} skipped. avg diagnostics: {avg_diag}")
    del surrogate
    torch.cuda.empty_cache()

    from types import SimpleNamespace

    all_rows = {}
    for tag in ("osfd_k1", "rrb_avg_k3", "rrb_cr_k3"):
        eval_args = SimpleNamespace(
            checkpoints_dir=args.checkpoints_dir,
            canvas=args.canvas,
            score_thr=args.score_thr,
            iou_thr=args.iou_thr,
            noise_dir=PROJECT_DIR / "results" / "noise" / args.manifest.stem,
            predictions_dir=PROJECT_DIR / "results" / "_n6a_predictions",
            force_clean=False,
            device=args.device,
            attacks=[f"n6a_{tag}"],
        )
        specs = MODEL_REGISTRY
        if args.models:
            by_name = {s.name: s for s in MODEL_REGISTRY}
            specs = [by_name[m] for m in args.models]
        gt_cache = evaluate_mod.build_gt_cache(coco, used_image_ids, img_dir, args.canvas)
        rows = []
        for spec in specs:
            rows.extend(evaluate_mod.evaluate_one_model(spec, eval_args, coco, used_image_ids, img_dir, gt_cache, logger))
        all_rows[tag] = [r for r in rows if r["attack"] == f"n6a_{tag}"]

        run_log_path = save_run_log(
            args.runs_dir,
            "run_attack",
            f"osfd_n6a_{tag}_{args.manifest.stem}_n{args.n_images}",
            {
                "attack": f"n6a_{tag}",
                "manifest": str(args.manifest),
                "seed": args.seed,
                "n_images": args.n_images,
                "config": vars(cfg),
                "diagnostics": avg_diag if tag == "rrb_cr_k3" else None,
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

    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    import csv

    fieldnames = ["model_name", "group", "ASR_osfd_k1", "ASR_rrb_avg_k3", "ASR_rrb_cr_k3",
                  "mAP_drop_osfd_k1", "mAP_drop_rrb_avg_k3", "mAP_drop_rrb_cr_k3", "delta_cr_vs_avg"]
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for tag in ("osfd_k1", "rrb_avg_k3", "rrb_cr_k3"):
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
            row[f"mAP_drop_{tag}"] = r.get("mAP_drop_pct")
        if row.get("ASR_rrb_avg_k3") is not None and row.get("ASR_rrb_cr_k3") is not None:
            row["delta_cr_vs_avg"] = row["ASR_rrb_cr_k3"] - row["ASR_rrb_avg_k3"]
        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info(f"=== N6-A pilot diagnostics (mean over steps/images): {avg_diag} ===")
    logger.info("=== N6-A pilot: ASR (%) osfd_k1 -> rrb_avg_k3 -> rrb_cr_k3 ===")
    for r in comparison:
        logger.info(
            f"{r['model_name']:20s} {r['group'] or '-':4s} "
            f"k1={r.get('ASR_osfd_k1', float('nan')):.1f} avg={r.get('ASR_rrb_avg_k3', float('nan')):.1f} "
            f"cr={r.get('ASR_rrb_cr_k3', float('nan')):.1f}  (cr-avg={r.get('delta_cr_vs_avg', float('nan')):+.1f})"
        )


if __name__ == "__main__":
    main()
