#!/usr/bin/env python
"""E3b: does RRB change the DIRECTION of the k=3 vs k=1 gradient (not just its
magnitude)? E3's factorial found k=3 has ~no standalone benefit without RRB
but a strong positive interaction with it (esp. mask_rcnn_swin_t +17.8,
yolox_l +18.3 ASR points beyond the additive prediction). This probes the
mechanism: at matched noise states along the REAL OSFD (k=3, RRB=on)
trajectory -- early/mid/late iteration -- freeze the noise and compute FOUR
gradients w.r.t. it (k in {1,3} x RRB in {off,on}) without touching the main
trajectory, then compare:

  cos(g_k1, g_k3)      -- direction agreement, computed separately under
                           no-RRB and under RRB (using the SAME augmented
                           view for both k's RRB-on probe, via an RNG
                           save/restore around the two calls, so the only
                           thing that differs between the pair is k)
  |g_k3| / |g_k1|       -- magnitude ratio, same-RRB-condition pairing

Reading key (as specified going in):
  - no-RRB: cos ~ 1, RRB: cos drops clearly       -> RRB redirects k=3's
                                                       gradient direction
  - cos similar in both conditions, ratio jumps    -> k=3 mostly amplifies
                                                       magnitude, not direction
  - neither cos nor ratio move much                -> the E3 synergy is an
                                                       accumulated-trajectory
                                                       effect, not visible in
                                                       any single instantaneous
                                                       gradient

This is on the SURROGATE (faster_rcnn_r50) only -- gradients are computed
during crafting, never on targets. No dependency on E1/E2/E3's saved noise
(needs per-step states, which weren't saved) -- runs its own short crafting
trajectories, but only for a small image subset (fast).

Example:
    python scripts/e3b_gradient_synergy.py --manifest data/manifests/dev_50.json --n-images 20
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--n-images", type=int, default=20, help="reference-trajectory images to probe (subset of manifest)")
    p.add_argument("--steps", type=int, default=100, help="reference trajectory length (k=3, RRB=on, i.e. baseline OSFD)")
    p.add_argument(
        "--checkpoint-steps", type=int, nargs="+", default=[1, 50, 100],
        help="1-indexed iteration numbers to probe gradients at",
    )
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "e3b_gradient_synergy.csv")
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


def compute_grad(x_clean, noise_snapshot, gt_boxes, feats_cln, model, normalize_fn, k: float, use_rrb: bool, cfg):
    from transfer_attack.augment import rrb_forward
    from transfer_attack.losses import osfd_loss

    noise_leaf = noise_snapshot.clone().requires_grad_(True)
    x_adv = (x_clean + noise_leaf).clamp(0.0, 255.0)
    aug = rrb_forward(x_adv.unsqueeze(0), gt_boxes, cfg) if use_rrb else x_adv.unsqueeze(0)
    feats_adv = model.backbone(normalize_fn(aug))
    loss = osfd_loss(feats_cln, feats_adv, k)
    loss.backward()
    return noise_leaf.grad.detach().clone()


def cos_and_ratio(g1, g3) -> tuple[float, float]:
    import torch.nn.functional as F

    f1, f3 = g1.flatten().unsqueeze(0), g3.flatten().unsqueeze(0)
    cos = F.cosine_similarity(f1, f3).item()
    ratio = f3.norm().item() / (f1.norm().item() + 1e-8)
    return cos, ratio


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

    from transfer_attack.attack import AttackConfig
    from transfer_attack.augment import rrb_forward
    from transfer_attack.constants import ALPHA, COCO_ANN_FILE, EPSILON, K, MU
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger
    from transfer_attack.losses import osfd_loss
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

    spec = get_spec("faster_rcnn_r50")
    handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)
    model = handle.model

    # Reference trajectory config = baseline OSFD (k=3, RRB on). Probes never
    # touch this trajectory's own noise/grad state -- they operate on cloned
    # snapshots.
    ref_cfg = AttackConfig(attack_type="osfd", k=K, use_rrb=True, steps=args.steps, canvas=args.canvas)
    checkpoint_set = set(args.checkpoint_steps)

    # checkpoint_step -> list of (cos_norrb, cos_rrb, ratio_norrb, ratio_rrb)
    per_checkpoint: dict[int, list[tuple[float, float, float, float]]] = {s: [] for s in args.checkpoint_steps}

    n_used = 0
    for image_id in image_ids:
        if n_used >= args.n_images:
            break
        gt_entries = gt_index[image_id]
        if not gt_entries:
            continue

        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
        gt_boxes, _ = gt_to_canvas(gt_entries, scale)
        x_clean = canvas_img.to(device)
        gt_boxes = gt_boxes.to(device)

        with torch.no_grad():
            feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))

        noise = torch.randint_like(x_clean, low=-2, high=3).float()
        g_mom = torch.zeros_like(x_clean)

        for step_idx in range(ref_cfg.steps):
            noise = noise.detach().requires_grad_(True)
            step_num = step_idx + 1
            if step_num in checkpoint_set:
                snapshot = noise.detach().clone()

                g_k1_norrb = compute_grad(x_clean, snapshot, gt_boxes, feats_cln, model, handle.normalize, 1.0, False, ref_cfg)
                g_k3_norrb = compute_grad(x_clean, snapshot, gt_boxes, feats_cln, model, handle.normalize, 3.0, False, ref_cfg)
                cos_norrb, ratio_norrb = cos_and_ratio(g_k1_norrb, g_k3_norrb)

                rng_state = capture_rng(device)
                g_k1_rrb = compute_grad(x_clean, snapshot, gt_boxes, feats_cln, model, handle.normalize, 1.0, True, ref_cfg)
                restore_rng(rng_state, device)
                g_k3_rrb = compute_grad(x_clean, snapshot, gt_boxes, feats_cln, model, handle.normalize, 3.0, True, ref_cfg)
                cos_rrb, ratio_rrb = cos_and_ratio(g_k1_rrb, g_k3_rrb)

                per_checkpoint[step_num].append((cos_norrb, cos_rrb, ratio_norrb, ratio_rrb))

            # ---- advance the REAL reference trajectory (k=3, RRB=on) ----
            x_adv = torch.clamp(x_clean + noise, 0.0, 255.0)
            aug = rrb_forward(x_adv.unsqueeze(0), gt_boxes, ref_cfg)
            feats_adv = model.backbone(handle.normalize(aug))
            loss = osfd_loss(feats_cln, feats_adv, ref_cfg.k)
            loss.backward()

            with torch.no_grad():
                g = noise.grad
                g_mom = ref_cfg.mu * g_mom + g / g.abs().mean(dim=[0, 1, 2], keepdim=True)
                noise = torch.clamp(noise + ref_cfg.alpha * torch.sign(g_mom), -ref_cfg.epsilon, ref_cfg.epsilon)

        n_used += 1
        logger.info(f"[{n_used}/{args.n_images}] image_id={image_id} done")

    logger.info("=== E3b: cos(g_k1, g_k3) and |g_k3|/|g_k1|, no-RRB vs RRB, by iteration ===")
    summary_rows = []
    for step_num in args.checkpoint_steps:
        vals = per_checkpoint[step_num]
        if not vals:
            continue
        mean_cos_norrb = sum(v[0] for v in vals) / len(vals)
        mean_cos_rrb = sum(v[1] for v in vals) / len(vals)
        mean_ratio_norrb = sum(v[2] for v in vals) / len(vals)
        mean_ratio_rrb = sum(v[3] for v in vals) / len(vals)
        summary_rows.append(
            {
                "iteration": step_num,
                "n_images": len(vals),
                "mean_cos_k1_k3_norrb": mean_cos_norrb,
                "mean_cos_k1_k3_rrb": mean_cos_rrb,
                "mean_norm_ratio_k3_over_k1_norrb": mean_ratio_norrb,
                "mean_norm_ratio_k3_over_k1_rrb": mean_ratio_rrb,
            }
        )
        logger.info(
            f"  iter={step_num:4d} n={len(vals):3d}  "
            f"cos(noRRB)={mean_cos_norrb:.4f}  cos(RRB)={mean_cos_rrb:.4f}  "
            f"ratio(noRRB)={mean_ratio_norrb:.4f}  ratio(RRB)={mean_ratio_rrb:.4f}"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
