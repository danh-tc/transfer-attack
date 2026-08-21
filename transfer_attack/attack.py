"""Shared I-FGSM + MI-momentum crafting loop, with two swappable loss
strategies: OSFD (backbone-feature MSE + RRB augmentation) and MI-FGSM
baseline (output/task loss, no augmentation).

Both attacks ASCEND their respective loss (matches Eq. 1 of the paper,
`argmax_x~ L`, and the reference code's `IFGSM.update_noise`:
`noise + alpha * sign(grad)`). This is not a bug -- do not "fix" the sign.
For OSFD this means ascending MSE(k*F(x_clean), F(T(x_adv))), which -- per the
paper's own derivation (limited equivariance + spatial consistency of
backbone features) -- ends up suppressing significant features at the object
and bleeding vicinal features into its neighborhood, rather than literally
pushing features to look like k*F(x_clean).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from transfer_attack.augment import rrb_forward
from transfer_attack.constants import ALPHA, CANVAS, EPSILON, K, L_S, MU, RHO, S_MAX, SIGMA, STEPS, THETA
from transfer_attack.losses import build_gt_data_sample, detector_task_loss, osfd_loss
from transfer_attack.models import ModelHandle


@dataclass
class AttackConfig:
    attack_type: Literal["osfd", "mi_fgsm"]
    epsilon: float = EPSILON
    alpha: float = ALPHA
    steps: int = STEPS
    mu: float = MU              # MI momentum decay, used by both attacks
    k: float = K                 # OSFD only
    theta: float = THETA          # RRB, OSFD only
    l_s: int = L_S
    rho: float = RHO
    s_max: float = S_MAX
    sigma: float = SIGMA
    canvas: int = CANVAS


def craft_one_image(
    handle: ModelHandle,
    x_clean: Tensor,
    gt_boxes: Tensor,
    gt_cat_ids: Tensor,
    cfg: AttackConfig,
    device: str = "cuda:0",
) -> tuple[Tensor, list[float]]:
    """x_clean: (3,H,W) canonical RGB [0,255] pixel-space image (already on the
    fixed canvas). gt_boxes: (M,4) xyxy canvas-space. gt_cat_ids: (M,) raw COCO
    category ids (mapped to this model's contiguous labels internally, only
    used by the mi_fgsm branch).

    Returns (noise[3,H,W] detached, per-step loss values).
    """
    model = handle.model
    x_clean = x_clean.to(device)
    gt_boxes = gt_boxes.to(device)

    noise = torch.randint_like(x_clean, low=-2, high=3).float()
    g_mom = torch.zeros_like(x_clean)

    feats_cln = None
    data_sample = None
    if cfg.attack_type == "osfd":
        with torch.no_grad():
            feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
    elif cfg.attack_type == "mi_fgsm":
        if gt_boxes.shape[0] == 0:
            raise ValueError("mi_fgsm crafting requires at least one GT box for this image")
        gt_labels = torch.tensor(
            [handle.cat_id_to_label[int(c)] for c in gt_cat_ids], dtype=torch.long, device=device
        )
        data_sample = build_gt_data_sample(gt_boxes, gt_labels, cfg.canvas)
    else:
        raise ValueError(f"Unknown attack_type {cfg.attack_type!r}")

    losses: list[float] = []
    for _ in range(cfg.steps):
        noise = noise.detach().requires_grad_(True)  # fresh leaf every step, keeps the
        # autograd graph from growing across all `steps` iterations.
        x_adv = torch.clamp(x_clean + noise, 0.0, 255.0)

        if cfg.attack_type == "osfd":
            aug = rrb_forward(x_adv.unsqueeze(0), gt_boxes, cfg)  # (2,3,H,W)
            feats_adv = model.backbone(handle.normalize(aug))
            loss = osfd_loss(feats_cln, feats_adv, cfg.k)
        else:
            x_norm = handle.normalize(x_adv.unsqueeze(0))
            loss = detector_task_loss(model, x_norm, data_sample)

        loss.backward()
        losses.append(float(loss.item()))

        with torch.no_grad():
            g = noise.grad
            g_mom = cfg.mu * g_mom + g / g.abs().mean(dim=[0, 1, 2], keepdim=True)
            noise = torch.clamp(noise + cfg.alpha * torch.sign(g_mom), -cfg.epsilon, cfg.epsilon)

    return noise.detach(), losses
