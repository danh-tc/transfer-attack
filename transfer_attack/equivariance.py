"""E11 -- Adversarial Equivariance Gap diagnostic support (RESEARCH.md Sec 32).

Tests a candidate attack principle (tentatively named CEFA -- Cross-view
Equivariance Failure Attack) BEFORE committing to building it: does OSFD's
known transfer driver (RRB) and N6-B's known transfer driver (path-averaged
gradient) coincide with an INCREASE in how badly a backbone's own feature
field violates a simple transformation-equivariance law, on exactly the hard
targets where those two interventions are known to help ASR? This module
provides only the shared geometric primitive; scripts/e11_equivariance_gap.py
does the craft/eval/compute/verdict orchestration.

Definition (see RESEARCH.md Sec 32 for the full derivation): for a mild
global transform tau (rotate a few degrees, or scale in/out, about the
image/feature-map's own center) and backbone feature F, the equivariance
residual at one (image, transform, stage) is

    E_F(x, tau) = F(tau(x)) - W_tau F(x)

where W_tau is the SAME geometric transform applied directly to the feature
tensor instead of the pixel tensor. A clean/well-behaved backbone keeps
||E_F(x, tau)|| small; the diagnostic asks whether an adversarial delta
grows this residual, and whether that growth tracks known ASR-increasing
interventions (RRB on/off; path-averaged vs instantaneous gradient).

Why ONE shared function suffices for both tau(x) and W_tau F(x) (unlike
transfer_attack/dbta.py's rotate_points, which had to independently
re-derive and empirically verify RRB's pixel-space torchvision.rotate
convention against synthetic images): here WE define tau ourselves, via
torch.nn.functional.affine_grid + grid_sample in NORMALIZED [-1,1]
coordinates. affine_grid's sampling grid is resolution-agnostic by
construction -- the identical affine matrix applied to a (1,3,800,800) pixel
tensor and a (1,C,25,25) backbone feature tensor produces geometrically the
SAME rotation/scale about each tensor's own center, regardless of its H,W.
So `warp(img, t)` (defining tau(x)) and `warp(feat, t)` (defining
W_tau F(x)) are, by construction, one shared geometric operator evaluated at
two different resolutions -- no separate point-tracking or empirical
sign-convention check needed. (This also means E11's tau is NOT required to
match RRB's own rotation convention/sign -- E11 defines its own transform
from scratch, purely for this diagnostic; only self-consistency between the
pixel-space and feature-space calls matters, not agreement with augment.py.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class Transform:
    name: str
    angle_deg: float = 0.0
    scale: float = 1.0


# Pre-registered transform set (RESEARCH.md Sec 32 Sec.9) -- mild global
# rotation/scale about the tensor's own center. Not swept/tuned.
TRANSFORM_SET: tuple[Transform, ...] = (
    Transform("rotate+5", angle_deg=5.0),
    Transform("rotate-5", angle_deg=-5.0),
    Transform("scale0.9", scale=0.9),
    Transform("scale1.1", scale=1.1),
)


def warp(t: Tensor, transform: Transform) -> Tensor:
    """t: (N,C,H,W), any resolution. Applies `transform` (rotation about
    center, then isotropic scale about center) via a single
    normalized-coordinate affine_grid + grid_sample call. Zero-padding at
    the border (padding_mode="zeros") is an accepted, symmetric approximation
    for this diagnostic -- it affects tau(x) and W_tau F(x) identically in
    spirit (both are edge artifacts of the same warp), and clean vs
    adversarial comparisons subtract most of that shared bias out."""
    n = t.shape[0]
    rad = math.radians(transform.angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    inv_scale = 1.0 / transform.scale
    theta = torch.tensor(
        [[cos_a * inv_scale, sin_a * inv_scale, 0.0],
         [-sin_a * inv_scale, cos_a * inv_scale, 0.0]],
        dtype=t.dtype, device=t.device,
    ).unsqueeze(0).expand(n, -1, -1)
    grid = F.affine_grid(theta, size=list(t.shape), align_corners=False)
    return F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def equivariance_relative_residual(feat_tau: Tensor, feat_orig_warped: Tensor, feat_orig: Tensor) -> float:
    """feat_tau = F(tau(x)); feat_orig_warped = W_tau F(x); feat_orig = F(x)
    -- all (1,C,H,W) at the same backbone stage. Returns
    ||feat_tau - feat_orig_warped||_2 / ||feat_orig||_2 (Frobenius norm over
    the whole tensor) -- the per-(image,stage,transform) relative
    equivariance residual from RESEARCH.md Sec 32's Q_m definition."""
    num = torch.linalg.vector_norm(feat_tau - feat_orig_warped).item()
    den = torch.linalg.vector_norm(feat_orig).item()
    return num / den if den > 0 else float("nan")


def compute_Q_for_image(
    model, normalize, x: Tensor, transforms: tuple[Transform, ...] = TRANSFORM_SET,
) -> dict:
    """x: (3,H,W) canvas-space pixel image (clean OR clean+noise, already
    clamped to [0,255]). Returns {"per_stage": [Q_stage_0, ..., Q_stage_L-1]
    (mean over `transforms`), "per_transform": {transform.name:
    [residual_stage_0, ...]}}. Runs entirely under no_grad (diagnostic only,
    not crafting -- no attack objective is defined here)."""
    x = x.unsqueeze(0)  # (1,3,H,W)
    with torch.no_grad():
        feats_orig = model.backbone(normalize(x))
        n_stages = len(feats_orig)
        per_transform: dict[str, list[float]] = {}
        for tr in transforms:
            x_tau = warp(x, tr)
            feats_tau = model.backbone(normalize(x_tau))
            residuals = []
            for stage_orig, stage_tau in zip(feats_orig, feats_tau):
                stage_orig_warped = warp(stage_orig, tr)
                residuals.append(equivariance_relative_residual(stage_tau, stage_orig_warped, stage_orig))
            per_transform[tr.name] = residuals
        per_stage = [
            sum(per_transform[tr.name][s] for tr in transforms) / len(transforms)
            for s in range(n_stages)
        ]
    return {"per_stage": per_stage, "per_transform": per_transform}
