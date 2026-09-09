"""BTFA -- Boundary Transition Field Attack (RESEARCH.md method-derivation,
proposed after DBTA's Sec 30: DBTA's dense per-POINT boundary sampling
(~24 discrete O/E/Bn triplets/object) is a competitive white-box objective
(unlike TGA's 3-scalar pooling, which collapsed even white-box), but it's
still spatially SPARSE -- only the perimeter contour, not the full boundary
NEIGHBORHOOD. BTFA replaces point sampling entirely with a continuous FIELD:
a signed-distance function around each GT box defines a boundary "band," and
the loss ascends disruption of the feature map's directional derivative
(along the field's normal direction) at every pixel in that band -- the
dense limit of DBTA's finite per-point O/E/Bn differences.

Design (RRB OFF only -- this module does not yet track RRB's rotate/resize
the way transfer_attack/dbta.py does for its point sampling; see "Known
simplifications" below):

  1. Signed-distance field S(x,y) for an axis-aligned box b=(x1,y1,x2,y2)
     (standard rectangle SDF, e.g. Inigo Quilez's formula): negative inside
     the box, zero exactly on its perimeter, positive outside. Evaluated
     directly at each backbone stage's own (H,W) resolution -- S is
     analytic, so no canvas-resolution mask + downsample step is needed
     (unlike transfer_attack/regions.py's boolean masks).
  2. Band mask: keep pixels with -shrink_frac*repr_dim <= S <= expand_frac*
     repr_dim, where repr_dim = sqrt(box_w*box_h) (a single isotropic scale
     per object -- SDF distance isn't axis-specific the way DBTA/E10's
     per-edge margins are, so this collapses the box's two margins into one
     representative dimension). shrink_frac/expand_frac reused verbatim from
     E10's frozen spec, same as TGA/DBTA.
  3. Normal direction n(p) = normalize(grad(S)(p)), grad(S) via a fixed
     Sobel-like finite-difference kernel (S is fixed geometry, no
     backprop needed through it -- computed once per image, like TGA's
     region masks or DBTA's boundary points).
  4. Directional derivative D_nF(p) = grad(F)(p) . n(p), grad(F) via the SAME
     Sobel kernel applied (depthwise) to the backbone feature map -- WITH
     grad, this is what carries the adversarial gradient signal.
  5. Loss: mean squared difference between D_nF_adv(p) and D_nF_clean(p)
     over every pixel in the band (and every channel), ascended -- same
     plain-MSE convention as osfd_loss/dbta_loss (TGA's Sec 29 diagnosis was
     about signal SPARSITY, not about rank-vs-magnitude, so no rank-based
     variant is tried here).

Known simplifications (documented, not silently ignored):
  - RRB is not supported (v0 scope, per the pre-registered experiment plan --
    DBTA's RRB integration required real point-transform machinery, Sec 30;
    BTFA would need the same for its dense field, deferred unless BTFA earns
    a GO at RRB-off first).
  - No exclusion of neighboring GT boxes from the band/field (same
    simplification DBTA's p_B sampling already accepted) -- a crowded image
    can have one object's band overlap another's box.
  - repr_dim collapses the box's two E10 margins (shrink_frac for w, h
    separately) into one isotropic SDF-distance scale; this is an
    adaptation of the point-based margin convention to a distance-field
    representation, not a value carried over unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Sobel-like finite-difference kernels (fixed, no learnable parameters) used
# for BOTH the geometric field's gradient (grad S, no-grad, fixed geometry)
# and the feature map's spatial gradient (grad F, WITH grad during crafting).
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
_SOBEL_Y = _SOBEL_X.t()


def _spatial_gradient(field: Tensor) -> tuple[Tensor, Tensor]:
    """field: (N,H,W) -- N independent channels/planes. Returns (gx,gy), each
    (N,H,W), via a depthwise Sobel-like conv (replicate padding to avoid
    zero-padding artifacts at the feature map's border). Differentiable
    w.r.t. field."""
    n = field.shape[0]
    kx = _SOBEL_X.to(field.device, field.dtype).view(1, 1, 3, 3).repeat(n, 1, 1, 1)
    ky = _SOBEL_Y.to(field.device, field.dtype).view(1, 1, 3, 3).repeat(n, 1, 1, 1)
    padded = F.pad(field.unsqueeze(0), (1, 1, 1, 1), mode="replicate")  # (1,N,H+2,W+2)
    gx = F.conv2d(padded, kx, groups=n)[0]  # (N,H,W)
    gy = F.conv2d(padded, ky, groups=n)[0]
    return gx, gy


def box_sdf(box: tuple[float, float, float, float], h: int, w: int, canvas: int, device=None) -> Tensor:
    """Signed-distance field for one axis-aligned box, evaluated at a (h,w)
    grid over the full canvas extent (cell centers). Standard rectangle SDF
    (negative inside, 0 on the perimeter, positive outside), in CANVAS-pixel
    distance units. No grad needed -- pure geometry from GT box coords."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) / 2.0, (y2 - y1) / 2.0

    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * canvas / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * canvas / w
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (h,w) each

    dx = (grid_x - cx).abs() - bw
    dy = (grid_y - cy).abs() - bh
    outside = torch.sqrt(dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2)
    inside = torch.maximum(dx, dy).clamp(max=0)
    return outside + inside  # (h,w)


def sdf_normal_and_band(
    box: tuple[float, float, float, float],
    h: int, w: int, canvas: int,
    shrink_frac: float = 0.125, expand_frac: float = 0.25,
    device=None,
) -> dict[str, Tensor] | None:
    """Returns {"nx":(h,w), "ny":(h,w), "mask":(h,w) bool} for one box at one
    stage's resolution, or None if the box is degenerate (zero-or-negative
    width/height). mask = band where -shrink_frac*repr_dim <= S <=
    expand_frac*repr_dim, repr_dim = sqrt(box_w*box_h)."""
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    repr_dim = (box_w * box_h) ** 0.5

    s = box_sdf(box, h, w, canvas, device=device)  # (h,w)
    sx, sy = _spatial_gradient(s.unsqueeze(0))  # each (1,h,w)
    sx, sy = sx[0], sy[0]
    norm = torch.sqrt(sx * sx + sy * sy + 1e-8)
    nx, ny = sx / norm, sy / norm

    mask = (s >= -shrink_frac * repr_dim) & (s <= expand_frac * repr_dim)
    return {"nx": nx, "ny": ny, "mask": mask}


def directional_derivative(feat: Tensor, nx: Tensor, ny: Tensor) -> Tensor:
    """feat: (C,H,W), WITH grad during crafting. nx,ny: (H,W), no grad (fixed
    geometry). Returns (C,H,W): grad(feat) . normal, at every spatial
    location -- differentiable w.r.t. feat."""
    fx, fy = _spatial_gradient(feat)  # each (C,H,W)
    return fx * nx.unsqueeze(0) + fy * ny.unsqueeze(0)


def btfa_loss(
    feats_cln: tuple[Tensor, ...],
    feats_adv_views: tuple[Tensor, ...],
    sdf_fields_per_stage: list[list[dict | None]],
    last_stage_only: bool = False,
) -> Tensor:
    """feats_cln: n-tuple of (1,C_i,H_i,W_i), no_grad, single clean view.
    feats_adv_views: n-tuple of (V,C_i,H_i,W_i), WITH grad -- V=1 (RRB not
    supported yet, see module docstring). sdf_fields_per_stage[stage_idx]
    [obj_idx] = {"nx","ny","mask"} at that stage's resolution, or None for a
    degenerate object -- built once per image (geometry only, independent of
    pixel values).

    Per stage, per object: D_nF is computed ONCE for the clean feature map
    and once per adversarial view, weighted by that object's own normal
    field; the loss is the mean squared difference over every pixel in the
    object's band (and every channel), ascended. Averaged over (view,
    object) pairs per stage, then over stages (or last-stage-only).
    """
    stage_items = list(zip(feats_cln, feats_adv_views, sdf_fields_per_stage))
    if last_stage_only:
        stage_items = stage_items[-1:]

    stage_losses: list[Tensor] = []
    for stage_cln, stage_adv, sdf_fields in stage_items:
        fx_c, fy_c = _spatial_gradient(stage_cln[0])  # (C,H,W) each, no-grad (stage_cln is no-grad)
        stage_total: Tensor | None = None
        stage_count = 0
        for obj_idx, sdf in enumerate(sdf_fields):
            if sdf is None:
                continue
            nx, ny, mask = sdf["nx"], sdf["ny"], sdf["mask"]
            d_cln = (fx_c * nx.unsqueeze(0) + fy_c * ny.unsqueeze(0)).detach()  # (C,H,W)
            n_mask_px = mask.sum().clamp(min=1)
            for v in range(stage_adv.shape[0]):
                d_adv = directional_derivative(stage_adv[v], nx, ny)  # (C,H,W)
                diff2 = (d_adv - d_cln) ** 2 * mask.unsqueeze(0)
                term = diff2.sum() / (diff2.shape[0] * n_mask_px)
                stage_total = term if stage_total is None else stage_total + term
                stage_count += 1
        if stage_count > 0:
            stage_losses.append(stage_total / stage_count)

    if not stage_losses:
        return feats_cln[0].new_zeros(())
    return sum(stage_losses) / len(stage_losses)
