"""DBTA -- Dense Boundary-Transition Attack (RESEARCH.md method-derivation,
proposed after TGA's Sec 29 NO-GO diagnosed the failure mode as "too sparse a
gradient signal": pooling O/E/Bn down to one (C,) mean vector per object
leaves only 3 scalar relations to optimize, regardless of whether the loss is
rank-based or magnitude-based (both NO-GO, see losses.py's TGA section).

DBTA keeps the exact same O/E/Bn/Bf region definitions and shrink_frac/
expand_frac margins as E10's frozen spec (transfer_attack/regions.py) and
TGA, but instead of pooling each region to a single vector, samples a DENSE
field of points along the GT box's own perimeter (the literal semantic edge
of the object -- also the shared border between E and Bn in the box-based
construction) and disrupts the per-point transition (interior -> edge ->
near-background) at each one. This trades E10's confirmed-invariant 3-scalar
summary for many more localized, gradient-carrying terms per object.

For each GT box b=(x1,y1,x2,y2) and each point p on b's perimeter, 3
canvas-space points are placed on CONCENTRIC contours using the same margins
build_region_masks uses:
  p_O = p shifted INWARD by shrink_frac * (box's own axis dimension)  -- lands
        on O's perimeter (interior side of the E ring).
  p_E = p itself                                                      -- the
        object's actual edge.
  p_B = p shifted OUTWARD by expand_frac * (box's own axis dimension) -- lands
        on Bn's outer perimeter (far side of the near-background ring).
Features at these points are read via bilinear F.grid_sample per backbone
stage -- differentiable, and stage-agnostic, since grid_sample's coordinates
are normalized to the input tensor's own H/W: the same canvas-space points
are reused at every stage, no per-stage mask rebuild needed (unlike
transfer_attack/regions.py's masks, which must be re-downsampled per stage).

RRB integration (added after round-1's RRB-off pilot passed white-box
sanity, and round-2's naive "reuse the same fixed canvas points under RRB"
attempt collapsed just as badly as TGA, -38 ASR points on the white-box
surrogate). Root cause: rrb_forward's rotate+resize physically moves object
content within the tensor, but the round-2 attempt kept sampling at the
UNTRANSFORMED canvas coordinates -- i.e. it was reading features from
whatever content happened to land there after the image moved, not from the
actual (now-relocated) object boundary. OSFD is immune to this because its
loss is a position-generic feature-map MSE; DBTA's per-point correspondence
is not. Fix: track the exact rotation angle/center and resize/pad parameters
each RRB draw uses (rrb_forward_with_params, `_draw_rotation`/`_draw_resize`
-- independent reimplementation of transfer_attack.augment's SAME
distributions, not a call into it, so existing OSFD RRB noise/results stay
byte-for-byte untouched) and apply the SAME transform to the canvas-space
boundary points before sampling each augmented view's feature map
(rotate_points, resize_pad_resize_points -- both point-transform formulas
verified empirically against torchvision.transforms.functional.rotate and
F.interpolate(align_corners=True) on synthetic marked-pixel images, not
derived from documentation alone, since a naive rotation-matrix guess for
`rotate`'s convention was checked and found WRONG in sign).

Known simplification (documented, not silently ignored): p_B can land inside
a NEIGHBORING GT box's region in crowded images -- transfer_attack/regions.py
explicitly excludes this for Bn/Bf's pooled means; this module does not,
since excluding it per-point would require per-point masking rather than a
single scalar validity check. Acceptable for a white-box sanity pilot;
revisit if DBTA gets a GO and moves toward a scaled/confirmed run. Similarly,
no degeneracy guard for objects small enough that shrink_frac's inward
margin overshoots the box's own center (build_region_masks drops such
objects entirely via its o_box check; this module does not, since it works
per-point, not per-region) -- acceptable noise for a sanity check.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor

from transfer_attack.constants import L_S, RHO, S_MAX, SIGMA, THETA

N_POINTS_DEFAULT = 24


# ---------------------------------------------------------------------------
# Boundary contour geometry (RRB-independent -- these are the CLEAN-image
# canvas-space points; RRB transform is applied on top, per step, see below)
# ---------------------------------------------------------------------------

def boundary_contour_points(
    gt_boxes: Tensor,
    canvas: int,
    shrink_frac: float = 0.125,
    expand_frac: float = 0.25,
    n_points: int = N_POINTS_DEFAULT,
) -> list[dict[str, Tensor] | None]:
    """gt_boxes: (M,4) xyxy canvas-space. Returns a list (len M) of dicts
    {"p_o":(n,2), "p_e":(n,2), "p_b":(n,2)} -- RAW canvas-space (pixel, NOT
    normalized to grid_sample's [-1,1]) sample coordinates for n boundary
    points around that object's perimeter -- or None if the box is
    degenerate (zero-or-negative width/height). n_points is split across the
    4 edges proportionally to their length (so a wide box gets more points on
    its top/bottom edges than its left/right ones). Normalization to
    grid_sample coords happens later, in `sample_at`, AFTER any RRB rotate/
    resize transform has been applied to these points (see module docstring)."""
    out: list[dict[str, Tensor] | None] = []
    for box in gt_boxes.tolist():
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            out.append(None)
            continue
        perimeter = 2 * (w + h)
        n_top = max(1, round(n_points * w / perimeter))
        n_right = max(1, round(n_points * h / perimeter))
        n_bottom = max(1, round(n_points * w / perimeter))
        n_left = max(1, round(n_points * h / perimeter))

        pts: list[tuple[float, float]] = []
        in_dir: list[tuple[float, float]] = []
        out_dir: list[tuple[float, float]] = []
        margin: list[float] = []

        def add_edge(n, x_fn, y_fn, inward, outward, m):
            for k in range(n):
                t = k / n
                pts.append((x_fn(t), y_fn(t)))
                in_dir.append(inward)
                out_dir.append(outward)
                margin.append(m)

        # top edge: y=y1, x: x1->x2; inward=+y (down, into box), outward=-y; margin scales with h
        add_edge(n_top, lambda t: x1 + t * w, lambda t: y1, (0.0, 1.0), (0.0, -1.0), h)
        # right edge: x=x2, y: y1->y2; inward=-x, outward=+x; margin scales with w
        add_edge(n_right, lambda t: x2, lambda t: y1 + t * h, (-1.0, 0.0), (1.0, 0.0), w)
        # bottom edge: y=y2, x: x2->x1; inward=-y, outward=+y; margin scales with h
        add_edge(n_bottom, lambda t: x2 - t * w, lambda t: y2, (0.0, -1.0), (0.0, 1.0), h)
        # left edge: x=x1, y: y2->y1; inward=+x, outward=-x; margin scales with w
        add_edge(n_left, lambda t: x1, lambda t: y2 - t * h, (1.0, 0.0), (-1.0, 0.0), w)

        pts_t = torch.tensor(pts, dtype=torch.float32)                    # (n,2) xy canvas
        in_dir_t = torch.tensor(in_dir, dtype=torch.float32)              # (n,2)
        out_dir_t = torch.tensor(out_dir, dtype=torch.float32)            # (n,2)
        margin_t = torch.tensor(margin, dtype=torch.float32).unsqueeze(1)  # (n,1)

        p_o = pts_t + in_dir_t * margin_t * shrink_frac
        p_e = pts_t
        p_b = pts_t + out_dir_t * margin_t * expand_frac
        out.append({"p_o": p_o, "p_e": p_e, "p_b": p_b})
    return out


def sample_at(feat: Tensor, points_xy: Tensor, canvas: int) -> Tensor:
    """feat: (C,H,W), WITH grad when called during crafting. points_xy: (n,2)
    RAW canvas-space (pixel) coords. Normalizes to grid_sample's [-1,1] range
    and bilinearly samples -- fully differentiable w.r.t. feat (points_xy
    carries no grad, it's fixed geometry from the GT boxes + RRB transform
    params, neither of which depends on the adversarial pixel values)."""
    p = points_xy.clamp(0.0, float(canvas))
    grid_xy = p / canvas * 2.0 - 1.0  # align_corners=False convention
    grid = grid_xy.to(feat.device).view(1, 1, -1, 2)
    sampled = F.grid_sample(feat.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled[0, :, 0, :].transpose(0, 1)  # (n,C)


# ---------------------------------------------------------------------------
# Point-transforms matching transfer_attack.augment's RRB operations exactly
# (verified empirically against torchvision/F.interpolate, see module
# docstring) -- so DBTA's boundary points can be moved into each augmented
# view's coordinate frame the same way the image content itself moved.
# ---------------------------------------------------------------------------

def rotate_points(points_xy: Tensor, angle_deg: float, center_xy: tuple[float, float]) -> Tensor:
    """points_xy: (n,2) canvas-space (x,y). Returns (n,2) transformed the SAME
    way torchvision.transforms.functional.rotate(img, angle_deg, center=
    center_xy) transforms image CONTENT -- if a bright pixel sat at
    points_xy[i] in the input, after rotation it sits at the returned
    coordinate in the output. This is NOT the standard CCW rotation matrix --
    verified empirically (synthetic marked-pixel image, 4 angle/center
    combinations) against torchvision's actual behavior, which differs in
    sign from a naive guess."""
    cx, cy = center_xy
    t = math.radians(angle_deg)
    dx = points_xy[:, 0] - cx
    dy = points_xy[:, 1] - cy
    cos_t, sin_t = math.cos(t), math.sin(t)
    nx = cx + dx * cos_t + dy * sin_t
    ny = cy - dx * sin_t + dy * cos_t
    return torch.stack([nx, ny], dim=1)


def resize_pad_resize_points(
    points_xy: Tensor,
    ori_w: int, ori_h: int,
    new_w: int, new_h: int,
    pad_left: int, pad_top: int,
    canvas_w: int, canvas_h: int,
) -> Tensor:
    """Point-transform matching transfer_attack.augment.adaptive_random_
    resizing: resize (ori_w,ori_h)->(new_w,new_h) with align_corners=True,
    pad by (pad_left,pad_top) into a (canvas_w,canvas_h) canvas, then resize
    canvas->(ori_w,ori_h) with align_corners=True. Verified empirically
    (synthetic marked-pixel image) against F.interpolate/F.pad's actual
    behavior with these exact settings."""
    x, y = points_xy[:, 0], points_xy[:, 1]
    xp = x * (new_w - 1) / (ori_w - 1) if ori_w > 1 else x
    yp = y * (new_h - 1) / (ori_h - 1) if ori_h > 1 else y
    xpp = xp + pad_left
    ypp = yp + pad_top
    xf = xpp * (ori_w - 1) / (canvas_w - 1) if canvas_w > 1 else xpp
    yf = ypp * (ori_h - 1) / (canvas_h - 1) if canvas_h > 1 else ypp
    return torch.stack([xf, yf], dim=1)


# ---------------------------------------------------------------------------
# RRB augmentation with tracked transform params (single image, batch=1).
# Independent reimplementation of transfer_attack.augment's random_axis_
# rotation/adaptive_random_resizing -- SAME distributions (draws the same
# random variables via the same calls, in the same order, for prob=1.0,
# which is the only path ever exercised by this codebase), but a separate
# function so existing OSFD RRB noise/results are untouched.
# ---------------------------------------------------------------------------

def _draw_rotation(gt_boxes: Tensor, h: int, w: int, theta: float, l_s: int) -> tuple[float, tuple[float, float]]:
    device = gt_boxes.device
    image_center = torch.tensor([[w // 2, h // 2]], dtype=torch.float32, device=device)
    if gt_boxes.numel() > 0:
        boxes_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0
        centers = torch.cat([boxes_centers, image_center], dim=0)
    else:
        centers = image_center
    centers_with_random = centers if l_s == 0 else centers + torch.randint_like(centers, low=-l_s, high=l_s)
    center_x, center_y = random.choice(centers_with_random)
    angle = random.random() * 2 * theta - theta
    return float(angle), (float(center_x), float(center_y))


def _draw_resize(gt_boxes: Tensor, ori_h: int, ori_w: int, rho: float, s_max: float) -> dict | None:
    if gt_boxes.numel() == 0:
        return None
    random_box_idx = random.randint(0, gt_boxes.shape[0] - 1)
    box = gt_boxes[random_box_idx]
    box_w = float(box[2] - box[0])
    box_h = float(box[3] - box[1])
    scale_h = min(1 + rho * (box_h / ori_h), s_max)
    scale_w = min(1 + rho * (box_w / ori_w), s_max)
    new_h = random.randint(ori_h, int(scale_h * ori_h))
    new_w = random.randint(ori_w, int(scale_w * ori_w))
    canvas_h = int(scale_h * ori_h)
    canvas_w = int(scale_w * ori_w)
    rem_h = canvas_h - new_h
    rem_w = canvas_w - new_w
    pad_left = random.randint(0, rem_w)
    pad_top = random.randint(0, rem_h)
    return {
        "ori_w": ori_w, "ori_h": ori_h, "new_w": new_w, "new_h": new_h,
        "pad_left": pad_left, "pad_top": pad_top, "canvas_w": canvas_w, "canvas_h": canvas_h,
    }


def rrb_forward_with_params(adv_img: Tensor, gt_boxes_canvas: Tensor, cfg=None) -> tuple[Tensor, list[dict]]:
    """adv_img: (1,3,H,W). Returns (aug: (2,3,H,W) [rotate-only, rotate-then-
    resize], both Gaussian-blurred -- same as transfer_attack.augment.
    rrb_forward -- PLUS transforms_per_view: a length-2 list, transforms_per_
    view[v] = {"angle_deg":..., "center_xy":..., "resize": {...}|None},
    ready to feed into `transform_view_points` to move DBTA's boundary points
    into view v's coordinate frame. view 0 (rotate-only) has resize=None;
    view 1 (rotate+resize) carries the SAME angle/center as view 0 plus the
    resize params -- matches augment.py's branch2 = resize(branch1)."""
    from torchvision.transforms.functional import rotate as tv_rotate

    theta = getattr(cfg, "theta", THETA) if cfg is not None else THETA
    l_s = getattr(cfg, "l_s", L_S) if cfg is not None else L_S
    rho = getattr(cfg, "rho", RHO) if cfg is not None else RHO
    s_max = getattr(cfg, "s_max", S_MAX) if cfg is not None else S_MAX
    sigma = getattr(cfg, "sigma", SIGMA) if cfg is not None else SIGMA

    _, _, h, w = adv_img.shape
    angle, center = _draw_rotation(gt_boxes_canvas, h, w, theta, l_s)
    branch1 = tv_rotate(adv_img, angle, center=[center[0], center[1]])

    resize_params = _draw_resize(gt_boxes_canvas, h, w, rho, s_max)
    if resize_params is None:
        branch2 = branch1
    else:
        rp = resize_params
        rescaled = F.interpolate(branch1, size=(rp["new_h"], rp["new_w"]), mode="bilinear", align_corners=True)
        rem_h = rp["canvas_h"] - rp["new_h"]
        rem_w = rp["canvas_w"] - rp["new_w"]
        padded = F.pad(rescaled, (rp["pad_left"], rem_w - rp["pad_left"], rp["pad_top"], rem_h - rp["pad_top"]), mode="constant", value=0.0)
        branch2 = F.interpolate(padded, size=(h, w), mode="bilinear", align_corners=True)

    combined = torch.cat([branch1, branch2], dim=0)
    aug = torch.clamp(combined + torch.randn_like(combined) * sigma, 0.0, 255.0)

    transforms_per_view = [
        {"angle_deg": angle, "center_xy": center, "resize": None},
        {"angle_deg": angle, "center_xy": center, "resize": resize_params},
    ]
    return aug, transforms_per_view


def transform_view_points(points_xy: Tensor, transform: dict) -> Tensor:
    """points_xy: (n,2) canvas-space, in the CLEAN (untransformed) frame.
    transform: one entry of rrb_forward_with_params's transforms_per_view.
    Returns points_xy moved into that view's coordinate frame."""
    p = rotate_points(points_xy, transform["angle_deg"], transform["center_xy"])
    if transform["resize"] is not None:
        p = resize_pad_resize_points(p, **transform["resize"])
    return p


def build_view_boundary_points(
    boundary_points_clean: list[dict | None], transforms_per_view: list[dict]
) -> list[list[dict | None]]:
    """boundary_points_clean: from boundary_contour_points (clean-frame, raw
    canvas coords). transforms_per_view: from rrb_forward_with_params.
    Returns boundary_points_per_view[v][obj_idx] = {"p_o","p_e","p_b"}
    transformed into view v's frame (or None, preserving degenerate-object
    slots)."""
    out: list[list[dict | None]] = []
    for transform in transforms_per_view:
        view_list: list[dict | None] = []
        for bp in boundary_points_clean:
            if bp is None:
                view_list.append(None)
                continue
            view_list.append({
                "p_o": transform_view_points(bp["p_o"], transform),
                "p_e": transform_view_points(bp["p_e"], transform),
                "p_b": transform_view_points(bp["p_b"], transform),
            })
        out.append(view_list)
    return out


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dbta_loss(
    feats_cln: tuple[Tensor, ...],
    feats_adv_views: tuple[Tensor, ...],
    boundary_points_clean: list[dict | None],
    boundary_points_per_view: list[list[dict | None]],
    canvas: int,
    last_stage_only: bool = False,
) -> Tensor:
    """feats_cln: n-tuple of (1,C_i,H_i,W_i), no_grad, single clean view.
    feats_adv_views: n-tuple of (V,C_i,H_i,W_i), WITH grad -- V=1 if RRB is
    off, V=2 if on (rotate-only/rotate+resize, same convention as osfd_loss/
    tga_loss). boundary_points_clean: from boundary_contour_points, the
    CLEAN-frame points (always used for the clean side -- the clean forward
    pass never goes through RRB). boundary_points_per_view: length must equal
    V; boundary_points_per_view[v] is per-object points ALREADY transformed
    into view v's coordinate frame (identical to boundary_points_clean, i.e.
    a 1-element list containing it, when RRB is off -- see attack.py).

    Per point p, the boundary-transition vector is
        T(p) = [f_E(p) - f_O(p) ; f_B(p) - f_E(p)]   (shape (2*C,))
    and the loss ascends MSE(T_adv(p), T_clean(p)) -- plain squared
    difference (same convention as osfd_loss's raw feature MSE, deliberately
    NOT normalized/rank-based, since TGA's Sec 29 diagnosis pointed at signal
    SPARSITY, not at the rank-vs-magnitude choice). Averaged over every
    (view, point) pair across every object, then over stages (or last-stage-
    only). An object with a degenerate box (None entry) contributes nothing.
    """
    stage_pairs = list(zip(feats_cln, feats_adv_views))
    if last_stage_only:
        stage_pairs = stage_pairs[-1:]

    stage_losses: list[Tensor] = []
    for stage_cln, stage_adv in stage_pairs:
        stage_total: Tensor | None = None
        stage_count = 0
        for obj_idx, bp_clean in enumerate(boundary_points_clean):
            if bp_clean is None:
                continue
            f_o_c = sample_at(stage_cln[0], bp_clean["p_o"], canvas)
            f_e_c = sample_at(stage_cln[0], bp_clean["p_e"], canvas)
            f_b_c = sample_at(stage_cln[0], bp_clean["p_b"], canvas)
            t_clean = torch.cat([f_e_c - f_o_c, f_b_c - f_e_c], dim=1).detach()  # (n,2C)
            for v in range(stage_adv.shape[0]):
                bp_v = boundary_points_per_view[v][obj_idx]
                if bp_v is None:
                    continue
                f_o_a = sample_at(stage_adv[v], bp_v["p_o"], canvas)
                f_e_a = sample_at(stage_adv[v], bp_v["p_e"], canvas)
                f_b_a = sample_at(stage_adv[v], bp_v["p_b"], canvas)
                t_adv = torch.cat([f_e_a - f_o_a, f_b_a - f_e_a], dim=1)  # (n,2C)
                term = F.mse_loss(t_adv, t_clean)
                stage_total = term if stage_total is None else stage_total + term
                stage_count += 1
        if stage_count > 0:
            stage_losses.append(stage_total / stage_count)

    if not stage_losses:
        return feats_cln[0].new_zeros(())
    return sum(stage_losses) / len(stage_losses)
