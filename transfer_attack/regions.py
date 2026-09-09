"""Semantic-region mask construction and pooling for an object's interior
(O) / boundary (E) / near-background (Bn) / far-background (Bf).

This is a verbatim port of scripts/e10_relational_geometry.py's FROZEN SPEC
(RESEARCH.md Sec 27-28, confirmed STRONG GO at N=296) -- shrink/expand
fractions, Bn/Bf's exclusion of every OTHER GT box's own rectangle, and the
MASK_MIN_STAGE_CELLS coverage threshold are copied exactly, not
reimplemented from scratch. Kept as a separate copy (not imported from the
scripts/ file) so E10's already-verified diagnostic script stays untouched --
if E10's frozen spec ever changes, this file must be updated to match by
hand, but E10 is closed/confirmed and not expected to change.

The one real difference from e10_relational_geometry.py: that script reduces
relational distances to Python floats (`.item()`) for CSV logging, since it
never backpropagates. This module keeps everything as tensors so
transfer_attack/losses.py::tga_loss can differentiate through it during
adversarial crafting.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

MASK_MIN_STAGE_CELLS = 1.0
_REGION_ORDER = ("O", "E", "Bn", "Bf")


def _clip_box(x1, y1, x2, y2, canvas):
    return max(0.0, x1), max(0.0, y1), min(float(canvas), x2), min(float(canvas), y2)


def shrink_box(box, frac, canvas):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return _clip_box(x1 + frac * w, y1 + frac * h, x2 - frac * w, y2 - frac * h, canvas)


def expand_box(box, frac, canvas):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return _clip_box(x1 - frac * w, y1 - frac * h, x2 + frac * w, y2 + frac * h, canvas)


def box_to_mask(box, canvas: int) -> Tensor:
    m = torch.zeros((canvas, canvas), dtype=torch.bool)
    x1, y1, x2, y2 = box
    ix1, iy1 = int(round(x1)), int(round(y1))
    ix2, iy2 = int(round(x2)), int(round(y2))
    if ix2 > ix1 and iy2 > iy1:
        m[iy1:iy2, ix1:ix2] = True
    return m


def build_region_masks(
    gt_boxes: Tensor, canvas: int, shrink_frac: float = 0.125, expand_frac: float = 0.25
) -> list[dict[str, Tensor] | None]:
    """gt_boxes: (M,4) xyxy canvas-space tensor. Returns a list (len M) of
    dicts {"O","E","Bn","Bf"} (each a (canvas,canvas) bool tensor), or None
    for objects whose shrunk interior is degenerate. Bn/Bf exclude the union
    of every OTHER GT box's own (raw, unexpanded) rectangle; O/E do not."""
    boxes = gt_boxes.tolist()
    n = len(boxes)
    box_masks = [box_to_mask(b, canvas) for b in boxes]

    out: list[dict[str, Tensor] | None] = []
    for i in range(n):
        b = boxes[i]
        o_box = shrink_box(b, shrink_frac, canvas)
        if o_box[2] <= o_box[0] or o_box[3] <= o_box[1]:
            out.append(None)
            continue
        other_mask = None
        for j in range(n):
            if j == i:
                continue
            other_mask = box_masks[j].clone() if other_mask is None else (other_mask | box_masks[j])
        if other_mask is None:
            other_mask = torch.zeros((canvas, canvas), dtype=torch.bool)

        o_mask = box_to_mask(o_box, canvas)
        e_mask = box_masks[i] & (~o_mask)
        bn_full = box_to_mask(expand_box(b, expand_frac, canvas), canvas)
        bn_mask = bn_full & (~box_masks[i]) & (~other_mask)
        bf_mask = (~bn_full) & (~other_mask)
        out.append({"O": o_mask, "E": e_mask, "Bn": bn_mask, "Bf": bf_mask})
    return out


def pool_regions_batched(feat: Tensor, region_masks_per_obj: list[dict | None]) -> list[dict | None]:
    """feat: (C,H,W) tensor for one stage -- WITH grad when called during
    crafting (masks/weights carry no grad, only feat does, so this is
    differentiable end-to-end). Returns a list the same length as
    region_masks_per_obj, entry i = {"O":(C,),...} pooled means for object i,
    or None if any region covers < MASK_MIN_STAGE_CELLS at this stage."""
    h, w = feat.shape[-2], feat.shape[-1]
    valid_idx = [i for i, m in enumerate(region_masks_per_obj) if m is not None]
    out: list[dict | None] = [None] * len(region_masks_per_obj)
    if not valid_idx:
        return out

    stacked = torch.stack(
        [region_masks_per_obj[i][name].float() for i in valid_idx for name in _REGION_ORDER]
    ).unsqueeze(1)  # (n_valid*4, 1, canvas, canvas)
    weights = F.interpolate(stacked, size=(h, w), mode="area")[:, 0].to(feat.device)  # (n_valid*4, h, w)

    for k, obj_idx in enumerate(valid_idx):
        obj_weights = weights[k * 4 : (k + 1) * 4]  # (4,h,w), order == _REGION_ORDER
        wsum = obj_weights.sum(dim=(1, 2))  # (4,)
        if bool((wsum < MASK_MIN_STAGE_CELLS).any()):
            continue
        pooled = {}
        for j, name in enumerate(_REGION_ORDER):
            pooled[name] = (feat * obj_weights[j].unsqueeze(0)).sum(dim=(1, 2)) / wsum[j]
        out[obj_idx] = pooled
    return out
