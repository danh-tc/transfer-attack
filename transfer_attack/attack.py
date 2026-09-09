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
from transfer_attack.btfa import btfa_loss, sdf_normal_and_band
from transfer_attack.constants import ALPHA, CANVAS, EPSILON, K, L_S, MU, RHO, S_MAX, SIGMA, STEPS, THETA
from transfer_attack.dbta import boundary_contour_points, build_view_boundary_points, dbta_loss, rrb_forward_with_params
from transfer_attack.losses import build_gt_data_sample, detector_task_loss, osfd_loss, tga_loss
from transfer_attack.models import ModelHandle
from transfer_attack.regions import build_region_masks, pool_regions_batched


@dataclass
class AttackConfig:
    attack_type: Literal["osfd", "mi_fgsm", "tga", "dbta", "btfa"]
    epsilon: float = EPSILON
    alpha: float = ALPHA
    steps: int = STEPS
    mu: float = MU              # MI momentum decay, used by all attacks
    k: float = K                 # OSFD only
    use_rrb: bool = True          # OSFD, TGA -- False: single-view feature loss, no augmentation (E3 factorial)
    theta: float = THETA          # RRB, OSFD/TGA only
    l_s: int = L_S
    rho: float = RHO
    s_max: float = S_MAX
    sigma: float = SIGMA
    canvas: int = CANVAS
    tga_shrink_frac: float = 0.125   # TGA only -- O/E/Bn region definition, verbatim from E10's frozen spec
    tga_expand_frac: float = 0.25    # TGA only
    tga_temperature: float = 1.0     # TGA only -- rank-loss softplus temperature, not tuned (see RESEARCH.md)
    tga_last_stage_only: bool = False  # TGA only -- restrict to last backbone stage (E10's confirmed scope) instead of averaging all stages
    tga_objective: str = "rank"        # TGA only -- "rank" | "geom" | "rank_geom", see losses.py::tga_loss
    tga_geom_lambda: float = 1.0       # TGA only -- weight of the geom term when tga_objective == "rank_geom"
    dbta_shrink_frac: float = 0.125    # DBTA only -- same margins as TGA/E10's frozen spec
    dbta_expand_frac: float = 0.25     # DBTA only
    dbta_n_points: int = 24            # DBTA only -- boundary contour points per object, see dbta.py
    dbta_last_stage_only: bool = False  # DBTA only -- restrict to last backbone stage instead of averaging all stages
    btfa_shrink_frac: float = 0.125    # BTFA only -- same margins as TGA/DBTA/E10's frozen spec
    btfa_expand_frac: float = 0.25     # BTFA only
    btfa_last_stage_only: bool = False  # BTFA only -- restrict to last backbone stage instead of averaging all stages


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
    region_masks_per_obj = None
    boundary_points_clean = None
    sdf_fields_per_stage = None
    if cfg.attack_type == "osfd":
        with torch.no_grad():
            feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
    elif cfg.attack_type == "tga":
        if gt_boxes.shape[0] == 0:
            raise ValueError("tga crafting requires at least one GT box for this image")
        with torch.no_grad():
            feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
        region_masks_per_obj = build_region_masks(gt_boxes, cfg.canvas, cfg.tga_shrink_frac, cfg.tga_expand_frac)
        # Coverage is a deterministic function of (gt_boxes, canvas, this model's
        # per-stage H/W) -- RRB off means it can't change step-to-step, so check
        # once up front rather than crashing on a zero-grad loss mid-loop if NO
        # object clears MASK_MIN_STAGE_CELLS at ANY stage (e.g. every GT box in
        # this image is too small to survive shrink+expand).
        stages_to_check = feats_cln[-1:] if cfg.tga_last_stage_only else feats_cln
        has_valid_region = any(
            pooled is not None
            for stage in stages_to_check
            for pooled in pool_regions_batched(stage[0], region_masks_per_obj)
        )
        if not has_valid_region:
            raise ValueError(
                "tga crafting: no (object, stage) clears the region-coverage threshold for "
                "this image -- every GT box is too small/degenerate after shrink+expand at every backbone stage"
            )
    elif cfg.attack_type == "dbta":
        if gt_boxes.shape[0] == 0:
            raise ValueError("dbta crafting requires at least one GT box for this image")
        with torch.no_grad():
            feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
        boundary_points_clean = boundary_contour_points(
            gt_boxes, cfg.canvas, cfg.dbta_shrink_frac, cfg.dbta_expand_frac, cfg.dbta_n_points
        )
        if not any(bp is not None for bp in boundary_points_clean):
            raise ValueError("dbta crafting: every GT box in this image is degenerate (zero-or-negative width/height)")
    elif cfg.attack_type == "btfa":
        if gt_boxes.shape[0] == 0:
            raise ValueError("btfa crafting requires at least one GT box for this image")
        if cfg.use_rrb:
            raise NotImplementedError(
                "btfa: RRB not supported yet -- the dense SDF/normal field is built once in the "
                "clean (untransformed) canvas frame and reused across steps; RRB's rotate/resize "
                "would require the same per-draw point-transform tracking dbta.py needed for its "
                "boundary points (see RESEARCH.md Sec 30), not yet ported to a dense field. Use "
                "use_rrb=False for the current v0 white-box-sanity pilot."
            )
        with torch.no_grad():
            feats_cln = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
        boxes_list = [tuple(b) for b in gt_boxes.tolist()]
        sdf_fields_per_stage = [
            [sdf_normal_and_band(b, stage.shape[-2], stage.shape[-1], cfg.canvas, cfg.btfa_shrink_frac, cfg.btfa_expand_frac, device=device) for b in boxes_list]
            for stage in feats_cln
        ]
        stages_to_check = sdf_fields_per_stage[-1:] if cfg.btfa_last_stage_only else sdf_fields_per_stage
        has_valid_band = any(sdf is not None and bool(sdf["mask"].any()) for stage_fields in stages_to_check for sdf in stage_fields)
        if not has_valid_band:
            raise ValueError(
                "btfa crafting: no object has a non-empty boundary band at any backbone stage -- "
                "every GT box is degenerate (zero-or-negative width/height) for this image"
            )
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
            # RRB on: 2 augmented views (rotate-only, rotate+resize), both blurred (rrb_forward).
            # RRB off (E3 factorial): single un-augmented view -- osfd_loss still works unchanged,
            # it just sums over 1 group instead of 2.
            aug = rrb_forward(x_adv.unsqueeze(0), gt_boxes, cfg) if cfg.use_rrb else x_adv.unsqueeze(0)
            feats_adv = model.backbone(handle.normalize(aug))
            loss = osfd_loss(feats_cln, feats_adv, cfg.k)
        elif cfg.attack_type == "tga":
            # Same RRB on/off convention as osfd above -- TGA v0 pilot runs with
            # use_rrb=False (single view); RRB support here is for the later
            # OSFD+RRB vs TGA+RRB comparison, not exercised by v0 itself.
            aug = rrb_forward(x_adv.unsqueeze(0), gt_boxes, cfg) if cfg.use_rrb else x_adv.unsqueeze(0)
            feats_adv = model.backbone(handle.normalize(aug))
            loss = tga_loss(
                feats_cln, feats_adv, region_masks_per_obj, cfg.tga_temperature, cfg.tga_last_stage_only,
                cfg.tga_objective, cfg.tga_geom_lambda,
            )
        elif cfg.attack_type == "dbta":
            if cfg.use_rrb:
                # RRB physically moves object content within the tensor (rotate+
                # resize) -- boundary_points_clean must be transformed by the SAME
                # per-draw params to stay pointed at the actual (now-relocated)
                # object boundary in each augmented view. Using dbta's OWN RRB
                # implementation (not augment.rrb_forward) because it additionally
                # returns those params -- see dbta.py's module docstring for why a
                # naive "reuse the untransformed points" attempt collapsed (-38 ASR
                # points on the white-box surrogate, same failure class as TGA).
                aug, transforms_per_view = rrb_forward_with_params(x_adv.unsqueeze(0), gt_boxes, cfg)
                boundary_points_per_view = build_view_boundary_points(boundary_points_clean, transforms_per_view)
            else:
                aug = x_adv.unsqueeze(0)
                boundary_points_per_view = [boundary_points_clean]
            feats_adv = model.backbone(handle.normalize(aug))
            loss = dbta_loss(feats_cln, feats_adv, boundary_points_clean, boundary_points_per_view, cfg.canvas, cfg.dbta_last_stage_only)
        elif cfg.attack_type == "btfa":
            # RRB off only (v0) -- see the NotImplementedError raised above for use_rrb=True.
            feats_adv = model.backbone(handle.normalize(x_adv.unsqueeze(0)))
            loss = btfa_loss(feats_cln, feats_adv, sdf_fields_per_stage, cfg.btfa_last_stage_only)
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
