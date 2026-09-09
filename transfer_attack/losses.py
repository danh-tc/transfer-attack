"""OSFD backbone-feature loss, the MI-FGSM baseline's output/task loss, and
TGA's relational-geometry rank loss (RESEARCH.md Sec 27-28 -> method
derivation, "TGA -- Semantic Transition Geometry Attack")."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def osfd_loss(feats_cln: tuple[Tensor, ...], feats_adv_2groups: tuple[Tensor, ...], k: float) -> Tensor:
    """feats_cln: n-tuple of (1,C_i,H_i,W_i) backbone stage tensors (no_grad).
    feats_adv_2groups: n-tuple of (2,C_i,H_i,W_i) tensors (group0=rotate-only,
    group1=rotate+resize), WITH grad.

    total = sum_i sum_g MSE(k * feats_cln[i], feats_adv_2groups[i][g])
    F.mse_loss's default 'mean' reduction implements the paper's (1/N_i)
    per-stage average (Eq. 2); summing over stages x 2 groups implements the
    outer double-sum from the reference code (TransferAttack.prepare_losses +
    IFGSM.combine_losses). No extra reweighting.
    """
    total = feats_cln[0].new_zeros(())
    for stage_cln, stage_adv in zip(feats_cln, feats_adv_2groups):
        target = k * stage_cln
        for g in range(stage_adv.shape[0]):
            total = total + F.mse_loss(target, stage_adv[g : g + 1])
    return total


def build_gt_data_sample(gt_boxes_canvas: Tensor, gt_labels: Tensor, canvas: int):
    """gt_boxes_canvas: (M,4) xyxy canvas-space. gt_labels: (M,) contiguous
    per-model label ids (already mapped from raw COCO category ids via the
    target ModelHandle's cat_id_to_label).

    ori_shape == img_shape == (canvas, canvas) is intentional: we've already
    done our own resize+pad, so there is nothing left for the model's internal
    rescale bookkeeping to do (scale_factor=(1.0, 1.0)).
    """
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData

    device = gt_boxes_canvas.device
    ds = DetDataSample()
    gt_instances = InstanceData()
    gt_instances.bboxes = gt_boxes_canvas
    gt_instances.labels = gt_labels.to(device)
    ds.gt_instances = gt_instances
    ds.set_metainfo(
        dict(
            img_shape=(canvas, canvas),
            ori_shape=(canvas, canvas),
            scale_factor=(1.0, 1.0),
            batch_input_shape=(canvas, canvas),
            # anchor-based heads' loss_by_feat -> get_anchors -> valid_flags
            # reads this (only on the .loss() path, not .predict()); our canvas
            # has no extra padding beyond itself, so it's (canvas, canvas, 3)
            # matching mmdet's usual (h, w, c) Pad-transform convention.
            pad_shape=(canvas, canvas, 3),
        )
    )
    return ds


def detector_task_loss(model: nn.Module, x_norm: Tensor, data_sample) -> Tensor:
    """x_norm: (1,3,H,W) already normalized with the SAME normalizer used
    elsewhere for this model. data_sample: a single DetDataSample (from
    build_gt_data_sample) carrying the GT instances for this image.

    Sums every loss_dict entry whose key contains "loss" (skips scalar
    diagnostics like accuracy). model.eval() is kept set by the caller for
    both this path and backbone-only extraction -- .loss() is a plain method
    call in mmdet 3.x, not gated by the train/eval flag, and keeping BatchNorm
    frozen (eval mode) is strictly better for batch-size-1 gradient crafting.
    """
    loss_dict = model.loss(x_norm, [data_sample])
    total = None
    for key, value in loss_dict.items():
        if "loss" not in key:
            continue
        if isinstance(value, (list, tuple)):
            term = sum(value)
        else:
            term = value
        total = term if total is None else total + term
    if total is None:
        raise RuntimeError(f"model.loss() returned no 'loss*' entries: keys={list(loss_dict.keys())}")
    return total


# ---------------------------------------------------------------------------
# TGA -- Semantic Transition Geometry Attack (method-discovery prototype,
# RESEARCH.md Sec 27-28's E10 relational-geometry invariant -> attack
# objective). v0 (this implementation) deliberately restricts to the O-E-Bn
# transition (drops Bf/far-background): E10 dev_300 confirmation (Sec 28)
# found O<->farBG was the ONLY one of 6 relations whose cross-family
# consistency fell below the pre-registered 0.30 threshold as N scaled
# 49->296, while O<->E and E<->nearBG -- both boundary-adjacent -- were the
# two strongest and most stable relations across both N. TGA v0 does not
# attack raw feature magnitude/direction (that's OSFD's suppress/amplify);
# it attacks the RELATIVE ordering of distances between O/E/Bn's pooled
# features, read off each clean image rather than hard-coded.
# ---------------------------------------------------------------------------

# The 3 pairwise relations among {O, E, Bn} -- see build_region_masks in
# transfer_attack/regions.py for the region definitions themselves.
TGA_RELATIONS = (("O", "E"), ("O", "Bn"), ("E", "Bn"))
_TGA_RELATION_KEYS = tuple(a + b for a, b in TGA_RELATIONS)
# Every unordered pair among the 3 relations above -- the rank loss penalizes
# an inversion of each pair's clean ordering, not just individual magnitudes.
_TGA_RANK_PAIRS = (("OE", "OBn"), ("OE", "EBn"), ("OBn", "EBn"))


def tga_relational_vector(pooled: dict[str, Tensor]) -> dict[str, Tensor]:
    """pooled: {"O":(C,),"E":(C,),"Bn":(C,),...}. Returns {"OE":scalar
    tensor,...} for the 3 TGA_RELATIONS, distance = 1 - cosine_similarity.
    Differentiable w.r.t. pooled's values (no .item() -- contrast with
    scripts/e10_relational_geometry.py's relational_vector, which is
    diagnostic-only and returns floats)."""

    def d(a: Tensor, b: Tensor) -> Tensor:
        return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))[0]

    return {a + b: d(pooled[a], pooled[b]) for a, b in TGA_RELATIONS}


def tga_rank_loss(r_clean: dict[str, Tensor], r_adv: dict[str, Tensor], temperature: float = 1.0) -> Tensor:
    """Relational-order-inversion loss (RESEARCH.md method-derivation note,
    "Relational-order inversion"). For every pair (i,j) of the 3 relations,
    read the clean sign s_ij = sign(r_clean_i - r_clean_j) and penalize the
    adversarial difference NOT having flipped past it:
        L = sum_{i<j} softplus(-s_ij * (r_adv_i - r_adv_j) / T)
    Ascending this loss (same convention as osfd_loss/detector_task_loss --
    see attack.py's module docstring) pushes r_adv_i - r_adv_j across zero
    and past it in the direction opposite the clean ordering, i.e. inverts
    which of the two relations is larger. A tied clean pair (s_ij == 0, no
    defined ordering to invert) is skipped rather than penalized either way.
    r_clean values are detached even if already no-grad, so this function is
    safe to call with a non-no-grad r_clean too."""
    total = None
    for i, j in _TGA_RANK_PAIRS:
        s_ij = torch.sign(r_clean[i].detach() - r_clean[j].detach())
        if s_ij.item() == 0.0:
            continue
        diff_adv = r_adv[i] - r_adv[j]
        term = F.softplus(-s_ij * diff_adv / temperature)
        total = term if total is None else total + term
    if total is None:
        return r_adv[_TGA_RELATION_KEYS[0]].new_zeros(())
    return total


def tga_geom_loss(r_clean: dict[str, Tensor], r_adv: dict[str, Tensor], eps: float = 1e-6) -> Tensor:
    """Magnitude-based alternative to tga_rank_loss (RESEARCH.md
    method-derivation note Sec 3, "geometry deformation"):
        r_hat = r / (||r||_2 + eps)
        L_geom = ||r_hat_adv - r_hat_clean||^2
    Tried after tga_rank_loss's v0 pilot came back a clean NO-GO on EVERY
    model, including a -22 to -30 ASR point loss on the white-box surrogate
    itself (not just a transfer failure) -- diagnosed as the rank loss's
    softplus(-s_ij*diff/T) having a vanishing gradient for any pair whose
    clean relations are already well-separated (z << 0 -> softplus(z) ~
    exp(z) ~ 0), so most objects contribute near-zero gradient throughout
    crafting regardless of stage choice (both all-stage and last-stage-only
    failed identically). L_geom's gradient is ~2*(r_hat_adv - r_hat_clean) --
    proportional to how far the normalized relational vector has already
    moved, never vanishing just because the clean gap was large. r_clean is
    detached (safe even if already no-grad)."""
    order = _TGA_RELATION_KEYS
    r_c = torch.stack([r_clean[k] for k in order]).detach()
    r_a = torch.stack([r_adv[k] for k in order])
    r_c_hat = r_c / (r_c.norm() + eps)
    r_a_hat = r_a / (r_a.norm() + eps)
    return ((r_a_hat - r_c_hat) ** 2).sum()


def tga_loss(
    feats_cln: tuple[Tensor, ...],
    feats_adv_views: tuple[Tensor, ...],
    region_masks_per_obj: list[dict | None],
    temperature: float = 1.0,
    last_stage_only: bool = False,
    objective: str = "rank",
    geom_lambda: float = 1.0,
) -> Tensor:
    """feats_cln: n-tuple of (1,C_i,H_i,W_i) backbone stage tensors (no_grad,
    single clean view). feats_adv_views: n-tuple of (V,C_i,H_i,W_i) tensors,
    WITH grad -- V=1 if RRB is off (TGA v0), V=2 if RRB is on (rotate-only /
    rotate+resize groups, same convention as osfd_loss's feats_adv_2groups,
    for the later OSFD+RRB vs TGA+RRB comparison). region_masks_per_obj: from
    transfer_attack.regions.build_region_masks, built once per image (masks
    only depend on GT boxes, not on the adversarial image).

    Per stage l, L_TGA^l = mean over every (view, object) pair that clears
    the mask-coverage threshold (transfer_attack.regions.pool_regions_batched)
    of the per-object term selected by `objective`:
      "rank"      -- tga_rank_loss(r_clean, r_adv, temperature) alone (v0).
      "geom"      -- tga_geom_loss(r_clean, r_adv) alone (tried after v0's
                     rank-only came back NO-GO on every model -- see
                     tga_geom_loss's docstring for the diagnosis).
      "rank_geom" -- tga_rank_loss + geom_lambda * tga_geom_loss (the
                     original combined design; not tried until each term is
                     checked in isolation first).
    Final loss = mean over stages of L_TGA^l (unweighted average across
    depth, no per-stage reweighting -- see RESEARCH.md method-derivation
    note Sec 4, "Multi-stage nhưng không feature engineering"). A stage/image
    with zero valid (view,object) pairs contributes nothing (skipped, not
    zero-padded into the stage average).

    last_stage_only: if True, restrict entirely to each model's LAST backbone
    stage (index len(feats)-1, architecture-relative) instead of averaging
    over all stages. This is the one pre-registered fallback carried over
    from E10 (RESEARCH.md Sec 27's limitation #4): the O<->E / E<->nearBG
    invariance was only ever confirmed at the last stage, never at shallower
    ones. Tried for "rank" (also NO-GO, identical pattern to all-stage) --
    kept as an option for other objectives too rather than hard-removed.
    """
    from transfer_attack.regions import pool_regions_batched

    if objective not in ("rank", "geom", "rank_geom"):
        raise ValueError(f"unknown TGA objective {objective!r}")

    stage_pairs = list(zip(feats_cln, feats_adv_views))
    if last_stage_only:
        stage_pairs = stage_pairs[-1:]

    stage_losses: list[Tensor] = []
    for stage_cln, stage_adv in stage_pairs:
        pooled_cln_list = pool_regions_batched(stage_cln[0], region_masks_per_obj)
        stage_total: Tensor | None = None
        stage_count = 0
        for v in range(stage_adv.shape[0]):
            pooled_adv_list = pool_regions_batched(stage_adv[v], region_masks_per_obj)
            for pooled_cln, pooled_adv in zip(pooled_cln_list, pooled_adv_list):
                if pooled_cln is None or pooled_adv is None:
                    continue
                r_cln = tga_relational_vector(pooled_cln)
                r_adv = tga_relational_vector(pooled_adv)
                if objective == "rank":
                    term = tga_rank_loss(r_cln, r_adv, temperature)
                elif objective == "geom":
                    term = tga_geom_loss(r_cln, r_adv)
                else:
                    term = tga_rank_loss(r_cln, r_adv, temperature) + geom_lambda * tga_geom_loss(r_cln, r_adv)
                stage_total = term if stage_total is None else stage_total + term
                stage_count += 1
        if stage_count > 0:
            stage_losses.append(stage_total / stage_count)

    if not stage_losses:
        return feats_cln[0].new_zeros(())
    return sum(stage_losses) / len(stage_losses)
