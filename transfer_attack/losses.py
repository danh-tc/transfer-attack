"""OSFD backbone-feature loss and the MI-FGSM baseline's output/task loss."""
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
