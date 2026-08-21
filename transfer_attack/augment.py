"""RRB (Rotation, Resizing, Blurring) data augmentation, ported from
reference-repo/OSFD/attack/base/RRB.py.

Faithful to the *code*, not the paper's figure: the two augmented views are
NOT independent parallel branches. branch1 = rotate(x); branch2 =
resize(branch1) (i.e. rotate-then-resize, applied sequentially on branch1's
own output). Both branches are then concatenated and Gaussian-blurred
together. See plan doc for the full derivation of this discrepancy.

All functions operate on raw pixel-space image tensors in [0,255] (not
normalized).
"""
from __future__ import annotations

import random

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms.functional import rotate

from transfer_attack.constants import L_S, RHO, S_MAX, SIGMA, THETA


def random_axis_rotation(
    imgs: Tensor, gt_boxes: list[Tensor], theta: float = THETA, l_s: int = L_S, prob: float = 1.0
) -> Tensor:
    """imgs: (N,3,H,W). gt_boxes: list of length N, each (Mi,4) xyxy canvas-space
    (may be empty -> falls back to the image-center candidate only)."""
    device = imgs.device
    result_list = []
    for idx in range(imgs.shape[0]):
        input_tensor = imgs[idx].unsqueeze(0)
        if random.random() >= prob:
            result_list.append(input_tensor)
            continue

        boxes = gt_boxes[idx]
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        image_center = torch.tensor([[w // 2, h // 2]], dtype=torch.float32, device=device)
        if boxes.numel() > 0:
            boxes_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            centers = torch.cat([boxes_centers, image_center], dim=0)
        else:
            centers = image_center

        if l_s == 0:
            centers_with_random = centers
        else:
            centers_with_random = centers + torch.randint_like(centers, low=-l_s, high=l_s)
        center_x, center_y = random.choice(centers_with_random)
        angle = random.random() * 2 * theta - theta
        result = rotate(input_tensor, angle, center=[float(center_x), float(center_y)])
        result_list.append(result)
    return torch.cat(result_list, dim=0)


def adaptive_random_resizing(
    imgs: Tensor, gt_boxes: list[Tensor], rho: float = RHO, s_max: float = S_MAX, prob: float = 1.0
) -> Tensor:
    """imgs: (N,3,H,W). gt_boxes: list of length N, each (Mi,4) xyxy canvas-space
    (may be empty -> identity pass-through for that image)."""
    padded_list = []
    for idx in range(imgs.shape[0]):
        input_tensor = imgs[idx].unsqueeze(0)
        boxes = gt_boxes[idx]
        if boxes.numel() == 0 or random.random() >= prob:
            padded_list.append(input_tensor)
            continue

        ori_size_h = input_tensor.shape[2]
        ori_size_w = input_tensor.shape[3]

        random_box_idx = random.randint(0, boxes.shape[0] - 1)
        box = boxes[random_box_idx]
        box_w = float(box[2] - box[0])
        box_h = float(box[3] - box[1])

        scale_h = min(1 + rho * (box_h / ori_size_h), s_max)
        scale_w = min(1 + rho * (box_w / ori_size_w), s_max)
        new_size_h = random.randint(ori_size_h, int(scale_h * ori_size_h))
        new_size_w = random.randint(ori_size_w, int(scale_w * ori_size_w))
        rescaled = F.interpolate(input_tensor, size=(new_size_h, new_size_w), mode="bilinear", align_corners=True)
        rem_h = int(scale_h * ori_size_h) - new_size_h
        rem_w = int(scale_w * ori_size_w) - new_size_w
        pad_left = random.randint(0, rem_w)
        pad_top = random.randint(0, rem_h)
        padded = F.pad(rescaled, (pad_left, rem_w - pad_left, pad_top, rem_h - pad_top), mode="constant", value=0.0)
        padded = F.interpolate(padded, size=(ori_size_h, ori_size_w), mode="bilinear", align_corners=True)
        padded_list.append(padded)
    return torch.cat(padded_list, dim=0)


def gaussian_blur(imgs: Tensor, sigma: float = SIGMA) -> Tensor:
    return torch.clamp(imgs + torch.randn_like(imgs) * sigma, 0.0, 255.0)


def rrb_forward(adv_img: Tensor, gt_boxes_canvas: Tensor, cfg=None) -> Tensor:
    """adv_img: (1,3,H,W). gt_boxes_canvas: (M,4) xyxy canvas-space GT boxes for
    this single image. Returns (2,3,H,W): [0]=rotate-only, [1]=rotate-then-resize,
    both Gaussian-blurred.
    """
    theta = getattr(cfg, "theta", THETA) if cfg is not None else THETA
    l_s = getattr(cfg, "l_s", L_S) if cfg is not None else L_S
    rho = getattr(cfg, "rho", RHO) if cfg is not None else RHO
    s_max = getattr(cfg, "s_max", S_MAX) if cfg is not None else S_MAX
    sigma = getattr(cfg, "sigma", SIGMA) if cfg is not None else SIGMA

    branch1 = random_axis_rotation(adv_img, [gt_boxes_canvas], theta=theta, l_s=l_s)
    branch2 = adaptive_random_resizing(branch1, [gt_boxes_canvas], rho=rho, s_max=s_max)
    combined = torch.cat([branch1, branch2], dim=0)  # (2,3,H,W)
    return gaussian_blur(combined, sigma=sigma)
