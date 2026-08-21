"""Per-model image normalization derived from each mmdet config's data_preprocessor.

Our canonical pixel-space image is always RGB, float32, range [0,255] (matching
how the original OSFD repo separates the pixel-space "clean image + noise"
being optimized from the normalized tensor fed to the backbone). This mirrors
reference-repo/OSFD/attack/utils/mmdet.py::imnormalize/imdenormalize, where
their `to_rgb` flag is the same concept as mmdet 3.x's `bgr_to_rgb`.

We read mean/std/bgr_to_rgb from the *config* (a plain dict), not from the live
ImgDataPreprocessor module's buffers, to avoid depending on that module's
private attribute names/shapes across mmdet versions.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def build_normalizer(model, device: str) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Returns (normalize, denormalize) closures operating on (N,3,H,W) tensors.

    normalize:   canonical RGB [0,255] pixel-space -> model input tensor
    denormalize: model input tensor -> canonical RGB [0,255] pixel-space
    """
    dp_cfg = model.cfg.model.data_preprocessor
    mean = torch.tensor(dp_cfg.get("mean", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
    std = torch.tensor(dp_cfg.get("std", [1.0, 1.0, 1.0]), dtype=torch.float32, device=device)
    bgr_to_rgb = bool(dp_cfg.get("bgr_to_rgb", False))
    mean = mean[..., None, None]
    std = std[..., None, None]

    def _flip_channels(x: Tensor) -> Tensor:
        # x is (N,3,H,W); swap R<->B.
        return x[:, [2, 1, 0], ...]

    def normalize(x_rgb_0_255: Tensor) -> Tensor:
        x = x_rgb_0_255
        if bgr_to_rgb:
            # config's mean/std are expressed in RGB order already -> no flip needed,
            # our canonical image is already RGB.
            pass
        else:
            # config's mean/std are expressed in BGR order (caffe-style configs) ->
            # flip our RGB image to BGR before applying them.
            x = _flip_channels(x)
        return (x - mean) / std

    def denormalize(x_model_input: Tensor) -> Tensor:
        x = x_model_input * std + mean
        if not bgr_to_rgb:
            # x is currently BGR -> flip back to canonical RGB.
            x = _flip_channels(x)
        return x

    return normalize, denormalize
