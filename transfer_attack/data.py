"""Manifest / COCO ground-truth loading and the fixed-canvas image pipeline.

Every image (surrogate crafting + all 7 models' clean/adversarial eval) is
resized keep-ratio then zero-padded bottom/right to a single square canvas
(CANVAS x CANVAS, see constants.py) so every model sees a bit-identical
perturbation -- see plan doc section 0 for the rationale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.nn import functional as F

from transfer_attack.constants import CANVAS


def load_manifest(path: Path) -> dict:
    with open(path, "r") as f:
        manifest = json.load(f)
    assert len(manifest["image_ids"]) == manifest["size"], (
        f"{path}: manifest size field ({manifest['size']}) does not match "
        f"len(image_ids) ({len(manifest['image_ids'])})"
    )
    return manifest


def load_coco(ann_file: Path) -> COCO:
    return COCO(str(ann_file))


def build_gt_index(coco: COCO, image_ids: list[int]) -> dict[int, list[dict]]:
    """Per image id: list of {"bbox_xyxy_orig": [x1,y1,x2,y2], "cat_id": int},
    dropping iscrowd=1 annotations and degenerate (zero-area) boxes."""
    index: dict[int, list[dict]] = {}
    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        entries = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            entries.append({"bbox_xyxy_orig": [x, y, x + w, y + h], "cat_id": ann["category_id"]})
        index[image_id] = entries
    return index


def load_canvas_image(
    img_dir: Path, coco: COCO, image_id: int, canvas: int = CANVAS
) -> tuple[Tensor, float, tuple[int, int], tuple[int, int]]:
    """Loads and resize+pads one image onto a `canvas x canvas` square.

    Returns (image[3,canvas,canvas] float32 RGB [0,255], scale, (pad_w,pad_h), (W0,H0)).
    `scale` maps original-image coordinates to canvas coordinates (padding is
    bottom/right only, so no coordinate-origin shift is needed).
    """
    info = coco.loadImgs([image_id])[0]
    file_path = img_dir / info["file_name"]
    img = Image.open(file_path).convert("RGB")
    w0, h0 = img.size
    scale = min(canvas / w0, canvas / h0)
    new_w, new_h = max(1, round(w0 * scale)), max(1, round(h0 * scale))

    arr = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)  # (1,3,H0,W0)
    resized = F.interpolate(arr, size=(new_h, new_w), mode="bilinear", align_corners=True)

    canvas_img = torch.zeros((1, 3, canvas, canvas), dtype=torch.float32)
    canvas_img[:, :, :new_h, :new_w] = resized
    pad_w, pad_h = canvas - new_w, canvas - new_h

    return canvas_img.squeeze(0), scale, (pad_w, pad_h), (w0, h0)


def gt_to_canvas(gt_entries: list[dict], scale: float) -> tuple[Tensor, Tensor]:
    """Converts a per-image GT-entry list (from build_gt_index) to canvas-space
    (boxes[M,4] xyxy, cat_ids[M] long) tensors. cat_ids are raw COCO category
    ids here; callers map to per-model contiguous labels via cat_id_to_label."""
    if not gt_entries:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    boxes = torch.tensor([e["bbox_xyxy_orig"] for e in gt_entries], dtype=torch.float32) * scale
    cat_ids = torch.tensor([e["cat_id"] for e in gt_entries], dtype=torch.long)
    return boxes, cat_ids


def boxes_canvas_to_original(boxes: Tensor, scale: float) -> Tensor:
    """Inverse of gt_to_canvas's box scaling (padding is bottom/right only, so
    no offset needed) -- used when emitting COCO-format results."""
    return boxes / scale
