#!/usr/bin/env python
"""Equivalence check for the two perf optimizations already baked into
scripts/e10_relational_geometry.py relative to the naive pilot implementation
it replaced (see that file's FROZEN SPEC comment + HANDOFF.md "Engineering
notes" for the two optimizations):

  1. Loop-order: build_region_masks(...) called ONCE per image (cached in
     `image_data`) and reused across all models, vs the naive pilot which
     rebuilt it once per (model, image) -- 6x redundant calls for 6 models.
  2. pool_regions_batched(...): one F.interpolate call per (stage, image)
     covering all objects x all 4 regions at once, vs pool_regions_one_stage
     called once per (object, region) in a Python loop.

Both are claimed to be *pure engineering* changes with NO effect on the
numeric output (mask construction is a deterministic function of GT boxes
only, independent of model/loop-order; F.interpolate(mode="area") treats each
batch element independently, so batching does not change any individual
element's result). This script proves that empirically rather than trusting
the docstring claim: it recomputes region masks and pooled relational vectors
two ways on the SAME loaded features (no RNG, no noise-crafting involved) and
reports the max absolute difference.

Does not touch or reinterpret the FROZEN SPEC -- same shrink/expand fracs,
same region definitions, same relational signature -- only compares two code
paths against each other on identical inputs.

Example:
    python scripts/e10_equivalence_check.py --manifest data/manifests/dev_50.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import e10_relational_geometry as e10  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--models", nargs="+", default=["faster_rcnn_r50", "mask_rcnn_swin_t"])
    p.add_argument("--shrink-frac", type=float, default=0.125)
    p.add_argument("--expand-frac", type=float, default=0.25)
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tol", type=float, default=1e-5)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger
    from transfer_attack.models import build_model_handle, get_spec

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"
    device = args.device

    # ---- Check 1: mask-building is loop-order independent -------------------
    # Build once "up front" (as the optimized script does) vs rebuild fresh
    # per (fake) model iteration -- must be bit-identical since it's a pure
    # function of (gt_boxes, canvas, shrink_frac, expand_frac) only.
    mask_mismatches = 0
    n_images_checked = 0
    per_image_masks: dict[int, list] = {}
    for image_id in image_ids:
        gt_entries = gt_index[image_id]
        if not gt_entries:
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        if gt_boxes.shape[0] == 0:
            continue
        masks_once = e10.build_region_masks(gt_boxes, args.canvas, args.shrink_frac, args.expand_frac)
        # simulate "rebuild per model" (naive pilot behavior) by calling again
        masks_rebuilt = e10.build_region_masks(gt_boxes, args.canvas, args.shrink_frac, args.expand_frac)
        n_images_checked += 1
        for m1, m2 in zip(masks_once, masks_rebuilt):
            if (m1 is None) != (m2 is None):
                mask_mismatches += 1
                continue
            if m1 is None:
                continue
            for name in ("O", "E", "Bn", "Bf"):
                if not torch.equal(m1[name], m2[name]):
                    mask_mismatches += 1
        per_image_masks[image_id] = masks_once
        canvas_img_cache = per_image_masks.setdefault("_canvas_cache", {})
        canvas_img_cache[image_id] = (canvas_img, gt_boxes, gt_cat_ids)
    canvas_cache = per_image_masks.pop("_canvas_cache")
    logger.info(
        f"[check 1: mask loop-order] {n_images_checked} images, region-mask mismatches (rebuild vs cached) = {mask_mismatches} "
        f"(expect exactly 0 -- pure function of GT boxes)"
    )

    # ---- Check 2: pool_regions_batched == per-object pool_regions_one_stage -
    max_abs_diff = 0.0
    n_compared = 0
    n_pooled_mismatch_none = 0
    for model_name in args.models:
        spec = get_spec(model_name)
        logger.info(f"=== equivalence check on {model_name} ===")
        handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)
        model = handle.model

        for image_id, region_masks_per_obj in per_image_masks.items():
            canvas_img, gt_boxes, gt_cat_ids = canvas_cache[image_id]
            x = canvas_img.to(device)
            with torch.no_grad():
                feats = model.backbone(handle.normalize(x.unsqueeze(0)))
            feats = [f[0] for f in feats]

            for stage_idx, feat in enumerate(feats):
                batched_out = e10.pool_regions_batched(feat, region_masks_per_obj)
                for obj_idx, rm in enumerate(region_masks_per_obj):
                    naive_out = e10.pool_regions_one_stage(feat, rm) if rm is not None else None
                    b_out = batched_out[obj_idx]
                    if (naive_out is None) != (b_out is None):
                        n_pooled_mismatch_none += 1
                        continue
                    if naive_out is None:
                        continue
                    naive_rel = e10.relational_vector(naive_out)
                    batched_rel = e10.relational_vector(b_out)
                    for rel in e10.RELATIONS:
                        diff = abs(naive_rel[rel] - batched_rel[rel])
                        max_abs_diff = max(max_abs_diff, diff)
                        n_compared += 1

        del handle, model
        torch.cuda.empty_cache()

    logger.info(
        f"[check 2: batched vs per-object pooling] {n_compared} relation-values compared across "
        f"{len(args.models)} model(s), max_abs_diff = {max_abs_diff:.3e} (tol={args.tol:.0e}), "
        f"none-mismatches = {n_pooled_mismatch_none} (expect 0)"
    )

    ok = mask_mismatches == 0 and n_pooled_mismatch_none == 0 and max_abs_diff < args.tol
    logger.info(f"===== EQUIVALENCE CHECK: {'PASS' if ok else 'FAIL'} =====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
