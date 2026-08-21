#!/usr/bin/env python
"""Visual sanity check: run clean predict_canvas() on a few dev_300 images for
one or more models and save the image with predicted boxes drawn on top, so
you can eyeball whether boxes land on real objects with plausible confidences
BEFORE trusting any crafted noise or mAP/ASR numbers.

This is the manual verification step called out as risk #2 in the plan
(normalize.py's mean/std/bgr_to_rgb handling was derived by symmetry with the
reference repo, not yet exercised against a live checkpoint).

Example:
    python scripts/sanity_check.py --models faster_rcnn_r50 --n 5 --out-dir results/sanity
    python scripts/sanity_check.py --models all --n 3 --score-thr 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["all"], help="model name(s) from MODEL_REGISTRY, or 'all'")
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_300.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "sanity")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--n", type=int, default=5, help="number of images to draw per model")
    p.add_argument("--score-thr", type=float, default=0.3, help="only draw boxes above this score")
    p.add_argument("--device", type=str, default="cuda:0")
    return p


def draw_predictions(canvas_img, pred: dict, cat_id_to_name: dict, score_thr: float):
    """canvas_img: (3,canvas,canvas) float32 [0,255] RGB tensor. Returns a PIL Image."""
    from PIL import Image, ImageDraw

    import numpy as np

    arr = canvas_img.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(np.ascontiguousarray(arr), mode="RGB")
    draw = ImageDraw.Draw(img)

    boxes, scores, cat_ids = pred["bboxes"], pred["scores"], pred["labels"]
    for box, score, cat_id in zip(boxes.tolist(), scores.tolist(), cat_ids.tolist()):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        name = cat_id_to_name.get(int(cat_id), f"cat{cat_id}")
        draw.text((x1 + 2, max(0, y1 - 12)), f"{name} {score:.2f}", fill=(255, 0, 0))
    return img


def main() -> None:
    args = build_arg_parser().parse_args()

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import load_canvas_image, load_coco, load_manifest
    from transfer_attack.eval_metrics import predict_canvas
    from transfer_attack.io_utils import get_logger
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    image_ids = load_manifest(args.manifest)["image_ids"][: args.n]
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    img_dir = args.data_dir / "val2017"

    if args.models == ["all"]:
        specs = MODEL_REGISTRY
    else:
        by_name = {s.name: s for s in MODEL_REGISTRY}
        specs = [by_name[m] for m in args.models]

    for spec in specs:
        logger.info(f"=== {spec.name} ===")
        handle = build_model_handle(spec, args.checkpoints_dir, device=args.device, coco=coco)
        out_dir = args.out_dir / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for image_id in image_ids:
            canvas_img, _, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
            pred = predict_canvas(handle, canvas_img, args.canvas, device=args.device)
            n_above_thr = int((pred["scores"] >= args.score_thr).sum())
            img = draw_predictions(canvas_img, pred, cat_id_to_name, args.score_thr)
            out_path = out_dir / f"{image_id}.png"
            img.save(out_path)
            logger.info(f"  image_id={image_id}: {n_above_thr} boxes >= {args.score_thr} -> {out_path}")

    logger.info(f"done -- open the PNGs under {args.out_dir} and eyeball them")


if __name__ == "__main__":
    main()
