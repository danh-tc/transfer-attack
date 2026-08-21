#!/usr/bin/env python
"""Environment/asset sanity check. Name and location are fixed -- setup_env.sh
invokes this script directly by path at the end of its bootstrap sequence.

Fast checks only by default (no full model instantiation, since this runs at
the tail of a multi-minute bootstrap). Pass --full to additionally build the
surrogate model and run one prediction, as a first real signal that the
mmdet-3.x model-construction/inference path works end to end.

Exits non-zero on any hard failure so setup_env.sh's `set -e` aborts here with
a clear message rather than failing mysteriously later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--full", action="store_true", help="also build the surrogate model and predict once")
    args = parser.parse_args()

    project_dir = args.project_dir
    checkpoints_dir = project_dir / "checkpoints"
    manifests_dir = project_dir / "data" / "manifests"

    # 1. Core imports + versions.
    try:
        import torch
        import mmcv
        import mmengine
        import mmdet
    except ImportError as e:
        fail(f"import failed: {e}")
        return
    print(f"  torch={torch.__version__} mmcv={mmcv.__version__} mmengine={mmengine.__version__} mmdet={mmdet.__version__}")
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
    ok(f"CUDA available: {torch.cuda.get_device_name(0)}")

    # 2. register_all_modules.
    from mmdet.utils import register_all_modules

    register_all_modules()
    ok("register_all_modules()")

    # 3. Checkpoints + config resolution for all 7 registered models.
    from transfer_attack.models import MODEL_REGISTRY, resolve_config_path

    for spec in MODEL_REGISTRY:
        ckpt_path = checkpoints_dir / spec.ckpt_filename
        if not ckpt_path.exists():
            fail(f"missing checkpoint for {spec.name!r}: {ckpt_path}")
        try:
            cfg_path = resolve_config_path(spec.config_name, checkpoints_dir)
        except FileNotFoundError as e:
            fail(f"could not resolve config for {spec.name!r}: {e}")
            return
        ok(f"{spec.name}: ckpt + config resolved ({cfg_path.name})")

    # 4. Manifests.
    for name in ("dev_300.json", "val_100.json"):
        path = manifests_dir / name
        if not path.exists():
            fail(f"missing manifest: {path}")
        with open(path) as f:
            manifest = json.load(f)
        if len(manifest["image_ids"]) != manifest["size"]:
            fail(f"{path}: size field ({manifest['size']}) != len(image_ids) ({len(manifest['image_ids'])})")
        ok(f"{name}: {manifest['size']} image ids")

    # 5. COCO annotations load.
    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import load_coco

    ann_file = project_dir / COCO_ANN_FILE
    if not ann_file.exists():
        fail(f"missing COCO annotation file: {ann_file}")
    coco = load_coco(ann_file)
    ok(f"COCO annotations loaded: {len(coco.getImgIds())} images, {len(coco.getCatIds())} categories")

    if args.full:
        from transfer_attack.constants import CANVAS, COCO_IMG_DIR
        from transfer_attack.data import gt_to_canvas, load_canvas_image
        from transfer_attack.models import build_model_handle, get_spec

        with open(manifests_dir / "dev_300.json") as f:
            dev_ids = json.load(f)["image_ids"]

        spec = get_spec("faster_rcnn_r50")
        handle = build_model_handle(spec, checkpoints_dir, device="cuda:0", coco=coco)
        ok(f"built model handle for {spec.name}")

        img_dir = project_dir / COCO_IMG_DIR
        image_id = dev_ids[0]
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, CANVAS)

        import torch as _torch
        from mmdet.structures import DetDataSample

        with _torch.no_grad():
            x_norm = handle.normalize(canvas_img.unsqueeze(0).to("cuda:0"))
            ds = DetDataSample()
            ds.set_metainfo(dict(img_shape=(CANVAS, CANVAS), ori_shape=(CANVAS, CANVAS), scale_factor=(1.0, 1.0)))
            out = handle.model.predict(x_norm, [ds], rescale=True)[0]
        inst = out.pred_instances
        n_boxes = inst.bboxes.shape[0]
        if n_boxes == 0:
            fail(f"surrogate produced 0 predictions on image_id={image_id} -- normalize/predict path likely broken")
        max_label = int(inst.labels.max().item()) if n_boxes > 0 else -1
        if not (0 <= max_label < 80):
            fail(f"surrogate predicted an out-of-range label id {max_label} (expected 0..79)")
        ok(f"surrogate predict() on image_id={image_id}: {n_boxes} boxes, label range OK")

    print("\n===== check_env.py: ALL CHECKS PASSED =====")


if __name__ == "__main__":
    main()
