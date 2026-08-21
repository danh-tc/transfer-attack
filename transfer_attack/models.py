"""Model registry (surrogate + 6 targets, verbatim from setup_env.sh) and
construction helpers using mmdet 3.x's official APIs directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from torch import Tensor, nn

from transfer_attack.normalize import build_normalizer


@dataclass(frozen=True)
class ModelSpec:
    name: str            # short id used throughout results/, e.g. "faster_rcnn_r50"
    role: str             # "surrogate" | "target"
    group: str | None     # "A" | "B" | "C" | None (surrogate)
    config_name: str      # mmdet config name, e.g. "faster-rcnn_r50_fpn_1x_coco"
    ckpt_filename: str    # checkpoint filename under checkpoints/


# Copied verbatim from setup_env.sh's `download_ckpt` calls.
MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec(
        name="faster_rcnn_r50",
        role="surrogate",
        group=None,
        config_name="faster-rcnn_r50_fpn_1x_coco",
        ckpt_filename="faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    ),
    # Group A -- ResNet-50 backbone, same family as the surrogate.
    ModelSpec(
        name="fcos_r50",
        role="target",
        group="A",
        config_name="fcos_r50-caffe_fpn_gn-head_1x_coco",
        ckpt_filename="fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth",
    ),
    ModelSpec(
        name="deformable_detr",
        role="target",
        group="A",
        config_name="deformable-detr_r50_16xb2-50e_coco",
        ckpt_filename="deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth",
    ),
    # Group B -- non-ResNet CNN backbone.
    ModelSpec(
        name="yolov3_d53",
        role="target",
        group="B",
        config_name="yolov3_d53_mstrain-608_273e_coco",
        ckpt_filename="yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth",
    ),
    ModelSpec(
        name="yolox_l",
        role="target",
        group="B",
        config_name="yolox_l_8x8_300e_coco",
        ckpt_filename="yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
    ),
    # Group C -- Swin Transformer backbone.
    ModelSpec(
        name="mask_rcnn_swin_t",
        role="target",
        group="C",
        config_name="mask-rcnn_swin-t-p4-w7_fpn_1x_coco",
        ckpt_filename="mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth",
    ),
    ModelSpec(
        name="dino_swin_l",
        role="target",
        group="C",
        config_name="dino-5scale_swin-l_8xb2-12e_coco",
        ckpt_filename="dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth",
    ),
]


def get_spec(name: str) -> ModelSpec:
    for spec in MODEL_REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown model name {name!r}; known: {[s.name for s in MODEL_REGISTRY]}")


def resolve_config_path(config_name: str, checkpoints_dir: Path) -> Path:
    """Locate the .py config file for `config_name`.

    Primary strategy: recursive glob of the *installed* mmdet package's bundled
    configs (site-packages/mmdet/.mim/configs/**/{config_name}.py) -- this is
    how `mim` itself resolves named configs, and it correctly follows that
    config's own `_base_` chain since it lives inside the real configs/ tree.

    Fallback: checkpoints_dir/{config_name}.py, which `mim download --dest`
    also writes as a side effect for some mim/mmdet version combinations.
    """
    import mmdet

    mmdet_pkg_dir = Path(mmdet.__file__).resolve().parent
    candidates = list(mmdet_pkg_dir.glob(f".mim/configs/**/{config_name}.py"))
    if not candidates:
        # Some installs vendor configs directly under mmdet/configs/ instead of .mim/.
        candidates = list(mmdet_pkg_dir.glob(f"configs/**/{config_name}.py"))
    if candidates:
        return candidates[0]

    fallback = checkpoints_dir / f"{config_name}.py"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not resolve config {config_name!r} under {mmdet_pkg_dir} "
        f"(.mim/configs or configs) nor at fallback {fallback}."
    )


@dataclass
class ModelHandle:
    name: str
    model: nn.Module
    normalize: Callable[[Tensor], Tensor]
    denormalize: Callable[[Tensor], Tensor]
    cat_id_to_label: dict[int, int]
    label_to_cat_id: dict[int, int]


def build_model_handle(
    spec: ModelSpec,
    checkpoints_dir: Path,
    device: str = "cuda:0",
    coco=None,
) -> ModelHandle:
    """Build+load one detector via mmdet.apis.init_detector and wrap it with
    the per-model normalizer and COCO category-id<->label mapping.

    `register_all_modules()` must already have been called once per process
    before this is invoked.
    """
    from mmdet.apis import init_detector

    config_path = resolve_config_path(spec.config_name, checkpoints_dir)
    ckpt_path = checkpoints_dir / spec.ckpt_filename
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {spec.name!r}: {ckpt_path}")

    model = init_detector(str(config_path), str(ckpt_path), device=device)
    model.eval()

    normalize, denormalize = build_normalizer(model, device)

    classes = tuple(model.dataset_meta["classes"])
    cat_id_to_label: dict[int, int] = {}
    label_to_cat_id: dict[int, int] = {}
    if coco is not None:
        # Some older mmdetection model-zoo checkpoints (pre-3.x) carry legacy
        # class names with underscores in place of spaces for multi-word COCO
        # categories (e.g. "traffic_light" vs. the COCO JSON's "traffic light").
        # pycocotools.getCatIds() does exact string matching, so normalize before
        # looking up -- verified this preserves label order (it's a naming quirk,
        # not a class-set/order mismatch).
        query_names = [c.replace("_", " ") for c in classes]
        cat_ids = coco.getCatIds(catNms=query_names)
        if len(cat_ids) != len(classes):
            raise RuntimeError(
                f"{spec.name}: expected {len(classes)} category ids from COCO for "
                f"{len(classes)} classes, got {len(cat_ids)} -- class-name mismatch "
                f"between model.dataset_meta and the COCO annotation file."
            )
        for label, cat_id in enumerate(cat_ids):
            cat_id_to_label[cat_id] = label
            label_to_cat_id[label] = cat_id

    return ModelHandle(
        name=spec.name,
        model=model,
        normalize=normalize,
        denormalize=denormalize,
        cat_id_to_label=cat_id_to_label,
        label_to_cat_id=label_to_cat_id,
    )
