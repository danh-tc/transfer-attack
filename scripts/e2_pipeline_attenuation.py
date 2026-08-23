#!/usr/bin/env python
"""E2 diagnostic: for the ONE pair of targets that share a backbone family but
differ completely in detector pipeline -- mask_rcnn_swin_t (two-stage:
backbone -> FPN neck -> RPN -> RoI head) vs dino_swin_l (transformer:
backbone -> neck/input-proj -> encoder -> decoder) -- measure clean-vs-adv
distortion at EVERY checkpoint along each model's own pipeline, using the
SAME already-crafted OSFD noise E1 used (no recrafting).

This directly answers E1's anomaly: mask_rcnn_swin_t showed LOWER backbone
distortion than dino_swin_l (E1: cos_dist 0.141 vs 0.225) yet a MUCH larger
mAP drop (72.2% vs 24.1%). If that gap is driven by the backbone alone, it
shouldn't be there -- so this script tracks damage through each model's own
downstream stages to see where the divergence actually appears.

Methodological note per model (read before interpreting the numbers):
  - mask_rcnn_swin_t: backbone/neck/rpn-raw checkpoints are deterministic
    given the input image (no proposal selection involved) so clean vs adv
    are directly comparable position-wise. The RoI checkpoint is NOT --
    RPN proposals selected on the clean image differ from those selected on
    the adv image, which would confound "region changed" with "feature at
    the same region changed". We remove that confound by reusing the SAME
    proposals (selected once, on the CLEAN image) for both the clean and adv
    RoI forward pass -- so the RoI-stage number is a clean "same region,
    different content" comparison.
  - dino_swin_l: backbone/neck/encoder-memory checkpoints are likewise
    deterministic and directly comparable. The decoder checkpoint is
    comparable PER QUERY SLOT (query content starts from the same fixed
    learned embedding at slot i for both clean and adv -- unlike RPN
    proposals, DINO's query content is not selected/gathered) but each
    slot's `reference_points` (which memory region it cross-attends to) ARE
    selected via input-dependent top-k over encoder output, so a slot may
    end up attending to a different region on the adv pass than on the clean
    pass. This is a softer comparison than the RoI fix above -- treat the
    decoder number as a pipeline-level distortion signal, not a strict
    same-region measurement.

Example:
    python scripts/e2_pipeline_attenuation.py --attack osfd --manifest data/manifests/dev_50.json
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["osfd", "mi_fgsm"], default="osfd")
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--noise-dir", type=Path, default=None, help="default: results/noise/<manifest-stem>/<attack>/")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "e2_pipeline_attenuation.csv")
    return p


def cos_dist_and_rel_l2(clean: "torch.Tensor", adv: "torch.Tensor") -> tuple[float, float]:
    import torch.nn.functional as F

    fc = clean.flatten().unsqueeze(0)
    fa = adv.flatten().unsqueeze(0)
    cos_sim = F.cosine_similarity(fc, fa).item()
    rel_l2 = (fa - fc).norm().item() / (fc.norm().item() + 1e-8)
    return 1.0 - cos_sim, rel_l2


def make_data_sample(canvas: int):
    from mmdet.structures import DetDataSample

    ds = DetDataSample()
    ds.set_metainfo(
        dict(
            img_shape=(canvas, canvas),
            ori_shape=(canvas, canvas),
            scale_factor=(1.0, 1.0),
            batch_input_shape=(canvas, canvas),
        )
    )
    return ds


def measure_mask_rcnn(model, handle, x_clean, x_adv, ds, device: str) -> dict:
    """Returns {checkpoint_name: (cos_dist, rel_l2)} for one image."""
    import torch
    from mmdet.structures.bbox import bbox2roi

    out = {}
    with torch.no_grad():
        feats_c = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
        feats_a = model.backbone(handle.normalize(x_adv.unsqueeze(0)))
        out["1_backbone"] = cos_dist_and_rel_l2(
            torch.cat([f.flatten() for f in feats_c]), torch.cat([f.flatten() for f in feats_a])
        )

        neck_c = model.neck(feats_c)
        neck_a = model.neck(feats_a)
        out["2_neck_fpn"] = cos_dist_and_rel_l2(
            torch.cat([f.flatten() for f in neck_c]), torch.cat([f.flatten() for f in neck_a])
        )

        rpn_cls_c, rpn_bbox_c = model.rpn_head(neck_c)
        rpn_cls_a, rpn_bbox_a = model.rpn_head(neck_a)
        rpn_flat_c = torch.cat([t.flatten() for t in (*rpn_cls_c, *rpn_bbox_c)])
        rpn_flat_a = torch.cat([t.flatten() for t in (*rpn_cls_a, *rpn_bbox_a)])
        out["3_rpn_raw"] = cos_dist_and_rel_l2(rpn_flat_c, rpn_flat_a)

        # Fixed proposals: selected ONCE on the clean neck features, reused
        # for both the clean and adv RoI forward pass (see module docstring).
        rpn_results_list = model.rpn_head.predict(neck_c, [ds], rescale=False)
        proposals = rpn_results_list[0].bboxes
        if proposals.shape[0] == 0:
            return out  # nothing to pool RoI features from for this image
        rois = bbox2roi([proposals]).to(device)

        bbox_c = model.roi_head._bbox_forward(neck_c, rois)
        bbox_a = model.roi_head._bbox_forward(neck_a, rois)
        out["4_roi_pooled_feats"] = cos_dist_and_rel_l2(bbox_c["bbox_feats"], bbox_a["bbox_feats"])
        out["5_roi_head_output"] = cos_dist_and_rel_l2(
            torch.cat([bbox_c["cls_score"].flatten(), bbox_c["bbox_pred"].flatten()]),
            torch.cat([bbox_a["cls_score"].flatten(), bbox_a["bbox_pred"].flatten()]),
        )
    return out


def measure_dino(model, handle, x_clean, x_adv, ds, device: str) -> dict:
    import torch

    out = {}
    with torch.no_grad():
        feats_c = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
        feats_a = model.backbone(handle.normalize(x_adv.unsqueeze(0)))
        out["1_backbone"] = cos_dist_and_rel_l2(
            torch.cat([f.flatten() for f in feats_c]), torch.cat([f.flatten() for f in feats_a])
        )

        neck_c = model.neck(feats_c) if model.with_neck else feats_c
        neck_a = model.neck(feats_a) if model.with_neck else feats_a
        out["2_neck_input_proj"] = cos_dist_and_rel_l2(
            torch.cat([f.flatten() for f in neck_c]), torch.cat([f.flatten() for f in neck_a])
        )

        enc_in_c, dec_in_c = model.pre_transformer(neck_c, [ds])
        enc_in_a, dec_in_a = model.pre_transformer(neck_a, [ds])
        enc_out_c = model.forward_encoder(**enc_in_c)
        enc_out_a = model.forward_encoder(**enc_in_a)
        out["3_encoder_memory"] = cos_dist_and_rel_l2(enc_out_c["memory"], enc_out_a["memory"])

        dec_in2_c, _ = model.pre_decoder(**enc_out_c)
        dec_in2_a, _ = model.pre_decoder(**enc_out_a)
        dec_out_c = model.forward_decoder(**{**dec_in_c, **dec_in2_c})
        dec_out_a = model.forward_decoder(**{**dec_in_a, **dec_in2_a})
        # last decoder layer only
        out["4_decoder_hidden_states"] = cos_dist_and_rel_l2(
            dec_out_c["hidden_states"][-1], dec_out_a["hidden_states"][-1]
        )
    return out


MEASURERS = {
    "mask_rcnn_swin_t": measure_mask_rcnn,
    "dino_swin_l": measure_dino,
}


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, load_noise
    from transfer_attack.models import get_spec, build_model_handle

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    noise_dir = args.noise_dir or (PROJECT_DIR / "results" / "noise" / args.manifest.stem / args.attack)
    if not noise_dir.exists():
        logger.error(f"noise dir not found: {noise_dir} -- craft it first (see scripts/craft.py)")
        sys.exit(1)

    img_dir = args.data_dir / "val2017"
    device = args.device
    ds = make_data_sample(args.canvas)

    detail_rows = []

    for model_name, measure_fn in MEASURERS.items():
        spec = get_spec(model_name)
        logger.info(f"=== {spec.name} (group={spec.group}) ===")
        handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)
        model = handle.model

        # checkpoint_name -> list of (cos_dist, rel_l2) across images
        per_checkpoint: dict[str, list[tuple[float, float]]] = {}
        n_images = 0
        n_missing_noise = 0

        for image_id in image_ids:
            noise_path = noise_dir / f"{image_id}.pt"
            if not noise_path.exists():
                n_missing_noise += 1
                continue

            canvas_img, _, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
            noise = load_noise(noise_path)
            x_clean = canvas_img.to(device)
            x_adv = (canvas_img + noise).clamp(0.0, 255.0).to(device)

            result = measure_fn(model, handle, x_clean, x_adv, ds, device)
            for ckpt_name, (cd, rl) in result.items():
                per_checkpoint.setdefault(ckpt_name, []).append((cd, rl))
            n_images += 1

        if n_missing_noise:
            logger.warning(f"  {n_missing_noise} images had no crafted noise under {noise_dir} -- skipped")

        for ckpt_name in sorted(per_checkpoint.keys()):
            values = per_checkpoint[ckpt_name]
            mean_cd = sum(v[0] for v in values) / len(values)
            mean_rl = sum(v[1] for v in values) / len(values)
            detail_rows.append(
                {
                    "model_name": spec.name,
                    "group": spec.group or "",
                    "checkpoint": ckpt_name,
                    "n_images": len(values),
                    "mean_cos_dist": mean_cd,
                    "mean_rel_l2": mean_rl,
                }
            )
            logger.info(f"  {ckpt_name:28s} n={len(values):3d} mean_cos_dist={mean_cd:.4f} mean_rel_l2={mean_rl:.4f}")

        del handle, model
        torch.cuda.empty_cache()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model_name", "group", "checkpoint", "n_images", "mean_cos_dist", "mean_rel_l2"]
        )
        writer.writeheader()
        writer.writerows(detail_rows)
    logger.info(f"wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
