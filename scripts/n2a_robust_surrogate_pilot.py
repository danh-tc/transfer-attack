#!/usr/bin/env python
"""Phase N2-A pilot (cheap control, not the main contribution): does crafting
OSFD against a surrogate whose BACKBONE is loaded from a publicly-available
ADVERSARIALLY-ROBUST ImageNet-pretrained ResNet-50 (Salman et al. 2020,
Linf eps=4/255, github.com/MadryLab/robustness) produce noise that transfers
better to the hard targets than the standard (non-robust) surrogate?

This is the classification-literature finding "robust surrogates craft more
transferable adversarial examples" (well-established for CNN->ViT transfer in
classification), ported to OD as a diagnostic, not a designed mechanism --
if it works, it's evidence the SURROGATE REPRESENTATION QUALITY is a real
bottleneck (which would also support N2-B); if it doesn't, that's a useful
negative finding too. Not intended as a standalone contribution on its own
("use a robust ResNet for OSFD" has weak novelty by itself).

Mechanism: identical craft loop to plain OSFD (transfer_attack.attack.
craft_one_image, unmodified -- k=3, RRB on, same epsilon/alpha/steps) -- the
ONLY change is which backbone weights compute feats_cln/feats_adv during
crafting. The robust checkpoint's state dict (prefix "module.model.") maps
1:1 onto mmdet's ResNet backbone key names (verified: 318/318 keys, 0
missing) -- only the BACKBONE is swapped for crafting; neck/rpn_head/roi_head
are irrelevant here since OSFD's loss never touches them. Evaluation (incl.
the surrogate row) always uses the STANDARD, unmodified faster_rcnn_r50
checkpoint from MODEL_REGISTRY -- exactly like every other target -- so the
"surrogate ASR" number means the same thing as in every other run (does this
noise fool a normal faster_rcnn_r50), not a mismatched robust-backbone/
COCO-head Frankenstein model.

Compare directly against N2-B's `osfd` baseline row (same manifest, seed,
N=20, steps=100, k=3, RRB on) -- no need to re-run it here.

Example:
    python scripts/n2a_robust_surrogate_pilot.py --manifest data/manifests/dev_50.json --n-images 20
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_ROBUST_CKPT = Path(
    "/tmp/claude-0/-workspace/db032472-a127-4e0f-ac2c-3d65c9b9e5a5/scratchpad/robust_ckpt/imagenet_linf_4.pt"
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--robust-ckpt", type=Path, default=DEFAULT_ROBUST_CKPT)
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "n2a_robust_surrogate_summary.csv")
    p.add_argument(
        "--models", nargs="+", default=["faster_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"],
    )
    return p


def build_robust_backbone_handle(checkpoints_dir: Path, robust_ckpt_path: Path, device: str, coco):
    import torch

    from transfer_attack.models import build_model_handle, get_spec

    spec = get_spec("faster_rcnn_r50")
    handle = build_model_handle(spec, checkpoints_dir, device=device, coco=coco)

    ckpt = torch.load(robust_ckpt_path, map_location="cpu")
    sd = ckpt["model"]
    prefix = "module.model."
    remapped = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    result = handle.model.backbone.load_state_dict(remapped, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"robust checkpoint missing backbone keys: {result.missing_keys}")
    handle.model.backbone.to(device)
    handle.model.backbone.eval()
    return handle


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch
    import evaluate as evaluate_mod

    from transfer_attack.attack import AttackConfig, craft_one_image
    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, save_noise, save_run_log
    from transfer_attack.models import MODEL_REGISTRY

    logger = get_logger()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"

    cfg = AttackConfig(attack_type="osfd", k=3.0, use_rrb=True, steps=args.steps, canvas=args.canvas)

    noise_dir = PROJECT_DIR / "results" / "noise" / args.manifest.stem / "n2a_robust"
    noise_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"loading robust backbone from {args.robust_ckpt}")
    robust_handle = build_robust_backbone_handle(args.checkpoints_dir, args.robust_ckpt, args.device, coco)
    logger.info("robust backbone loaded (318/318 keys matched, 0 missing) -- crafting with it now")

    n_crafted, n_skipped = 0, 0
    t0 = time.time()
    used_image_ids = []
    for image_id in image_ids:
        if len(used_image_ids) >= args.n_images:
            break
        gt_entries = gt_index[image_id]
        if not gt_entries:
            n_skipped += 1
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, cfg.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        noise, step_losses = craft_one_image(robust_handle, canvas_img, gt_boxes, gt_cat_ids, cfg, device=args.device)
        save_noise(noise_dir / f"{image_id}.pt", noise)
        used_image_ids.append(image_id)
        n_crafted += 1
        if n_crafted % args.log_every == 0:
            logger.info(f"[{n_crafted}/{args.n_images}] elapsed={time.time() - t0:.1f}s loss[-1]={step_losses[-1]:.4f}")
    craft_elapsed = time.time() - t0
    logger.info(f"craft finished: {n_crafted} crafted, {n_skipped} skipped -> {noise_dir}")
    del robust_handle
    torch.cuda.empty_cache()

    from types import SimpleNamespace

    eval_args = SimpleNamespace(
        checkpoints_dir=args.checkpoints_dir,
        canvas=args.canvas,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        noise_dir=PROJECT_DIR / "results" / "noise" / args.manifest.stem,
        predictions_dir=PROJECT_DIR / "results" / "_n2a_predictions",
        force_clean=False,
        device=args.device,
        attacks=["n2a_robust"],
    )

    specs = MODEL_REGISTRY
    if args.models:
        by_name = {s.name: s for s in MODEL_REGISTRY}
        specs = [by_name[m] for m in args.models]

    t_eval0 = time.time()
    gt_cache = evaluate_mod.build_gt_cache(coco, used_image_ids, img_dir, args.canvas)
    rows = []
    for spec in specs:
        rows.extend(evaluate_mod.evaluate_one_model(spec, eval_args, coco, used_image_ids, img_dir, gt_cache, logger))
    eval_elapsed = time.time() - t_eval0
    logger.info(f"eval finished {len(specs)} models in {eval_elapsed:.1f}s")

    run_log_path = save_run_log(
        args.runs_dir,
        "run_attack",
        f"osfd_n2a_robust_{args.manifest.stem}_n{args.n_images}",
        {
            "attack": "n2a_robust",
            "manifest": str(args.manifest),
            "seed": args.seed,
            "n_images": args.n_images,
            "score_thr": args.score_thr,
            "iou_thr": args.iou_thr,
            "config": {**vars(cfg), "robust_ckpt": str(args.robust_ckpt)},
            "results": {
                "n_images_used": len(used_image_ids),
                "n_crafted": n_crafted,
                "n_skipped": n_skipped,
                "craft_elapsed_sec": round(craft_elapsed, 1),
                "eval_elapsed_sec": round(eval_elapsed, 1),
                "rows": [r for r in rows if r["attack"] == "n2a_robust"],
            },
        },
    )
    logger.info(f"run log written -> {run_log_path}")

    import csv

    out_rows = [r for r in rows if r["attack"] == "n2a_robust"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "model_name", "group", "mAP_clean", "mAP_adv", "mAP_drop_pct", "AP50_clean", "AP50_adv", "ASR"])
        writer.writeheader()
        writer.writerows(out_rows)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info("=== N2-A: robust-surrogate-crafted OSFD, ASR (%) ===")
    for r in out_rows:
        logger.info(f"  {r['model_name']:20s} {r.get('group') or '-':4s} ASR={r['ASR']:.1f} mAP_drop={r['mAP_drop_pct']:.1f}")


if __name__ == "__main__":
    main()
