#!/usr/bin/env python
"""Phase M pilot: Model-space View Consistency (MVC) -- candidate mechanism 3
from the E1-E3b diagnostic chain. E3 found INPUT-space view consistency (RRB)
is the dominant transfer driver on groups B/C. This tests the symmetric
hypothesis: does MODEL-space view consistency (multiple lightweight backbone
"variants" via post-hoc channel masking, forced to agree on which features
get suppressed) add anything BEYOND naive model-space ensembling?

Three variants, same budget (k=3, RRB on, epsilon/alpha/steps unchanged):
  osfd      -- baseline, no model-space perturbation at all (n_variants=1).
  mvc_avg   -- N=2 masked backbone-output variants per step, loss = mean of
               the two variants' independent osfd_loss (naive ensemble).
  mvc_cons  -- same as mvc_avg, PLUS an explicit consistency term that
               penalizes disagreement between the two variants' adv feature
               maps (MSE, restricted to channels kept by BOTH variants'
               masks, to avoid the confound of comparing a kept channel in
               one variant against a zeroed one in the other).

Masking mechanics: a forward hook on backbone.layer2 zeroes a random subset
of that layer's OUTPUT channels (keep_prob=0.9, ~10% dropped) before it feeds
into layer3 -- so layer3/layer4 (and hence the returned stage-2/3 features)
are recomputed INDEPENDENTLY per variant, not just re-windowed copies of one
shared tensor. This requires N separate backbone(aug) forward passes per step
(one per variant), not one shared pass with post-hoc output masking -- an
earlier version of this script tried the cheaper post-hoc-masking shortcut
and it was a dead end: at any channel kept by both variants' masks, the value
is IDENTICAL by construction (same underlying tensor, just windowed
differently), so the "disagreement" between variants was mathematically
guaranteed to be ~0 regardless of the consistency term's weight -- caught via
a debug print (`disagree_mse` was exactly 0.000000 at every stage) before
wasting a full run on it. Masking mid-backbone forces genuine divergence
downstream of the branch point. One mask is sampled PER VARIANT PER IMAGE
(not resampled every step -- a variant is one fixed stochastic sub-network
used for that image's whole crafting run) and reused for both the
clean-reference and adv forward passes throughout that image's crafting, so a
variant's loss never mixes "channel was zeroed" with "channel content
changed" across clean vs adv.

GO/NO-GO reading (decided BEFORE running, per the point of this pilot):
  mvc_avg > osfd but mvc_cons ~= mvc_avg   -> gain is just from naive
                                               ensembling (T-SEA/RaPA-like
                                               territory) -> NO-GO on novelty.
  mvc_cons > mvc_avg > osfd, esp. on
  YOLOX-L / mask_rcnn_swin_t / dino_swin_l  -> explicit consistency adds
                                                real value -> worth pursuing.
This pilot is NOT trying to prove MVC beats OSFD outright -- only whether the
consistency term earns its keep over naive model-space ensembling.

Example:
    python scripts/m_mvc_pilot.py --manifest data/manifests/dev_50.json --n-images 20
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

VARIANTS = [
    {"tag": "osfd", "n_variants": 1, "use_consistency": False},
    {"tag": "mvc_avg", "n_variants": 2, "use_consistency": False},
    {"tag": "mvc_cons", "n_variants": 2, "use_consistency": True},
]

KEEP_PROB = 0.9          # light channel masking: drop ~10% of backbone-output channels per variant
# disagree (post-branch MSE, ~0.03-0.04) sits ~300-350x below osfd_loss's own scale (~11.2)
# with the debug image checked -- lambda_cons=100 brings the consistency term to ~24-31% of
# main-loss magnitude: a meaningful but non-dominant regularization strength (not grid-searched).
LAMBDA_CONS = 100.0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-prob", type=float, default=KEEP_PROB)
    p.add_argument("--lambda-cons", type=float, default=LAMBDA_CONS)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "m_mvc_pilot_summary.csv")
    p.add_argument(
        "--run-tags", nargs="+", default=None,
        help="subset of VARIANTS tags to craft+eval (default: all 3). mvc_cons is auto-suffixed "
             "with _lam<N> so different --lambda-cons sanity checks don't clobber each other's noise/run-log.",
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help="subset of MODEL_REGISTRY names to evaluate (default: all 7, incl. surrogate)",
    )
    return p


class _BranchMaskHook:
    """Forward hook on backbone.layer2 (see MASK_LAYER_NAME). When `.mask` is
    set, zeroes the masked-out channels of layer2's output BEFORE it is fed
    into layer3 -- so layer3/layer4 (and therefore the returned stage-2/3
    features) are recomputed independently per variant, not just re-windowed
    copies of one shared tensor. `.mask = None` is a passthrough (used for
    the unmasked `n_variants == 1` / plain-OSFD path, and for the one-time
    channel-count probe forward)."""

    def __init__(self):
        self.mask = None
        self.last_channels = None

    def __call__(self, module, inputs, output):
        self.last_channels = output.shape[1]
        if self.mask is None:
            return output
        return output * self.mask.view(1, -1, 1, 1).to(output.dtype)


MASK_LAYER_NAME = "layer2"  # branch point: layer1 output identical across variants (expected), layer3/layer4 diverge


def craft_one_image_mvc(handle, x_clean, gt_boxes, cfg, n_variants, use_consistency, keep_prob, lambda_cons, device):
    import torch
    import torch.nn.functional as F

    from transfer_attack.augment import rrb_forward
    from transfer_attack.losses import osfd_loss

    model = handle.model
    x_clean = x_clean.to(device)
    gt_boxes = gt_boxes.to(device)

    hook_obj = _BranchMaskHook()
    hook_handle = getattr(model.backbone, MASK_LAYER_NAME).register_forward_hook(hook_obj)
    try:
        if n_variants == 1:
            with torch.no_grad():
                feats_cln_variants = [model.backbone(handle.normalize(x_clean.unsqueeze(0)))]
            masks = [None]
        else:
            # Probe layer2's channel count with one unmasked pass, then sample one
            # fixed random mask PER VARIANT PER IMAGE (not resampled every step --
            # a variant = one fixed stochastic sub-network used for this image's
            # whole crafting run, closer to "model neighborhood" than to RRB's
            # per-step input resampling).
            with torch.no_grad():
                model.backbone(handle.normalize(x_clean.unsqueeze(0)))
            n_channels = hook_obj.last_channels
            masks = [torch.rand(n_channels, device=device) < keep_prob for _ in range(n_variants)]

            feats_cln_variants = []
            for m in masks:
                hook_obj.mask = m
                with torch.no_grad():
                    feats_cln_variants.append(model.backbone(handle.normalize(x_clean.unsqueeze(0))))
            hook_obj.mask = None

        noise = torch.randint_like(x_clean, low=-2, high=3).float()
        g_mom = torch.zeros_like(x_clean)

        for _ in range(cfg.steps):
            noise = noise.detach().requires_grad_(True)
            x_adv = torch.clamp(x_clean + noise, 0.0, 255.0)
            aug = rrb_forward(x_adv.unsqueeze(0), gt_boxes, cfg)  # (2,3,H,W) -- RRB always on, shared across variants

            if n_variants == 1:
                hook_obj.mask = None
                feats_adv_raw = model.backbone(handle.normalize(aug))
                loss = osfd_loss(feats_cln_variants[0], feats_adv_raw, cfg.k)
            else:
                variant_losses = []
                variant_feats_adv = []
                for v in range(n_variants):
                    hook_obj.mask = masks[v]
                    feats_adv_v = model.backbone(handle.normalize(aug))  # genuinely independent forward pass
                    variant_losses.append(osfd_loss(feats_cln_variants[v], feats_adv_v, cfg.k))
                    variant_feats_adv.append(feats_adv_v)
                hook_obj.mask = None
                loss = sum(variant_losses) / n_variants

                if use_consistency:
                    assert n_variants == 2, "consistency term only implemented for N=2 variants"
                    disagree = feats_cln_variants[0][0].new_zeros(())
                    for stage_i in range(len(feats_cln_variants[0])):
                        disagree = disagree + F.mse_loss(variant_feats_adv[0][stage_i], variant_feats_adv[1][stage_i])
                    loss = loss - lambda_cons * (disagree / len(feats_cln_variants[0]))

            loss.backward()
            with torch.no_grad():
                g = noise.grad
                g_mom = cfg.mu * g_mom + g / g.abs().mean(dim=[0, 1, 2], keepdim=True)
                noise = torch.clamp(noise + cfg.alpha * torch.sign(g_mom), -cfg.epsilon, cfg.epsilon)
    finally:
        hook_handle.remove()

    return noise.detach()


def craft_and_evaluate_variant(variant: dict, args, coco, image_ids, gt_index, img_dir, logger) -> list[dict]:
    from types import SimpleNamespace

    import torch
    import evaluate as evaluate_mod

    from transfer_attack.attack import AttackConfig
    from transfer_attack.data import gt_to_canvas, load_canvas_image
    from transfer_attack.io_utils import save_noise, save_run_log
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle, get_spec

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tag = variant["tag"]
    cfg = AttackConfig(attack_type="osfd", k=3.0, use_rrb=True, steps=args.steps, canvas=args.canvas)

    noise_dir = PROJECT_DIR / "results" / "noise" / args.manifest.stem / f"m_{tag}"
    noise_dir.mkdir(parents=True, exist_ok=True)

    surrogate_spec = get_spec("faster_rcnn_r50")
    surrogate = build_model_handle(surrogate_spec, args.checkpoints_dir, device=args.device, coco=coco)
    logger.info(f"[{tag}] craft: n_variants={variant['n_variants']} use_consistency={variant['use_consistency']}")

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
        gt_boxes, _ = gt_to_canvas(gt_entries, scale)
        noise = craft_one_image_mvc(
            surrogate, canvas_img, gt_boxes, cfg,
            variant["n_variants"], variant["use_consistency"], args.keep_prob, args.lambda_cons, args.device,
        )
        save_noise(noise_dir / f"{image_id}.pt", noise)
        used_image_ids.append(image_id)
        n_crafted += 1
        if n_crafted % args.log_every == 0:
            logger.info(f"[{tag}] [{n_crafted}/{args.n_images}] elapsed={time.time() - t0:.1f}s")
    craft_elapsed = time.time() - t0
    logger.info(f"[{tag}] craft finished: {n_crafted} crafted, {n_skipped} skipped -> {noise_dir}")
    del surrogate
    torch.cuda.empty_cache()

    eval_args = SimpleNamespace(
        checkpoints_dir=args.checkpoints_dir,
        canvas=args.canvas,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        noise_dir=PROJECT_DIR / "results" / "noise" / args.manifest.stem,
        predictions_dir=PROJECT_DIR / "results" / "_m_pilot_predictions",
        force_clean=False,
        device=args.device,
        attacks=[f"m_{tag}"],
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
    logger.info(f"[{tag}] eval finished {len(specs)} models in {eval_elapsed:.1f}s")

    run_log_path = save_run_log(
        args.runs_dir,
        "run_attack",
        f"osfd_mvc_{tag}_{args.manifest.stem}_n{args.n_images}",
        {
            "attack": f"m_{tag}",
            "manifest": str(args.manifest),
            "seed": args.seed,
            "n_images": args.n_images,
            "score_thr": args.score_thr,
            "iou_thr": args.iou_thr,
            "config": {**vars(cfg), **variant, "keep_prob": args.keep_prob, "lambda_cons": args.lambda_cons},
            "results": {
                "n_images_used": len(used_image_ids),
                "n_crafted": n_crafted,
                "n_skipped": n_skipped,
                "craft_elapsed_sec": round(craft_elapsed, 1),
                "eval_elapsed_sec": round(eval_elapsed, 1),
                "rows": rows,
            },
        },
    )
    logger.info(f"[{tag}] run log written -> {run_log_path}")
    return [r for r in rows if r["attack"] == f"m_{tag}"]


def main() -> None:
    args = build_arg_parser().parse_args()

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"

    run_variants = [v for v in VARIANTS if not args.run_tags or v["tag"] in args.run_tags]

    all_rows: dict[str, list[dict]] = {}
    for variant in run_variants:
        if variant["tag"] == "mvc_cons":
            # auto-suffix so a different --lambda-cons sanity check doesn't clobber the
            # noise/run-log of a previous mvc_cons run at a different lambda.
            variant = {**variant, "tag": f"mvc_cons_lam{int(args.lambda_cons)}"}
        all_rows[variant["tag"]] = craft_and_evaluate_variant(variant, args, coco, image_ids, gt_index, img_dir, logger)

    extra_tags = [t for t in all_rows if t not in ("osfd", "mvc_avg", "mvc_cons")]
    if extra_tags:
        logger.info(f"=== extra tag(s) not in the standard osfd/mvc_avg/mvc_cons comparison: {extra_tags} ===")
        for t in extra_tags:
            for r in all_rows[t]:
                logger.info(f"  [{t}] {r['model_name']:20s} {r.get('group') or '-':4s} ASR={r['ASR']:.1f} mAP_drop={r['mAP_drop_pct']:.1f}")

    by_model: dict[str, dict[str, dict]] = {}
    for tag, rows in all_rows.items():
        for r in rows:
            by_model.setdefault(r["model_name"], {})[tag] = r

    import csv

    fieldnames = ["model_name", "group", "ASR_osfd", "ASR_mvc_avg", "ASR_mvc_cons",
                  "mAP_drop_osfd", "mAP_drop_mvc_avg", "mAP_drop_mvc_cons",
                  "delta_avg_vs_osfd_ASR", "delta_cons_vs_avg_ASR"]
    comparison = []
    for model_name, per_tag in by_model.items():
        group = next(iter(per_tag.values())).get("group", "")
        row = {"model_name": model_name, "group": group}
        for tag in ("osfd", "mvc_avg", "mvc_cons"):
            r = per_tag.get(tag, {})
            row[f"ASR_{tag}"] = r.get("ASR")
            row[f"mAP_drop_{tag}"] = r.get("mAP_drop_pct")
        if None not in (row["ASR_osfd"], row["ASR_mvc_avg"], row["ASR_mvc_cons"]):
            row["delta_avg_vs_osfd_ASR"] = row["ASR_mvc_avg"] - row["ASR_osfd"]
            row["delta_cons_vs_avg_ASR"] = row["ASR_mvc_cons"] - row["ASR_mvc_avg"]
        comparison.append(row)
    comparison.sort(key=lambda r: (r["group"] or "", r["model_name"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    logger.info(f"wrote -> {args.out_csv}")

    logger.info("=== MVC pilot: ASR (%) osfd -> mvc_avg -> mvc_cons ===")
    for r in comparison:
        logger.info(
            f"{r['model_name']:20s} {r['group'] or '-':4s} "
            f"osfd={r['ASR_osfd']:.1f} avg={r['ASR_mvc_avg']:.1f} cons={r['ASR_mvc_cons']:.1f}  "
            f"(avg-osfd={r.get('delta_avg_vs_osfd_ASR', float('nan')):+.1f}, "
            f"cons-avg={r.get('delta_cons_vs_avg_ASR', float('nan')):+.1f})"
        )


if __name__ == "__main__":
    main()
