#!/usr/bin/env python
"""E1 diagnostic: does OSFD's backbone-feature damage survive the surrogate ->
target architecture change, and does the amount of surviving damage correlate
with the black-box mAP drop already measured in runs/?

For each registered model (surrogate + all targets), forward the SAME
already-crafted x_clean / x_adv = x_clean + noise pair through that model's
OWN backbone (no recrafting -- this only reuses noise already on disk under
results/noise/<manifest>/<attack>/) and measure, per backbone stage:

  cos_dist  = 1 - cosine_similarity(F(x_clean), F(x_adv))   (flattened per image)
  rel_l2    = ||F(x_adv) - F(x_clean)||_2 / ||F(x_clean)||_2

averaged over images. If cos_dist/rel_l2 fall off sharply from group A -> B ->
C and track the mAP-drop numbers in runs/*.json, that's direct evidence the
adversarial feature distortion doesn't survive the architecture change
(representation mismatch is the bottleneck) rather than e.g. the detector head
just being more robust to equally-large feature damage (that's E2's question).

Example:
    python scripts/e1_feature_damage.py --attack osfd --manifest data/manifests/dev_50.json
"""
from __future__ import annotations

import argparse
import csv
import json
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
    p.add_argument("--models", nargs="+", default=["all"], help="model name(s) from MODEL_REGISTRY, or 'all'")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "e1_feature_damage.csv")
    p.add_argument("--out-summary-csv", type=Path, default=PROJECT_DIR / "results" / "e1_feature_damage_summary.csv")
    p.add_argument(
        "--runs-dir", type=Path, default=PROJECT_DIR / "runs",
        help="used to look up the matching run_attack log for mAP_drop_pct/ASR context in the summary",
    )
    return p


def find_matching_run_log(runs_dir: Path, attack: str, manifest_stem: str) -> dict | None:
    """Most recent runs/run_attack_<attack>_<manifest_stem>_*.json, if any."""
    candidates = sorted(runs_dir.glob(f"run_attack_{attack}_{manifest_stem}_*.json"))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch
    import torch.nn.functional as F

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, load_noise
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle

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
        logger.error(
            f"noise dir not found: {noise_dir} -- craft it first, e.g.:\n"
            f"  python scripts/craft.py --attack {args.attack} --manifest {args.manifest} "
            f"--out-dir {noise_dir} --steps 100"
        )
        sys.exit(1)

    if args.models == ["all"]:
        specs = MODEL_REGISTRY
    else:
        by_name = {s.name: s for s in MODEL_REGISTRY}
        specs = [by_name[m] for m in args.models]

    img_dir = args.data_dir / "val2017"
    device = args.device

    detail_rows = []  # one row per (model, stage)
    summary_rows = []  # one row per model

    run_log = find_matching_run_log(args.runs_dir, args.attack, args.manifest.stem)
    map_drop_by_model = {}
    asr_by_model = {}
    if run_log is not None:
        for row in run_log.get("results", {}).get("rows", []):
            if row["attack"] == args.attack:
                map_drop_by_model[row["model_name"]] = row.get("mAP_drop_pct")
                asr_by_model[row["model_name"]] = row.get("ASR")
        logger.info(f"matched run log for context: {run_log.get('timestamp')}")
    else:
        logger.warning(
            f"no runs/run_attack_{args.attack}_{args.manifest.stem}_*.json found -- "
            f"summary will have empty mAP_drop_pct/ASR columns"
        )

    for spec in specs:
        logger.info(f"=== {spec.name} (role={spec.role}, group={spec.group}) ===")
        handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)
        model = handle.model

        n_images = 0
        n_missing_noise = 0
        # per-stage accumulators: stage_idx -> list of per-image cos_dist / rel_l2
        cos_dists: dict[int, list[float]] = {}
        rel_l2s: dict[int, list[float]] = {}

        for image_id in image_ids:
            noise_path = noise_dir / f"{image_id}.pt"
            if not noise_path.exists():
                n_missing_noise += 1
                continue

            canvas_img, _, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
            noise = load_noise(noise_path)
            x_clean = canvas_img.to(device)
            x_adv = (canvas_img + noise).clamp(0.0, 255.0).to(device)

            with torch.no_grad():
                feats_clean = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
                feats_adv = model.backbone(handle.normalize(x_adv.unsqueeze(0)))

            for stage_idx, (fc, fa) in enumerate(zip(feats_clean, feats_adv)):
                fc_flat = fc.flatten().unsqueeze(0)
                fa_flat = fa.flatten().unsqueeze(0)
                cos_sim = F.cosine_similarity(fc_flat, fa_flat).item()
                fc_norm = fc_flat.norm().item()
                rel_l2 = (fa_flat - fc_flat).norm().item() / (fc_norm + 1e-8)
                cos_dists.setdefault(stage_idx, []).append(1.0 - cos_sim)
                rel_l2s.setdefault(stage_idx, []).append(rel_l2)

            n_images += 1

        if n_missing_noise:
            logger.warning(f"  {n_missing_noise} images had no crafted noise under {noise_dir} -- skipped")

        n_stages = len(cos_dists)
        all_stage_cos = []
        all_stage_l2 = []
        for stage_idx in range(n_stages):
            cd = cos_dists[stage_idx]
            rl = rel_l2s[stage_idx]
            mean_cd = sum(cd) / len(cd)
            mean_rl = sum(rl) / len(rl)
            all_stage_cos.append(mean_cd)
            all_stage_l2.append(mean_rl)
            detail_rows.append(
                {
                    "model_name": spec.name,
                    "role": spec.role,
                    "group": spec.group or "",
                    "stage_idx": stage_idx,
                    "n_images": n_images,
                    "mean_cos_dist": mean_cd,
                    "mean_rel_l2": mean_rl,
                }
            )

        mean_cos_all_stages = sum(all_stage_cos) / n_stages if n_stages else float("nan")
        mean_l2_all_stages = sum(all_stage_l2) / n_stages if n_stages else float("nan")
        last_stage_cos = all_stage_cos[-1] if all_stage_cos else float("nan")
        last_stage_l2 = all_stage_l2[-1] if all_stage_l2 else float("nan")

        summary_rows.append(
            {
                "model_name": spec.name,
                "role": spec.role,
                "group": spec.group or "",
                "n_images": n_images,
                "n_stages": n_stages,
                "mean_cos_dist_all_stages": mean_cos_all_stages,
                "mean_rel_l2_all_stages": mean_l2_all_stages,
                "last_stage_cos_dist": last_stage_cos,
                "last_stage_rel_l2": last_stage_l2,
                "mAP_drop_pct": map_drop_by_model.get(spec.name),
                "ASR": asr_by_model.get(spec.name),
            }
        )
        logger.info(
            f"  n_images={n_images} n_stages={n_stages} "
            f"mean_cos_dist(all stages)={mean_cos_all_stages:.4f} "
            f"last_stage_cos_dist={last_stage_cos:.4f}"
        )

        del handle, model
        torch.cuda.empty_cache()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model_name", "role", "group", "stage_idx", "n_images", "mean_cos_dist", "mean_rel_l2"]
        )
        writer.writeheader()
        writer.writerows(detail_rows)
    logger.info(f"wrote per-stage detail -> {args.out_csv}")

    with open(args.out_summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name", "role", "group", "n_images", "n_stages",
                "mean_cos_dist_all_stages", "mean_rel_l2_all_stages",
                "last_stage_cos_dist", "last_stage_rel_l2", "mAP_drop_pct", "ASR",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"wrote per-model summary -> {args.out_summary_csv}")

    # Quick correlation read (targets only, i.e. group != "") against mAP_drop_pct,
    # if we found a matching run log -- purely a diagnostic printout, not saved.
    target_rows = [r for r in summary_rows if r["group"] and r["mAP_drop_pct"] is not None]
    if len(target_rows) >= 3:
        import numpy as np

        cos = np.array([r["mean_cos_dist_all_stages"] for r in target_rows])
        drop = np.array([r["mAP_drop_pct"] for r in target_rows])
        corr = float(np.corrcoef(cos, drop)[0, 1])
        logger.info(
            f"[diagnostic] Pearson corr(mean_cos_dist_all_stages, mAP_drop_pct) across "
            f"{len(target_rows)} target models = {corr:.3f} "
            f"(close to +1 => feature damage tracks mAP drop => representation mismatch "
            f"is a plausible bottleneck; close to 0 => damage doesn't explain the drop => "
            f"look at E2's downstream-pipeline-attenuation hypothesis instead)"
        )
    else:
        logger.warning("not enough target rows with matched mAP_drop_pct to print a correlation (need >= 3)")


if __name__ == "__main__":
    main()
