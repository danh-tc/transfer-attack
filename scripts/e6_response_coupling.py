#!/usr/bin/env python
"""E6 diagnostic: Backbone Adversarial Response Coupling.

Question (RESEARCH.md gap): why does the same OSFD perturbation transfer
very differently across targets, especially CNN -> Swin, even when the
detector HEAD is held fixed (matched-pair controls: DINO-R50 vs DINO-Swin-L,
MaskRCNN-R50 vs MaskRCNN-Swin-T)? Prior diagnostics ruled out raw ||delta_F||
(E1) and instantaneous gradient cosine (N6-B mechanism-proof, RESEARCH.md
S19) as explanations. This tests a different quantity: does a target's
backbone RESPOND to the surrogate's perturbation in a way that is more
SIMILAR IN SHAPE (not magnitude) to how the surrogate's own backbone
responds, on targets where that perturbation transfers well?

No new attack is crafted here -- this is a forward-only diagnostic that reads
noise already saved under results/noise/<manifest-stem>/<attack>/ (produced
by run_attack.py/craft.py) and runs model.backbone(...) under torch.no_grad()
for both x_clean and x_adv, exactly like scripts/e1_feature_damage.py.

Per model m, per image i:
    F_m(x_i), F_m(x_i + delta_i)             -- backbone(normalize(x)) tuple of stages
    dF_m,i = concat_stages(pool(F_m(x_i+delta_i))) - concat_stages(pool(F_m(x_i)))
    dF_hat_m,i = dF_m,i / (||dF_m,i|| + eps)   -- per-image L2 normalize so no single
                                                    image's magnitude dominates the Gram
Each backbone stage is spatially average-pooled to a fixed --pool-size x
--pool-size grid before flattening+concatenating across stages, so the
per-image vector stays a tractable size (independent per-model dimensionality
is fine -- CKA needs equal N across models, not equal feature dimension).

Similarity between surrogate and target uses linear CKA computed in Gram
(sample x sample) space (Kornblith et al. 2019's HSIC formulation), which is
exactly what tolerates the different channel widths/spatial sizes/stage
counts of ResNet vs Swin vs Darknet/CSPDarknet backbones:
    C_response(s,t) = CKA(Gram(dF_hat_s), Gram(dF_hat_t))       -- main quantity
    S_clean(s,t)     = CKA(Gram(F_hat_s(x)), Gram(F_hat_t(x)))  -- clean-feature control

GO criteria (pre-registered, see conversation): strong GO requires (1) both
matched pairs show C_response(R50) > C_response(Swin), (2) bootstrap 95% CI
of that within-pair difference excludes 0 for both pairs, (3) across all
targets corr(C_response, ASR) > 0, and (4) corr(C_response, ASR) >
corr(S_clean, ASR). Weak GO: both matched pairs point the right direction but
(2)-(4) aren't all clean. NO-GO: a matched pair goes the wrong way, or
corr(C_response, ASR) ~= 0, or S_clean explains transfer at least as well as
C_response -- in which case do NOT reach for CKA-variant/subspace fixes to
rescue the hypothesis.

Example (pilot):
    python scripts/e6_response_coupling.py --attack osfd --manifest data/manifests/dev_50.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# (r50_name, swin_name) -- same detector head/decoder, only backbone changes.
MATCHED_PAIRS = [
    ("dino", "dino_r50", "dino_swin_l"),
    ("mask_rcnn", "mask_rcnn_r50", "mask_rcnn_swin_t"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["osfd", "mi_fgsm"], default="osfd")
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--noise-dir", type=Path, default=None, help="default: results/noise/<manifest-stem>/<attack>/")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--surrogate", type=str, default="faster_rcnn_r50")
    p.add_argument("--targets", nargs="+", default=["all"], help="target model name(s), or 'all'")
    p.add_argument("--pool-size", type=int, default=7, help="adaptive-avg-pool spatial grid per backbone stage")
    p.add_argument(
        "--pool-mode", choices=["mean", "rms"], default="mean",
        help="mean: linear avg pool. rms: sqrt(mean(x**2)) per cell, sensitivity-check alternative",
    )
    p.add_argument(
        "--bootstrap-frac-sweep", type=str, default=None,
        help="comma-separated list of subsample fractions, e.g. '0.7,0.8,0.9' -- computes Gram ONCE, "
        "reports matched-pair/correlation CI for each frac (no re-extraction). Overrides --bootstrap-frac.",
    )
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument(
        "--bootstrap-frac", type=float, default=0.8,
        help="subsample fraction WITHOUT replacement per draw (see module docstring on why not classic bootstrap)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, default=PROJECT_DIR / "results" / "e6_response_coupling.csv")
    p.add_argument(
        "--out-summary-csv", type=Path, default=PROJECT_DIR / "results" / "e6_response_coupling_summary.csv"
    )
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    p.add_argument(
        "--save-gram-npz", type=Path, default=None,
        help="save all Gram matrices (surrogate + targets, clean + delta) to this .npz for reuse "
        "without re-running feature extraction (e.g. future jackknife/CI audits)",
    )
    return p


def find_matching_run_log(runs_dir: Path, attack: str, manifest_stem: str) -> dict | None:
    """Most recent runs/run_attack_<attack>_<manifest_stem>_*.json, if any."""
    candidates = sorted(runs_dir.glob(f"run_attack_{attack}_{manifest_stem}_*.json"))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Linear CKA in Gram (sample x sample) space -- Kornblith et al. 2019.
# Works across models with different per-image feature dimensionality since
# only the N x N Gram matrix (never the raw feature matrix) is compared.
# ---------------------------------------------------------------------------
def center_gram(k):
    """Double-center a Gram matrix: Kc = H K H, H = I - (1/n) * ones.

    Computed via the standard O(n^2) row/col/grand-mean formula (verified
    numerically identical to the naive O(n^3) H@K@H matmul, max abs diff
    ~1e-14) instead of explicit matrix products -- matters here because this
    runs inside a 2000-draw resampling loop, once per target per draw; at
    N=300 the O(n^3) version would make the CI step the dominant cost of the
    whole diagnostic (observed ~15min just for bootstrap+corr at N=120,
    cubic scaling would put N=300 close to 2h on that step alone).
    """
    rowmean = k.mean(axis=1, keepdims=True)
    colmean = k.mean(axis=0, keepdims=True)
    grandmean = k.mean()
    return k - rowmean - colmean + grandmean


def linear_cka_from_gram(k, l) -> float:
    import numpy as np

    kc = center_gram(k)
    lc = center_gram(l)
    hsic_kl = float(np.sum(kc * lc))
    hsic_kk = float(np.sum(kc * kc))
    hsic_ll = float(np.sum(lc * lc))
    denom = (hsic_kk * hsic_ll) ** 0.5
    if denom < 1e-12:
        return float("nan")
    return hsic_kl / denom


def gram_from_vectors(mat) -> "object":
    """mat: (N, D) numpy array of already-L2-normalized rows -> (N, N) Gram."""
    return mat @ mat.T


# ---------------------------------------------------------------------------
# Feature extraction: per-model, per-image pooled+concatenated backbone
# response vectors (clean and delta), both L2-normalized per image.
# ---------------------------------------------------------------------------
def _pool_stage(x, pool_size, mode):
    """x: (1,C,H,W) -> flattened (C*pool_size*pool_size,) pooled vector.
    mode='mean': plain adaptive average pool (linear -- pool(a)-pool(b) == pool(a-b)).
    mode='rms': sqrt(mean(x**2)) per cell -- captures magnitude of change
    regardless of sign within a cell; NOT linear, so must be applied directly
    to the quantity of interest (the raw diff map for dF, the raw clean map
    for F), never pooled separately then subtracted.
    """
    import torch.nn.functional as Fnn

    if mode == "mean":
        return Fnn.adaptive_avg_pool2d(x, (pool_size, pool_size)).flatten()
    if mode == "rms":
        return Fnn.adaptive_avg_pool2d(x * x, (pool_size, pool_size)).clamp_min(0).sqrt().flatten()
    raise ValueError(f"unknown pool mode {mode!r}")


def extract_pooled_response(handle, image_ids, noise_dir, img_dir, coco, canvas, pool_size, device, pool_mode="mean"):
    import numpy as np
    import torch
    import torch.nn.functional as Fnn

    from transfer_attack.data import load_canvas_image
    from transfer_attack.io_utils import load_noise

    model = handle.model
    clean_rows: list = []
    delta_rows: list = []

    for image_id in image_ids:
        noise_path = noise_dir / f"{image_id}.pt"
        canvas_img, _, _, _ = load_canvas_image(img_dir, coco, image_id, canvas)
        noise = load_noise(noise_path)
        x_clean = canvas_img.to(device)
        x_adv = (canvas_img + noise).clamp(0.0, 255.0).to(device)

        with torch.no_grad():
            feats_clean = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
            feats_adv = model.backbone(handle.normalize(x_adv.unsqueeze(0)))

        clean_parts = []
        delta_parts = []
        for fc, fa in zip(feats_clean, feats_adv):
            clean_parts.append(_pool_stage(fc, pool_size, pool_mode))
            # Pool the raw diff map directly (correct for both mean, where it's
            # equivalent to pooling separately and subtracting, and rms, where
            # it is NOT equivalent -- see _pool_stage docstring).
            delta_parts.append(_pool_stage(fa - fc, pool_size, pool_mode))
        clean_vec = torch.cat(clean_parts).cpu().numpy().astype(np.float64)
        delta_vec = torch.cat(delta_parts).cpu().numpy().astype(np.float64)

        clean_vec = clean_vec / (np.linalg.norm(clean_vec) + 1e-8)
        delta_vec = delta_vec / (np.linalg.norm(delta_vec) + 1e-8)
        clean_rows.append(clean_vec)
        delta_rows.append(delta_vec)

    return np.stack(clean_rows, axis=0), np.stack(delta_rows, axis=0)


# ---------------------------------------------------------------------------
# Resampling for CI: subsample images WITHOUT replacement (m = bootstrap_frac
# * n, default 80%), NOT the usual bootstrap-with-replacement.
#
# Verified empirically (synthetic test, see conversation) that naive
# with-replacement bootstrap is systematically biased upward for a Gram/CKA
# statistic: resampling with replacement duplicates rows, and a duplicated
# image has similarity 1.0 with its own copy by construction -- this isn't
# noise, it inflates every resampled Gram matrix's mean similarity relative to
# the true (unique-rows) statistic. On synthetic correlated data (point CKA
# 0.894), with-replacement bootstrap put the point estimate OUTSIDE its own
# 95% CI ([0.902, 0.936]); m-out-of-n subsampling without replacement (m=0.8n)
# recovered a CI that correctly contains the point estimate ([0.892, 0.912]).
# This is the standard fix for kernel/U-statistics under resampling (bootstrap
# is unreliable for statistics that aren't smooth/linear in the empirical
# Gram matrix; subsampling without replacement is the documented alternative).
# ---------------------------------------------------------------------------
def bootstrap_matched_pair(k_sur, k_r50, k_swin, n_draws, seed, frac=0.8):
    import numpy as np

    rng = random.Random(seed)
    n = k_sur.shape[0]
    m = max(2, round(frac * n))
    point_r50 = linear_cka_from_gram(k_sur, k_r50)
    point_swin = linear_cka_from_gram(k_sur, k_swin)
    point_delta = point_r50 - point_swin

    deltas = []
    for _ in range(n_draws):
        idx = np.array(rng.sample(range(n), k=m))
        sub = np.ix_(idx, idx)
        c_r50 = linear_cka_from_gram(k_sur[sub], k_r50[sub])
        c_swin = linear_cka_from_gram(k_sur[sub], k_swin[sub])
        deltas.append(c_r50 - c_swin)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]
    same_side = sum(1 for d in deltas if (d >= 0) == (point_delta >= 0)) / len(deltas)
    return point_r50, point_swin, point_delta, lo, hi, same_side


def jackknife_matched_pair_delta(k_sur, k_r50, k_swin):
    """Delete-1 jackknife SE for theta = CKA(sur,r50) - CKA(sur,swin), computed
    directly on the difference statistic (not on each CKA separately) so the
    shared-image dependence between the two CKA terms is preserved.

    theta_full = point estimate at the full N.
    theta_(-i) = same statistic recomputed with image i removed from all 3
    Gram matrices (leave-one-out).
    SE_jack = sqrt((n-1)/n * sum_i (theta_(-i) - mean_i theta_(-i))^2).

    Cheaper than the bootstrap: n leave-one-out folds vs n_draws (2000)
    resamples, and does not carry the m-out-of-n subsampling's own subtlety
    (a raw percentile CI from m<n draws estimates the sampling distribution
    of the m-sample statistic, not the n-sample one, without a sqrt(m/n)
    rescale) -- jackknife SE is computed directly at the actual N used.
    """
    import numpy as np

    n = k_sur.shape[0]
    theta_full = linear_cka_from_gram(k_sur, k_r50) - linear_cka_from_gram(k_sur, k_swin)

    idx_all = np.arange(n)
    thetas = np.empty(n)
    for i in range(n):
        idx = idx_all[idx_all != i]
        sub = np.ix_(idx, idx)
        thetas[i] = linear_cka_from_gram(k_sur[sub], k_r50[sub]) - linear_cka_from_gram(k_sur[sub], k_swin[sub])

    theta_bar = thetas.mean()
    se_jack = float(np.sqrt((n - 1) / n * np.sum((thetas - theta_bar) ** 2)))
    ci_lo = theta_full - 1.96 * se_jack
    ci_hi = theta_full + 1.96 * se_jack
    # bias check: jackknife estimate of bias = (n-1)*(theta_bar - theta_full);
    # large relative to se_jack would flag the statistic as poorly behaved for
    # delete-1 jackknife (e.g. near a non-smooth point) -- report, don't hide.
    bias_jack = (n - 1) * (theta_bar - theta_full)
    return theta_full, se_jack, ci_lo, ci_hi, bias_jack


def bootstrap_cross_target_corr(k_sur, k_targets: dict, stat_by_name: dict, n_draws, seed, frac=0.8):
    """stat_by_name: e.g. ASR per target name, held fixed (not resampled --
    it's an external aggregate context value, same simplification e1 uses).
    """
    import numpy as np

    names = [n for n in k_targets if n in stat_by_name and stat_by_name[n] is not None]
    y = np.array([stat_by_name[n] for n in names])
    point_vals = np.array([linear_cka_from_gram(k_sur, k_targets[n]) for n in names])
    point_corr = float(np.corrcoef(point_vals, y)[0, 1])

    rng = random.Random(seed)
    n = k_sur.shape[0]
    m = max(2, round(frac * n))
    corrs = []
    for _ in range(n_draws):
        idx = np.array(rng.sample(range(n), k=m))
        sub = np.ix_(idx, idx)
        k_sur_sub = k_sur[sub]
        vals = np.array([linear_cka_from_gram(k_sur_sub, k_targets[n][sub]) for n in names])
        if np.std(vals) < 1e-9:
            continue
        corrs.append(float(np.corrcoef(vals, y)[0, 1]))
    corrs.sort()
    lo = corrs[int(0.025 * len(corrs))]
    hi = corrs[min(int(0.975 * len(corrs)), len(corrs) - 1)]
    same_side = sum(1 for c in corrs if (c >= 0) == (point_corr >= 0)) / len(corrs)
    return point_corr, lo, hi, same_side, names


def main() -> None:
    args = build_arg_parser().parse_args()

    import numpy as np
    import torch

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import load_coco, load_manifest
    from transfer_attack.io_utils import get_logger
    from transfer_attack.models import MODEL_REGISTRY, build_model_handle, get_spec

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    all_image_ids = manifest["image_ids"]
    if args.limit is not None:
        all_image_ids = all_image_ids[: args.limit]

    noise_dir = args.noise_dir or (PROJECT_DIR / "results" / "noise" / args.manifest.stem / args.attack)
    if not noise_dir.exists():
        logger.error(
            f"noise dir not found: {noise_dir} -- craft it first, e.g.:\n"
            f"  python scripts/run_attack.py --attack {args.attack} --manifest {args.manifest} --steps 100"
        )
        sys.exit(1)

    # Fix the exact image set + order ONCE (model-independent -- same noise
    # file per image_id for every model), so every Gram matrix is indexed
    # identically and bootstrap resampling is valid across models.
    used_image_ids = [iid for iid in all_image_ids if (noise_dir / f"{iid}.pt").exists()]
    n_missing = len(all_image_ids) - len(used_image_ids)
    if n_missing:
        logger.warning(f"{n_missing}/{len(all_image_ids)} images had no crafted noise under {noise_dir} -- skipped")
    n = len(used_image_ids)
    logger.info(f"using N={n} images (order fixed across all models)")

    if args.targets == ["all"]:
        target_specs = [s for s in MODEL_REGISTRY if s.role == "target"]
    else:
        target_specs = [get_spec(m) for m in args.targets]

    img_dir = args.data_dir / "val2017"
    device = args.device

    run_log = find_matching_run_log(args.runs_dir, args.attack, args.manifest.stem)
    map_drop_by_model, asr_by_model = {}, {}
    if run_log is not None:
        for row in run_log.get("results", {}).get("rows", []):
            if row["attack"] == args.attack:
                map_drop_by_model[row["model_name"]] = row.get("mAP_drop_pct")
                asr_by_model[row["model_name"]] = row.get("ASR")
        logger.info(f"matched run log for ASR/mAP_drop context: {run_log.get('timestamp')}")
    else:
        logger.warning(f"no runs/run_attack_{args.attack}_{args.manifest.stem}_*.json found -- ASR/mAP columns empty")

    # ---- Surrogate pass ----
    surrogate_spec = get_spec(args.surrogate)
    logger.info(f"=== surrogate {surrogate_spec.name} ===")
    sur_handle = build_model_handle(surrogate_spec, args.checkpoints_dir, device=device, coco=coco)
    sur_clean, sur_delta = extract_pooled_response(
        sur_handle, used_image_ids, noise_dir, img_dir, coco, args.canvas, args.pool_size, device, args.pool_mode
    )
    k_sur_clean = gram_from_vectors(sur_clean)
    k_sur_delta = gram_from_vectors(sur_delta)
    del sur_handle
    torch.cuda.empty_cache()

    # ---- Target passes ----
    k_clean_by_target: dict[str, "object"] = {}
    k_delta_by_target: dict[str, "object"] = {}
    group_by_target: dict[str, str] = {}

    for spec in target_specs:
        logger.info(f"=== target {spec.name} (group={spec.group}) ===")
        handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)
        clean_mat, delta_mat = extract_pooled_response(
            handle, used_image_ids, noise_dir, img_dir, coco, args.canvas, args.pool_size, device, args.pool_mode
        )
        k_clean_by_target[spec.name] = gram_from_vectors(clean_mat)
        k_delta_by_target[spec.name] = gram_from_vectors(delta_mat)
        group_by_target[spec.name] = spec.group or ""
        del handle
        torch.cuda.empty_cache()

    # ---- Per-target C_response / S_clean + write CSVs ----
    detail_rows = []
    for name in k_delta_by_target:
        c_response = linear_cka_from_gram(k_sur_delta, k_delta_by_target[name])
        s_clean = linear_cka_from_gram(k_sur_clean, k_clean_by_target[name])
        detail_rows.append(
            {
                "target": name,
                "group": group_by_target[name],
                "n_images": n,
                "S_clean": s_clean,
                "C_response": c_response,
                "ASR": asr_by_model.get(name),
                "mAP_drop_pct": map_drop_by_model.get(name),
            }
        )
        logger.info(f"  {name}: S_clean={s_clean:.4f} C_response={c_response:.4f} ASR={asr_by_model.get(name)}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["target", "group", "n_images", "S_clean", "C_response", "ASR", "mAP_drop_pct"]
        )
        writer.writeheader()
        writer.writerows(detail_rows)
    logger.info(f"wrote per-target table -> {args.out_csv}")
    with open(args.out_summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["target", "group", "n_images", "S_clean", "C_response", "ASR", "mAP_drop_pct"]
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    if args.save_gram_npz:
        args.save_gram_npz.parent.mkdir(parents=True, exist_ok=True)
        gram_payload = {"k_sur_clean": k_sur_clean, "k_sur_delta": k_sur_delta, "image_ids": np.array(used_image_ids)}
        for name in k_delta_by_target:
            gram_payload[f"k_clean__{name}"] = k_clean_by_target[name]
            gram_payload[f"k_delta__{name}"] = k_delta_by_target[name]
        np.savez_compressed(args.save_gram_npz, **gram_payload)
        logger.info(f"saved Gram matrix cache -> {args.save_gram_npz} (reuse for future audits, no re-extraction)")

    # ---- Matched-pair jackknife (delete-1, direct on the difference statistic) ----
    # Preferred over the m-out-of-n subsampling CI below for the "is the CI at
    # the actual N trustworthy" question: subsampling with m<n estimates the
    # sampling distribution of the m-sample statistic, not the n-sample one,
    # without a sqrt(m/n) rescale -- jackknife SE is computed directly at N.
    logger.info("=== Matched-pair jackknife (delete-1 SE, computed at full N) ===")
    for pair_name, r50_name, swin_name in MATCHED_PAIRS:
        if r50_name not in k_delta_by_target or swin_name not in k_delta_by_target:
            continue
        theta, se_jack, jk_lo, jk_hi, bias_jack = jackknife_matched_pair_delta(
            k_sur_delta, k_delta_by_target[r50_name], k_delta_by_target[swin_name]
        )
        logger.info(
            f"  {pair_name}: theta={theta:+.4f} SE_jack={se_jack:.4f} "
            f"95% CI=[{jk_lo:+.4f},{jk_hi:+.4f}] jackknife_bias_est={bias_jack:+.5f}"
        )

    if args.bootstrap_frac_sweep:
        fracs = [float(x) for x in args.bootstrap_frac_sweep.split(",")]
    else:
        fracs = [args.bootstrap_frac]

    sweep_verdicts = []
    for frac in fracs:
        logger.info(f"########## bootstrap_frac={frac} ##########")

        # ---- Matched-pair bootstrap ----
        logger.info("=== Matched-pair response coupling (R50 vs Swin, same head) ===")
        pair_results = {}
        for pair_name, r50_name, swin_name in MATCHED_PAIRS:
            if r50_name not in k_delta_by_target or swin_name not in k_delta_by_target:
                logger.warning(f"  {pair_name}: missing {r50_name} or {swin_name} in targets -- skipped")
                continue
            c_r50, c_swin, delta, lo, hi, same_side = bootstrap_matched_pair(
                k_sur_delta, k_delta_by_target[r50_name], k_delta_by_target[swin_name],
                args.n_bootstrap, args.seed, frac,
            )
            pair_results[pair_name] = (c_r50, c_swin, delta, lo, hi, same_side)
            logger.info(
                f"  {pair_name}: C({r50_name})={c_r50:.4f} C({swin_name})={c_swin:.4f} "
                f"delta={delta:+.4f} 95% CI=[{lo:+.4f},{hi:+.4f}] same_side_frac={same_side:.3f}"
            )

        # ---- Cross-target correlation ----
        logger.info("=== Cross-target correlation with ASR ===")
        corr_response = bootstrap_cross_target_corr(
            k_sur_delta, k_delta_by_target, asr_by_model, args.n_bootstrap, args.seed, frac
        )
        corr_clean = bootstrap_cross_target_corr(
            k_sur_clean, k_clean_by_target, asr_by_model, args.n_bootstrap, args.seed, frac
        )
        logger.info(
            f"  corr(C_response, ASR) = {corr_response[0]:+.3f} 95% CI=[{corr_response[1]:+.3f},{corr_response[2]:+.3f}] "
            f"same_side_frac={corr_response[3]:.3f} (n_targets={len(corr_response[4])})"
        )
        logger.info(
            f"  corr(S_clean, ASR)    = {corr_clean[0]:+.3f} 95% CI=[{corr_clean[1]:+.3f},{corr_clean[2]:+.3f}] "
            f"same_side_frac={corr_clean[3]:.3f} (n_targets={len(corr_clean[4])})"
        )

        # ---- Verdict ----
        logger.info("=== Verdict ===")
        both_pairs_present = len(pair_results) == len(MATCHED_PAIRS)
        direction_ok = both_pairs_present and all(v[2] > 0 for v in pair_results.values())
        ci_excludes_zero = both_pairs_present and all((v[3] > 0) or (v[4] < 0) for v in pair_results.values())
        corr_positive = corr_response[0] > 0
        response_beats_clean = corr_response[0] > corr_clean[0]

        if not both_pairs_present:
            verdict = "inconclusive"
            logger.warning("VERDICT: inconclusive -- one or both matched pairs missing from --targets")
        elif not direction_ok:
            verdict = "NO-GO"
            logger.warning(
                "VERDICT: NO-GO -- a matched pair went the wrong direction "
                "(C_response(R50) <= C_response(Swin)). Do not reach for CKA-variant/subspace fixes."
            )
        elif abs(corr_response[0]) < 0.05:
            verdict = "NO-GO"
            logger.warning(
                "VERDICT: NO-GO -- corr(C_response, ASR) ~= 0 across targets; response coupling does not "
                "relate to transfer even though matched-pair direction was correct."
            )
        elif corr_clean[0] >= corr_response[0]:
            verdict = "NO-GO"
            logger.warning(
                "VERDICT: NO-GO -- clean-feature similarity (S_clean) explains transfer at least as well as "
                "C_response; response coupling adds no evidence over the trivial clean-similarity control."
            )
        elif ci_excludes_zero and corr_positive and response_beats_clean:
            verdict = "STRONG GO"
            logger.info(
                "VERDICT: STRONG GO -- both matched pairs correct direction with CI excluding 0, "
                "corr(C_response,ASR)>0, and C_response beats S_clean as a transfer predictor."
            )
        else:
            verdict = "WEAK GO"
            logger.info(
                "VERDICT: WEAK GO -- both matched pairs point the right direction, but CI/correlation "
                "criteria are not all clean yet. Do not claim mechanism; gather more evidence "
                "(additional surrogate/perturbation source) before scaling compute."
            )
        sweep_verdicts.append(
            {
                "frac": frac,
                "verdict": verdict,
                "delta_dino": pair_results.get("dino", (None,) * 6)[2],
                "delta_mask_rcnn": pair_results.get("mask_rcnn", (None,) * 6)[2],
                "corr_response": corr_response[0],
                "corr_clean": corr_clean[0],
            }
        )

    if len(fracs) > 1:
        logger.info("########## sweep summary (robustness across bootstrap_frac) ##########")
        for row in sweep_verdicts:
            logger.info(
                f"  frac={row['frac']}: delta_dino={row['delta_dino']:+.4f} "
                f"delta_mask_rcnn={row['delta_mask_rcnn']:+.4f} "
                f"corr_response={row['corr_response']:+.3f} corr_clean={row['corr_clean']:+.3f} "
                f"verdict={row['verdict']}"
            )
        dino_signs = {row["delta_dino"] > 0 for row in sweep_verdicts}
        mask_signs = {row["delta_mask_rcnn"] > 0 for row in sweep_verdicts}
        beats_clean = {row["corr_response"] > row["corr_clean"] for row in sweep_verdicts}
        logger.info(
            f"  robustness: delta_dino sign stable={len(dino_signs) == 1} "
            f"delta_mask_rcnn sign stable={len(mask_signs) == 1} "
            f"C_response>S_clean stable={len(beats_clean) == 1}"
        )


if __name__ == "__main__":
    main()
