#!/usr/bin/env python
"""E10 -- Cross-Architecture Semantic-Region Relational Geometry (discovery
experiment, RESEARCH.md Phase N continuation after E6-E9's H1 support /
C_response-proxy dead end).

Question: does a stable RELATIONAL geometry between an object's interior (O),
boundary (E), near background (Bn) and far background (Bf) exist across
heterogeneous backbone families (ResNet, CSPDarknet, Swin), and does OSFD's
noise -- and specifically RRB -- disrupt that shared relational structure in a
way that tracks black-box transfer? This does NOT compare raw feature vectors
across architectures (E1 already showed that's not the bottleneck) -- it
compares the STRUCTURE of a 6-dim relational signature
r = [d(O,E), d(O,Bn), d(O,Bf), d(E,Bn), d(E,Bf), d(Bn,Bf)], d = 1-cosine_sim,
which is architecture-agnostic by construction (no cross-model vector
comparison anywhere).

Three questions, each with a pre-registered falsification criterion (no
metric-shopping after a NO-GO -- see module-level GO/NO-GO thresholds below):

  Q1 (existence)   -- is r's cross-object variation a REPRODUCIBLE per-object
                       signal across model families (not just within one)?
                       Table A: per-relation Spearman consistency of clean r
                       across model pairs, grouped by backbone family.
  Q2 (relevance)    -- does OSFD's clean->adv delta-r move consistently across
                       targets, and does the magnitude of that disruption
                       RESTRICTED to Q1's shared/invariant relations track ASR
                       (matched-pair: same head, R50 backbone vs Swin backbone)?
                       Table B.
  Q3 (mechanism)    -- does RRB (vs no-RRB, both at k=1, reusing E3's factorial
                       noise) increase disruption specifically of those SAME
                       shared relations, in the direction of its known ASR gain?
                       Table C.

No recrafting beyond what this script's own setup needs (osfd default +
E3 k1_norrb/k1_rrb arms on dev_50, via scripts/craft.py). All backbone forward
passes are gradient-free, single image at a time -- same cost class as
E1/E2/E2b/E2c/N4.

Example:
    python scripts/e10_relational_geometry.py --manifest data/manifests/dev_50.json

------------------------------------------------------------------------------
FROZEN SPEC (RESEARCH.md dev_50 pilot, STRONG GO -- see RESEARCH.md Sec 27).
Everything below is semantics/math, pre-registered and NOT to be changed for
the dev_300 confirmation run. Only engineering (loop order, batching, caching)
may change from this point on; any edit to this file must leave every one of
these bullets true bit-for-bit (verified by a separate equivalence check on
dev_50 against the pilot's saved reference CSVs under
results/e10_pilot_reference/, tolerance ~1e-5, before dev_300 is ever run):

  - Regions per GT box b=(x1,y1,x2,y2), all rectangles, canvas-resolution:
      O  = shrink(b, shrink_frac=0.125 per side)      -- CLI --shrink-frac
      E  = b MINUS O
      Bn = expand(b, expand_frac=0.25 per side) MINUS b   -- CLI --expand-frac
      Bf = full canvas MINUS expand(b, expand_frac)
    Bn and Bf additionally EXCLUDE the union of every OTHER GT box's own
    rectangle in the same image (raw box, not expanded/shrunk). O and E do
    NOT get this exclusion (see build_region_masks docstring).
  - An object is dropped entirely (not just at one stage) if shrink(b) is
    degenerate (zero-or-negative width/height) -- see build_region_masks.
  - Per stage l (backbone output tuple index, architecture-relative -- i.e.
    "last stage" = index len(feats)-1 for THAT model, not an absolute index):
    each region mask is downsampled from canvas resolution to that stage's
    (h,w) via F.interpolate(mode="area") on the {0,1} mask -- giving a
    fractional-coverage weight in [0,1] per stage cell, NOT a hard threshold.
    Region pooled feature = weighted mean over (h,w): sum(F*w)/sum(w).
  - An (object, stage) is dropped if ANY of the 4 regions' weight sums to
    < MASK_MIN_STAGE_CELLS=1.0 stage-cell-equivalent at that stage.
  - Relational signature (6 components, symmetric distance d(a,b)=1-cosine_sim
    between the two regions' pooled (C,)-vectors): r = [d(O,E), d(O,Bn),
    d(O,Bf), d(E,Bn), d(E,Bf), d(Bn,Bf)]. Order/definition fixed by RELATIONS.
  - ALL summary tables (A/B/C) use ONLY the last stage (see above) -- shallower
    stages are recorded in the detail CSVs but never analyzed for the verdict.
  - Q1/Table A: per relation, per model-pair, Spearman rho of that relation's
    raw value across matched (image,object) pairs valid for both models (>=5
    matched objects required, else NaN for that pair). Aggregated as an
    unweighted mean rho over: within-R50-family pairs, within-CSP-family pairs
    (NaN with 1 registered CSP model), within-Swin-family pairs, cross-family
    pairs (family_1 != family_2), and all pairs ("global"). Family assignment:
    BACKBONE_FAMILY dict (R50: faster_rcnn_r50/dino_r50/mask_rcnn_r50; CSP:
    yolov3_d53/yolox_l; Swin: mask_rcnn_swin_t/dino_swin_l).
  - Shared/invariant relation = global mean rho > GO1_CONSISTENCY_THR=0.30 AND
    cross-family mean rho > GO1_CONSISTENCY_THR=0.30 (both must hold).
  - Q2/Table B: disrupt_shared(model) = mean over (image,object) of the L2 norm
    of delta_r = r_adv - r_clean, restricted to the columns in the shared set
    (all 6 in the pilot). ASR fetched fresh via evaluate.py's own machinery on
    THIS run's noise (not read from old run logs).
  - Q3/Table C: identical disrupt_shared computation, applied separately to the
    e3_k1_norrb and e3_k1_rrb noise variants (k=1 both arms, RRB off/on); ASR
    likewise fetched fresh for both arms.
  - Models: exactly {faster_rcnn_r50, dino_r50, mask_rcnn_r50, yolox_l,
    mask_rcnn_swin_t, dino_swin_l} (DEFAULT_MODELS) for the frozen result;
    canvas=800, dataset=dev_50 for the pilot (dev_300 for the confirmation run
    -- same manifest schema, same image-loading/GT pipeline as every other
    script in this repo, nothing E10-specific there).
------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

RELATIONS = ["OE", "OBn", "OBf", "EBn", "EBf", "BnBf"]
RELATION_LABEL = {
    "OE": "O<->E", "OBn": "O<->nearBG", "OBf": "O<->farBG",
    "EBn": "E<->nearBG", "EBf": "E<->farBG", "BnBf": "nearBG<->farBG",
}

# Which noise dir (under results/noise/<manifest-stem>/) each "variant" tag reads from.
VARIANTS = {
    "osfd": "osfd",              # main OSFD baseline: k=3, RRB on
    "k1_norrb": "e3_k1_norrb",    # E3 arm: k=1, RRB off
    "k1_rrb": "e3_k1_rrb",        # E3 arm: k=1, RRB on
}

DEFAULT_MODELS = ["faster_rcnn_r50", "dino_r50", "mask_rcnn_r50", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"]

# backbone-family grouping used for Table A/B (independent of MODEL_REGISTRY's
# A/B/C/D "group" field, which encodes a different axis -- see models.py).
BACKBONE_FAMILY = {
    "faster_rcnn_r50": "R50",
    "dino_r50": "R50",
    "mask_rcnn_r50": "R50",
    "yolov3_d53": "CSP",
    "yolox_l": "CSP",
    "mask_rcnn_swin_t": "Swin",
    "dino_swin_l": "Swin",
}

# Matched pairs (same detector head/decoder, R50 vs Swin backbone) -- same
# pairing RESEARCH.md's H1/E6-E9 chain uses for its strongest evidence.
MATCHED_PAIRS = [
    ("dino", "dino_r50", "dino_swin_l"),
    ("mask_rcnn", "mask_rcnn_r50", "mask_rcnn_swin_t"),
]

# Pre-registered GO thresholds (do not change after seeing results).
GO1_CONSISTENCY_THR = 0.30   # mean Spearman rho across model pairs, per relation
GO1_MIN_RELATIONS = 2        # need >=2 of 6 relations clearing GO1_CONSISTENCY_THR,
                              # AND their cross-family consistency specifically must
                              # also clear this bar (not just within-family)
MASK_MIN_STAGE_CELLS = 1.0   # region must cover >= 1 stage-cell-equivalent to be poolable


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifests" / "dev_50.json")
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    p.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "coco")
    p.add_argument("--noise-root", type=Path, default=None, help="default: results/noise/<manifest-stem>")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--shrink-frac", type=float, default=0.125, help="object-interior shrink, fraction of box w/h per side")
    p.add_argument("--expand-frac", type=float, default=0.25, help="near-background expand, fraction of box w/h per side")
    p.add_argument("--canvas", type=int, default=800)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results")
    p.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    p.add_argument("--predictions-dir", type=Path, default=PROJECT_DIR / "results" / "_e10_predictions")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    return p


# ---------------------------------------------------------------------------
# Region mask construction (canvas-resolution, model-independent)
# ---------------------------------------------------------------------------

def _clip_box(x1, y1, x2, y2, canvas):
    return max(0.0, x1), max(0.0, y1), min(float(canvas), x2), min(float(canvas), y2)


def shrink_box(box, frac, canvas):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return _clip_box(x1 + frac * w, y1 + frac * h, x2 - frac * w, y2 - frac * h, canvas)


def expand_box(box, frac, canvas):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return _clip_box(x1 - frac * w, y1 - frac * h, x2 + frac * w, y2 + frac * h, canvas)


def box_to_mask(box, canvas: int):
    import torch

    m = torch.zeros((canvas, canvas), dtype=torch.bool)
    x1, y1, x2, y2 = box
    ix1, iy1 = int(round(x1)), int(round(y1))
    ix2, iy2 = int(round(x2)), int(round(y2))
    if ix2 > ix1 and iy2 > iy1:
        m[iy1:iy2, ix1:ix2] = True
    return m


def build_region_masks(gt_boxes, canvas: int, shrink_frac: float, expand_frac: float):
    """gt_boxes: (M,4) xyxy canvas-space tensor. Returns a list (len M) of
    dicts {"O":mask,"E":mask,"Bn":mask,"Bf":mask} (each a (canvas,canvas) bool
    tensor), or None for objects whose shrunk interior is degenerate.
    Bn/Bf explicitly exclude every OTHER GT box's own rectangle (E10 spec
    step 1's "loại vùng overlap với GT objects khác")."""
    boxes = gt_boxes.tolist()
    n = len(boxes)
    box_masks = [box_to_mask(b, canvas) for b in boxes]

    out = []
    for i in range(n):
        b = boxes[i]
        o_box = shrink_box(b, shrink_frac, canvas)
        if o_box[2] <= o_box[0] or o_box[3] <= o_box[1]:
            out.append(None)
            continue
        other_mask = None
        for j in range(n):
            if j == i:
                continue
            other_mask = box_masks[j].clone() if other_mask is None else (other_mask | box_masks[j])
        if other_mask is None:
            import torch
            other_mask = torch.zeros((canvas, canvas), dtype=torch.bool)

        o_mask = box_to_mask(o_box, canvas)
        e_mask = box_masks[i] & (~o_mask)
        bn_full = box_to_mask(expand_box(b, expand_frac, canvas), canvas)
        bn_mask = bn_full & (~box_masks[i]) & (~other_mask)
        bf_mask = (~bn_full) & (~other_mask)
        out.append({"O": o_mask, "E": e_mask, "Bn": bn_mask, "Bf": bf_mask})
    return out


def pool_regions_one_stage(feat, region_masks: dict) -> dict | None:
    """feat: (C,H,W) tensor for one stage. region_masks: {"O":...,"Bf":mask}
    canvas-res bool masks. Returns {"O":(C,),...} pooled means, or None if any
    region maps to < MASK_MIN_STAGE_CELLS worth of coverage at this stage
    resolution (too small to pool reliably)."""
    import torch.nn.functional as F

    h, w = feat.shape[-2], feat.shape[-1]
    pooled = {}
    for name, mask in region_masks.items():
        weight = F.interpolate(mask.float()[None, None], size=(h, w), mode="area")[0, 0].to(feat.device)
        wsum = weight.sum()
        if wsum.item() < MASK_MIN_STAGE_CELLS:
            return None
        pooled[name] = (feat * weight.unsqueeze(0)).sum(dim=(1, 2)) / wsum
    return pooled


_REGION_ORDER = ("O", "E", "Bn", "Bf")


def pool_regions_batched(feat, region_masks_per_obj: list[dict | None]) -> list[dict | None]:
    """Batched, numerically-equivalent version of calling pool_regions_one_stage
    once per object in region_masks_per_obj (perf optimization only -- see the
    FROZEN SPEC comment at the top of this file: mode="area" interpolation is
    independent per batch element, so stacking every object's 4 region masks
    into one F.interpolate call and unstacking produces the identical
    per-element result as one call per mask, just far fewer Python/CUDA
    dispatch round-trips). Returns a list the same length as
    region_masks_per_obj, entry i = pooled dict for object i or None."""
    import torch
    import torch.nn.functional as F

    h, w = feat.shape[-2], feat.shape[-1]
    valid_idx = [i for i, m in enumerate(region_masks_per_obj) if m is not None]
    out: list[dict | None] = [None] * len(region_masks_per_obj)
    if not valid_idx:
        return out

    stacked = torch.stack(
        [region_masks_per_obj[i][name].float() for i in valid_idx for name in _REGION_ORDER]
    ).unsqueeze(1)  # (n_valid*4, 1, canvas, canvas)
    weights = F.interpolate(stacked, size=(h, w), mode="area")[:, 0].to(feat.device)  # (n_valid*4, h, w)

    for k, obj_idx in enumerate(valid_idx):
        obj_weights = weights[k * 4:(k + 1) * 4]  # (4,h,w), order == _REGION_ORDER
        wsum = obj_weights.sum(dim=(1, 2))  # (4,)
        if bool((wsum < MASK_MIN_STAGE_CELLS).any()):
            continue
        pooled = {}
        for j, name in enumerate(_REGION_ORDER):
            pooled[name] = (feat * obj_weights[j].unsqueeze(0)).sum(dim=(1, 2)) / wsum[j]
        out[obj_idx] = pooled
    return out


def relational_vector(pooled: dict) -> dict:
    """pooled: {"O":(C,),"E":(C,),"Bn":(C,),"Bf":(C,)}. Returns {"OE":float,...}."""
    import torch.nn.functional as F

    def d(a, b):
        return float((1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))).item())

    return {
        "OE": d(pooled["O"], pooled["E"]),
        "OBn": d(pooled["O"], pooled["Bn"]),
        "OBf": d(pooled["O"], pooled["Bf"]),
        "EBn": d(pooled["E"], pooled["Bn"]),
        "EBf": d(pooled["E"], pooled["Bf"]),
        "BnBf": d(pooled["Bn"], pooled["Bf"]),
    }


# ---------------------------------------------------------------------------
# Small stats helpers (no scipy dependency, matches project convention --
# see e5_success_vs_failure.py / n4_object_level_diagnostic.py)
# ---------------------------------------------------------------------------

def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / (vx ** 0.5 * vy ** 0.5)


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return pearson(_rank(xs), _rank(ys))


# ---------------------------------------------------------------------------
# Feature extraction main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()

    import torch

    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, gt_to_canvas, load_canvas_image, load_coco, load_manifest
    from transfer_attack.io_utils import get_logger, load_noise
    from transfer_attack.models import build_model_handle, get_spec

    logger = get_logger()

    from mmdet.utils import register_all_modules

    register_all_modules()

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    noise_root = args.noise_root or (PROJECT_DIR / "results" / "noise" / args.manifest.stem)
    variant_dirs = {tag: noise_root / subdir for tag, subdir in VARIANTS.items()}
    for tag, d in variant_dirs.items():
        if not d.exists():
            logger.error(f"noise dir for variant {tag!r} not found: {d} -- craft it first (see scripts/craft.py)")
            sys.exit(1)

    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"
    device = args.device

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean_csv_path = args.out_dir / "e10_relational_clean.csv"
    delta_csv_path = args.out_dir / "e10_relational_delta.csv"
    clean_fieldnames = ["image_id", "gt_idx", "gt_cat_id", "object_area", "model_name", "family", "stage_idx", "n_stages"] + [f"d_{r}" for r in RELATIONS]
    delta_fieldnames = (
        ["image_id", "gt_idx", "gt_cat_id", "object_area", "model_name", "family", "variant", "stage_idx", "n_stages"]
        + [f"clean_{r}" for r in RELATIONS] + [f"adv_{r}" for r in RELATIONS] + [f"delta_{r}" for r in RELATIONS]
    )

    # ------------------------------------------------------------------
    # Precompute per-image data ONCE (masks depend only on GT boxes, not on
    # model) -- perf fix vs the pilot, which rebuilt these once per model (6x
    # redundant CPU work). Iterating image_ids in order and only inserting
    # entries that pass the same "has GT / non-empty gt_boxes" filter the
    # pilot applied inline preserves the exact same (model, image) row order
    # in the output CSVs as before, for the dev_50 equivalence check.
    # ------------------------------------------------------------------
    image_data: dict[int, dict] = {}
    for image_id in image_ids:
        gt_entries = gt_index[image_id]
        if not gt_entries:
            continue
        canvas_img, scale, _, _ = load_canvas_image(img_dir, coco, image_id, args.canvas)
        gt_boxes, gt_cat_ids = gt_to_canvas(gt_entries, scale)
        if gt_boxes.shape[0] == 0:
            continue
        region_masks_per_obj = build_region_masks(gt_boxes, args.canvas, args.shrink_frac, args.expand_frac)
        areas = []
        cat_ids = []
        for obj_idx in range(gt_boxes.shape[0]):
            x1, y1, x2, y2 = gt_boxes[obj_idx].tolist()
            areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
            cat_ids.append(int(gt_cat_ids[obj_idx].item()))
        image_data[image_id] = {
            "canvas_img": canvas_img, "region_masks_per_obj": region_masks_per_obj,
            "areas": areas, "cat_ids": cat_ids,
        }

    n_obj_total = sum(1 for d in image_data.values() for m in d["region_masks_per_obj"] if m is not None)
    n_skipped_degenerate_interior = sum(1 for d in image_data.values() for m in d["region_masks_per_obj"] if m is None)

    with open(clean_csv_path, "w", newline="") as fc, open(delta_csv_path, "w", newline="") as fd:
        clean_writer = csv.DictWriter(fc, fieldnames=clean_fieldnames)
        clean_writer.writeheader()
        delta_writer = csv.DictWriter(fd, fieldnames=delta_fieldnames)
        delta_writer.writeheader()

        for model_name in args.models:
            spec = get_spec(model_name)
            family = BACKBONE_FAMILY.get(model_name, "OTHER")
            logger.info(f"=== {model_name} (family={family}) ===")
            handle = build_model_handle(spec, args.checkpoints_dir, device=device, coco=coco)
            model = handle.model

            n_images = 0
            for image_id, data in image_data.items():
                canvas_img = data["canvas_img"]
                region_masks_per_obj = data["region_masks_per_obj"]
                areas, cat_ids = data["areas"], data["cat_ids"]

                x_clean = canvas_img.to(device)
                with torch.no_grad():
                    feats_clean = model.backbone(handle.normalize(x_clean.unsqueeze(0)))
                feats_clean = [f[0] for f in feats_clean]  # drop batch dim -> list of (C,h,w)
                n_stages = len(feats_clean)

                # cache per-(object,stage) clean relational vector, reused
                # across all 3 adv variants below (avoids recomputation).
                clean_rel_cache: dict[tuple[int, int], dict] = {}

                for stage_idx, fc_stage in enumerate(feats_clean):
                    pooled_list = pool_regions_batched(fc_stage, region_masks_per_obj)
                    for obj_idx, pooled in enumerate(pooled_list):
                        if pooled is None:
                            continue
                        rel = relational_vector(pooled)
                        clean_rel_cache[(obj_idx, stage_idx)] = rel
                        clean_writer.writerow(
                            {
                                "image_id": image_id, "gt_idx": obj_idx, "gt_cat_id": cat_ids[obj_idx],
                                "object_area": areas[obj_idx],
                                "model_name": model_name, "family": family, "stage_idx": stage_idx, "n_stages": n_stages,
                                **{f"d_{r}": rel[r] for r in RELATIONS},
                            }
                        )

                for tag, vdir in variant_dirs.items():
                    noise_path = vdir / f"{image_id}.pt"
                    if not noise_path.exists():
                        continue
                    noise = load_noise(noise_path)
                    x_adv = (canvas_img + noise).clamp(0.0, 255.0).to(device)
                    with torch.no_grad():
                        feats_adv = model.backbone(handle.normalize(x_adv.unsqueeze(0)))
                    feats_adv = [f[0] for f in feats_adv]

                    for stage_idx, fa_stage in enumerate(feats_adv):
                        pooled_adv_list = pool_regions_batched(fa_stage, region_masks_per_obj)
                        for obj_idx, pooled_adv in enumerate(pooled_adv_list):
                            key = (obj_idx, stage_idx)
                            if key not in clean_rel_cache:
                                continue  # clean side wasn't poolable at this stage -- can't take a delta
                            if pooled_adv is None:
                                continue
                            rel_adv = relational_vector(pooled_adv)
                            rel_clean = clean_rel_cache[key]
                            delta_writer.writerow(
                                {
                                    "image_id": image_id, "gt_idx": obj_idx, "gt_cat_id": cat_ids[obj_idx],
                                    "object_area": areas[obj_idx],
                                    "model_name": model_name, "family": family, "variant": tag,
                                    "stage_idx": stage_idx, "n_stages": len(feats_adv),
                                    **{f"clean_{r}": rel_clean[r] for r in RELATIONS},
                                    **{f"adv_{r}": rel_adv[r] for r in RELATIONS},
                                    **{f"delta_{r}": rel_adv[r] - rel_clean[r] for r in RELATIONS},
                                }
                            )

                n_images += 1

            logger.info(f"  {model_name}: {n_images} images processed")
            del handle, model
            torch.cuda.empty_cache()

    logger.info(
        f"objects: {n_obj_total} usable, {n_skipped_degenerate_interior} skipped "
        f"(degenerate interior after shrink -- box too small for shrink_frac={args.shrink_frac})"
    )
    logger.info(f"wrote clean detail -> {clean_csv_path}")
    logger.info(f"wrote delta detail -> {delta_csv_path}")

    # ------------------------------------------------------------------
    # Analysis pass -- reads the two CSVs just written (kept as a separate
    # in-process step, not a second script invocation, so the whole thing is
    # one command; still fully separable if this needs to be rerun on saved
    # CSVs alone later).
    # ------------------------------------------------------------------
    analyze(args, clean_csv_path, delta_csv_path, logger)


# ---------------------------------------------------------------------------
# Analysis: Q1 (Table A), Q2 (Table B), Q3 (Table C)
# ---------------------------------------------------------------------------

def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def last_stage_only(rows: list[dict]) -> list[dict]:
    """Keep only stage_idx == n_stages-1 rows (deepest/most semantic stage per
    model -- architecture-agnostic since it's relative to each model's own
    stage count, not an absolute index)."""
    out = []
    for r in rows:
        if int(r["stage_idx"]) == int(r["n_stages"]) - 1:
            out.append(r)
    return out


def fetch_asr(args, tag: str, attack_dir_name: str, models: list[str], coco, image_ids, gt_index, img_dir, logger) -> dict:
    """ASR per model for one variant, via evaluate.py's own machinery (same
    pattern e3_osfd_rrb_factorial.py uses -- attack tag is just a noise-dir/
    prediction-cache namespace here, not validated against evaluate.py's CLI
    choices list)."""
    from types import SimpleNamespace

    import evaluate as evaluate_mod
    from transfer_attack.models import get_spec

    noise_root = args.noise_root or (PROJECT_DIR / "results" / "noise" / args.manifest.stem)
    eval_args = SimpleNamespace(
        checkpoints_dir=args.checkpoints_dir,
        canvas=args.canvas,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        noise_dir=noise_root,
        predictions_dir=args.predictions_dir,
        force_clean=False,
        device=args.device,
        attacks=[attack_dir_name],
    )
    gt_cache = evaluate_mod.build_gt_cache(coco, image_ids, img_dir, args.canvas)
    asr_by_model = {}
    for model_name in models:
        spec = get_spec(model_name)
        rows = evaluate_mod.evaluate_one_model(spec, eval_args, coco, image_ids, img_dir, gt_cache, logger)
        for r in rows:
            if r["attack"] == attack_dir_name:
                asr_by_model[model_name] = r["ASR"]
    return asr_by_model


def table_a_clean_invariance(clean_rows: list[dict], models: list[str], out_path: Path, logger) -> dict:
    """Returns shared_set: {relation: {"global_rho": float, "is_shared": bool}}."""
    last = last_stage_only(clean_rows)
    logger.info(f"Table A: {len(last)} (model,object) rows at last-stage after mask-coverage filtering")
    # object key -> model -> {relation: value}
    by_obj: dict[tuple, dict] = {}
    for r in last:
        key = (r["image_id"], r["gt_idx"])
        by_obj.setdefault(key, {})[r["model_name"]] = {rel: float(r[f"d_{rel}"]) for rel in RELATIONS}

    n_all_models = sum(1 for per_model in by_obj.values() if len(per_model) == len(models))
    logger.info(f"Table A: {len(by_obj)} distinct objects, {n_all_models} with valid last-stage data for ALL {len(models)} models")

    pairs = list(itertools.combinations(models, 2))
    family_of = {m: BACKBONE_FAMILY.get(m, "OTHER") for m in models}

    table = []
    shared_set = {}
    for rel in RELATIONS:
        pair_rhos = {}
        for m1, m2 in pairs:
            xs, ys = [], []
            for obj, per_model in by_obj.items():
                if m1 in per_model and m2 in per_model:
                    xs.append(per_model[m1][rel])
                    ys.append(per_model[m2][rel])
            pair_rhos[(m1, m2)] = spearman(xs, ys) if len(xs) >= 5 else float("nan")

        def _mean(vals):
            vals = [v for v in vals if v == v]
            return sum(vals) / len(vals) if vals else float("nan")

        r50_vals = [rho for (m1, m2), rho in pair_rhos.items() if family_of[m1] == "R50" and family_of[m2] == "R50"]
        swin_vals = [rho for (m1, m2), rho in pair_rhos.items() if family_of[m1] == "Swin" and family_of[m2] == "Swin"]
        csp_vals = [rho for (m1, m2), rho in pair_rhos.items() if family_of[m1] == "CSP" and family_of[m2] == "CSP"]
        cross_vals = [rho for (m1, m2), rho in pair_rhos.items() if family_of[m1] != family_of[m2]]
        global_vals = list(pair_rhos.values())

        row = {
            "relation": RELATION_LABEL[rel],
            "R50_consistency": _mean(r50_vals),
            "CSP_consistency": _mean(csp_vals),
            "Swin_consistency": _mean(swin_vals),
            "cross_family_consistency": _mean(cross_vals),
            "global": _mean(global_vals),
        }
        table.append(row)
        is_shared = (row["global"] == row["global"] and row["global"] > GO1_CONSISTENCY_THR
                     and row["cross_family_consistency"] == row["cross_family_consistency"]
                     and row["cross_family_consistency"] > GO1_CONSISTENCY_THR)
        shared_set[rel] = {"global_rho": row["global"], "cross_family_rho": row["cross_family_consistency"], "is_shared": is_shared}

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relation", "R50_consistency", "CSP_consistency", "Swin_consistency", "cross_family_consistency", "global"])
        writer.writeheader()
        writer.writerows(table)
    logger.info(f"wrote Table A -> {out_path}")
    logger.info("=== Table A: clean relational invariance (Spearman rho, last stage) ===")
    for row in table:
        logger.info(
            f"  {row['relation']:14s} R50={row['R50_consistency']:.3f} CSP={row['CSP_consistency']:.3f} "
            f"Swin={row['Swin_consistency']:.3f} cross_family={row['cross_family_consistency']:.3f} global={row['global']:.3f}"
        )
    n_shared = sum(1 for v in shared_set.values() if v["is_shared"])
    logger.info(f"shared/invariant relations (global>{GO1_CONSISTENCY_THR} AND cross_family>{GO1_CONSISTENCY_THR}): {n_shared}/6")
    return shared_set


def disrupt_norm(row: dict, relations: list[str]) -> float:
    return sum(float(row[f"delta_{r}"]) ** 2 for r in relations) ** 0.5


def table_b_attack_disruption(delta_rows: list[dict], models: list[str], shared_set: dict, asr_osfd: dict, out_path: Path, logger) -> None:
    last = last_stage_only([r for r in delta_rows if r["variant"] == "osfd"])
    shared_rels = [r for r, v in shared_set.items() if v["is_shared"]]

    per_model_deltas: dict[str, dict[str, list[float]]] = {m: {rel: [] for rel in RELATIONS} for m in models}
    per_model_shared_disrupt: dict[str, list[float]] = {m: [] for m in models}
    per_model_all_disrupt: dict[str, list[float]] = {m: [] for m in models}

    for r in last:
        m = r["model_name"]
        if m not in per_model_deltas:
            continue
        for rel in RELATIONS:
            per_model_deltas[m][rel].append(float(r[f"delta_{rel}"]))
        per_model_all_disrupt[m].append(disrupt_norm(r, RELATIONS))
        if shared_rels:
            per_model_shared_disrupt[m].append(disrupt_norm(r, shared_rels))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    table = []
    for m in models:
        row = {"model_name": m, "family": BACKBONE_FAMILY.get(m, "OTHER"), "ASR": asr_osfd.get(m)}
        for rel in RELATIONS:
            row[f"mean_delta_{rel}"] = mean(per_model_deltas[m][rel])
        row["disrupt_all6"] = mean(per_model_all_disrupt[m])
        row["disrupt_shared"] = mean(per_model_shared_disrupt[m]) if shared_rels else float("nan")
        table.append(row)

    fieldnames = ["model_name", "family", "ASR"] + [f"mean_delta_{r}" for r in RELATIONS] + ["disrupt_all6", "disrupt_shared"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    logger.info(f"wrote Table B -> {out_path}")

    logger.info(f"=== Table B: OSFD attack disruption (shared relations = {[RELATION_LABEL[r] for r in shared_rels]}) ===")
    for row in table:
        logger.info(
            f"  {row['model_name']:20s} ASR={row['ASR']:.1f} disrupt_shared={row['disrupt_shared']:.4f} "
            f"disrupt_all6={row['disrupt_all6']:.4f}"
        )
    # cross-target correlation, ASR vs shared disruption
    xs = [row["ASR"] for row in table if row["ASR"] == row["ASR"]]
    ys = [row["disrupt_shared"] for row in table if row["ASR"] == row["ASR"]]
    if len(xs) >= 3 and shared_rels:
        logger.info(f"corr(ASR, disrupt_shared) across {len(xs)} targets = {pearson(xs, ys):.3f}")
    # matched pairs
    by_model = {row["model_name"]: row for row in table}
    for pair_name, r50_name, swin_name in MATCHED_PAIRS:
        if r50_name in by_model and swin_name in by_model:
            r50, swin = by_model[r50_name], by_model[swin_name]
            logger.info(
                f"matched pair [{pair_name}]: ASR {r50_name}={r50['ASR']:.1f} vs {swin_name}={swin['ASR']:.1f} | "
                f"disrupt_shared {r50_name}={r50['disrupt_shared']:.4f} vs {swin_name}={swin['disrupt_shared']:.4f} -> "
                f"{'MATCHES direction' if (r50['ASR'] > swin['ASR']) == (r50['disrupt_shared'] > swin['disrupt_shared']) else 'DOES NOT MATCH direction'}"
            )


def table_c_rrb_mechanism(delta_rows: list[dict], models: list[str], shared_set: dict, asr_norrb: dict, asr_rrb: dict, out_path: Path, logger) -> None:
    shared_rels = [r for r, v in shared_set.items() if v["is_shared"]]
    last_norrb = last_stage_only([r for r in delta_rows if r["variant"] == "k1_norrb"])
    last_rrb = last_stage_only([r for r in delta_rows if r["variant"] == "k1_rrb"])

    def per_model_mean_disrupt(rows, relations):
        acc: dict[str, list[float]] = {m: [] for m in models}
        for r in rows:
            m = r["model_name"]
            if m in acc:
                acc[m].append(disrupt_norm(r, relations))
        return {m: (sum(v) / len(v) if v else float("nan")) for m, v in acc.items()}

    norrb_disrupt = per_model_mean_disrupt(last_norrb, shared_rels if shared_rels else RELATIONS)
    rrb_disrupt = per_model_mean_disrupt(last_rrb, shared_rels if shared_rels else RELATIONS)

    table = []
    for m in models:
        d_norrb, d_rrb = norrb_disrupt.get(m, float("nan")), rrb_disrupt.get(m, float("nan"))
        a_norrb, a_rrb = asr_norrb.get(m), asr_rrb.get(m)
        row = {
            "model_name": m, "family": BACKBONE_FAMILY.get(m, "OTHER"),
            "noRRB_disruption_shared": d_norrb, "RRB_disruption_shared": d_rrb,
            "delta_disruption": (d_rrb - d_norrb) if (d_norrb == d_norrb and d_rrb == d_rrb) else float("nan"),
            "ASR_noRRB": a_norrb, "ASR_RRB": a_rrb,
            "delta_ASR": (a_rrb - a_norrb) if (a_norrb is not None and a_rrb is not None) else None,
        }
        table.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "family", "noRRB_disruption_shared", "RRB_disruption_shared", "delta_disruption", "ASR_noRRB", "ASR_RRB", "delta_ASR"])
        writer.writeheader()
        writer.writerows(table)
    logger.info(f"wrote Table C -> {out_path}")

    logger.info(f"=== Table C: RRB mechanism (disruption restricted to shared relations = {[RELATION_LABEL[r] for r in shared_rels]}) ===")
    n_match = 0
    n_total = 0
    for row in table:
        if row["delta_ASR"] is None or row["delta_disruption"] != row["delta_disruption"]:
            continue
        n_total += 1
        matches = (row["delta_ASR"] > 0) == (row["delta_disruption"] > 0)
        n_match += int(matches)
        logger.info(
            f"  {row['model_name']:20s} delta_disruption={row['delta_disruption']:+.4f} "
            f"delta_ASR={row['delta_ASR']:+.1f} {'MATCH' if matches else 'MISMATCH'}"
        )
    if n_total:
        logger.info(f"RRB hypothesis (delta_disruption and delta_ASR same sign): {n_match}/{n_total} models")


def analyze(args, clean_csv_path: Path, delta_csv_path: Path, logger) -> None:
    from transfer_attack.constants import COCO_ANN_FILE
    from transfer_attack.data import build_gt_index, load_coco, load_manifest

    coco = load_coco(PROJECT_DIR / COCO_ANN_FILE)
    manifest = load_manifest(args.manifest)
    image_ids = manifest["image_ids"]
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    gt_index = build_gt_index(coco, image_ids)
    img_dir = args.data_dir / "val2017"

    clean_rows = load_csv_rows(clean_csv_path)
    delta_rows = load_csv_rows(delta_csv_path)

    shared_set = table_a_clean_invariance(clean_rows, args.models, args.out_dir / "e10_table_A_clean_invariance.csv", logger)

    asr_osfd = fetch_asr(args, "osfd", "osfd", args.models, coco, image_ids, gt_index, img_dir, logger)
    table_b_attack_disruption(delta_rows, args.models, shared_set, asr_osfd, args.out_dir / "e10_table_B_attack_disruption.csv", logger)

    asr_norrb = fetch_asr(args, "k1_norrb", "e3_k1_norrb", args.models, coco, image_ids, gt_index, img_dir, logger)
    asr_rrb = fetch_asr(args, "k1_rrb", "e3_k1_rrb", args.models, coco, image_ids, gt_index, img_dir, logger)
    table_c_rrb_mechanism(delta_rows, args.models, shared_set, asr_norrb, asr_rrb, args.out_dir / "e10_table_C_rrb_mechanism.csv", logger)

    n_shared = sum(1 for v in shared_set.values() if v["is_shared"])
    logger.info("=== E10 verdict (pre-registered criteria, RESEARCH.md to be updated with full reasoning) ===")
    logger.info(f"GO-A (existence): {n_shared}/6 relations shared (need >= {GO1_MIN_RELATIONS}) -> {'PASS' if n_shared >= GO1_MIN_RELATIONS else 'FAIL'}")


if __name__ == "__main__":
    main()
