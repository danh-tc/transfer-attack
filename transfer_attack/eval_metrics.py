"""ASR (evasion success rate) and COCO mAP computation.

ASR definition (confirmed with user): for a given target model, first find the
GT instances correctly detected on the CLEAN image (matching class, score >=
score_thr, IoU >= iou_thr against that GT box, standard greedy
highest-IoU/highest-score matching). Then, on the ADVERSARIAL image, check
whether each of those instances still has a matching prediction under the same
rule -- if not, it counts as "evaded". ASR = evaded / |clean-correct set|,
aggregated over all images.
"""
from __future__ import annotations

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor

from transfer_attack.constants import DEFAULT_IOU_THR, DEFAULT_SCORE_THR
from transfer_attack.models import ModelHandle


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """boxes1: (N,4), boxes2: (M,4), both xyxy. Returns (N,M) IoU matrix."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def greedy_match(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    pred_labels: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    iou_thr: float = DEFAULT_IOU_THR,
    score_thr: float = DEFAULT_SCORE_THR,
) -> list[int | None]:
    """Returns, per GT index, the matched prediction index (into the ORIGINAL
    pred_* tensors) or None. Standard COCO-style greedy assignment: filter
    preds by score>=thr, sort desc by score, for each pred find the
    highest-IoU unmatched GT of the same label with IoU>=iou_thr."""
    n_gt = gt_boxes.shape[0]
    matched: list[int | None] = [None] * n_gt
    if n_gt == 0 or pred_boxes.shape[0] == 0:
        return matched

    keep = (pred_scores >= score_thr).nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        return matched
    order = keep[torch.argsort(pred_scores[keep], descending=True)]

    iou = box_iou(pred_boxes, gt_boxes)  # (P,G)
    gt_taken = torch.zeros(n_gt, dtype=torch.bool)

    for p_idx in order.tolist():
        same_label = gt_labels == pred_labels[p_idx]
        candidate_iou = iou[p_idx].clone()
        candidate_iou[~same_label] = -1.0
        candidate_iou[gt_taken] = -1.0
        best_gt = int(torch.argmax(candidate_iou).item())
        if candidate_iou[best_gt] >= iou_thr:
            matched[best_gt] = p_idx
            gt_taken[best_gt] = True

    return matched


def compute_asr_for_image(
    clean_pred: dict,
    adv_pred: dict,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    iou_thr: float = DEFAULT_IOU_THR,
    score_thr: float = DEFAULT_SCORE_THR,
) -> tuple[int, int]:
    """clean_pred/adv_pred: {"bboxes": Tensor[P,4], "scores": Tensor[P], "labels": Tensor[P]},
    all in the SAME coordinate space as gt_boxes (canvas-space; IoU is
    scale-invariant within one comparison so no coordinate conversion needed).

    Returns (num_evaded, num_clean_correct) for this image -- callers sum both
    across the dataset before dividing.
    """
    clean_match = greedy_match(
        clean_pred["bboxes"], clean_pred["scores"], clean_pred["labels"], gt_boxes, gt_labels, iou_thr, score_thr
    )
    clean_correct_idx = [i for i, m in enumerate(clean_match) if m is not None]
    if not clean_correct_idx:
        return 0, 0

    adv_match = greedy_match(
        adv_pred["bboxes"], adv_pred["scores"], adv_pred["labels"], gt_boxes, gt_labels, iou_thr, score_thr
    )
    evaded = sum(1 for i in clean_correct_idx if adv_match[i] is None)
    return evaded, len(clean_correct_idx)


def to_coco_results(
    preds_by_image_id: dict[int, dict],
    label_to_cat_id: dict[int, int],
    scale_by_image_id: dict[int, float],
) -> list[dict]:
    """Converts canvas-space (boxes,scores,labels) predictions -> original-
    image-space COCO result dicts {image_id, category_id, bbox:[x,y,w,h], score}."""
    results = []
    for image_id, pred in preds_by_image_id.items():
        scale = scale_by_image_id[image_id]
        boxes = pred["bboxes"] / scale
        for box, score, label in zip(boxes.tolist(), pred["scores"].tolist(), pred["labels"].tolist()):
            x1, y1, x2, y2 = box
            results.append(
                {
                    "image_id": image_id,
                    "category_id": label_to_cat_id[int(label)],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
    return results


def compute_coco_map(coco_gt: COCO, results: list[dict], img_ids: list[int]) -> dict:
    if not results:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
    }


@torch.no_grad()
def predict_canvas(handle: ModelHandle, x_canvas: Tensor, canvas: int, device: str = "cuda:0") -> dict:
    """x_canvas: (3,canvas,canvas) canonical RGB [0,255] pixel-space image.

    Deliberately NOT using mmdet.apis.inference_detector -- that helper re-runs
    the model's own configured test pipeline (its own resize scale, its own
    numpy/BGR conventions), which would silently defeat the fixed-canvas
    design. model.predict(batch_inputs, batch_data_samples, rescale=True) is
    the documented lower-level entry point BaseDetector.forward(mode='predict')
    dispatches to, and is the correct extension point here. ori_shape ==
    img_shape == canvas and scale_factor=(1,1) make `rescale=True` a no-op,
    since we've already done our own resize+pad.

    Returns {"bboxes": Tensor[P,4] xyxy canvas-space, "scores": Tensor[P],
    "labels": Tensor[P]} with labels remapped to RAW COCO category ids (via
    handle.label_to_cat_id) so predictions from different models are directly
    comparable against the shared GT category-id space.
    """
    from mmdet.structures import DetDataSample

    x_norm = handle.normalize(x_canvas.unsqueeze(0).to(device))
    ds = DetDataSample()
    ds.set_metainfo(
        dict(
            img_shape=(canvas, canvas),
            ori_shape=(canvas, canvas),
            scale_factor=(1.0, 1.0),
            # DETR-family detectors (deformable_detr, dino_*) read this off
            # batch_data_samples[0] to build the transformer padding mask;
            # non-DETR detectors ignore it. Our canvas is already a fixed
            # padded square, so it's just (canvas, canvas).
            batch_input_shape=(canvas, canvas),
        )
    )
    out = handle.model.predict(x_norm, [ds], rescale=True)[0]
    inst = out.pred_instances
    labels = inst.labels.cpu()
    cat_ids = torch.tensor([handle.label_to_cat_id[int(l)] for l in labels], dtype=torch.long)
    return {"bboxes": inst.bboxes.cpu(), "scores": inst.scores.cpu(), "labels": cat_ids}
