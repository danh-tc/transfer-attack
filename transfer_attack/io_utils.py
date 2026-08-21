"""Noise tensor / prediction (de)serialization and lightweight logging helpers."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor


def get_logger(name: str = "transfer_attack") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_run_log(runs_dir: Path, run_type: str, tag: str, payload: dict) -> Path:
    """Write one JSON log per invocation of craft.py/evaluate.py under `runs_dir`
    (repo-tracked, unlike results/ and logs/ which hold large regenerable
    artifacts and are gitignored). `payload` should carry the full config used
    plus a compact result summary -- NOT bulky per-image tensors/predictions,
    those stay under results/.

    File name: {run_type}_{tag}_{UTC timestamp}.json, e.g.
    craft_osfd_20260821T150352Z.json -- timestamp makes repeated runs additive
    (each run gets its own log entry) rather than overwriting history.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = runs_dir / f"{run_type}_{tag}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump({"timestamp": timestamp, **payload}, f, indent=2)
    return path


def save_noise(path: Path, noise: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(noise.cpu(), path)


def load_noise(path: Path, device: str = "cpu") -> Tensor:
    return torch.load(path, map_location=device)


def save_predictions(path: Path, preds_by_image_id: dict[int, dict]) -> None:
    """preds_by_image_id: {image_id: {"bboxes": Tensor[P,4], "scores": Tensor[P], "labels": Tensor[P]}}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        str(image_id): {
            "bboxes": pred["bboxes"].tolist(),
            "scores": pred["scores"].tolist(),
            "labels": pred["labels"].tolist(),
        }
        for image_id, pred in preds_by_image_id.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_predictions(path: Path) -> dict[int, dict]:
    with open(path, "r") as f:
        raw = json.load(f)
    out = {}
    for image_id_str, pred in raw.items():
        out[int(image_id_str)] = {
            "bboxes": torch.tensor(pred["bboxes"], dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(pred["scores"], dtype=torch.float32),
            "labels": torch.tensor(pred["labels"], dtype=torch.long),
        }
    return out
