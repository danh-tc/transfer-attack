"""Noise tensor / prediction (de)serialization and lightweight logging helpers."""
from __future__ import annotations

import json
import logging
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
