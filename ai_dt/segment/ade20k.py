"""Landscape classes from an ADE20K semantic model (SegFormer or OneFormer).

``pip install transformers torch`` (CPU wheels are fine on Windows). The
model is ``AI_DT_ADE20K_MODEL``, default
``nvidia/segformer-b2-finetuned-ade-512-512`` (about 100 MB, a few seconds
per preview on a laptop CPU). A OneFormer id such as
``shi-labs/oneformer_ade20k_swin_tiny`` works too and is slower. The
label map is read from the model's ``id2label``, so any ADE20K model with
the usual names maps onto the same classes.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np

from ai_dt.segment.base import LANDSCAPE_CLASSES, SegmenterUnavailable

DEFAULT_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"

# ADE20K names (first word before a comma or semicolon) for each class.
CLASS_LABELS: dict[str, frozenset[str]] = {
    "sky": frozenset({"sky"}),
    "vegetation": frozenset({"tree", "grass", "plant", "flower", "palm", "field", "flora"}),
    "water": frozenset({"water", "sea", "river", "lake", "swimming pool", "waterfall", "fountain"}),
    "architecture": frozenset({
        "building", "house", "skyscraper", "tower", "bridge", "wall", "column", "fence",
        "door", "windowpane", "window", "roof", "railing", "awning", "stairs", "stairway",
        "step", "booth", "arcade", "temple", "castle", "church", "hovel",
    }),
    "ground": frozenset({
        "road", "floor", "earth", "ground", "sand", "rock", "stone", "path", "sidewalk",
        "dirt track", "land", "hill", "mountain", "runway", "grandstand", "pavement",
    }),
    "people": frozenset({"person", "people"}),
}


def _base_name(label: str) -> str:
    return re.split(r"[,;]", str(label).lower())[0].strip()


def class_ids(id2label: dict) -> dict[str, set[int]]:
    found: dict[str, set[int]] = {name: set() for name in CLASS_LABELS}
    for raw_id, label in id2label.items():
        name = _base_name(label)
        for cls, labels in CLASS_LABELS.items():
            if name in labels:
                found[cls].add(int(raw_id))
    return found


def masks_from_labels(label_map: np.ndarray, id2label: dict, classes) -> dict[str, np.ndarray]:
    ids = class_ids(id2label)
    out: dict[str, np.ndarray] = {}
    for cls in classes:
        wanted = ids.get(cls)
        if not wanted:
            continue
        mask = np.isin(label_map, list(wanted)).astype(np.float32)
        if mask.any():
            out[cls] = mask
    return out


@lru_cache(maxsize=1)
def _load(model_id: str):
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except Exception as exc:
        raise SegmenterUnavailable(f"transformers/torch are not installed ({exc})") from exc
    if "oneformer" in model_id.lower():
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        return "oneformer", OneFormerProcessor.from_pretrained(model_id), OneFormerForUniversalSegmentation.from_pretrained(model_id)
    return "semantic", AutoImageProcessor.from_pretrained(model_id), AutoModelForSemanticSegmentation.from_pretrained(model_id)


class Ade20kSegmenter:
    name = "ade20k"
    classes = (*LANDSCAPE_CLASSES, "people")

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.environ.get("AI_DT_ADE20K_MODEL", DEFAULT_MODEL)

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        return True

    def segment(self, image, classes, *, boxes: list) -> dict[str, np.ndarray]:
        import torch

        kind, processor, model = _load(self.model_id)
        target = [(image.height, image.width)]
        with torch.no_grad():
            if kind == "oneformer":
                inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
                outputs = model(**inputs)
                label_map = processor.post_process_semantic_segmentation(outputs, target_sizes=target)[0]
            else:
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                label_map = processor.post_process_semantic_segmentation(outputs, target_sizes=target)[0]
        return masks_from_labels(np.asarray(label_map.cpu().numpy()), model.config.id2label, classes)
