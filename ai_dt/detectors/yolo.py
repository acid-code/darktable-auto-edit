"""Ultralytics YOLOv8 detector.

Swap the weights by passing another ``model_path`` (or the
``AI_EDITOR_YOLO_MODEL`` environment variable / Darktable preference).
A custom model that labels faces or eyes is picked up by name: any class
containing ``face`` or ``eye`` is mapped to that category. Other classes are
mapped through :data:`COCO_CATEGORIES`. Edit that dictionary to follow a
different label set. Classes that are not listed are ignored on purpose, so a
COCO model does not turn chairs and bottles into local edits.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from ai_dt.errors import DetectorUnavailable
from ai_dt.schema import Detection

# COCO names this plugin is willing to treat as a photographic subject.
COCO_CATEGORIES = {
    "person": "person",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",
    "bird": "animal",
    "bicycle": "subject",
    "motorcycle": "subject",
    "train": "subject",
    "bus": "subject",
    "car": "subject",
    "boat": "subject",
}


def category_for(class_name: str) -> str | None:
    key = class_name.lower().strip()
    if key in COCO_CATEGORIES:
        return COCO_CATEGORIES[key]
    if "eye" in key:
        return "eye"
    if "face" in key:
        return "face"
    if "building" in key or "facade" in key or "architecture" in key:
        return "architecture"
    return None


# yolov8n scores a partly hidden cat, bike, or person under the usual 0.35
# bar. These categories stay, down to this floor. Architecture does not.
_RECALL_FLOOR = {
    "animal": 0.15,
    "person": 0.15,
    "face": 0.15,
    "eye": 0.15,
    "subject": 0.15,
}


def predict_confidence(min_confidence: float = 0.35) -> float:
    """The threshold passed to the model. The model drops boxes before we see them."""

    if min_confidence > 0.35:
        return float(min_confidence)
    return float(min(min_confidence, min(_RECALL_FLOOR.values())))


def passes_confidence(class_name: str, confidence: float, min_confidence: float = 0.35) -> bool:
    """Keep a mapped subject. A partly hidden one can sit under 0.35."""

    category = category_for(class_name)
    if category is None:
        return False
    if min_confidence > 0.35 or category not in _RECALL_FLOOR:
        needed = min_confidence
    else:
        needed = min(min_confidence, _RECALL_FLOOR[category])
    return float(confidence) >= needed


def _model_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    base = Path(root) if root else Path.home()
    path = base / "darktable" / "ai-editor-models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_weights(model_path: str) -> str:
    """Load a bare model name from the editor cache, not the working directory."""

    path = Path(model_path)
    if path.is_absolute() and path.is_file():
        return str(path)
    if path.parent == Path("."):
        cached = _model_dir() / path.name
        stray = Path.cwd() / path.name
        if cached.is_file():
            if stray.is_file() and stray.resolve() != cached.resolve():
                stray.unlink()
            return str(cached)
        if stray.is_file():
            stray.replace(cached)
            return str(cached)
    if path.is_file():
        return str(path.resolve())
    return model_path


class YoloDetector:
    name = "yolo"

    def __init__(self, model_path: str = "yolov8n.pt", min_confidence: float = 0.35) -> None:
        self.model_path = model_path
        self.min_confidence = min_confidence

    def detect(self, image: Image.Image) -> list[Detection]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailable(
                "ultralytics is not installed. Install requirements-ai.txt or "
                "choose the heuristic detector."
            ) from exc

        model_path = resolve_weights(self.model_path)
        try:
            model = YOLO(model_path)
            results = model.predict(
                source=np.asarray(image.convert("RGB")),
                conf=predict_confidence(self.min_confidence),
                verbose=False,
            )
        except Exception as exc:
            raise DetectorUnavailable(f"YOLO model {model_path!r} failed: {exc}") from exc

        names = model.names or {}
        found: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = str(names.get(class_id, class_id))
                if not passes_confidence(class_name, confidence, self.min_confidence):
                    continue
                category = category_for(class_name)
                if category is None:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxyn[0].tolist())
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width < 0.01 or height < 0.01:
                    continue
                found.append(
                    Detection(
                        label=class_name,
                        category=category,
                        confidence=confidence,
                        bbox=(x1, y1, width, height),
                        source=self.name,
                    )
                )
        return found
