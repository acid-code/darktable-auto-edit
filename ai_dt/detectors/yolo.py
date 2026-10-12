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

        try:
            model = YOLO(self.model_path)
            results = model.predict(source=np.asarray(image.convert("RGB")), verbose=False)
        except Exception as exc:
            raise DetectorUnavailable(f"YOLO model {self.model_path!r} failed: {exc}") from exc

        names = model.names or {}
        found: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.min_confidence:
                    continue
                class_id = int(box.cls[0])
                class_name = str(names.get(class_id, class_id))
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
