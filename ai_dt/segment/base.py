"""The segmenter interface shared by every backend."""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

LANDSCAPE_CLASSES = ("sky", "architecture", "vegetation", "water", "ground")
BOX_CLASSES = ("subject", "people")
ALL_CLASSES = ("subject", "background", "people", *LANDSCAPE_CLASSES)

# A mask that covers almost nothing, or the whole frame, is not a region.
MIN_COVER = 0.01
MAX_COVER = 0.97


class SegmenterUnavailable(RuntimeError):
    """The backend's package, model files, or key are missing."""


class Segmenter(Protocol):
    name: str
    classes: tuple[str, ...]

    def available(self) -> bool: ...

    def segment(self, image, classes: Iterable[str], *, boxes: list) -> dict[str, np.ndarray]: ...


def usable(mask: np.ndarray) -> bool:
    cover = float((np.asarray(mask) >= 0.5).mean())
    return MIN_COVER <= cover <= MAX_COVER


def box_of(item) -> tuple[str, tuple[float, float, float, float]] | None:
    """``(category, (x, y, w, h))`` from a Detection or a dict, or None."""

    if isinstance(item, dict):
        category = str(item.get("category") or "")
        raw = item.get("bbox")
    else:
        category = str(getattr(item, "category", "") or "")
        raw = getattr(item, "bbox", None)
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x, y, w, h = (float(part) for part in raw)
    except (TypeError, ValueError):
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    return category, (x, y, w, h)


def boxes_for(boxes: list, categories: set[str]) -> list[tuple[float, float, float, float]]:
    found = []
    for item in boxes or []:
        parsed = box_of(item)
        if parsed and parsed[0] in categories:
            found.append(parsed[1])
    return found


SUBJECT_CATEGORIES = {"subject", "person", "animal", "face"}
PEOPLE_CATEGORIES = {"person", "face"}
