"""Semantic masks for drawn-mask steps: subject, sky, background, people, landscape classes.

``segment(image, classes)`` asks each backend in turn and keeps the first
usable mask for every class. Backends are optional and imported lazily:

- ``darktable-sam``: darktable 5.6's own SAM 2.1 ONNX files (subject and
  people, from a detector box). Needs ``onnxruntime`` + ``opencv``.
- ``rembg``: subject / background (and people with ``u2net_human_seg``).
  ``pip install rembg``. CPU ONNX.
- ``ade20k``: SegFormer or OneFormer trained on ADE20K, for sky,
  architecture, vegetation, water, ground and people.
  ``pip install transformers torch`` (CPU wheels work on Windows).
- ``gemini``: cloud segmentation. Off unless asked for with
  ``AI_DT_SEGMENT_GEMINI=1`` or ``backend="gemini"``, because it uploads
  the preview. Needs ``GEMINI_API_KEY``.
- ``heuristic``: NumPy only. Always last, always available.

Every mask is float32 0..1 at the image size. A class with no usable
region is missing from the result rather than an empty mask.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ai_dt.segment.base import (
    ALL_CLASSES,
    BOX_CLASSES,
    LANDSCAPE_CLASSES,
    Segmenter,
    SegmenterUnavailable,
    usable,
)

BACKENDS = ("auto", "darktable-sam", "rembg", "ade20k", "gemini", "heuristic")


@dataclass
class SegmentResult:
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def get(self, name: str) -> np.ndarray | None:
        return self.masks.get(name)


def _make(name: str) -> Segmenter:
    if name == "heuristic":
        from ai_dt.segment.heuristic import HeuristicSegmenter

        return HeuristicSegmenter()
    if name == "darktable-sam":
        from ai_dt.segment.darktable_sam import DarktableSamSegmenter

        return DarktableSamSegmenter()
    if name == "rembg":
        from ai_dt.segment.rembg_backend import RembgSegmenter

        return RembgSegmenter()
    if name == "ade20k":
        from ai_dt.segment.ade20k import Ade20kSegmenter

        return Ade20kSegmenter()
    if name == "gemini":
        from ai_dt.segment.gemini import GeminiSegmenter

        return GeminiSegmenter()
    raise ValueError(f"unknown segmenter {name!r}. Use: {', '.join(BACKENDS)}")


def backend_order(backend: str = "auto") -> list[str]:
    choice = (backend or os.environ.get("AI_DT_SEGMENTER") or "auto").strip().lower()
    if choice != "auto":
        return [choice] if choice == "heuristic" else [choice, "heuristic"]
    order = ["darktable-sam", "rembg", "ade20k"]
    if os.environ.get("AI_DT_SEGMENT_GEMINI", "").strip() in {"1", "true", "yes"}:
        order.append("gemini")
    order.append("heuristic")
    return order


def segment(
    image: Image.Image,
    classes: list[str] | tuple[str, ...],
    *,
    boxes: list | None = None,
    backend: str = "auto",
    segmenters: list[Segmenter] | None = None,
) -> SegmentResult:
    """Masks for ``classes``. ``boxes`` are detector ``Detection``-like objects or dicts."""

    wanted = [name for name in dict.fromkeys(classes) if name in ALL_CLASSES]
    result = SegmentResult()
    if not wanted:
        return result
    rgb = image.convert("RGB")
    chain = segmenters if segmenters is not None else []
    if segmenters is None:
        for name in backend_order(backend):
            try:
                chain.append(_make(name))
            except ValueError:
                raise
            except Exception as exc:  # an optional backend that cannot even load
                result.notes.append(f"{name}: {exc}")
    # Background is the inverse of the subject. Ask for the subject instead.
    need = set(wanted)
    if "background" in need:
        need.add("subject")
    for segmenter in chain:
        missing = [name for name in ALL_CLASSES if name in need and name not in result.masks]
        offered = [name for name in missing if name in segmenter.classes]
        if not offered:
            continue
        try:
            if not segmenter.available():
                result.notes.append(f"{segmenter.name}: not installed")
                continue
            found = segmenter.segment(rgb, offered, boxes=boxes or [])
        except SegmenterUnavailable as exc:
            result.notes.append(f"{segmenter.name}: {exc}")
            continue
        except Exception as exc:  # a model that fails at run time is skipped
            result.notes.append(f"{segmenter.name} failed: {exc}")
            continue
        for name, mask in (found or {}).items():
            if name in offered and name not in result.masks:
                fitted = _fit(mask, rgb.size)
                if fitted is not None and usable(fitted):
                    result.masks[name] = fitted
                    result.sources[name] = segmenter.name
    if "background" in need and "background" not in result.masks and "subject" in result.masks:
        result.masks["background"] = (1.0 - result.masks["subject"]).astype(np.float32)
        result.sources["background"] = result.sources["subject"] + " (inverse subject)"
    for name in list(result.masks):
        if name not in wanted:
            del result.masks[name]
            result.sources.pop(name, None)
    return result


def _fit(mask: np.ndarray | None, size: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    values = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if values.ndim != 2:
        return None
    width, height = size
    if values.shape != (height, width):
        values = np.asarray(
            Image.fromarray(values, mode="F").resize((width, height), Image.BILINEAR),
            dtype=np.float32,
        )
    return values


__all__ = [
    "ALL_CLASSES",
    "BACKENDS",
    "BOX_CLASSES",
    "LANDSCAPE_CLASSES",
    "SegmentResult",
    "Segmenter",
    "SegmenterUnavailable",
    "backend_order",
    "segment",
]
