"""Cloud segmentation with Gemini. Off unless asked for, because it uploads the preview.

Turn it on with ``AI_DT_SEGMENT_GEMINI=1`` (in ``.env`` or the
environment) or ``--segmenter gemini``. Needs ``GEMINI_API_KEY``. The
model is ``AI_DT_GEMINI_SEGMENT_MODEL`` (default ``gemini-2.5-flash``).

Gemini has answered in two shapes: ``box_2d`` (0..1000, ``[y0, x0, y1, x1]``)
with a base64 PNG probability ``mask`` cropped to that box, and, in the
current docs, a ``polygon`` of ``[x, y]`` points on the same 0..1000
scale. ``masks_from_reply`` reads both.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from ai_dt.segment.base import ALL_CLASSES, SegmenterUnavailable

DEFAULT_MODEL = "gemini-2.5-flash"
_PROMPT_NAMES = {
    "subject": "the main subject",
    "people": "people",
    "sky": "sky",
    "architecture": "buildings and architecture",
    "vegetation": "trees, grass and plants",
    "water": "water",
    "ground": "ground, rocks and roads",
}


def prompt_for(classes) -> str:
    wanted = [name for name in classes if name in _PROMPT_NAMES]
    lines = "\n".join(f'- "{name}": {_PROMPT_NAMES[name]}' for name in wanted)
    return (
        "Give the segmentation masks for these regions of the photo. "
        "Reply with a JSON list only. Each entry has \"label\" (one of the keys below), "
        "\"box_2d\" as [y0, x0, y1, x1] on a 0-1000 scale, and either \"mask\" (a base64 PNG "
        "probability map for that box) or \"polygon\" (a list of [x, y] points on a 0-1000 scale). "
        "Leave a region out when it is not in the photo.\n" + lines
    )


def _json_list(text: str) -> list:
    cleaned = str(text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = data.get("masks") or data.get("items") or []
    return data if isinstance(data, list) else []


def _png_mask(encoded: str) -> np.ndarray | None:
    raw = str(encoded or "")
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        with Image.open(io.BytesIO(base64.b64decode(raw))) as image:
            return np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    except Exception:
        return None


def masks_from_reply(text: str, width: int, height: int, classes=ALL_CLASSES) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for item in _json_list(text):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        if label not in classes:
            continue
        canvas = np.zeros((height, width), dtype=np.float32)
        polygon = item.get("polygon") or item.get("points")
        box = item.get("box_2d")
        if isinstance(polygon, list) and len(polygon) >= 3:
            try:
                points = [(float(p[0]) / 1000.0 * width, float(p[1]) / 1000.0 * height) for p in polygon]
            except (TypeError, ValueError, IndexError):
                continue
            image = Image.new("L", (width, height), 0)
            ImageDraw.Draw(image).polygon(points, fill=255)
            canvas = np.asarray(image, dtype=np.float32) / 255.0
        elif isinstance(box, list) and len(box) == 4:
            try:
                y0, x0, y1, x1 = (float(v) / 1000.0 for v in box)
            except (TypeError, ValueError):
                continue
            px0, px1 = int(round(x0 * width)), int(round(x1 * width))
            py0, py1 = int(round(y0 * height)), int(round(y1 * height))
            if px1 - px0 < 1 or py1 - py0 < 1:
                continue
            patch = _png_mask(item.get("mask")) if item.get("mask") else None
            if patch is None:
                canvas[py0:py1, px0:px1] = 1.0
            else:
                resized = Image.fromarray(patch, mode="F").resize((px1 - px0, py1 - py0), Image.BILINEAR)
                canvas[py0:py1, px0:px1] = np.asarray(resized, dtype=np.float32)
        else:
            continue
        out[label] = np.maximum(out[label], canvas) if label in out else canvas
    return out


def _jpeg_base64(image: Image.Image) -> str:
    copy = image.convert("RGB")
    copy.thumbnail((1024, 1024))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class GeminiSegmenter:
    name = "gemini"
    classes = tuple(_PROMPT_NAMES)

    def __init__(self, model: str | None = None, timeout: float = 60.0) -> None:
        self.model = model or os.environ.get("AI_DT_GEMINI_SEGMENT_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())

    def _post(self, body: dict, key: str) -> dict:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SegmenterUnavailable(f"Gemini segmentation failed ({exc})") from exc

    def segment(self, image, classes, *, boxes: list) -> dict[str, np.ndarray]:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise SegmenterUnavailable("GEMINI_API_KEY is not set")
        body = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": _jpeg_base64(image)}},
                    {"text": prompt_for(classes)},
                ]
            }],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        payload = self._post(body, key)
        text = ""
        for candidate in payload.get("candidates") or []:
            for part in ((candidate.get("content") or {}).get("parts") or []):
                text += str(part.get("text") or "")
        return masks_from_reply(text, image.width, image.height, tuple(classes))
