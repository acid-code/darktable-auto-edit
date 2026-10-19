"""Dependency-free masks from color, texture and position.

This is the fallback every run has. It is meant for broad regions, the way
a landscape edit uses them: an open sky above the horizon, green
vegetation, smooth blue water under the horizon, line-filled buildings and
the ground at the bottom. A subject comes from a detector box refined by a
color model of the box against its surround, or from a center-weighted
color saliency when there is no box. Edges are coarse on purpose; the
drawn-mask feather softens them.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ai_dt.dt_shapes import label_components
from ai_dt.segment.base import (
    LANDSCAPE_CLASSES,
    PEOPLE_CATEGORIES,
    SUBJECT_CATEGORIES,
    boxes_for,
)

_WORK = 160


def _blur(values: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return values
    padded = np.pad(values, radius, mode="edge")
    integral = padded.cumsum(0).cumsum(1)
    integral = np.pad(integral, ((1, 0), (1, 0)))
    size = 2 * radius + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return (total / float(size * size)).astype(np.float32)


def _small(image: Image.Image) -> np.ndarray:
    scale = _WORK / float(max(image.size))
    if scale < 1.0:
        size = (max(8, int(round(image.width * scale))), max(8, int(round(image.height * scale))))
        image = image.resize(size, Image.BILINEAR)
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _connected_to(binary: np.ndarray, edge: str) -> np.ndarray:
    labels, count = label_components(binary)
    if count == 0:
        return np.zeros_like(binary)
    if edge == "top":
        touching = np.unique(labels[0])
    elif edge == "bottom":
        touching = np.unique(labels[-1])
    else:
        touching = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    touching = touching[touching > 0]
    return np.isin(labels, touching)


def _largest(binary: np.ndarray, near: tuple[float, float] | None = None) -> np.ndarray:
    labels, count = label_components(binary)
    if count == 0:
        return binary
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    chosen = int(np.argmax(sizes))
    if near is not None:
        height, width = binary.shape
        cy = min(height - 1, max(0, int(near[1] * height)))
        cx = min(width - 1, max(0, int(near[0] * width)))
        at = int(labels[cy, cx])
        if at and sizes[at] >= 0.25 * sizes[chosen]:
            chosen = at
    return labels == chosen


def _close(binary: np.ndarray, radius: int = 2) -> np.ndarray:
    grown = _blur(binary.astype(np.float32), radius) > 0.05
    return _blur(grown.astype(np.float32), radius) > 0.95


def _features(pixels: np.ndarray) -> dict[str, np.ndarray]:
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    high = pixels.max(axis=2)
    low = pixels.min(axis=2)
    sat = np.where(high > 1e-4, (high - low) / np.maximum(high, 1e-4), 0.0)
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    gx[:, 1:-1] = luma[:, 2:] - luma[:, :-2]
    gy[1:-1] = luma[2:] - luma[:-2]
    edges = np.hypot(gx, gy)
    texture = _blur(edges, 2)
    straight = (np.abs(gx) > 2.5 * np.abs(gy)) | (np.abs(gy) > 2.5 * np.abs(gx))
    strong = (edges > 0.06).astype(np.float32)
    density = _blur(strong, 3)
    # Buildings are many strong edges that run horizontally or vertically.
    # Noise and foliage have edges too, but in every direction.
    coherence = _blur(strong * straight, 3) / np.maximum(density, 1e-4)
    # A single horizon or shore is one straight line. A facade has both
    # vertical and horizontal edges close together.
    vertical = _blur(strong * (np.abs(gx) > 2.5 * np.abs(gy)), 4)
    horizontal = _blur(strong * (np.abs(gy) > 2.5 * np.abs(gx)), 4)
    both = (vertical > 0.04) & (horizontal > 0.04)
    lines = np.where((coherence > 0.6) & (density > 0.12) & both, density, 0.0).astype(np.float32)
    height = pixels.shape[0]
    rows = np.broadcast_to(np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None], luma.shape)
    return {
        "red": red, "green": green, "blue": blue, "luma": luma, "sat": sat,
        "texture": texture, "lines": lines, "rows": rows, "edges": edges,
    }


def sky_mask(pixels: np.ndarray, f: dict | None = None) -> np.ndarray:
    f = f or _features(pixels)
    smooth = f["texture"] < 0.045
    blue_sky = (f["blue"] > f["red"] + 0.015) & (f["luma"] > 0.28)
    pale = (f["luma"] > 0.55) & (f["sat"] < 0.3)
    sunset = (
        (f["luma"] > 0.4)
        & (f["red"] >= f["green"])
        & (f["red"] > f["blue"] + 0.05)
        & (f["texture"] < 0.025)
    )
    green = (f["green"] > f["red"] * 1.04) & (f["green"] > f["blue"] * 1.04) & (f["sat"] > 0.12)
    candidate = smooth & (blue_sky | pale | sunset) & ~green & (f["rows"] < 0.8)
    sky = _connected_to(candidate, "top")
    sky = _close(sky, 1) & (f["rows"] < 0.85)
    if sky.mean() < 0.02:
        return np.zeros_like(f["luma"], dtype=np.float32)
    return sky.astype(np.float32)


def horizon_row(sky: np.ndarray) -> float:
    """Where the sky ends, 0..1 from the top. 0.45 when there is no sky."""

    columns = sky >= 0.5
    if not columns.any():
        return 0.45
    height = sky.shape[0]
    bottoms = []
    for column in columns.T:
        rows = np.flatnonzero(column)
        if rows.size:
            bottoms.append(rows.max() + 1)
    return float(np.median(bottoms)) / float(height) if bottoms else 0.45


def landscape_masks(pixels: np.ndarray) -> dict[str, np.ndarray]:
    f = _features(pixels)
    sky = sky_mask(pixels, f) >= 0.5
    horizon = horizon_row(sky.astype(np.float32))
    below = f["rows"] >= horizon - 0.02
    vegetation = (
        (f["green"] > f["red"] * 1.04)
        & (f["green"] > f["blue"] * 1.04)
        & (f["sat"] > 0.12)
        & (f["luma"] > 0.04)
        & ~sky
    )
    vegetation = _blur(vegetation.astype(np.float32), 2) > 0.5
    water = (
        below
        & ~sky
        & ~vegetation
        & (f["blue"] >= f["red"] - 0.01)
        & (f["texture"] < 0.06)
        & (f["luma"] > 0.06)
    )
    water = _close(water, 1)
    water = water & (_blur(water.astype(np.float32), 4) > 0.35)
    architecture = (
        ~sky
        & ~vegetation
        & ~water
        & (f["lines"] > 0.18)
        & (f["sat"] < 0.45)
    )
    architecture = _blur(architecture.astype(np.float32), 3) > 0.4
    architecture &= ~sky
    ground = below & ~sky & ~vegetation & ~water & ~architecture
    ground = _connected_to(ground, "bottom")
    out = {
        "sky": sky,
        "vegetation": vegetation & ~sky,
        "water": water,
        "architecture": architecture & ~water & ~vegetation,
        "ground": ground,
    }
    return {name: mask.astype(np.float32) for name, mask in out.items()}


def _color_codes(pixels: np.ndarray) -> np.ndarray:
    quant = np.clip((pixels * 8).astype(int), 0, 7)
    return quant[..., 0] * 64 + quant[..., 1] * 8 + quant[..., 2]


def _box_mask(pixels: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = pixels.shape[:2]
    x, y, w, h = box
    x0, x1 = int(max(0, x * width)), int(min(width, (x + w) * width + 0.999))
    y0, y1 = int(max(0, y * height)), int(min(height, (y + h) * height + 0.999))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    codes = _color_codes(pixels)
    inner = np.zeros((height, width), dtype=bool)
    ix0 = x0 + (x1 - x0) // 5
    ix1 = x1 - (x1 - x0) // 5
    iy0 = y0 + (y1 - y0) // 5
    iy1 = y1 - (y1 - y0) // 5
    inner[iy0:max(iy1, iy0 + 1), ix0:max(ix1, ix0 + 1)] = True
    pad_x = max(2, (x1 - x0) // 3)
    pad_y = max(2, (y1 - y0) // 3)
    ring = np.zeros((height, width), dtype=bool)
    ring[max(0, y0 - pad_y):min(height, y1 + pad_y), max(0, x0 - pad_x):min(width, x1 + pad_x)] = True
    ring[y0:y1, x0:x1] = False
    if ring.sum() < 8:
        ring = np.ones((height, width), dtype=bool)
        ring[y0:y1, x0:x1] = False
        if ring.sum() < 8:
            return None
    fg = np.bincount(codes[inner], minlength=512).astype(np.float64) + 0.5
    bg = np.bincount(codes[ring], minlength=512).astype(np.float64) + 0.5
    fg /= fg.sum()
    bg /= bg.sum()
    likely = (fg[codes] > bg[codes] * 1.2)
    inside = np.zeros((height, width), dtype=bool)
    inside[y0:y1, x0:x1] = True
    blob = likely & inside
    blob = _blur(blob.astype(np.float32), 1) > 0.45
    if blob.sum() < 0.12 * (x1 - x0) * (y1 - y0):
        # Nothing separates by color: keep a soft oval inside the box.
        yy, xx = np.mgrid[0:height, 0:width]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rx, ry = max(1.0, (x1 - x0) / 2.0), max(1.0, (y1 - y0) / 2.0)
        blob = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    blob = _largest(blob, (x + w / 2.0, y + h / 2.0))
    return _close(blob, 1)


def _span_fill(blob: np.ndarray, axis: int) -> np.ndarray:
    """Fill each row (axis=1) or column (axis=0) between its first and last set pixel."""

    index = np.arange(blob.shape[axis])
    shape = [1, 1]
    shape[axis] = -1
    index = index.reshape(shape)
    big = blob.shape[axis]
    first = np.where(blob, index, big).min(axis=axis, keepdims=True)
    last = np.where(blob, index, -1).max(axis=axis, keepdims=True)
    return (index >= first) & (index <= last)


def _saliency_subject(pixels: np.ndarray) -> np.ndarray | None:
    """The patch that stands out most from its own surroundings, near the middle.

    Local contrast (each pixel against a wide blur of its neighbourhood) keeps a
    large area that merely differs from the border colour, such as a patch of
    water, from beating a small bird in front of it.
    """

    height, width = pixels.shape[:2]
    local = np.stack([_blur(pixels[..., c], 12) for c in range(3)], axis=-1)
    distance = np.sqrt(((pixels - local) ** 2).sum(axis=2))
    yy, xx = np.mgrid[0:height, 0:width]
    center = np.exp(-(((xx / width - 0.5) / 0.35) ** 2 + ((yy / height - 0.5) / 0.35) ** 2))
    score = _blur((distance * center).astype(np.float32), 2)
    blob = score > float(score.mean() + 2.0 * score.std())
    if not blob.any():
        return None
    labels, count = label_components(blob)
    edge = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
    best, best_score = 0, 0.0
    for index in range(1, count + 1):
        if index in edge:
            continue
        part = labels == index
        size = int(part.sum())
        if size < 0.004 * blob.size:
            continue
        ys, xs = np.nonzero(part)
        span_x, span_y = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        # A wide flat band (a shoreline, a hedge, a horizon) is not a subject.
        if max(span_x, span_y) > 4 * min(span_x, span_y):
            continue
        strength = float(score[part].mean()) * size ** 0.25
        if strength > best_score:
            best, best_score = index, strength
    if not best:
        return None
    blob = _close(labels == best, 1)
    # Local contrast lights up a flat subject's outline more than its middle,
    # so fill whatever the outline encloses.
    blob = blob | _span_fill(blob, 1) | _span_fill(blob, 0)
    cover = float(blob.mean())
    if cover < 0.005 or cover > 0.6:
        return None
    return blob


class HeuristicSegmenter:
    name = "heuristic"
    classes = ("subject", "people", *LANDSCAPE_CLASSES)

    def available(self) -> bool:
        return True

    def segment(self, image: Image.Image, classes, *, boxes: list) -> dict[str, np.ndarray]:
        pixels = _small(image)
        wanted = set(classes)
        out: dict[str, np.ndarray] = {}
        if wanted & set(LANDSCAPE_CLASSES):
            for name, mask in landscape_masks(pixels).items():
                if name in wanted:
                    out[name] = mask
        if "subject" in wanted:
            found = None
            for box in boxes_for(boxes, SUBJECT_CATEGORIES):
                blob = _box_mask(pixels, box)
                if blob is not None:
                    found = blob if found is None else (found | blob)
            if found is None:
                found = _saliency_subject(pixels)
            if found is not None:
                out["subject"] = found.astype(np.float32)
        if "people" in wanted:
            found = None
            for box in boxes_for(boxes, PEOPLE_CATEGORIES):
                x, y, w, h = box
                category_face = w < 0.5 and h < 0.5
                # A face box stands for the head; reach down for the shoulders.
                grown = (max(0.0, x - 0.5 * w), y, min(1.0, 2.0 * w), min(1.0 - y, 2.6 * h)) if category_face else box
                blob = _box_mask(pixels, grown)
                if blob is not None:
                    found = blob if found is None else (found | blob)
            if found is not None:
                out["people"] = found.astype(np.float32)
        return {name: _soften(mask) for name, mask in out.items()}


def _soften(mask: np.ndarray) -> np.ndarray:
    return np.clip(_blur(mask.astype(np.float32), 1), 0.0, 1.0)
