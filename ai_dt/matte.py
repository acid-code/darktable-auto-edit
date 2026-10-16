"""A subject mask seeded by a detector box.

The box says where to look. Color inside that box says which pixels are
the subject. A box that is already one color keeps the whole box, because
there is no edge to follow.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def seeded_matte(pixels: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Soft mask in 0..1. Corners outside the subject color stay dark."""

    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    gate = _ellipse(height, width, box, expand=1.15, inner=0.72, feather=0.45)
    seed = _crop(pixels, box, inset=0.28)
    if seed.size == 0:
        return gate
    color = np.median(seed.reshape(-1, 3), axis=0)
    distance = np.sqrt(np.sum((pixels.astype(np.float32) - color) ** 2, axis=2))
    seed_distance = _crop(distance, box, inset=0.28)
    if seed_distance.size == 0:
        return gate
    limit = max(0.08, float(np.percentile(seed_distance, 80)) * 2.5)
    similar = distance <= limit
    inside = gate > 0.5
    if not np.any(inside):
        return gate
    coverage = float(similar[inside].mean())
    if coverage > 0.85:
        return _feather(gate)
    # Eyes are a different color, and they sit inside the subject.
    # Fill those holes. Pixels that touch the outside of the frame stay out,
    # so a loose box does not paint its corners.
    matte = np.where(similar, gate, np.float32(0.0)).astype(np.float32)
    matte = _fill_enclosed(matte)
    return _feather(np.minimum(matte, gate))


def _crop(values: np.ndarray, box: tuple[float, float, float, float], *, inset: float) -> np.ndarray:
    height, width = int(values.shape[0]), int(values.shape[1])
    x, y, bw, bh = (float(item) for item in box)
    x0 = int(np.clip((x + bw * inset) * width, 0, width - 1))
    y0 = int(np.clip((y + bh * inset) * height, 0, height - 1))
    x1 = int(np.clip((x + bw * (1.0 - inset)) * width, x0 + 1, width))
    y1 = int(np.clip((y + bh * (1.0 - inset)) * height, y0 + 1, height))
    return values[y0:y1, x0:x1]


def _ellipse(
    height: int,
    width: int,
    bbox: tuple[float, float, float, float],
    *,
    expand: float,
    inner: float,
    feather: float,
) -> np.ndarray:
    x, y, bw, bh = bbox
    cx = (x + bw * 0.5) * width
    cy = (y + bh * 0.5) * height
    rx = max(2.0, bw * width * 0.5 * expand)
    ry = max(2.0, bh * height * 0.5 * expand)
    yy, xx = np.ogrid[0:height, 0:width]
    distance = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    edge = np.clip((distance - inner) / max(feather, 1e-3), 0.0, 1.0)
    edge = edge * edge * (3.0 - 2.0 * edge)
    return (1.0 - edge).astype(np.float32)


def _fill_enclosed(mask: np.ndarray) -> np.ndarray:
    """Turn holes inside the subject back on. The outside stays off."""

    binary = np.where(mask > 0.45, np.uint8(255), np.uint8(0))
    padded = np.pad(binary, 1, constant_values=np.uint8(0))
    image = Image.fromarray(padded, mode="L")
    ImageDraw.floodfill(image, (0, 0), 128)
    filled = np.asarray(image)[1:-1, 1:-1]
    result = mask.copy()
    result[filled == 0] = np.float32(1.0)
    return result


def _feather(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="edge")
    height, width = mask.shape
    total = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + height, dx : dx + width]
    return (total / np.float32(9.0)).astype(np.float32)
