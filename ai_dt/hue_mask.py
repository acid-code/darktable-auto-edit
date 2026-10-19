"""Find compact regions of one named hue on a developed preview.

The curve then peaks at the hue those pixels actually have, and the shapes
limit the module to those regions. A style cannot store a brush, so the
shapes are ellipses the sidecar splice can attach.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_MAX_SIDE = 240
_MIN_AREA = 0.002
_MAX_AREA = 0.12


def hue_sample(image: Image.Image, hue_name: str) -> dict | None:
    """Ellipses and the median hue for one named color, or None.

    ``shapes`` is at most four compact blobs. A green patch is not a red
    one, and a speck smaller than ``_MIN_AREA`` is ignored.
    """

    name = str(hue_name or "").strip().lower()
    if name not in {"red", "orange", "yellow", "green", "blue"}:
        return None
    small = _shrink(image.convert("RGB"))
    rgb = np.asarray(small).astype(np.float32) / 255.0
    hue = _lab_hue(rgb)
    mask = _hue_mask(rgb, hue, name)
    if not mask.any():
        return None
    height, width = mask.shape
    frame = float(height * width)
    blobs = []
    for ys, xs in _components(mask):
        area = float(len(xs)) / frame
        if area < _MIN_AREA or area > _MAX_AREA:
            continue
        blob_hue = hue[ys, xs]
        blobs.append((area, float(np.median(blob_hue)), ys, xs))
    if not blobs:
        return None
    blobs.sort(key=lambda item: item[0], reverse=True)
    anchor = blobs[0][1]
    kept = [item for item in blobs if _hue_distance(item[1], anchor) <= 0.05][:4]
    if not kept:
        return None
    shapes = []
    samples = []
    for _area, _median, ys, xs in kept:
        samples.append(hue[ys, xs])
        cx = float(xs.mean() + 0.5) / width
        cy = float(ys.mean() + 0.5) / height
        rx = max((float(xs.max() - xs.min()) + 1.0) / width * 0.62, 0.02)
        ry = max((float(ys.max() - ys.min()) + 1.0) / height * 0.62, 0.02)
        rx = min(rx, cx, 1.0 - cx)
        ry = min(ry, cy, 1.0 - cy)
        if rx <= 0.0 or ry <= 0.0:
            continue
        shapes.append(
            {
                "cx": round(cx, 5),
                "cy": round(cy, 5),
                "rx": round(rx, 5),
                "ry": round(ry, 5),
            }
        )
    if not shapes:
        return None
    center = float(np.median(np.concatenate(samples)))
    return {"center": center, "shapes": shapes}


def _shrink(image: Image.Image) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= _MAX_SIDE:
        return image
    ratio = _MAX_SIDE / float(longest)
    return image.resize(
        (max(1, int(width * ratio)), max(1, int(height * ratio))),
        Image.Resampling.BOX,
    )


def _lab_hue(rgb: np.ndarray) -> np.ndarray:
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    red, green, blue = linear[..., 0], linear[..., 1], linear[..., 2]
    x = red * 0.4124564 + green * 0.3575761 + blue * 0.1804375
    y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
    z = red * 0.0193339 + green * 0.1191920 + blue * 0.9503041
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def bend(channel: np.ndarray) -> np.ndarray:
        delta = 6.0 / 29.0
        return np.where(channel > delta**3, np.cbrt(channel), channel / (3 * delta * delta) + 4.0 / 29.0)

    a = 500.0 * (bend(x / xn) - bend(y / yn))
    b = 200.0 * (bend(y / yn) - bend(z / zn))
    return (np.arctan2(b, a) / (2.0 * np.pi)) % 1.0


def _hue_mask(rgb: np.ndarray, hue: np.ndarray, name: str) -> np.ndarray:
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    peak = np.maximum(np.maximum(red, green), blue)
    saturation = (peak - np.minimum(np.minimum(red, green), blue)) / np.maximum(peak, 1e-6)
    colorful = (saturation > 0.35) & (peak > 0.15)
    if name == "red":
        select = (red >= green * 1.08) & (red >= blue * 1.2) & ((hue <= 0.14) | (hue >= 0.94))
    elif name == "orange":
        select = (red > green) & (green > blue * 1.1) & (hue >= 0.03) & (hue <= 0.18)
    elif name == "yellow":
        select = (red > blue * 1.3) & (green > blue * 1.3) & (hue >= 0.10) & (hue <= 0.28)
    elif name == "green":
        select = (green >= red * 1.12) & (green >= blue) & (hue >= 0.18) & (hue <= 0.48)
    else:
        select = (blue >= red * 1.12) & (blue >= green) & (hue >= 0.50) & (hue <= 0.80)
    return colorful & select


def _hue_distance(left: float, right: float) -> float:
    gap = abs(left - right) % 1.0
    return min(gap, 1.0 - gap)


def _components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent = [0]
    rank = [0]

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def unite(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left == right:
            return
        if rank[left] < rank[right]:
            parent[left] = right
        elif rank[right] < rank[left]:
            parent[right] = left
        else:
            parent[right] = left
            rank[left] += 1

    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            neighbors = []
            if x and labels[y, x - 1]:
                neighbors.append(int(labels[y, x - 1]))
            if y and labels[y - 1, x]:
                neighbors.append(int(labels[y - 1, x]))
            if neighbors:
                labels[y, x] = neighbors[0]
                for other in neighbors[1:]:
                    unite(neighbors[0], other)
            else:
                count += 1
                parent.append(count)
                rank.append(0)
                labels[y, x] = count
    groups: dict[int, list[tuple[int, int]]] = {}
    ys, xs = np.where(labels > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        root = find(int(labels[y, x]))
        groups.setdefault(root, []).append((y, x))
    found = []
    for points in groups.values():
        arr = np.array(points)
        found.append((arr[:, 0], arr[:, 1]))
    return found
