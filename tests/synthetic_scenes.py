"""Small synthetic photos for the segmenter and mask-step tests."""

from __future__ import annotations

import numpy as np
from PIL import Image


def landscape(height: int = 240, width: int = 360) -> Image.Image:
    """Sky over trees and a facade, water below, a strip of ground at the bottom.

    Rows: sky 0-90, trees (left) and building (right) 90-170, water
    170-215, ground 215-240.
    """

    rng = np.random.default_rng(1)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:90] = (0.45, 0.62, 0.9)
    image[90:170, :150] = np.array([0.15, 0.45, 0.12]) + rng.random((80, 150, 3)) * 0.12
    image[90:170, 150:] = 0.62
    for y in range(96, 166, 18):
        for x in range(158, 354, 18):
            image[y : y + 9, x : x + 8] = 0.18
    image[170:] = (0.2, 0.35, 0.55)
    image[215:] = np.array([0.45, 0.35, 0.25]) + rng.random((height - 215, width, 3)) * 0.2
    return Image.fromarray((np.clip(image, 0, 1) * 255).astype("uint8"))


def subject_on_foliage(height: int = 240, width: int = 320) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """A red block on green noise, and its detector box (x, y, w, h)."""

    rng = np.random.default_rng(2)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[...] = np.array([0.15, 0.4, 0.12]) + rng.random((height, width, 3)) * 0.2
    image[70:190, 120:220] = (0.8, 0.15, 0.1)
    box = (120 / width, 70 / height, 100 / width, 120 / height)
    return Image.fromarray((np.clip(image, 0, 1) * 255).astype("uint8")), box


def region(mask: np.ndarray, rows: slice, cols: slice = slice(None)) -> float:
    return float((np.asarray(mask)[rows, cols] >= 0.5).mean())
