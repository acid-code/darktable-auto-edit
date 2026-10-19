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


def _ellipse(height: int, width: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def _box(truth: np.ndarray, label: str, confidence: float) -> dict:
    ys, xs = np.nonzero(truth)
    height, width = truth.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return {
        "label": label,
        "category": "animal",
        "confidence": confidence,
        "bbox": [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height],
        "source": "test",
    }


def animals_in_field(height: int = 240, width: int = 360):
    """Sky over a grass field with a standing deer, a lying sheep, and a bird in flight.

    Returns ``(image, boxes, truths)``; truths are boolean full-frame masks
    in the order of the boxes.
    """

    rng = np.random.default_rng(3)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:100] = (0.5, 0.68, 0.92)
    image[100:] = np.array([0.2, 0.48, 0.15]) + rng.random((height - 100, width, 3)) * 0.12
    # Standing deer: body plus four legs and a neck/head.
    deer = _ellipse(height, width, 90, 140, 38, 18)
    for leg_x in (64, 76, 104, 116):
        deer[150:195, leg_x : leg_x + 5] = True
    deer[105:130, 118:128] = True
    deer |= _ellipse(height, width, 128, 102, 10, 8)
    image[deer] = (0.55, 0.33, 0.16)
    # Lying sheep: a low wide shape.
    sheep = _ellipse(height, width, 260, 190, 34, 14)
    image[sheep] = (0.9, 0.88, 0.82)
    # Bird in flight: dark body with spread wings, in the sky.
    bird = _ellipse(height, width, 250, 45, 6, 4)
    bird |= _ellipse(height, width, 238, 40, 10, 3) | _ellipse(height, width, 262, 40, 10, 3)
    image[bird] = (0.12, 0.1, 0.1)
    picture = Image.fromarray((np.clip(image, 0, 1) * 255).astype("uint8"))
    truths = [deer, sheep, bird]
    boxes = [_box(deer, "deer", 0.8), _box(sheep, "sheep", 0.6), _box(bird, "bird", 0.3)]
    boxes[0]["label"] = "horse"  # COCO has no deer; YOLO calls it a horse or a cow
    return picture, boxes, truths


def occluded_animal(height: int = 240, width: int = 320):
    """A brown dog behind tall grass blades. Returns ``(image, box, truth)``."""

    rng = np.random.default_rng(4)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[...] = np.array([0.22, 0.5, 0.16]) + rng.random((height, width, 3)) * 0.1
    truth = _ellipse(height, width, 160, 120, 70, 45)
    image[truth] = (0.6, 0.38, 0.2)
    # Grass blades in front of the lower half of the dog.
    for x in range(92, 232, 12):
        image[115:200, x : x + 3] = (0.2, 0.5, 0.15)
    box = _box(truth, "dog", 0.45)
    return Image.fromarray((np.clip(image, 0, 1) * 255).astype("uint8")), box, truth


def tiny_bird(height: int = 240, width: int = 360):
    """A distant bird a few pixels wide in a clear sky. Returns ``(image, box, truth)``."""

    image = np.zeros((height, width, 3), dtype=np.float32)
    image[...] = (0.55, 0.7, 0.93)
    image[180:] = (0.3, 0.45, 0.25)
    truth = _ellipse(height, width, 200, 60, 6, 3)
    image[truth] = (0.1, 0.1, 0.12)
    box = _box(truth, "bird", 0.18)
    return Image.fromarray((np.clip(image, 0, 1) * 255).astype("uint8")), box, truth
