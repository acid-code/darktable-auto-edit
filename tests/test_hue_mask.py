"""A named hue is measured on the preview and limited to its own shapes."""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from ai_dt.hue_mask import hue_sample
from ai_dt.modules_extra import colorzones_hue_params


def _frame() -> Image.Image:
    image = Image.new("RGB", (200, 160), (18, 18, 28))
    pixels = image.load()
    for x in range(30, 70):
        for y in range(20, 56):
            pixels[x, y] = (190, 40, 35)
    for x in range(120, 170):
        for y in range(100, 146):
            pixels[x, y] = (200, 48, 40)
    for x in range(20, 50):
        for y in range(100, 140):
            pixels[x, y] = (30, 150, 40)
    for x in range(90, 94):
        for y in range(10, 14):
            pixels[x, y] = (220, 20, 20)
    return image


def _lightness(blob: bytes) -> list[tuple[float, float]]:
    count = struct.unpack_from("<i", blob, 484)[0]
    nodes = []
    for node in range(count):
        nodes.append(struct.unpack_from("<ff", blob, 4 + node * 8))
    return nodes


class HueSampleTests(unittest.TestCase):
    def test_two_red_patches_are_two_shapes_and_green_is_not(self) -> None:
        sample = hue_sample(_frame(), "red")
        self.assertIsNotNone(sample)
        assert sample is not None
        shapes = sample["shapes"]
        self.assertEqual(len(shapes), 2)
        centers = sorted((item["cx"], item["cy"]) for item in shapes)
        self.assertAlmostEqual(centers[0][0], 0.25, delta=0.06)
        self.assertAlmostEqual(centers[0][1], 0.24, delta=0.06)
        self.assertAlmostEqual(centers[1][0], 0.72, delta=0.06)
        self.assertAlmostEqual(centers[1][1], 0.77, delta=0.06)
        self.assertGreater(sample["center"], 0.0)
        self.assertLess(sample["center"], 0.16)

    def test_the_lightness_curve_peaks_at_the_measured_hue(self) -> None:
        blob = colorzones_hue_params("red", center=0.075)
        self.assertIsNotNone(blob)
        assert blob is not None
        self.assertEqual(struct.unpack_from("<i", blob, 0)[0], 2)
        self.assertEqual(struct.unpack_from("<f", blob, 508)[0], 0.0)
        nodes = _lightness(blob)
        peak = max(nodes, key=lambda item: item[1])
        self.assertAlmostEqual(peak[0], 0.075, delta=0.02)
        self.assertGreater(peak[1], 0.75)
        flat = [y for x, y in nodes if 0.4 < x < 0.6]
        self.assertTrue(flat)
        self.assertTrue(all(abs(y - 0.5) < 0.02 for y in flat))

    def test_too_much_red_lowers_chroma_and_leaves_lightness(self) -> None:
        blob = colorzones_hue_params("red", center=0.075, reduce=True)
        self.assertIsNotNone(blob)
        assert blob is not None
        lightness = _lightness(blob)
        self.assertTrue(all(abs(y - 0.5) < 0.02 for _x, y in lightness))
        count = struct.unpack_from("<i", blob, 488)[0]
        chroma = [struct.unpack_from("<ff", blob, 4 + (20 + node) * 8) for node in range(count)]
        dip = min(chroma, key=lambda item: item[1])
        self.assertAlmostEqual(dip[0], 0.075, delta=0.02)
        self.assertLess(dip[1], 0.4)
