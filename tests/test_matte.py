"""A subject mask follows color inside the detector box."""

from __future__ import annotations

import unittest

import numpy as np

from ai_dt.matte import seeded_matte


class SeededMatteTests(unittest.TestCase):
    def test_a_loose_box_does_not_paint_its_corners(self) -> None:
        pixels = np.zeros((80, 80, 3), dtype=np.float32)
        pixels[..., 1] = 0.45
        pixels[..., 2] = 0.12
        yy, xx = np.ogrid[0:80, 0:80]
        disc = (yy - 40) ** 2 + (xx - 40) ** 2 <= 12**2
        pixels[disc] = (0.75, 0.12, 0.08)
        # The box is much wider than the disc. Its corners are background.
        matte = seeded_matte(pixels, (0.1, 0.1, 0.8, 0.8))
        self.assertGreater(float(matte[40, 40]), 0.8)
        self.assertLess(float(matte[10, 10]), 0.15)
        self.assertLess(float(matte[0, 0]), 0.05)

    def test_a_box_filled_by_one_color_still_covers_that_box(self) -> None:
        pixels = np.full((40, 40, 3), 0.2, dtype=np.float32)
        pixels[8:32, 10:30] = (0.62, 0.40, 0.32)
        matte = seeded_matte(pixels, (10 / 40, 8 / 40, 20 / 40, 24 / 40))
        self.assertGreater(float(matte[20, 20]), 0.7)
        self.assertLess(float(matte[1, 1]), 0.15)


if __name__ == "__main__":
    unittest.main()