from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from ai_dt import mask_steps


class AnimalFallbackTests(unittest.TestCase):
    def test_missed_animal_falls_back_to_the_main_subject(self):
        # A plain field with one bright blob the animal detector won't name.
        pixels = np.full((240, 320, 3), 90, dtype=np.uint8)
        pixels[90:160, 130:190] = (230, 200, 40)
        image = Image.fromarray(pixels)
        forms, source = mask_steps.step_forms({}, "mask_animal", image, "heuristic")
        self.assertTrue(forms)
        self.assertIn("no animal detected", source)


if __name__ == "__main__":
    unittest.main()
