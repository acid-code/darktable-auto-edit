"""A loose animals outline over water must not join a tight parrot outline."""

import unittest
from unittest import mock

import numpy as np
from PIL import Image

from ai_dt import mask_steps
from ai_dt.segment.animals import Animal, AnimalResult


def _animal(mask, label, area):
    h, w = mask.shape
    return Animal(label=label, confidence=0.9, bbox=(0, 0, area, 1.0), mask=mask, source="fake",
                  crop=(0.0, 0.0, 1.0, 1.0), crop_mask=mask)


def _scene():
    h, w = 120, 160
    parrot = np.zeros((h, w), np.float32)
    parrot[30:70, 60:90] = 1.0
    water = np.zeros((h, w), np.float32)
    water[25:, :] = 1.0  # bloated: runs to the frame edge, covers the parrot
    return parrot, water


class WaterBleedTest(unittest.TestCase):
    def test_bloated_water_form_is_dropped(self):
        parrot, water = _scene()
        kept = mask_steps._drop_scenery_animals([_animal(water, "animal", 0.9), _animal(parrot, "parrot", 0.1)])
        self.assertEqual([a.label for a in kept], ["parrot"])

    def test_loose_tight_form_swallowing_parrot_is_dropped(self):
        parrot, _ = _scene()
        loose = np.zeros_like(parrot)
        loose[15:95, 40:120] = 1.0  # inside the frame, but swallows the parrot
        kept = mask_steps._drop_scenery_animals([_animal(loose, "animal", 0.4), _animal(parrot, "parrot", 0.1)])
        self.assertEqual([a.label for a in kept], ["parrot"])

    def test_single_closeup_is_kept(self):
        _, water = _scene()
        self.assertEqual(len(mask_steps._drop_scenery_animals([_animal(water, "dog", 0.9)])), 1)

    def test_animals_step_returns_only_tight_parrot(self):
        parrot, water = _scene()
        result = AnimalResult(animals=[_animal(water, "animal", 0.9), _animal(parrot, "parrot", 0.1)])
        image = Image.new("RGB", (160, 120), (40, 90, 140))
        recipe = {"subjects": [{"label": "parrot", "category": "animal", "bbox": [0.37, 0.25, 0.2, 0.33]}]}
        for step in ("mask_animals", "mask_animal"):
            with mock.patch("ai_dt.segment.animals.segment_animals", return_value=result), \
                 mock.patch.object(mask_steps, "_clear_subject_elsewhere", return_value=False):
                forms, _ = mask_steps.step_forms(recipe, step, image)
            self.assertTrue(forms)
            pts = np.array([p for f in forms for p in f["points"]])
            self.assertGreater(pts[:, 0].min(), 0.3)
            self.assertLess(pts[:, 0].max(), 0.62)
            self.assertLess(pts[:, 1].max(), 0.65)


if __name__ == "__main__":
    unittest.main()
