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


class BoxlessSubjectLiftTests(unittest.TestCase):
    def test_a_subject_lift_without_a_box_is_drawn_not_whole_frame(self):
        import json
        import tempfile
        from pathlib import Path

        from ai_dt.features import commit_waiting_step

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pixels = np.full((240, 320, 3), 90, dtype=np.uint8)
            pixels[90:160, 130:190] = (230, 200, 40)
            Image.fromarray(pixels).save(directory / "preview.jpg")
            recipe = {
                "features_next": "subject",
                "features_next_title": "Lift the parrot a little",
                "plan": {"features_done": [], "features_path": ["subject"], "reasoning": {}},
                "segmenter": "heuristic",
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            saved = commit_waiting_step(directory)
            self.assertIn("step_mask", saved)
            self.assertFalse((directory / "step.dtstyle").is_file())
