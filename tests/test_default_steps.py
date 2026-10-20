"""Default editor steps: clipping, dehaze and subject sharpening."""

from __future__ import annotations

import struct
import unittest

import numpy as np
from PIL import Image

from ai_dt import default_steps as d
from ai_dt.applied import _name_to_step
from ai_dt.mapping import step_history_name
from ai_dt.mask_steps import STEPS, module_blob, capped_grade


def _image(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype("uint8"))


def _colour_ramp(low: float, high: float) -> np.ndarray:
    ramp = np.tile(np.linspace(low, high, 512), (256, 1))
    rng = np.random.default_rng(1)
    hue = rng.random((256, 512)) * 0.0
    return np.clip(np.stack([ramp, ramp * 0.55 + hue, ramp * 0.25], -1), 0, 1)


class DefaultStepsTests(unittest.TestCase):
    def test_clipped_highlights_are_pulled_until_they_almost_vanish(self) -> None:
        found = d.measure(_image(_colour_ramp(0.2, 1.03)))
        clip = found[d.CLIPPING_ID]
        self.assertLess(clip["highlights"], 0)
        self.assertGreater(clip["clipped_before"], d.CLIP_HI_TRIGGER)
        self.assertLess(clip["clipped_after"], clip["clipped_before"] * 0.3)

    def test_crushed_shadows_are_lifted(self) -> None:
        clip = d.measure(_image(_colour_ramp(-0.1, 0.9)))[d.CLIPPING_ID]
        self.assertGreater(clip["shadows"], 0)
        self.assertLess(clip["crushed_after"], clip["crushed_before"])

    def test_a_well_exposed_photo_gets_no_clipping_step(self) -> None:
        self.assertNotIn(d.CLIPPING_ID, d.measure(_image(_colour_ramp(0.15, 0.85))))

    def test_contrast_floor_holds(self) -> None:
        luma = d._luma(np.asarray(_colour_ramp(-0.3, 1.3), dtype=np.float32))
        clip = d.solve_clipping(luma)
        out = d.simulate(luma, clip["shadows"], clip["highlights"])
        self.assertGreaterEqual(d._spread(out), d.CONTRAST_FLOOR * d._spread(luma) - 1e-6)

    def test_haze_only_when_the_dark_channel_is_lifted(self) -> None:
        clear = _colour_ramp(0.05, 0.9)
        self.assertNotIn(d.DEHAZE_ID, d.measure(_image(clear)))
        hazy = clear * 0.4 + 0.45
        found = d.measure(_image(hazy))[d.DEHAZE_ID]
        self.assertGreater(found["strength"], 0.1)
        self.assertLessEqual(found["strength"], 0.4)
        light = d.haze_score(np.asarray(clear * 0.7 + 0.2, dtype=np.float32))
        self.assertLess(light, found["score"])

    def test_sharpening_is_reduced_then_skipped_with_noise(self) -> None:
        base = _colour_ramp(0.1, 0.9)
        self.assertEqual(d.measure(_image(base))[d.SHARPEN_ID]["action"], "lift")
        rng = np.random.default_rng(0)
        mild = base + rng.normal(0, 0.016, base.shape[:2] + (1,))
        self.assertEqual(d.measure(_image(mild))[d.SHARPEN_ID]["action"], "hold")
        heavy = base + rng.normal(0, 0.05, base.shape[:2] + (1,))
        self.assertNotIn(d.SHARPEN_ID, d.measure(_image(heavy)))
        self.assertEqual(d.measure(_image(base), iso=6400)[d.SHARPEN_ID]["action"], "hold")

    def test_defaults_go_first_without_duplicates_or_applied_steps(self) -> None:
        found = {d.CLIPPING_ID: {"shadows": 0.3, "highlights": -0.5}, d.SHARPEN_ID: {"action": "lift"}}
        plan = {"step_queue": [{"id": "mask_sky", "action": "darken", "title": "Sky"}], "features_next": "mask_sky"}
        out = d.add_defaults(dict(plan), found)
        self.assertEqual([i["id"] for i in out["step_queue"]], [d.CLIPPING_ID, d.SHARPEN_ID, "mask_sky"])
        self.assertEqual(out["features_next"], d.CLIPPING_ID)
        again = d.add_defaults(out, found)
        self.assertEqual(len(again["step_queue"]), 3)
        skipped = d.add_defaults(dict(plan), found, [d.CLIPPING_ID])
        self.assertEqual(skipped["features_next"], d.SHARPEN_ID)

    def test_module_blobs_match_darktable_54_and_56(self) -> None:
        style = d.style_for(d.CLIPPING_ID, {"shadows": 0.4, "highlights": -0.6})
        self.assertIn("<operation>toneequal</operation>", style)
        self.assertEqual(len(d.tone_blob(0.4, -0.6)), 72)
        haze = d.style_for(d.DEHAZE_ID, {"strength": 0.3, "distance": 0.2})
        self.assertIn("<operation>hazeremoval</operation>", haze)
        self.assertIn("<module>3</module>", haze)
        self.assertEqual(len(d.haze_blob(0.3, 0.2)), 16)
        step = STEPS[d.SHARPEN_ID]
        blob, version = module_blob(step, capped_grade(step, "hold"))
        self.assertEqual(version, 1)
        radius, amount, threshold = struct.unpack("<fff", blob)
        self.assertLess(amount, 0.5)
        self.assertGreater(threshold, 1.0)
        self.assertIsNone(d.style_for(d.SHARPEN_ID, {}))

    def test_history_names_map_back_to_the_steps(self) -> None:
        names = _name_to_step()
        for ident in d.DEFAULT_IDS:
            self.assertEqual(names[step_history_name(ident)], ident)


if __name__ == "__main__":
    unittest.main()
