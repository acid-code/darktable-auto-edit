"""The rendered TIFF isolates a subject instead of grading the whole frame."""

from __future__ import annotations

import unittest

import numpy as np

from ai_dt.pipeline import analyze, write_outputs
from ai_dt.schema import Atmosphere
from ai_dt.separate import render_finish
from ai_dt.simulate import sample_image


def _texture(pixels: np.ndarray) -> float:
    values = pixels.astype(np.float32)
    return float(np.mean(np.abs(values[:, 1:] - values[:, :-1])))


def _bird() -> np.ndarray:
    rng = np.random.default_rng(4)
    height, width = 480, 720
    noise = rng.random((height, width, 3), dtype=np.float32)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = 0.12 + 0.18 * noise[..., 0]
    image[..., 1] = 0.32 + 0.40 * noise[..., 1]
    image[..., 2] = 0.10 + 0.12 * noise[..., 2]
    image[:, ::17, :] *= 0.3
    image[::13, :, :] *= 0.4
    image[180:420, 300:480] = (0.10, 0.10, 0.11)
    image[150:250, 330:470] = (0.78, 0.12, 0.08)
    image[260:360, 340:460] = (0.86, 0.70, 0.10)
    image[188:206, 400:418] = (0.04, 0.03, 0.03)
    image[193:198, 406:411] = (0.92, 0.92, 0.90)
    return np.clip(image, 0.0, 1.0)


class SeparationTests(unittest.TestCase):
    def test_a_bird_keeps_its_color_and_the_background_goes_soft(self) -> None:
        original = _bird()
        finished, info = render_finish(original, Atmosphere(), [])
        before = _texture(original[16:90, 16:120])
        after = _texture(finished[16:90, 16:120])
        head = finished[170:220, 360:430]
        self.assertTrue(info["applied"])
        self.assertGreaterEqual(info["eyes"], 1)
        self.assertLess(after, before * 0.45)
        self.assertGreater(float(head[..., 0].mean()), float(head[..., 1].mean()) + 0.15)
        self.assertGreater(float(head[..., 0].mean()), 0.45)

    def test_a_facade_is_not_treated_as_a_subject(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "facade.png"
            sample_image("architecture").save(path)
            recipe = analyze(path, detector="heuristic", advisor="histogram", output_dir=directory)
            _, _, edited = write_outputs(recipe, directory)
            self.assertFalse(recipe["separation"]["applied"])
            self.assertEqual(recipe["separation"]["background"], "unchanged")
            self.assertTrue(edited.is_file())

    def test_a_portrait_gets_an_eye_mask(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "face.png"
            sample_image("portrait").save(path)
            recipe = analyze(path, detector="heuristic", advisor="histogram")
        self.assertTrue(recipe["separation"]["applied"])
        self.assertGreaterEqual(recipe["separation"]["eyes"], 1)
        self.assertEqual(recipe["input"]["kind"], "raster")


if __name__ == "__main__":
    unittest.main()
