"""Linear gradient exposure, and the caps around it.

The calls pass stage intents. They do not go through steer sentences.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from ai_dt.guides.decide import Picture
from ai_dt.schema import Atmosphere, Detection
from ai_dt.separate import render_finish


def _luma(pixels: np.ndarray) -> float:
    values = pixels.astype(np.float32)
    return float((0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]).mean())


def _picture(kind: str, **overrides: object) -> Picture:
    values: dict[str, object] = {
        "type": kind,
        "reason": "A fixture picture.",
        "brightness": 0.45,
        "warmth": 0.0,
        "saturation": 0.12,
        "edges": 0.03,
        "line_fraction": 0.01,
        "sky": 0.45,
        "skin": 0.0,
        "highlight_specks": 0.0,
        "compact_area": 0.04,
        "compact_fill": 0.6,
        "subject_box": None,
    }
    values.update(overrides)
    return Picture(**values)  # type: ignore[arg-type]


def _sky_over_land() -> np.ndarray:
    rng = np.random.default_rng(2)
    height, width = 480, 720
    frame = np.zeros((height, width, 3), dtype=np.float32)
    frame[:220] = (0.45, 0.62, 0.86)
    frame[220:] = (0.28, 0.38, 0.16)
    frame[220:] += rng.random((height - 220, width, 3), dtype=np.float32) * np.float32(0.08)
    return np.clip(frame, 0.0, 1.0)


def _flat_field() -> np.ndarray:
    return np.full((420, 640, 3), (0.32, 0.46, 0.20), dtype=np.float32)


def _creature_on_rock() -> np.ndarray:
    """A daylight body on a brown rock, with a brighter sky behind it."""

    image = np.full((480, 720, 3), (0.30, 0.32, 0.28), dtype=np.float32)
    image[20:120, 180:540] = (0.90, 0.92, 0.96)
    image[358:420, 60:660] = (0.55, 0.32, 0.14)
    image[160:362, 270:470] = (0.15, 0.45, 0.72)
    return image


def _dark_land_with_subject() -> tuple[np.ndarray, Picture, list[Detection]]:
    """Bright sky, a body in that sky, and a land that is already as dark."""

    pixels = np.zeros((480, 720, 3), dtype=np.float32)
    pixels[:220] = (0.84, 0.86, 0.90)
    pixels[220:] = (0.08, 0.08, 0.08)
    pixels[40:180, 260:460] = (0.72, 0.48, 0.28)
    box = (260 / 720, 40 / 480, 200 / 720, 140 / 480)
    picture = _picture("animal", subject_box=box, compact_area=0.05)
    detections = [Detection("animal", "animal", 0.9, box, "test")]
    return pixels, picture, detections


_GRADIENT = {"stage": "gradient", "action": "darken"}
_CREATURE_INTENTS = [
    {"stage": "subject", "action": "darken"},
    {"stage": "rock", "action": "darken"},
    {"stage": "behind", "action": "lift"},
    _GRADIENT,
    {"stage": "eyes", "action": "skip"},
]


def _finish(pixels: np.ndarray, **kwargs: object) -> tuple[np.ndarray, dict]:
    detections = kwargs.pop("detections", [])
    return render_finish(pixels, Atmosphere(), list(detections), **kwargs)  # type: ignore[arg-type]


def _dramatic(subject: float, rock: float, behind: float, before_subject: float) -> bool:
    return (
        subject < before_subject - 0.06
        and rock < subject - 0.08
        and behind > subject + 0.06
    )


class GradientWhenTests(unittest.TestCase):
    def test_a_bright_sky_gets_one_soft_field_that_spares_the_land(self) -> None:
        pixels = _sky_over_land()
        graded, info = _finish(pixels, intents=[_GRADIENT])
        plain, plain_info = _finish(pixels, omit={"gradient"})
        sky = _luma(graded[:80]) - _luma(plain[:80])
        middle = _luma(graded[200:280]) - _luma(plain[200:280])
        land = _luma(graded[360:460]) - _luma(plain[360:460])
        self.assertEqual(info["stages_ran"].count("gradient"), 1)
        self.assertNotIn("gradient", plain_info["stages_ran"])
        self.assertLess(sky, -0.01)
        self.assertLess(sky, middle)
        self.assertLess(middle, land + 0.005)
        self.assertGreaterEqual(info["gradient"]["strength"], -0.35)
        self.assertLessEqual(info["gradient"]["strength"], 0.35)

    def test_a_close_portrait_a_night_frame_and_a_flat_field_are_skipped(self) -> None:
        sky = _sky_over_land()
        close = _picture("portrait", compact_area=0.2, skin=0.4, subject_box=(0.2, 0.1, 0.5, 0.7))
        dark = _picture("night", brightness=0.12, compact_area=0.0)
        portrait, portrait_info = _finish(sky, picture=close, intents=[_GRADIENT])
        night, night_info = _finish(sky, picture=dark, intents=[_GRADIENT])
        field, field_info = _finish(_flat_field(), intents=[_GRADIENT])
        self.assertNotIn("gradient", portrait_info["stages_ran"])
        self.assertIn("portrait", str(portrait_info["gradient"].get("reason") or "").lower())
        self.assertNotIn("gradient", night_info["stages_ran"])
        self.assertIn("night", str(night_info["gradient"].get("reason") or "").lower())
        self.assertNotIn("gradient", field_info["stages_ran"])
        self.assertIn("falloff", str(field_info["gradient"].get("reason") or "").lower())
        self.assertTrue(np.allclose(portrait, _finish(sky, picture=close, omit={"gradient"})[0]))
        self.assertTrue(np.allclose(night, _finish(sky, picture=dark, omit={"gradient"})[0]))
        self.assertTrue(np.allclose(field, _finish(_flat_field(), omit={"gradient"})[0]))


class GradientStopTests(unittest.TestCase):
    def test_a_gradient_stops_when_the_dark_end_matches_the_subject(self) -> None:
        pixels, picture, detections = _dark_land_with_subject()
        _graded, info = _finish(pixels, picture=picture, detections=detections, intents=[_GRADIENT])
        self.assertNotIn("gradient", info["stages_ran"])
        self.assertIn("dark end", str(info["gradient"].get("reason") or "").lower())

    def test_a_negligible_move_is_put_back(self) -> None:
        pixels = _sky_over_land()
        weak = {
            "axis": "vertical",
            "bright": "top",
            "strength": -0.02,
            "reason": "A weak falloff.",
        }
        with patch("ai_dt.reading.linear_falloff", return_value=weak):
            _graded, info = _finish(pixels, intents=[_GRADIENT])
        self.assertNotIn("gradient", info["stages_ran"])
        self.assertIn("negligible", str(info["gradient"].get("reason") or "").lower())

    def test_the_subject_stays_out_of_the_darkening(self) -> None:
        pixels = _sky_over_land()
        pixels = pixels.copy()
        # Darker than the land, so the dark-end stop does not fire, and sitting in the sky.
        pixels[36:150, 80:210] = (0.05, 0.06, 0.07)
        box = (80 / 720, 36 / 480, 130 / 720, 114 / 480)
        picture = _picture("animal", subject_box=box, compact_area=0.04)
        detections = [Detection("animal", "animal", 0.9, box, "test")]
        intents = [_GRADIENT, {"stage": "eyes", "action": "skip"}]
        graded, info = _finish(pixels, picture=picture, detections=detections, intents=intents)
        plain, _plain_info = _finish(pixels, picture=picture, detections=detections, omit={"gradient", "eyes"})
        subject = _luma(graded[50:140, 100:190]) - _luma(plain[50:140, 100:190])
        sky = _luma(graded[:30, 400:680]) - _luma(plain[:30, 400:680])
        self.assertIn("gradient", info["stages_ran"])
        self.assertGreater(subject, -0.02)
        self.assertLess(sky, subject - 0.01)

    def test_crushing_the_subject_puts_the_gradient_back(self) -> None:
        pixels = _sky_over_land()
        pixels = pixels.copy()
        pixels[10:180, 50:250] = (0.08, 0.09, 0.10)
        box = (80 / 720, 36 / 480, 130 / 720, 114 / 480)
        picture = _picture("animal", subject_box=box, compact_area=0.04)
        detections = [Detection("animal", "animal", 0.9, box, "test")]
        intents = [_GRADIENT, {"stage": "eyes", "action": "skip"}]
        from ai_dt.separate import _grade_region as real_grade

        def _leak(pixels: np.ndarray, mask: np.ndarray, plan: object) -> np.ndarray:
            # A gradient that ignores its mask and darkens the body.
            if abs(float(getattr(plan, "exposure", 0.0))) > 0.05 and float(getattr(plan, "contrast", 1.0)) == 0.0:
                return np.clip(pixels * np.float32(0.45), 0.0, 1.0)
            return real_grade(pixels, mask, plan)  # type: ignore[arg-type]

        with patch("ai_dt.separate._grade_region", side_effect=_leak):
            graded, info = _finish(pixels, picture=picture, detections=detections, intents=intents)
        plain, _plain_info = _finish(pixels, picture=picture, detections=detections, omit={"gradient", "eyes"})
        self.assertNotIn("gradient", info["stages_ran"])
        self.assertIn("crush", str(info["gradient"].get("reason") or "").lower())
        self.assertLess(abs(_luma(graded[50:140, 100:190]) - _luma(plain[50:140, 100:190])), 0.02)


class CapTests(unittest.TestCase):
    def _means(self, rendered: np.ndarray) -> tuple[float, float, float]:
        subject = _luma(rendered[190:320, 300:440])
        rock = _luma(rendered[375:415, 80:220])
        behind = _luma(rendered[30:100, 220:500])
        return subject, rock, behind

    def test_open_caps_can_separate_a_creature_on_a_rock(self) -> None:
        pixels = _creature_on_rock()
        before_subject = _luma(pixels[190:320, 300:440])
        graded, info = _finish(pixels, caps_open=True, intents=_CREATURE_INTENTS)
        subject, rock, behind = self._means(graded)
        self.assertTrue(info["strong"])
        self.assertTrue(_dramatic(subject, rock, behind, before_subject))
        self.assertLess(subject, before_subject - 0.06)
        self.assertLess(rock, subject - 0.08)
        self.assertGreater(behind, subject + 0.06)

    def test_closed_caps_do_not_separate_that_creature(self) -> None:
        pixels = _creature_on_rock()
        before_subject = _luma(pixels[190:320, 300:440])
        before_rock = _luma(pixels[375:415, 80:220])
        graded, info = _finish(pixels, intents=_CREATURE_INTENTS)
        explicit, explicit_info = _finish(pixels, caps_open=False, intents=_CREATURE_INTENTS)
        subject, rock, behind = self._means(graded)
        self.assertFalse(info["strong"])
        self.assertFalse(explicit_info["strong"])
        self.assertFalse(_dramatic(subject, rock, behind, before_subject))
        self.assertGreater(subject, before_subject * (2.0 ** -0.35) - 0.02)
        self.assertGreater(rock, before_rock * (2.0 ** -0.35) - 0.02)
        self.assertGreater(rock, subject - 0.08)
        self.assertTrue(np.allclose(graded, explicit, atol=1e-5))

    def test_open_caps_may_push_the_gradient_past_a_third_of_a_stop(self) -> None:
        pixels = _sky_over_land()
        opened, info = _finish(pixels, caps_open=True, intents=[_GRADIENT])
        closed, closed_info = _finish(pixels, caps_open=False, intents=[_GRADIENT])
        self.assertLess(info["gradient"]["strength"], -0.35)
        self.assertGreaterEqual(closed_info["gradient"]["strength"], -0.35)
        self.assertLess(_luma(opened[:80]), _luma(closed[:80]) - 0.01)

    def test_the_default_call_leaves_the_caps_closed(self) -> None:
        pixels = _creature_on_rock()
        before_subject = _luma(pixels[190:320, 300:440])
        before_rock = _luma(pixels[375:415, 80:220])
        graded, info = _finish(pixels)
        subject, rock, _behind = self._means(graded)
        self.assertFalse(info["strong"])
        self.assertGreater(subject, before_subject - 0.05)
        self.assertGreater(rock, before_rock - 0.08)


if __name__ == "__main__":
    unittest.main()
