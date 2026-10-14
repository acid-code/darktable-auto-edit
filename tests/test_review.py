"""Faithful night grades stay off a red cast, and invented water is dropped."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from ai_dt.modules_extra import channelmixer_params
from ai_dt.pipeline import analyze
from ai_dt.review import review_plan
from ai_dt.steer import apply_proposal, empty_plan


def _decode_mixer(blob: bytes) -> dict:
    red = struct.unpack_from("<f", blob, 0)[0]
    blue = struct.unpack_from("<f", blob, 40)[0]
    illuminant, _fluo, _led, adaptation = struct.unpack_from("<4i", blob, 120)
    _x, _y, kelvin, _gamut = struct.unpack_from("<4f", blob, 136)
    return {
        "red": red,
        "blue": blue,
        "illuminant": illuminant,
        "adaptation": adaptation,
        "kelvin": kelvin,
    }


class ChannelMixerTests(unittest.TestCase):
    def test_the_train_warmth_does_not_replace_the_camera_white_balance(self) -> None:
        # Job 48 packed warmth +0.07 and tint -0.0407 as a custom daylight
        # illuminant. Darktable then used that as the pipe white balance.
        blob = channelmixer_params(0.07, -0.0407)
        self.assertIsNotNone(blob)
        decoded = _decode_mixer(blob or b"")
        self.assertEqual(decoded["illuminant"], 0)
        self.assertEqual(decoded["adaptation"], 4)
        self.assertGreater(decoded["kelvin"], 4800)
        self.assertLess(decoded["red"] / decoded["blue"], 1.12)

    def test_a_large_warmth_stays_a_small_channel_gain(self) -> None:
        blob = channelmixer_params(0.8, -0.5)
        self.assertIsNotNone(blob)
        decoded = _decode_mixer(blob or b"")
        self.assertEqual(decoded["illuminant"], 0)
        self.assertLess(decoded["red"], 1.1)
        self.assertGreater(decoded["blue"], 0.9)


class FaithfulNightTests(unittest.TestCase):
    def test_a_faithful_night_grade_does_not_take_a_red_cast_or_invent_water(self) -> None:
        image = Image.new("RGB", (240, 160), (8, 14, 36))
        proposal = {
            "status": "gemini",
            "backend": "gemini-3.5-flash-lite",
            "note": "Stages proposed by Gemini.",
            "paragraph": (
                "the train needs attention to recover details in the blown highlights "
                "while keeping the water and reflection natural."
            ),
            "stages": [
                {
                    "stage": "subject",
                    "action": "lift",
                    "reason": "recover detail in the bright metallic highlights of the train",
                },
                {
                    "stage": "reflection",
                    "action": "lift",
                    "reason": "to clarify the reflection and prevent it from being lost",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "night.png"
            image.save(path)
            recipe = analyze(
                path,
                detector="heuristic",
                advisor="histogram",
                steer="the main theme here is the train",
                look="faithful",
                saved_advice={
                    "suggestion": {
                        "temperature_shift": 0.8,
                        "tint": -0.5,
                        "mood": "warm",
                        "rationale": "needs warmth",
                    },
                    "advisor_notes": [],
                },
                proposal=proposal,
            )
        self.assertEqual(recipe["picture"]["type"], "night")
        atmosphere = recipe["analysis"]["atmosphere"]
        self.assertLess(abs(atmosphere["temperature_shift"]), 0.08)
        self.assertGreater(atmosphere["tint"], -0.05)
        self.assertLess(atmosphere["temperature_shift"], 0.2)
        spoken = " ".join(
            [
                str(recipe["plan"].get("label") or ""),
                str(recipe["plan"].get("text") or ""),
                str((recipe["plan"].get("reasoning") or {}).get("paragraph") or ""),
            ]
        ).lower()
        self.assertNotIn("water", spoken)
        self.assertNotIn("reflection", spoken)
        self.assertIn("train", spoken)
        self.assertFalse(recipe["plan"]["caps_open"])
        asks = recipe["plan"]["asks"]
        self.assertFalse(any(item.get("target") == "reflection" for item in asks))
        for module in recipe["modules"]:
            if module["operation"] == "colorbalancergb":
                self.assertLess(abs(module["params"].get("global_C", 0.0)), 0.03)
        for plugin in recipe["_plugins"]:
            if plugin["operation"] != "channelmixerrgb":
                continue
            decoded = _decode_mixer(bytes.fromhex(plugin["op_params"]))
            self.assertEqual(decoded["illuminant"], 0)
            self.assertLess(decoded["red"] / max(decoded["blue"], 1e-6), 1.12)


class ClaimTests(unittest.TestCase):
    def test_water_in_the_steer_is_kept(self) -> None:
        plan = empty_plan(["grade"])
        plan = apply_proposal(
            plan,
            {
                "status": "gemini",
                "paragraph": "Darken the water and lift the reflection.",
                "stages": [
                    {"stage": "foreground", "action": "darken", "reason": "The water should go darker."},
                    {"stage": "reflection", "action": "lift", "reason": "The reflection should get clearer."},
                ],
            },
        )
        picture = type("Picture", (), {"type": "landscape", "reason": "Sky over a ridge."})()
        reading = type("Reading", (), {"sentence": "The ridge is the subject.", "regions": [], "problems": [], "leave_alone": []})()
        reviewed = review_plan(plan, picture, reading, "darken the water and lift the reflection")
        paragraph = str(reviewed["reasoning"]["paragraph"]).lower()
        self.assertIn("water", paragraph)
        self.assertIn("reflection", paragraph)
        self.assertFalse(reviewed["caps_open"])


class WarmerNightTests(unittest.TestCase):
    def test_warmer_on_a_night_frame_does_not_install_a_red_cast(self) -> None:
        image = Image.new("RGB", (240, 160), (8, 14, 36))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "night.png"
            image.save(path)
            recipe = analyze(
                path,
                detector="heuristic",
                advisor="histogram",
                steer="the train",
                look="warmer",
                saved_advice={
                    "suggestion": {
                        "temperature_shift": 0.8,
                        "tint": -0.5,
                        "mood": "warm",
                        "rationale": "needs warmth",
                    },
                    "advisor_notes": [],
                },
            )
        self.assertEqual(recipe["picture"]["type"], "night")
        self.assertFalse(recipe["plan"]["caps_open"])
        shift = recipe["analysis"]["atmosphere"]["temperature_shift"]
        self.assertGreater(shift, 0.08)
        self.assertLess(shift, 0.2)
        color = next(item for item in recipe["modules"] if item["operation"] == "colorbalancergb")
        self.assertEqual(color["params"]["global_C"], 0.0)
        self.assertEqual(color["params"]["global_H"], 0.0)
        found = False
        for plugin in recipe["_plugins"]:
            if plugin["operation"] != "channelmixerrgb":
                continue
            found = True
            decoded = _decode_mixer(bytes.fromhex(plugin["op_params"]))
            self.assertEqual(decoded["illuminant"], 0)
            self.assertEqual(decoded["adaptation"], 4)
            self.assertGreater(decoded["kelvin"], 4800)
            self.assertLess(decoded["red"] / max(decoded["blue"], 1e-6), 1.12)
        self.assertTrue(found)


class ShadowLiftTests(unittest.TestCase):
    def test_open_shadows_follows_the_picture_and_does_not_add_fifty_percent(self) -> None:
        image = Image.new("RGB", (240, 160), (36, 22, 10))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warm-dark.png"
            image.save(path)
            recipe = analyze(
                path,
                detector="heuristic",
                advisor="histogram",
                look="shadow-lift",
            )
        tone = next(item for item in recipe["modules"] if item["operation"] == "toneequal")
        color = next(item for item in recipe["modules"] if item["operation"] == "colorbalancergb")
        self.assertGreater(tone["params"]["shadows"], 0.3)
        self.assertLess(tone["params"]["shadows"], 0.5)
        self.assertLessEqual(color["params"]["brilliance_shadows"], 0.12)
        self.assertNotEqual(round(color["params"]["brilliance_shadows"], 2), 0.5)
        self.assertEqual(color["params"]["global_H"], 0.0)
        self.assertEqual(color["params"]["global_C"], 0.0)


class GradientPlacementTests(unittest.TestCase):
    def test_an_upper_bright_frame_is_not_a_center_line(self) -> None:
        image = Image.new("RGB", (240, 160), (18, 22, 28))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 239, 47), fill=(230, 228, 220))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sky.png"
            image.save(path)
            recipe = analyze(
                path,
                detector="heuristic",
                advisor="histogram",
                steer="gradient",
                look="",
            )
        gradient = next(item for item in recipe["modules"] if item["operation"] == "graduatednd")
        offset = float(gradient["params"]["offset"])
        rotation = float(gradient["params"]["rotation"])
        self.assertGreater(gradient["params"]["density"], 0.4)
        self.assertGreater(gradient["params"]["hardness"], 20.0)
        self.assertFalse(abs(rotation) < 1.0 and abs(offset - 50.0) < 2.0)
        self.assertLess(offset, 42.0)
