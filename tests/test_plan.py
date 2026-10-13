"""Steer grammar, opened caps, gradient, and the text reasoner.

The reasoner is stubbed. These tests do not call Ollama or Gemini.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ai_dt.pipeline import analyze, write_outputs
from ai_dt.steer import parse_steer
from tests.test_guides import _flowers_beside_a_bird, _heron_on_water, _landscape, _save_array, _skin_head


def _luma(pixels: np.ndarray) -> float:
    values = pixels.astype(np.float32)
    return float((0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]).mean())


def _dry_creature() -> np.ndarray:
    """An olive body on grass. No water and no reflection."""

    image = np.full((480, 720, 3), (0.24, 0.40, 0.18), dtype=np.float32)
    image[170:360, 270:450] = (0.40, 0.36, 0.16)
    return image


def _creature_on_rock() -> np.ndarray:
    """A daylight body on a brown rock, with a brighter sky behind it."""

    image = np.full((480, 720, 3), (0.30, 0.32, 0.28), dtype=np.float32)
    # Bright sky sits behind the body and stays off the corners.
    image[20:120, 180:540] = (0.90, 0.92, 0.96)
    # A long thin rock is a perch, not a second creature.
    image[358:420, 60:660] = (0.55, 0.32, 0.14)
    image[160:362, 270:470] = (0.15, 0.45, 0.72)
    return image


def _flat_field() -> np.ndarray:
    return np.full((420, 640, 3), (0.32, 0.46, 0.20), dtype=np.float32)


def _analyze(pixels: np.ndarray | Image.Image, **kwargs) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frame.png"
        if isinstance(pixels, Image.Image):
            pixels.save(path)
        else:
            _save_array(pixels, path)
        if "output_dir" not in kwargs and kwargs.get("steer"):
            kwargs["output_dir"] = Path(tmp)
        return analyze(path, detector="heuristic", advisor="histogram", **kwargs)


class GrammarTests(unittest.TestCase):
    def test_epid_is_ignored_and_epic_opens_caps(self) -> None:
        ignored = parse_steer("epid")
        self.assertFalse(ignored["strong"])
        self.assertEqual(ignored["summary"], "Ignored: epid.")
        opened = parse_steer("epic, darken the subject")
        self.assertTrue(opened["strong"])
        self.assertEqual(opened["strong_word"], "epic")
        self.assertEqual(opened["asks"], [{"verb": "darken", "target": "subject", "source": "user"}])

    def test_the_reel_sentence_names_three_asks(self) -> None:
        parsed = parse_steer("strong, darken the subject, darken the foreground, light behind the subject")
        self.assertTrue(parsed["strong"])
        self.assertEqual(
            [(item["verb"], item["target"]) for item in parsed["asks"]],
            [("darken", "subject"), ("darken", "foreground"), ("lift", "behind")],
        )


class AnimalAndGradientTests(unittest.TestCase):
    def test_a_dry_creature_is_an_animal_not_a_face(self) -> None:
        recipe = _analyze(_dry_creature())
        names = [item["multi_name"] for item in recipe["modules"]]
        self.assertEqual(recipe["picture"]["type"], "animal")
        self.assertNotIn("AI face", names)
        vignettes = [item for item in recipe["modules"] if item["operation"] == "vignette"]
        self.assertTrue(vignettes)
        self.assertEqual(vignettes[0]["multi_name"], "AI animal")

    def test_a_warm_cluster_beside_a_bird_is_not_a_face(self) -> None:
        recipe = _analyze(_flowers_beside_a_bird())
        names = [item["multi_name"] for item in recipe["modules"]]
        self.assertNotIn("AI face", names)
        self.assertEqual(recipe["picture"]["type"], "animal")

    def test_a_bright_sky_gets_a_gradient_that_spares_the_land(self) -> None:
        pixels = _landscape()
        graded = _analyze(pixels)
        plain = _analyze(pixels, steer="skip the gradient")
        with_sky = _luma(np.array(graded["_rendered"])[:80])
        without_sky = _luma(np.array(plain["_rendered"])[:80])
        with_land = _luma(np.array(graded["_rendered"])[300:420])
        without_land = _luma(np.array(plain["_rendered"])[300:420])
        self.assertIn("gradient", graded["separation"]["stages_ran"])
        self.assertNotIn("gradient", plain["separation"]["stages_ran"])
        self.assertLess(with_sky, without_sky - 0.01)
        self.assertGreater(with_land - without_land, with_sky - without_sky)

    def test_a_close_portrait_and_a_flat_field_skip_the_gradient(self) -> None:
        portrait = _analyze(_skin_head())
        field = _analyze(_flat_field())
        self.assertNotIn("gradient", portrait["separation"]["stages_ran"])
        self.assertEqual(portrait["picture"]["type"], "portrait")
        self.assertNotIn("gradient", field["separation"]["stages_ran"])

    def test_a_gradient_stops_when_the_dark_end_matches_the_subject(self) -> None:
        pixels = np.zeros((480, 720, 3), dtype=np.float32)
        pixels[:220] = (0.84, 0.86, 0.90)
        pixels[220:] = (0.08, 0.08, 0.08)
        pixels[40:180, 260:460] = (0.72, 0.48, 0.28)
        recipe = _analyze(pixels)
        info = recipe["separation"].get("gradient") or {}
        self.assertNotIn("gradient", recipe["separation"]["stages_ran"])
        self.assertIn("dark end", str(info.get("reason") or "").lower())


class CapTests(unittest.TestCase):
    def _means(self, recipe: dict, source: np.ndarray) -> tuple[float, float, float]:
        rendered = np.array(recipe["_rendered"])
        subject = _luma(rendered[190:320, 300:440])
        rock = _luma(rendered[375:415, 80:220])
        behind = _luma(rendered[30:100, 220:500])
        return subject, rock, behind

    def test_epic_reel_steer_darkens_the_subject_and_the_rock(self) -> None:
        pixels = _creature_on_rock()
        before_subject = _luma(pixels[190:320, 300:440])
        recipe = _analyze(
            pixels,
            steer="epic, darken the subject, darken the foreground, light behind the subject, skip the eyes",
        )
        subject, rock, behind = self._means(recipe, pixels)
        self.assertTrue(recipe["plan"]["strong"])
        self.assertLess(subject, before_subject - 0.06)
        self.assertLess(rock, subject - 0.08)
        self.assertGreater(behind, subject + 0.06)
        text = recipe["plan"]["text"]
        self.assertIn("Caps open because you said epic.", text)
        self.assertIn("subject", text)
        self.assertIn("surround", text)

    def test_blank_and_moody_do_not_crush(self) -> None:
        pixels = _creature_on_rock()
        before_subject = _luma(pixels[190:320, 300:440])
        before_rock = _luma(pixels[375:415, 80:220])
        blank = _analyze(pixels)
        moody = _analyze(pixels, vibe="moody dark")
        for recipe in (blank, moody):
            subject, rock, _behind = self._means(recipe, pixels)
            self.assertFalse(recipe["plan"]["strong"])
            self.assertGreater(subject, before_subject - 0.05)
            self.assertGreater(rock, before_rock - 0.08)

    def test_reverting_the_surround_keeps_the_subject(self) -> None:
        pixels = _dry_creature()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "creature.png"
            _save_array(pixels, path)
            first = analyze(path, detector="heuristic", advisor="histogram", output_dir=directory)
            write_outputs(first, directory)
            second = analyze(
                path,
                detector="heuristic",
                advisor="histogram",
                output_dir=directory,
                steer="revert the surround",
            )
        self.assertIn("background", second["plan"]["skip"])
        self.assertNotIn("background", second["separation"]["stages_ran"])
        self.assertIn("subject", second["separation"]["stages_ran"])
        self.assertTrue(any(line["stage"] == "surround" and line["skipped"] for line in second["plan"]["lines"]))

    def test_epid_is_listed_and_does_not_open_caps(self) -> None:
        def skipped(*_args, **_kwargs) -> dict:
            return {
                "status": "skipped",
                "backend": "",
                "note": "Reasoning was skipped.",
                "stages": [],
            }

        with patch("ai_dt.pipeline.propose_stages", skipped):
            recipe = _analyze(_dry_creature(), steer="epid")
        self.assertFalse(recipe["plan"]["strong"])
        self.assertIn("Ignored: epid.", recipe["plan"]["summary"])
        self.assertIn("Ignored: epid.", recipe["plan"]["text"])
        self.assertIn("Caps closed.", recipe["plan"]["text"])


class ReasonerTests(unittest.TestCase):
    def test_a_stubbed_reasoner_is_overridden_by_steer(self) -> None:
        def fake(*_args, **_kwargs) -> dict:
            return {
                "status": "ollama",
                "backend": "llama3.2",
                "note": "Stages proposed by the local model llama3.2.",
                "stages": [
                    {"stage": "subject", "action": "lift", "reason": "Lift the body."},
                    {"stage": "surround", "action": "darken", "reason": "Quiet the surround."},
                ],
            }

        with patch("ai_dt.pipeline.propose_stages", fake):
            recipe = _analyze(_dry_creature(), steer="darken the subject")
        subject = next(line for line in recipe["plan"]["lines"] if line["stage"] == "subject")
        self.assertEqual(subject["source"], "user")
        self.assertTrue(subject["override"])
        self.assertFalse(recipe["plan"]["strong"])
        self.assertIn("llama3.2", recipe["plan"]["text"])

    def test_epic_opens_caps_and_epid_does_not(self) -> None:
        def fake(*_args, **_kwargs) -> dict:
            return {
                "status": "ollama",
                "backend": "llama3.2",
                "note": "Stages proposed by the local model llama3.2.",
                "stages": [{"stage": "subject", "action": "darken", "reason": "Darken the body."}],
            }

        with patch("ai_dt.pipeline.propose_stages", fake):
            epic = _analyze(_dry_creature(), steer="epic")
            epid = _analyze(_dry_creature(), steer="epid")
        self.assertTrue(epic["plan"]["strong"])
        self.assertIn("Caps open because you said epic.", epic["plan"]["text"])
        self.assertFalse(epid["plan"]["strong"])
        self.assertIn("Ignored: epid.", epid["plan"]["text"])

    def test_steer_still_applies_when_reasoning_is_skipped(self) -> None:
        def skipped(*_args, **_kwargs) -> dict:
            return {
                "status": "skipped",
                "backend": "",
                "note": "Reasoning was skipped.",
                "stages": [],
            }

        with patch("ai_dt.pipeline.propose_stages", skipped):
            recipe = _analyze(_skin_head(), steer="skip the eyes")
        self.assertIn("eyes", recipe["plan"]["skip"])
        self.assertNotIn("eyes", recipe["separation"]["stages_ran"])
        self.assertIn("Reasoning was skipped.", recipe["plan"]["text"])


class HeronSentenceTests(unittest.TestCase):
    def test_the_sentence_is_a_plan_with_the_reflection_running(self) -> None:
        sentence = "epic, dark end result with the bird as the main subject. Make the reflection clearer"

        def fake(*_args, **_kwargs) -> dict:
            return {
                "status": "ollama",
                "backend": "llama3.2",
                "note": "Stages proposed by the local model llama3.2.",
                "paragraph": "",
                "stages": [
                    {"stage": "subject", "action": "lift", "reason": "one short sentence"},
                    {"stage": "gradient", "action": "darken", "reason": "The sky side is brighter."},
                    {"stage": "sky", "action": "darken", "reason": "Darken the sky."},
                ],
            }

        with patch("ai_dt.pipeline.propose_stages", fake):
            recipe = _analyze(_heron_on_water(), steer=sentence)
        text = recipe["plan"]["text"]
        paragraph = str(recipe["plan"]["reasoning"].get("paragraph") or "")
        self.assertIn("bird", paragraph.lower())
        self.assertIn("reflection", paragraph.lower())
        self.assertIn(paragraph, text)
        self.assertNotIn("one short sentence", text.lower())
        self.assertNotIn("reflection", recipe["plan"]["ignored"])
        self.assertNotIn("dark", recipe["plan"]["ignored"])
        self.assertNotIn("Ignored:", text)
        self.assertTrue(recipe["plan"]["caps_open"])
        self.assertIn("reflection", recipe["separation"]["stages_ran"])
        self.assertNotIn("reflection — skipped", text)
        self.assertNotIn("gradient", recipe["separation"]["stages_ran"])
        self.assertNotIn("gradient", text.lower())
        self.assertNotIn("\nsky —", text.lower())
        self.assertNotIn("Skipped:", text)
        asks = {(item["verb"], item["target"]) for item in recipe["plan"]["asks"]}
        self.assertIn(("darken", "foreground"), asks)
        self.assertIn(("lift", "reflection"), asks)

    def test_the_heron_photo_runs_the_reflection_and_does_not_skip_it(self) -> None:
        sentence = "epic, dark end result with the bird as the main subject. Make the reflection clearer"
        path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "heron.jpg"

        def fake(*_args, **_kwargs) -> dict:
            return {
                "status": "ollama",
                "backend": "llama3.2",
                "note": "Stages proposed by the local model llama3.2.",
                "paragraph": "",
                "stages": [
                    {"stage": "reflection", "action": "skip", "reason": "Skip the mirror."},
                    {"stage": "foreground", "action": "darken", "reason": "The land is graded on its own."},
                    {"stage": "gradient", "action": "darken", "reason": "The sky side is brighter."},
                ],
            }

        with patch("ai_dt.pipeline.propose_stages", fake):
            recipe = analyze(path, detector="heuristic", advisor="histogram", steer=sentence)
        text = recipe["plan"]["text"]
        self.assertEqual(recipe["picture"]["type"], "animal")
        self.assertIn("reflection", recipe["separation"]["stages_ran"])
        self.assertNotIn("foreground", recipe["separation"]["stages_ran"])
        self.assertNotIn("reflection — skipped", text)
        self.assertNotIn("Skipped:", text)
        self.assertNotIn("the land is graded", text.lower())
        self.assertNotIn("could not be found", text.lower())
        source = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        rendered = np.asarray(recipe["_rendered"]).astype(np.float32)
        echo = rendered[500:640, 540:660]
        beside = rendered[500:640, 200:320]
        source_echo = source[500:640, 540:660]
        source_beside = source[500:640, 200:320]
        echo_gain = _luma(echo) - _luma(source_echo)
        beside_gain = _luma(beside) - _luma(source_beside)
        self.assertGreater(echo_gain, beside_gain + 0.02)
        self.assertGreater(echo_gain, 0.02)


if __name__ == "__main__":
    unittest.main()
