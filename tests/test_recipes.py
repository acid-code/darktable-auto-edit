"""Nearest-neighbor vibe matching. No network and no Darktable."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from ai_dt.pipeline import analyze
from ai_dt.recipes.ingest import ingest_note
from ai_dt.recipes.library import load_library, match_vibe
from ai_dt.simulate import render_preview, sample_image


def _note(recipe_id: str, scene: str, vibe: str) -> dict:
    return {
        "id": recipe_id,
        "title": "Bleach portrait",
        "scenes": [scene],
        "vibes": [vibe, "faded"],
        "description": "A faded high-key portrait with pale color and lifted shadows.",
        "stages": {
            "exposure_ev": 0.1,
            "contrast": -0.2,
            "temperature_shift": 0.1,
            "tint": 0.0,
            "saturation": -0.3,
            "shadows": 0.4,
            "highlights": 0.05,
            "vibrance": -0.1,
            "vignette_strength": 0.2,
        },
    }


class LibraryTests(unittest.TestCase):
    def test_shipped_library_loads_without_warnings(self) -> None:
        recipes, warnings = load_library()
        self.assertEqual(warnings, [])
        ids = {item.id for item in recipes}
        self.assertIn("portrait-faithful", ids)
        self.assertIn("portrait-moody", ids)
        self.assertIn("architecture-crisp", ids)
        self.assertGreaterEqual(len(recipes), 10)

    def test_blank_vibe_on_a_portrait_stays_faithful(self) -> None:
        match = match_vibe("portrait", "")
        self.assertEqual(match.recipe.id if match.recipe else None, "portrait-faithful")
        self.assertTrue(match.applied)
        self.assertIn("faithful", match.query)
        self.assertEqual(match.recipe.stages["exposure_ev"], 0.0)
        self.assertEqual(match.vectorizer, "term-frequency")
        self.assertEqual(match.method, "nearest-neighbor")

    def test_moody_dark_portrait_beats_architecture(self) -> None:
        match = match_vibe("portrait", "moody dark portrait")
        self.assertEqual(match.recipe.id if match.recipe else None, "portrait-moody")
        neighbor_ids = [item.id for item in match.neighbors]
        self.assertEqual(neighbor_ids[0], "portrait-moody")
        self.assertNotIn("architecture-crisp", neighbor_ids[:1])

    def test_crisp_building_hits_the_architecture_recipe(self) -> None:
        match = match_vibe("architecture", "crisp building")
        self.assertEqual(match.recipe.id if match.recipe else None, "architecture-crisp")

    def test_quiet_night_on_a_general_frame_finds_the_night_recipe(self) -> None:
        match = match_vibe("general", "quiet night")
        self.assertEqual(match.recipe.id if match.recipe else None, "night-quiet")

    def test_nonsense_falls_back_to_faithful(self) -> None:
        match = match_vibe("portrait", "zzzz qqqq xxxx")
        self.assertEqual(match.recipe.id if match.recipe else None, "portrait-faithful")
        self.assertIn("faithful", match.reason)

    def test_folder_name_is_not_required_for_a_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp)
            odd = library / "not-a-scene"
            odd.mkdir()
            payload = _note("portrait-bleach", "portrait", "bleach")
            (odd / "note.json").write_text(json.dumps(payload), encoding="utf-8")
            match = match_vibe("portrait", "bleach faded", directory=library)
        self.assertEqual(match.recipe.id if match.recipe else None, "portrait-bleach")


class IngestTests(unittest.TestCase):
    def test_a_note_is_filed_under_its_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note = root / "note.json"
            library = root / "recipes"
            note.write_text(json.dumps(_note("portrait-bleach", "portrait", "bleach")), encoding="utf-8")
            destination = ingest_note(note, library)
            self.assertEqual(destination, library / "portrait" / "portrait-bleach.json")
            recipes, warnings = load_library(library)
        self.assertEqual(warnings, [])
        self.assertEqual(recipes[0].id, "portrait-bleach")

    def test_a_broken_note_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note = root / "note.json"
            note.write_text('{"id": "nope", "title": ""}', encoding="utf-8")
            with self.assertRaises(ValueError):
                ingest_note(note, root / "recipes")

    def test_an_existing_id_is_not_replaced_quietly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note = root / "note.json"
            library = root / "recipes"
            note.write_text(json.dumps(_note("portrait-bleach", "portrait", "bleach")), encoding="utf-8")
            ingest_note(note, library)
            with self.assertRaises(ValueError):
                ingest_note(note, library)


class PipelineVibeTests(unittest.TestCase):
    def test_a_dark_frame_keeps_its_exposure_under_the_faithful_look(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "dark.png"
            Image.new("RGB", (80, 60), (18, 18, 18)).save(path)
            recipe = analyze(
                path,
                detector="heuristic",
                advisor="histogram",
                output_dir=directory,
                style_name="AI Edit dark",
            )
        self.assertGreater(recipe["analysis"]["atmosphere"]["exposure_ev"], 0.8)
        self.assertEqual(recipe["recipe_match"]["id"], "general-faithful")
        self.assertEqual(recipe["recipe_match"]["stages"]["contrast"], 0.0)

    def test_moody_changes_the_grade_without_taking_over_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "portrait.png"
            sample_image("portrait").save(path)
            plain = analyze(
                path,
                detector="heuristic",
                advisor="histogram",
                output_dir=directory,
                style_name="AI Edit plain",
            )
            moody = analyze(
                path,
                detector="heuristic",
                advisor="histogram",
                vibe="moody dark",
                output_dir=directory,
                style_name="AI Edit moody",
            )
        self.assertEqual(moody["recipe_match"]["id"], "portrait-moody")
        plain_exposure = plain["analysis"]["atmosphere"]["exposure_ev"]
        moody_exposure = moody["analysis"]["atmosphere"]["exposure_ev"]
        self.assertLess(abs(moody_exposure - plain_exposure), 0.25)
        self.assertLess(
            moody["analysis"]["atmosphere"]["saturation"],
            plain["analysis"]["atmosphere"]["saturation"],
        )
        vignette = next(item for item in moody["modules"] if item["operation"] == "vignette")
        plain_vignette = next(item for item in plain["modules"] if item["operation"] == "vignette")
        self.assertLess(vignette["params"]["brightness"], plain_vignette["params"]["brightness"])
        after = render_preview(sample_image("portrait"), moody)
        before = sample_image("portrait")
        self.assertNotEqual(list(after.getdata())[0], list(before.getdata())[0])


if __name__ == "__main__":
    unittest.main()
