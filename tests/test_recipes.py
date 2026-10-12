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

from ai_dt.guides.decide import Picture
from ai_dt.guides.library import match_guide
from ai_dt.pipeline import analyze
from ai_dt.recipes.ingest import ingest_note
from ai_dt.recipes.library import load_library, match_vibe
from ai_dt.recipes.reason import preset_phrase, reason_about
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
        self.assertIn("eye-detail", ids)
        self.assertGreaterEqual(len(recipes), 11)

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

    def test_velvia_finds_the_fuji_landscape_preset(self) -> None:
        match = match_vibe("landscape", "velvia vivid")
        self.assertEqual(match.recipe.id if match.recipe else None, "fuji-velvia")

    def test_portra_finds_the_portrait_preset(self) -> None:
        match = match_vibe("portrait", "portra")
        self.assertTrue((match.recipe.id if match.recipe else "").startswith("kodak-portra"))

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


def _picture(
    picture_type: str,
    brightness: float = 0.5,
    warmth: float = 0.0,
    saturation: float = 0.2,
) -> Picture:
    return Picture(
        type=picture_type,
        reason=f"This is a {picture_type}.",
        brightness=brightness,
        warmth=warmth,
        saturation=saturation,
        edges=0.1,
        line_fraction=0.02,
        sky=0.7 if picture_type == "landscape" else 0.1,
        skin=0.4 if picture_type == "portrait" else 0.0,
        highlight_specks=0.0,
        compact_area=0.2,
        compact_fill=0.5,
    )


class ReasonTests(unittest.TestCase):
    def test_a_dark_portrait_keeps_the_frame_and_masks_the_eyes(self) -> None:
        picture = _picture("portrait", brightness=0.12, warmth=-0.02)
        guide = match_guide("portrait").guide
        match = match_vibe("portrait", "")
        reasoning = reason_about(picture, [], "", guide, match)
        used = {item["id"]: item for item in reasoning.used}
        aside = {item["id"] for item in reasoning.set_aside}
        self.assertIn("portrait-faithful", used)
        self.assertEqual(used["portrait-faithful"]["place"], "frame")
        self.assertIn("eye-detail", used)
        self.assertEqual(used["eye-detail"]["place"], "eyes")
        self.assertIn("eyes", reasoning.description)
        self.assertIn("night-quiet", aside)
        self.assertIn("portrait-moody", aside)
        self.assertIsNotNone(reasoning.frame_stages)
        assert reasoning.frame_stages is not None
        self.assertEqual(reasoning.frame_stages["temperature_shift"], 0.0)
        self.assertEqual(reasoning.frame_stages["saturation"], 0.0)
        self.assertIn("eyes", reasoning.nudges)
        self.assertLessEqual(abs(reasoning.nudges["eyes"].saturation), 0.02)
        self.assertNotIn("horizon", reasoning.nudges)

    def test_a_golden_sunset_warms_the_line_and_not_the_frame(self) -> None:
        picture = _picture("landscape", brightness=0.55, warmth=0.08)
        guide = match_guide("landscape").guide
        match = match_vibe("landscape", "golden sunset")
        self.assertEqual(match.recipe.place if match.recipe else None, "horizon")
        reasoning = reason_about(picture, [], "golden sunset", guide, match)
        self.assertIsNone(reasoning.frame_stages)
        self.assertIn("horizon", reasoning.nudges)
        self.assertLessEqual(reasoning.nudges["horizon"].saturation, 0.0)
        self.assertGreater(reasoning.nudges["horizon"].warmth, 0.0)
        places = {item["id"]: item["place"] for item in reasoning.used}
        self.assertEqual(places.get("landscape-golden"), "horizon")
        self.assertNotIn("eye-detail", places)

    def test_crisp_architecture_is_a_line_mask(self) -> None:
        picture = _picture("architecture", brightness=0.5, warmth=0.0)
        guide = match_guide("architecture").guide
        match = match_vibe("architecture", "crisp building")
        self.assertEqual(match.recipe.place if match.recipe else None, "lines")
        reasoning = reason_about(picture, [], "crisp building", guide, match)
        self.assertIsNone(reasoning.frame_stages)
        self.assertIn("foreground", reasoning.nudges)
        self.assertLessEqual(abs(reasoning.nudges["foreground"].saturation), 0.03)
        self.assertGreater(reasoning.nudges["foreground"].contrast, 0.0)

    def test_blank_vibe_does_not_select_a_fuji_recipe(self) -> None:
        picture = _picture("landscape", saturation=0.05)
        guide = match_guide("landscape").guide
        match = match_vibe("landscape", "")
        reasoning = reason_about(picture, [], "", guide, match)
        fuji = [item["id"] for item in reasoning.used if item["id"].startswith("fuji-")]
        self.assertEqual(fuji, [])
        self.assertFalse(match.recipe.id.startswith("fuji-") if match.recipe else False)
        self.assertNotEqual(match.method, "fuji-simulation")

    def test_velvia_selects_fuji_velvia(self) -> None:
        picture = _picture("portrait", warmth=0.1)
        guide = match_guide("portrait").guide
        match = match_vibe("portrait", "velvia")
        reasoning = reason_about(picture, [], "velvia", guide, match)
        self.assertEqual(match.recipe.id if match.recipe else None, "fuji-velvia")
        frame = [item["id"] for item in reasoning.used if item["place"] == "frame"]
        self.assertEqual(frame, ["fuji-velvia"])
        self.assertIn("Velvia", reasoning.decision)
        self.assertIn("eye-detail", {item["id"] for item in reasoning.used})

    def test_fuji_on_a_landscape_selects_the_landscape_sim(self) -> None:
        guide = match_guide("landscape").guide
        open_land = _picture("landscape", saturation=0.05)
        match = match_vibe("general", "fuji")
        reasoning = reason_about(open_land, [], "fuji", guide, match)
        self.assertEqual(match.recipe.id if match.recipe else None, "fuji-velvia")
        frame = [item["id"] for item in reasoning.used if item["place"] == "frame"]
        self.assertEqual(frame, ["fuji-velvia"])
        self.assertIn("Velvia", reasoning.decision)

        saturated = _picture("landscape", saturation=0.3)
        held = match_vibe("general", "fujicam")
        held_reason = reason_about(saturated, [], "fujicam", guide, held)
        self.assertEqual(held.recipe.id if held.recipe else None, "fuji-provia")
        self.assertEqual(
            [item["id"] for item in held_reason.used if item["place"] == "frame"],
            ["fuji-provia"],
        )
        self.assertIn("Provia", held_reason.decision)

    def test_fuji_on_a_portrait_does_not_select_velvia_or_cool(self) -> None:
        picture = _picture("portrait", brightness=0.55, warmth=0.12)
        guide = match_guide("portrait").guide
        match = match_vibe("portrait", "fuji")
        reasoning = reason_about(picture, [], "fuji", guide, match)
        self.assertEqual(match.recipe.id if match.recipe else None, "fuji-astia")
        self.assertNotEqual(match.recipe.id if match.recipe else None, "fuji-velvia")
        frame = [item["id"] for item in reasoning.used if item["place"] == "frame"]
        self.assertEqual(frame, ["fuji-astia"])
        self.assertIsNotNone(reasoning.frame_stages)
        assert reasoning.frame_stages is not None
        self.assertGreaterEqual(reasoning.frame_stages["temperature_shift"], -0.02)
        self.assertGreaterEqual(reasoning.frame_stages["tint"], 0.0)
        self.assertIn("eye-detail", {item["id"] for item in reasoning.used})
        self.assertEqual(
            next(item["place"] for item in reasoning.used if item["id"] == "eye-detail"),
            "eyes",
        )
        self.assertIn("Astia", reasoning.decision)

    def test_a_preset_phrase_is_the_only_frame_look(self) -> None:
        recipes, warnings = load_library()
        self.assertEqual(warnings, [])
        presets = [item for item in recipes if "/presets/" in item.path.replace("\\", "/")]
        self.assertGreaterEqual(len(presets), 12)
        picture = _picture("portrait", brightness=0.55, warmth=0.12)
        guide = match_guide("portrait").guide
        phrases = {preset_phrase(item.id) for item in presets}
        self.assertEqual(len(phrases), len(presets))
        self.assertEqual(preset_phrase("fuji-velvia"), "fuji velvia")
        self.assertEqual(preset_phrase("fuji-classic-chrome"), "fuji classic chrome")
        self.assertEqual(preset_phrase("kodak-portra-400"), "kodak portra 400")
        for recipe in presets:
            phrase = preset_phrase(recipe.id)
            match = match_vibe("portrait", phrase, recipes=recipes)
            reasoning = reason_about(picture, [], phrase, guide, match, recipes=recipes)
            frame = [item["id"] for item in reasoning.used if item["place"] == "frame"]
            self.assertEqual(frame, [recipe.id], phrase)
            self.assertEqual(match.method, "preset")
            self.assertIn(recipe.title, reasoning.decision)
        eyes = reason_about(
            picture,
            [],
            preset_phrase("fuji-velvia"),
            guide,
            match_vibe("portrait", preset_phrase("fuji-velvia"), recipes=recipes),
            recipes=recipes,
        )
        self.assertEqual(
            [item["id"] for item in eyes.used if item["place"] == "frame"],
            ["fuji-velvia"],
        )
        self.assertIn("eye-detail", {item["id"] for item in eyes.used})
        faithful = reason_about(picture, [], "", guide, match_vibe("portrait", "", recipes=recipes), recipes=recipes)
        self.assertFalse(any(item["id"].startswith("fuji-") or item["id"].startswith("kodak-") for item in faithful.used))


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
        self.assertIn("description", recipe["reasoning"])
        self.assertNotIn("eye-detail", {item["id"] for item in recipe["reasoning"]["used"]})

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
