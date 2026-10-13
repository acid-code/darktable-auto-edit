"""Picture type decides the guide. A vibe cannot turn a portrait into wildlife."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ai_dt.cli import main
from ai_dt.detectors.heuristic import gate_faces
from ai_dt.guides.decide import Picture, decide_picture
from ai_dt.guides.ingest import ingest_note
from ai_dt.guides.library import load_guides, match_guide, validate_guide
from ai_dt.guides.support import reflected_subject, water_reflection
from ai_dt.pipeline import analyze, write_outputs
from ai_dt.schema import Atmosphere, Detection
from ai_dt.separate import render_finish
from ai_dt.simulate import sample_image

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ROOT / "guides"


def _luma_mean(pixels: np.ndarray) -> float:
    values = pixels.astype(np.float32)
    return float((0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]).mean())


def _heron_on_water() -> np.ndarray:
    """A grey bird over dark water, with pale flowers off to the right."""

    height, width = 480, 720
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:150] = (0.07, 0.12, 0.06)
    image[150:310] = (0.14, 0.26, 0.11)
    image[310:] = (0.045, 0.05, 0.05)
    image[220:330, 240:430] = (0.40, 0.42, 0.45)
    image[175:250, 340:450] = (0.52, 0.53, 0.56)
    image[198:214, 400:418] = (0.06, 0.06, 0.07)
    image[340:450, 300:370] = (0.16, 0.17, 0.18)
    yy, xx = np.ogrid[0:height, 0:width]
    for cx, cy in ((560, 200), (610, 220), (655, 195), (585, 245), (650, 240)):
        blob = (xx - cx) ** 2 / 18**2 + (yy - cy) ** 2 / 14**2 <= 1.0
        image[blob] = (0.90, 0.78, 0.47)
    return image


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


def _save_array(pixels: np.ndarray, path: Path) -> None:
    image = Image.fromarray(np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    image.save(path)


def _line_grid() -> Image.Image:
    image = Image.new("RGB", (640, 480), (186, 184, 176))
    draw = ImageDraw.Draw(image)
    for x in range(16, 300, 14):
        draw.line((x, 24, x, 450), fill=(18, 18, 18), width=4)
    for y in range(24, 450, 14):
        draw.line((16, y, 300, y), fill=(18, 18, 18), width=4)
    return image


def _night() -> np.ndarray:
    rng = np.random.default_rng(1)
    frame = np.zeros((480, 720, 3), dtype=np.float32)
    frame[..., 0] = 0.04
    frame[..., 1] = 0.07
    frame[..., 2] = 0.16
    frame += rng.random(frame.shape, dtype=np.float32) * np.float32(0.02)
    for x, y in ((80, 100), (200, 140), (400, 90), (560, 160)):
        frame[y : y + 18, x : x + 14] = (0.85, 0.62, 0.22)
    return np.clip(frame, 0.0, 1.0)


def _blue_subject() -> np.ndarray:
    """A compact subject that is not red or yellow."""

    frame = np.full((480, 720, 3), (0.45, 0.48, 0.42), dtype=np.float32)
    frame[160:340, 260:460] = (0.15, 0.35, 0.75)
    return frame


def _landscape() -> np.ndarray:
    rng = np.random.default_rng(2)
    height, width = 480, 720
    frame = np.zeros((height, width, 3), dtype=np.float32)
    frame[:220] = (0.45, 0.62, 0.86)
    frame[220:] = (0.28, 0.38, 0.16)
    frame[220:] += rng.random((height - 220, width, 3), dtype=np.float32) * np.float32(0.08)
    return np.clip(frame, 0.0, 1.0)


def _mountain_over_water() -> np.ndarray:
    """A wide ridge over dark water, with the same ridges echoed below."""

    height, width = 480, 720
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:240] = (0.72, 0.82, 0.93)
    image[240:] = (0.07, 0.09, 0.11)
    image[90:240, 80:540] = (0.58, 0.56, 0.52)
    image[245:390, 80:540] = (0.22, 0.23, 0.24)
    for x in range(110, 520, 34):
        image[110:235, x : x + 5] = (0.34, 0.33, 0.30)
        image[255:375, x : x + 5] = (0.12, 0.13, 0.14)
    return image


def _house_over_water() -> np.ndarray:
    """A windowed facade and the darker copy of it in the water below."""

    height, width = 480, 720
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:250] = (0.62, 0.74, 0.90)
    image[250:] = (0.07, 0.09, 0.11)
    image[40:250, 40:500] = (0.78, 0.74, 0.68)
    for x in range(52, 490, 12):
        image[52:242, x : x + 3] = (0.18, 0.20, 0.22)
    for y in range(58, 236, 14):
        image[y : y + 3, 48:494] = (0.18, 0.20, 0.22)
    building = image[40:250, 40:500].copy()
    image[252:462, 40:500] = np.clip(building[::-1] * np.float32(0.38), 0.0, 1.0)
    image[250:, 540:] = (0.07, 0.09, 0.11)
    return image


def _sunset() -> np.ndarray:
    height, width = 480, 720
    frame = np.zeros((height, width, 3), dtype=np.float32)
    frame[:180] = (0.62, 0.70, 0.88)
    frame[180:230] = (0.86, 0.42, 0.18)
    frame[230:] = (0.24, 0.30, 0.16)
    return frame


def _note(guide_id: str = "tide-pool", picture_type: str = "tide-pool") -> dict:
    skip = "Skip this mask. A tide pool keeps the global grade."
    empty = {
        "build": "never",
        "skip": skip,
        "exposure": 0.0,
        "contrast": 0.0,
        "saturation": 0.0,
        "clarity": 0.0,
        "defocus": 0.0,
        "mute": 0.0,
    }
    return {
        "id": guide_id,
        "title": "Tide pool",
        "types": [picture_type, "tide"],
        "words": ["tide", "pool", "water", "rocks"],
        "description": "A small tide pool. Keep the rocks sharp and do not invent a creature from the reflections.",
        "reason": "Water and rock stay as they are, so this picture keeps the global grade.",
        "stages": ["grade"],
        "masks": {
            name: dict(empty)
            for name in ("subject", "eyes", "skin", "sky", "background", "foreground")
        },
    }


class GuideLibraryTests(unittest.TestCase):
    def test_shipped_guides_load_without_warnings(self) -> None:
        guides, warnings = load_guides()
        self.assertEqual(warnings, [])
        ids = {item.id for item in guides}
        for guide_id in (
            "portrait",
            "group",
            "animal",
            "landscape",
            "architecture",
            "night",
            "food",
            "still-life",
            "street",
            "macro",
            "general",
        ):
            self.assertIn(guide_id, ids)
        self.assertGreaterEqual(len(guides), 11)

    def test_each_picture_type_finds_its_own_guide(self) -> None:
        expected = {
            "portrait": "portrait",
            "group": "group",
            "animal": "animal",
            "wildlife": "animal",
            "landscape": "landscape",
            "architecture": "architecture",
            "building": "architecture",
            "night": "night",
            "food": "food",
            "still-life": "still-life",
            "street": "street",
            "macro": "macro",
            "general": "general",
        }
        for picture_type, guide_id in expected.items():
            match = match_guide(picture_type)
            self.assertEqual(match.guide.id, guide_id, picture_type)
            self.assertFalse(match.fallback, picture_type)
            self.assertEqual(match.vectorizer, "term-frequency")
            self.assertEqual(match.method, "nearest-neighbor")

    def test_a_nonsense_type_uses_the_faithful_general_guide(self) -> None:
        match = match_guide("zzzz qqqq")
        self.assertTrue(match.fallback)
        self.assertEqual(match.guide.id, "general")
        self.assertIn("faithful general guide", match.reason)
        self.assertIn("grade", match.guide.stages)
        self.assertNotIn("subject", match.guide.stages)

    def test_a_guide_cannot_ask_for_a_blowout(self) -> None:
        note = _note()
        note["masks"]["subject"]["exposure"] = 4.0
        note["masks"]["subject"]["build"] = "subject"
        with self.assertRaises(ValueError):
            validate_guide(note)

    def test_only_the_background_may_defocus(self) -> None:
        note = _note()
        note["masks"]["foreground"]["defocus"] = 0.8
        note["masks"]["foreground"]["build"] = "foreground"
        with self.assertRaises(ValueError):
            validate_guide(note)

    def test_folder_name_is_not_required_for_a_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp)
            odd = library / "not-a-type"
            odd.mkdir()
            (odd / "note.json").write_text(json.dumps(_note()), encoding="utf-8")
            match = match_guide("tide-pool", directory=library)
        self.assertEqual(match.guide.id, "tide-pool")
        self.assertFalse(match.fallback)


class GuideIngestTests(unittest.TestCase):
    def test_a_json_note_is_filed_under_its_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note = root / "note.json"
            library = root / "guides"
            note.write_text(json.dumps(_note()), encoding="utf-8")
            destination = ingest_note(note, library)
            self.assertEqual(destination, library / "tide-pool" / "tide-pool.json")
            guides, warnings = load_guides(library)
        self.assertEqual(warnings, [])
        self.assertEqual(guides[0].id, "tide-pool")

    def test_a_text_note_is_filed_too(self) -> None:
        text = """
id: tide-pool
title: Tide pool
types: tide-pool, tide
words: tide, pool, water, rocks
description: A small tide pool. Keep the rocks sharp and do not invent a creature from the reflections.
reason: Water and rock stay as they are, so this picture keeps the global grade.
stages: grade

mask subject
build: never
skip: Skip this mask. A tide pool keeps the global grade.
exposure: 0

mask eyes
build: never
skip: Skip this mask. A tide pool keeps the global grade.

mask skin
build: never
skip: Skip this mask. A tide pool keeps the global grade.

mask sky
build: never
skip: Skip this mask. A tide pool keeps the global grade.

mask background
build: never
skip: Skip this mask. A tide pool keeps the global grade.

mask foreground
build: never
skip: Skip this mask. A tide pool keeps the global grade.
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note = root / "note.txt"
            library = root / "guides"
            note.write_text(text, encoding="utf-8")
            destination = ingest_note(note, library)
            self.assertTrue(destination.is_file())
            match = match_guide("tide", directory=library)
        self.assertEqual(match.guide.id, "tide-pool")

    def test_a_broken_note_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note = root / "note.json"
            note.write_text('{"id": "nope", "title": ""}', encoding="utf-8")
            with self.assertRaises(ValueError):
                ingest_note(note, root / "guides")

    def test_an_existing_id_is_not_replaced_quietly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note = root / "note.json"
            library = root / "guides"
            note.write_text(json.dumps(_note()), encoding="utf-8")
            ingest_note(note, library)
            with self.assertRaises(ValueError):
                ingest_note(note, library)


class PictureTypeTests(unittest.TestCase):
    def test_a_face_portrait_uses_the_portrait_guide_and_keeps_the_eyes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "face.png"
            sample_image("portrait").save(path)
            before = {item.relative_to(GUIDES) for item in GUIDES.rglob("*") if item.is_file()}
            recipe = analyze(path, detector="heuristic", advisor="histogram", vibe="wildlife")
            after = {item.relative_to(GUIDES) for item in GUIDES.rglob("*") if item.is_file()}
            rendered = np.array(recipe["_rendered"])
            write_outputs(recipe, directory)
            payload = json.loads((directory / "output.json").read_text(encoding="utf-8"))
        self.assertEqual(before, after)
        self.assertEqual(recipe["picture"]["type"], "portrait")
        self.assertEqual(recipe["guide"]["id"], "portrait")
        self.assertEqual(recipe["guide"]["file"].replace("\\", "/"), "guides/portrait.json")
        self.assertFalse(recipe["guide"]["fallback"])
        self.assertIn("face", recipe["guide"]["reason"].lower())
        ran = recipe["guide"]["stages_ran"]
        self.assertLess(ran.index("subject"), ran.index("eyes"))
        self.assertLess(ran.index("eyes"), ran.index("background"))
        self.assertTrue(recipe["separation"]["applied"])
        self.assertGreaterEqual(recipe["separation"]["eyes"], 1)
        self.assertEqual(recipe["separation"]["background"], "defocus")
        self.assertEqual(payload["picture"]["type"], "portrait")
        self.assertEqual(payload["guide"]["id"], "portrait")
        self.assertEqual(payload["features"], ["person"])
        self.assertNotIn("seat", payload["features"])
        self.assertNotIn("perch", payload["features"])
        # The background is softer than the original dark field, and the face is still there.
        self.assertGreater(float(rendered[80:200, 80:180].std()), 0.0)
        self.assertNotEqual(recipe["guide"]["id"], "animal")

    def test_a_wildlife_vibe_does_not_change_a_portrait(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "face.png"
            sample_image("portrait").save(path)
            recipe = analyze(
                path,
                detector="heuristic",
                advisor="histogram",
                vibe="wildlife",
            )
        self.assertEqual(recipe["picture"]["type"], "portrait")
        self.assertEqual(recipe["guide"]["id"], "portrait")
        self.assertNotIn("animal", recipe["guide"]["stages"])

    def test_a_line_grid_facade_is_architecture_and_is_not_defocused(self) -> None:
        image = _line_grid()
        original = np.asarray(image).astype(np.float32) / 255.0
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "facade.png"
            image.save(path)
            recipe = analyze(path, detector="heuristic", advisor="histogram", vibe="wildlife")
            rendered = np.array(recipe["_rendered"])
        self.assertEqual(recipe["picture"]["type"], "architecture")
        self.assertEqual(recipe["guide"]["id"], "architecture")
        self.assertFalse(recipe["separation"]["applied"])
        self.assertEqual(recipe["separation"]["background"], "unchanged")
        self.assertEqual(recipe["separation"]["subject_masks"], 0)
        self.assertNotIn("background", recipe["guide"]["stages_ran"])
        self.assertNotIn("subject", recipe["guide"]["stages_ran"])
        self.assertEqual(recipe["features"], ["building"])
        self.assertNotIn("perch", recipe["features"])
        crop = (slice(40, 420), slice(24, 280))
        edited = _texture(rendered[crop])
        blurred = (
            np.asarray(image.crop((24, 40, 280, 420)).filter(ImageFilter.GaussianBlur(radius=8))).astype(
                np.float32
            )
            / 255.0
        )
        self.assertGreater(edited, _texture(blurred) * 2.0)
        self.assertGreater(edited, _texture(original[crop]) * 0.7)

    def test_a_dark_blue_night_does_not_invent_a_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "night.png"
            _save_array(_night(), path)
            recipe = analyze(path, detector="heuristic", advisor="histogram", vibe="wildlife")
        self.assertEqual(recipe["picture"]["type"], "night")
        self.assertEqual(recipe["guide"]["id"], "night")
        self.assertIn("guides/night.json", recipe["guide"]["file"].replace("\\", "/"))
        self.assertFalse(recipe["separation"]["applied"])
        self.assertEqual(recipe["separation"]["subject_masks"], 0)
        self.assertEqual(recipe["separation"]["background"], "unchanged")
        self.assertNotIn("subject", recipe["guide"]["stages_ran"])
        self.assertNotIn("background", recipe["guide"]["stages_ran"])
        self.assertIn("grade", recipe["guide"]["stages_ran"])
        self.assertEqual(recipe["features"], [])

    def test_a_red_yellow_bird_uses_the_wildlife_guide_and_defocuses(self) -> None:
        pixels = _bird()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "bird.png"
            _save_array(pixels, path)
            recipe = analyze(path, detector="heuristic", advisor="histogram", vibe="crisp building")
            rendered = np.array(recipe["_rendered"])
        self.assertEqual(recipe["picture"]["type"], "animal")
        self.assertEqual(recipe["guide"]["id"], "animal")
        self.assertEqual(recipe["guide"]["title"], "Wildlife")
        self.assertIn("wildlife", recipe["guide"]["types"])
        self.assertTrue(recipe["separation"]["applied"])
        self.assertGreaterEqual(recipe["separation"]["eyes"], 1)
        self.assertEqual(recipe["separation"]["background"], "defocus")
        self.assertIn("subject", recipe["guide"]["stages_ran"])
        self.assertIn("background", recipe["guide"]["stages_ran"])
        before = _texture(pixels[16:90, 16:120])
        after = _texture(rendered[16:90, 16:120])
        head = rendered[170:220, 360:430]
        self.assertLess(after, before * 0.45)
        self.assertGreater(float(head[..., 0].mean()), float(head[..., 1].mean()) + 0.15)
        self.assertIn("animal", recipe["features"])
        self.assertNotIn("perch", recipe["features"])

    def test_a_subject_that_is_not_red_or_yellow_is_still_an_animal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "blue.png"
            _save_array(_blue_subject(), path)
            recipe = analyze(path, detector="heuristic", advisor="histogram")
        self.assertEqual(recipe["picture"]["type"], "animal")
        self.assertEqual(recipe["guide"]["id"], "animal")
        self.assertTrue(recipe["separation"]["applied"])
        self.assertEqual(recipe["separation"]["background"], "defocus")

    def test_a_landscape_is_not_blurred_like_a_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "land.png"
            _save_array(_landscape(), path)
            recipe = analyze(path, detector="heuristic", advisor="histogram")
        self.assertEqual(recipe["picture"]["type"], "landscape")
        self.assertEqual(recipe["guide"]["id"], "landscape")
        self.assertFalse(recipe["separation"]["applied"])
        self.assertEqual(recipe["separation"]["background"], "unchanged")
        self.assertNotIn("background", recipe["guide"]["stages_ran"])
        self.assertIn("sky", recipe["guide"]["stages_ran"])
        self.assertEqual(recipe["features"], ["land", "sky"])
        self.assertNotIn("perch", recipe["features"])
        self.assertNotIn("horizon", recipe["guide"]["stages_ran"])
        self.assertNotIn("reflection", recipe["guide"]["stages_ran"])

    def test_a_green_field_with_a_warm_horizon_is_not_a_face(self) -> None:
        image = Image.new("RGB", (640, 420), (78, 148, 46))
        draw = ImageDraw.Draw(image)
        for y in range(90, 400, 16):
            draw.line((0, y, 640, y), fill=(52, 112, 32), width=3)
        draw.rectangle((0, 28, 639, 108), fill=(186, 154, 96))
        draw.ellipse((36, 52, 78, 78), fill=(36, 24, 16))
        draw.ellipse((548, 58, 590, 84), fill=(28, 20, 14))
        draw.ellipse((300, 160, 348, 214), fill=(24, 78, 28))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "fields.png"
            image.save(path)
            recipe = analyze(path, detector="heuristic", advisor="histogram")
        categories = {item["category"] for item in recipe["analysis"]["subjects"]}
        self.assertEqual(recipe["picture"]["type"], "landscape")
        self.assertEqual(recipe["guide"]["id"], "landscape")
        self.assertNotIn("face", categories)
        self.assertNotIn("eye", categories)
        self.assertFalse(recipe["separation"]["applied"])
        self.assertEqual(recipe["separation"]["background"], "unchanged")
        self.assertNotIn("subject", recipe["guide"]["stages_ran"])
        self.assertNotIn("eyes", recipe["guide"]["stages_ran"])
        self.assertNotIn("reflection", recipe["guide"]["stages_ran"])

    def test_a_bird_in_water_is_lifted_with_its_eye_and_reflection(self) -> None:
        pixels = _heron_on_water()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heron.png"
            _save_array(pixels, path)
            recipe = analyze(path, detector="heuristic", advisor="histogram")
            rendered = np.array(recipe["_rendered"])
        categories = {item["category"] for item in recipe["analysis"]["subjects"]}
        self.assertEqual(recipe["picture"]["type"], "animal")
        self.assertNotIn("face", categories)
        self.assertIn("animal", categories)
        self.assertIn("subject", recipe["guide"]["stages_ran"])
        self.assertIn("eyes", recipe["guide"]["stages_ran"])
        self.assertIn("reflection", recipe["guide"]["stages_ran"])
        vignette = next(item for item in recipe["modules"] if item["operation"] == "vignette")
        self.assertEqual(vignette["multi_name"], "AI animal")
        self.assertLess(vignette["params"]["center_x"], 0.15)
        bird = _luma_mean(rendered[200:330, 240:430]) - _luma_mean(pixels[200:330, 240:430])
        flowers = _luma_mean(rendered[180:260, 530:690]) - _luma_mean(pixels[180:260, 530:690])
        mirror = _luma_mean(rendered[360:430, 310:360]) - _luma_mean(pixels[360:430, 310:360])
        beside = _luma_mean(rendered[360:430, 180:230]) - _luma_mean(pixels[360:430, 180:230])
        self.assertGreater(bird, flowers + 0.02)
        self.assertGreater(mirror, beside + 0.01)
        self.assertIn("Lifted together", recipe["reasoning"]["description"])

    def test_a_mountain_or_house_reflection_is_sharpened_without_becoming_an_animal(self) -> None:
        mountain = _mountain_over_water()
        house = _house_over_water()
        self.assertIsNone(reflected_subject(mountain))
        self.assertIsNone(reflected_subject(house))
        self.assertIsNotNone(water_reflection(mountain))
        self.assertIsNotNone(water_reflection(house))
        for pixels, kind, picture_type in (
            (mountain, "mountain", "landscape"),
            (house, "house", "architecture"),
        ):
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / f"{kind}.png"
                    _save_array(pixels, path)
                    recipe = analyze(path, detector="heuristic", advisor="histogram")
                    rendered = np.array(recipe["_rendered"])
                self.assertEqual(recipe["picture"]["type"], picture_type)
                self.assertNotEqual(recipe["picture"]["type"], "animal")
                categories = {item["category"] for item in recipe["analysis"]["subjects"]}
                self.assertNotIn("animal", categories)
                self.assertFalse(recipe["separation"]["applied"])
                self.assertEqual(recipe["separation"]["background"], "unchanged")
                self.assertNotIn("background", recipe["guide"]["stages_ran"])
                self.assertNotIn("subject", recipe["guide"]["stages_ran"])
                self.assertIn("reflection", recipe["guide"]["stages_ran"])
                mirror = rendered[260:340, 180:420]
                beside = rendered[260:340, 580:700]
                source_mirror = pixels[260:340, 180:420]
                source_beside = pixels[260:340, 580:700]
                mirror_gain = _texture(mirror) - _texture(source_mirror)
                beside_gain = _texture(beside) - _texture(source_beside)
                self.assertGreater(mirror_gain, beside_gain + 0.008)
                land = rendered[140:210, 180:420]
                self.assertGreater(_texture(land), _texture(pixels[140:210, 180:420]) * 0.75)

    def test_a_sunset_line_is_warmed_without_recoloring_the_land(self) -> None:
        source = _sunset()
        picture = decide_picture(source, [])
        self.assertEqual(picture.type, "landscape")
        rendered, info = render_finish(source, Atmosphere(), [], picture=picture)
        self.assertIn("horizon", info["stages_ran"])
        line = rendered[185:220].mean(axis=(0, 1))
        land = rendered[360:430].mean(axis=(0, 1))
        source_line = source[185:220].mean(axis=(0, 1))
        source_land = source[360:430].mean(axis=(0, 1))
        line_warmth = float((line[0] - line[2]) - (source_line[0] - source_line[2]))
        land_warmth = float((land[0] - land[2]) - (source_land[0] - source_land[2]))
        self.assertGreater(line_warmth, 0.03)
        self.assertLess(land_warmth, line_warmth * 0.45)

    def test_two_faces_are_a_group(self) -> None:
        pixels = np.full((120, 160, 3), 0.42, dtype=np.float32)
        detections = [
            Detection("face", "face", 0.8, (0.08, 0.12, 0.34, 0.6), "test"),
            Detection("face", "face", 0.8, (0.55, 0.14, 0.32, 0.58), "test"),
        ]
        picture = decide_picture(pixels, detections)
        self.assertEqual(picture.type, "group")
        match = match_guide(picture.type)
        self.assertEqual(match.guide.id, "group")

    def test_the_command_line_names_the_type_and_the_guide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "face.png"
            sample_image("portrait").save(path)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(
                    [
                        str(path),
                        "--output-dir",
                        str(directory),
                        "--detector",
                        "heuristic",
                        "--advisor",
                        "histogram",
                    ]
                )
        self.assertEqual(code, 0)
        first = buffer.getvalue().splitlines()[0]
        self.assertEqual(first, "portrait — Portrait guide, kept person")


def _bird_on_branch() -> np.ndarray:
    """A colored bird on a thick brown branch, against fine green needles.

    This is a synthetic stand-in. The user's photograph is not stored here.
    """

    rng = np.random.default_rng(9)
    height, width = 480, 720
    noise = rng.random((height, width, 3), dtype=np.float32)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = 0.08 + 0.14 * noise[..., 0]
    image[..., 1] = 0.34 + 0.42 * noise[..., 1]
    image[..., 2] = 0.06 + 0.10 * noise[..., 2]
    image[:, ::3, 1] *= 0.55
    image[::2, :, 1] *= 0.72
    grain = rng.random((80, width), dtype=np.float32)
    branch = np.zeros((80, width, 3), dtype=np.float32)
    branch[..., 0] = 0.40 + 0.10 * grain
    branch[..., 1] = 0.26 + 0.06 * grain
    branch[..., 2] = 0.13 + 0.04 * grain
    branch[:, ::11, :] *= 0.72
    image[250:330] = branch
    # A thin twig, disconnected from the perch, should fall soft with the needles.
    image[36:40, 80:420] = (0.36, 0.22, 0.12)
    image[36:40, 80:420:6] *= 0.7
    image[168:272, 300:460] = (0.66, 0.72, 0.80)
    image[150:196, 330:440] = (0.06, 0.06, 0.08)
    image[248:470, 404:438] = (0.42, 0.60, 0.86)
    tail = image[340:460, 406:436]
    image[340:460, 406:436] = np.clip(tail + (rng.random(tail.shape, dtype=np.float32) - 0.5) * 0.08, 0.0, 1.0)
    body = image[200:250, 320:430]
    image[200:250, 320:430] = np.clip(body + (rng.random(body.shape, dtype=np.float32) - 0.5) * 0.05, 0.0, 1.0)
    return np.clip(image, 0.0, 1.0)


def _person_on_chair() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    rng = np.random.default_rng(3)
    height, width = 480, 720
    noise = rng.random((height, width, 3), dtype=np.float32)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = 0.10 + 0.12 * noise[..., 0]
    image[..., 1] = 0.36 + 0.36 * noise[..., 1]
    image[..., 2] = 0.08 + 0.08 * noise[..., 2]
    image[:, ::4, 1] *= 0.6
    grain = rng.random((180, 340), dtype=np.float32)
    chair = np.zeros((180, 340, 3), dtype=np.float32)
    chair[..., 0] = 0.42 + 0.08 * grain
    chair[..., 1] = 0.26 + 0.05 * grain
    chair[..., 2] = 0.14 + 0.03 * grain
    chair[:, ::9, :] *= 0.75
    image[220:400, 190:530] = chair
    image[60:250, 280:460] = (0.80, 0.58, 0.46)
    image[100:132, 320:360] = (0.08, 0.06, 0.05)
    image[100:132, 390:430] = (0.08, 0.06, 0.05)
    return image, (280 / width, 60 / height, 180 / width, 190 / height)


class KeptFeatureTests(unittest.TestCase):
    def test_the_animal_guide_names_the_perch_in_plain_language(self) -> None:
        match = match_guide("animal")
        guide = match.guide
        self.assertEqual(list(guide.keep), ["animal", "perch"])
        self.assertEqual(list(guide.quiet), ["background"])
        self.assertIn("branch", guide.features["perch"].lower())
        self.assertIn("soft", guide.features["background"].lower())
        portrait = match_guide("portrait").guide
        self.assertEqual(list(portrait.keep), ["person", "held", "seat"])
        self.assertIn("chair", portrait.features["seat"].lower())
        architecture = match_guide("architecture").guide
        self.assertEqual(list(architecture.keep), ["building"])
        self.assertEqual(list(architecture.quiet), ["sky"])
        landscape = match_guide("landscape").guide
        self.assertEqual(list(landscape.keep), ["land", "sky"])
        self.assertEqual(list(landscape.quiet), [])

    def test_a_bird_and_its_branch_stay_sharp_and_the_needles_do_not(self) -> None:
        pixels = _bird_on_branch()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "perch.png"
            _save_array(pixels, path)
            recipe = analyze(path, detector="heuristic", advisor="histogram", vibe="crisp building")
            rendered = np.array(recipe["_rendered"])
            write_outputs(recipe, directory)
            payload = json.loads((directory / "output.json").read_text(encoding="utf-8"))
        self.assertEqual(recipe["picture"]["type"], "animal")
        self.assertEqual(recipe["guide"]["id"], "animal")
        self.assertEqual(recipe["features"], ["animal", "perch"])
        self.assertEqual(payload["features"], ["animal", "perch"])
        self.assertEqual(recipe["separation"]["background"], "defocus")
        branch = rendered[258:322, 30:180]
        needles = rendered[48:110, 30:180]
        bird = rendered[205:245, 330:420]
        tail = rendered[360:440, 408:434]
        twig = rendered[30:48, 100:300]
        self.assertGreater(_texture(branch), _texture(needles) * 1.8)
        self.assertGreater(_texture(branch), _texture(pixels[258:322, 30:180]) * 0.45)
        self.assertLess(_texture(needles), _texture(pixels[48:110, 30:180]) * 0.5)
        self.assertGreater(_texture(bird), _texture(needles))
        self.assertGreater(float(bird[..., 2].mean()), float(bird[..., 0].mean()))
        self.assertGreater(float(bird[..., 2].mean()), 0.55)
        self.assertGreater(float(tail[..., 2].mean()), float(tail[..., 1].mean()))
        self.assertGreater(_texture(tail), _texture(needles))
        self.assertLess(_texture(twig), _texture(branch) * 0.55)

    def test_a_chair_under_a_person_stays_sharp(self) -> None:
        pixels, face = _person_on_chair()
        picture = Picture(
            type="portrait",
            reason="A face is the clear subject, so this is a portrait.",
            brightness=0.4,
            warmth=0.05,
            saturation=0.2,
            edges=0.1,
            line_fraction=0.02,
            sky=0.0,
            skin=0.2,
            highlight_specks=0.0,
            compact_area=0.12,
            compact_fill=0.7,
            subject_box=face,
        )
        finished, info = render_finish(
            pixels,
            Atmosphere(),
            [Detection("face", "face", 0.9, face, "test")],
            guide=match_guide("portrait").guide,
            picture=picture,
        )
        self.assertEqual(info["features"], ["person", "seat"])
        self.assertNotIn("perch", info["features"])
        chair = finished[300:380, 210:500]
        needles = finished[20:80, 20:160]
        self.assertGreater(_texture(chair), _texture(needles) * 1.6)
        self.assertGreater(float(finished[120:200, 340:420, 0].mean()), 0.45)


def _flowers_beside_a_bird() -> np.ndarray:
    height, width = 480, 720
    image = np.full((height, width, 3), (0.20, 0.46, 0.18), dtype=np.float32)
    image[160:310, 40:170] = (0.12, 0.28, 0.62)
    image[185:215, 90:125] = (0.03, 0.03, 0.04)
    image[40:450, 220:700] = (0.90, 0.55, 0.42)
    vein = np.array([0.50, 0.28, 0.20], dtype=np.float32)
    image[40:450, 220:700:4] = vein
    image[40:450:4, 220:700] = vein
    return image


def _open_field() -> Image.Image:
    image = Image.new("RGB", (640, 420), (78, 148, 46))
    draw = ImageDraw.Draw(image)
    for y in range(90, 400, 16):
        draw.line((0, y, 640, y), fill=(52, 112, 32), width=3)
    draw.rectangle((0, 28, 639, 108), fill=(186, 154, 96))
    return image


def _warm_facade() -> np.ndarray:
    facade = np.full((480, 640, 3), (0.55, 0.72, 0.86), dtype=np.float32)
    facade[30:460, 20:420] = (0.84, 0.58, 0.40)
    mortar = np.array([0.48, 0.28, 0.20], dtype=np.float32)
    facade[30:460, 20:420:4] = mortar
    facade[30:460:4, 20:420] = mortar
    return facade


def _skin_head() -> Image.Image:
    image = Image.new("RGB", (800, 600), (28, 28, 32))
    draw = ImageDraw.Draw(image)
    draw.ellipse((250, 80, 560, 480), fill=(214, 162, 127))
    draw.ellipse((330, 200, 370, 240), fill=(25, 18, 18))
    draw.ellipse((450, 200, 490, 240), fill=(25, 18, 18))
    return image


class FaceGateTests(unittest.TestCase):
    def _modules(self, pixels: np.ndarray | Image.Image) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            if isinstance(pixels, Image.Image):
                pixels.save(path)
            else:
                _save_array(pixels, path)
            return analyze(path, detector="heuristic", advisor="histogram")

    def test_warm_flowers_beside_a_bird_are_not_a_face(self) -> None:
        recipe = self._modules(_flowers_beside_a_bird())
        names = [item["multi_name"] for item in recipe["modules"]]
        self.assertNotIn("AI face", names)
        vignettes = [item for item in recipe["modules"] if item["operation"] == "vignette"]
        if vignettes:
            self.assertEqual(vignettes[0]["multi_name"], "AI animal")
            self.assertLess(vignettes[0]["params"]["center_x"], -0.2)

    def test_an_open_field_has_no_vignette(self) -> None:
        recipe = self._modules(_open_field())
        names = [item["multi_name"] for item in recipe["modules"]]
        self.assertNotIn("AI face", names)
        self.assertFalse(any(item["operation"] == "vignette" for item in recipe["modules"]))
        self.assertEqual(recipe["picture"]["measurements"]["compact_area"], 0.0)

    def test_a_warm_gridded_facade_is_not_a_face(self) -> None:
        recipe = self._modules(_warm_facade())
        names = [item["multi_name"] for item in recipe["modules"]]
        categories = {item["category"] for item in recipe["analysis"]["subjects"]}
        self.assertNotIn("AI face", names)
        self.assertNotIn("face", categories)

    def test_a_dry_compact_body_is_an_animal(self) -> None:
        image = np.full((480, 720, 3), (0.24, 0.40, 0.18), dtype=np.float32)
        image[170:360, 270:450] = (0.40, 0.36, 0.16)
        recipe = self._modules(image)
        categories = {item["category"] for item in recipe["analysis"]["subjects"]}
        self.assertEqual(recipe["picture"]["type"], "animal")
        self.assertIn("animal", categories)
        self.assertNotIn("AI face", [item["multi_name"] for item in recipe["modules"]])

    def test_a_head_with_two_eyes_is_still_a_face(self) -> None:
        recipe = self._modules(_skin_head())
        names = [item["multi_name"] for item in recipe["modules"]]
        categories = {item["category"] for item in recipe["analysis"]["subjects"]}
        self.assertIn("AI face", names)
        self.assertIn("face", categories)
        self.assertIn("eye", categories)

    def test_a_person_is_not_promoted_by_a_nearby_skin_blob(self) -> None:
        pixels = _flowers_beside_a_bird()
        image = Image.fromarray(np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
        from ai_dt.detectors.heuristic import HeuristicDetector

        found = HeuristicDetector().detect(image)
        person = Detection("person", "person", 0.9, (0.02, 0.25, 0.16, 0.5), "yolo")
        gated = gate_faces(pixels, found + [person])
        self.assertTrue(any(item.category == "person" and item.source == "yolo" for item in gated))
        self.assertFalse(any(item.category == "face" for item in gated))
        self.assertEqual(person.bbox, next(item.bbox for item in gated if item.category == "person"))

    def test_a_mediapipe_face_stays(self) -> None:
        face = Detection("face", "face", 0.92, (0.3, 0.2, 0.22, 0.4), "mediapipe")
        pixels = np.full((80, 80, 3), 0.4, dtype=np.float32)
        gated = gate_faces(pixels, [face])
        self.assertEqual([item.source for item in gated], ["mediapipe"])


class LookbackTests(unittest.TestCase):
    def test_an_eye_hotter_than_the_face_stops_the_stage(self) -> None:
        pixels = np.full((160, 160, 3), 0.18, dtype=np.float32)
        pixels[30:140, 40:120] = (0.55, 0.40, 0.32)
        pixels[50:70, 55:75] = (0.92, 0.90, 0.88)
        pixels[50:70, 90:110] = (0.92, 0.90, 0.88)
        face = (40 / 160, 30 / 160, 80 / 160, 110 / 160)
        detections = [
            Detection("face", "face", 0.9, face, "test"),
            Detection("eye", "eye", 0.8, (55 / 160, 50 / 160, 20 / 160, 20 / 160), "test"),
            Detection("eye", "eye", 0.8, (90 / 160, 50 / 160, 20 / 160, 20 / 160), "test"),
        ]
        picture = Picture(
            type="portrait",
            reason="A face is the clear subject, so this is a portrait.",
            brightness=0.4,
            warmth=0.08,
            saturation=0.1,
            edges=0.04,
            line_fraction=0.01,
            sky=0.0,
            skin=0.4,
            highlight_specks=0.0,
            compact_area=0.2,
            compact_fill=0.8,
            subject_box=face,
        )
        _rendered, info = render_finish(
            pixels,
            Atmosphere(),
            detections,
            guide=match_guide("portrait").guide,
            picture=picture,
            stages=["grade", "subject", "eyes", "background"],
        )
        self.assertIn("eye", info["stop_reason"].lower())
        self.assertNotIn("eyes", info["stages_ran"])
        self.assertNotIn("background", info["stages_ran"])
        self.assertGreaterEqual(len(info["measurements"]), 2)
        self.assertTrue(any(item["stage"] == "eyes" and item["kept"] is False for item in info["measurements"]))

    def test_a_weak_reflection_is_held(self) -> None:
        pixels = _heron_on_water()
        pixels = pixels.copy()
        pixels[340:450, 300:370] = (0.08, 0.085, 0.09)
        picture = decide_picture(pixels, [])
        detections = []
        if picture.subject_box is not None:
            detections = [Detection("animal", "animal", 0.8, picture.subject_box, "test")]
        _rendered, info = render_finish(pixels, Atmosphere(), detections, picture=picture)
        self.assertEqual(info["reflection"], "hold")
        self.assertIn("reflection", info["stages_ran"])
        mirror = _luma_mean(_rendered[360:430, 310:360]) - _luma_mean(pixels[360:430, 310:360])
        self.assertLess(abs(mirror), 0.02)


class RegionTests(unittest.TestCase):
    def test_a_bright_window_is_protected_from_the_surround(self) -> None:
        pixels = np.full((200, 280, 3), (0.28, 0.40, 0.24), dtype=np.float32)
        pixels[40:160, 24:100] = (0.12, 0.32, 0.70)
        pixels[16:184, 140:260] = (0.72, 0.70, 0.62)
        pixels[28:68, 168:214] = (0.97, 0.97, 0.98)
        picture = decide_picture(pixels, [])
        rendered, info = render_finish(pixels, Atmosphere(), [], picture=picture)
        self.assertEqual(info["theme"], "window")
        window = _luma_mean(rendered[28:68, 168:214]) - _luma_mean(pixels[28:68, 168:214])
        ground = _luma_mean(rendered[90:160, 220:255]) - _luma_mean(pixels[90:160, 220:255])
        self.assertGreater(window, ground)


if __name__ == "__main__":
    unittest.main()
