"""An animal photo is planned with the animal steps, never skin or person.

The duck test: a mallard on a bright concrete ledge by green water, request
"Make the duck the main theme". YOLO put a low "person" box on the duck's
head, the photo went to the portrait guide, and Gemini planned subject, skin,
eyes, background. Mistral and Gemini are mocked; no network, no keys.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from ai_dt import catalog
from ai_dt.animal_plan import (
    animal_noun,
    animal_plan,
    bright_ground,
    consistent_regions,
    generic_species,
    retype_from_text,
    subject_is_animal,
)
from ai_dt.cli import build_parser
from ai_dt.detectors.heuristic import drop_animal_heads
from ai_dt.features import plan_chat
from ai_dt.guides.decide import decide_picture
from ai_dt.schema import Detection
from ai_dt.status_text import description_changed, plan_origin
from ai_dt.text_io import clean_text, read_text_arg, repair_mojibake

BIRD = Detection("bird", "animal", 0.9, (0.527, 0.335, 0.246, 0.617), "yolo")
HEAD = Detection("person", "person", 0.49, (0.6, 0.34, 0.08, 0.12), "yolo")
PERSON_STEPS = {"skin", "face", "person", "mask_people", "eyes"}

READING = "A male Common Eider duck\u2019s plumage stands on a bright concrete ledge in front of green rippled water."


def _gemini(*ids: str, paragraph: str = "") -> dict:
    return {
        "ask_ollama": False,
        "proposal": {
            "status": "gemini",
            "backend": "gemini-3.5-flash-lite",
            "paragraph": paragraph
            or "The duck is darker than the bright ground, so lifting the subject and darkening the background "
            "makes the eider the main theme as requested, keeping the water and ledge untouched.",
            "stages": [{"stage": ident, "action": "lift", "reason": ident} for ident in ids],
            "operations": [
                {"id": ident, "action": "include", "reason": ident, "title": "Make the duck the main theme"}
                for ident in ids
            ],
            "features": [],
        },
    }


def _duck_recipe(**plan_extra) -> dict:
    plan = {
        "guide_stages": ["grade", "subject", "skin", "eyes", "background"],
        "skip": [],
        "features_done": [],
        "features_skipped": [],
        "features_path": [],
        "step_log": [],
        "reasoning": {},
        "asks": [],
    }
    plan.update(plan_extra)
    return {
        "features_next": "",
        "picture": {"type": "portrait"},
        "reading": {"sentence": "A duck on a ledge."},
        "scene": {"summary": READING, "source": "mistral"},
        "analysis": {"atmosphere": {}, "subjects": [BIRD.to_dict()]},
        "separation": {"stages_ran": ["grade"]},
        "plan": plan,
    }


class DetectionRoutingTests(unittest.TestCase):
    def test_a_person_box_on_the_ducks_head_is_dropped(self) -> None:
        kept = drop_animal_heads([BIRD, HEAD])
        self.assertEqual([item.category for item in kept], ["animal"])
        pixels = np.full((60, 80, 3), 0.4, dtype=np.float32)
        self.assertEqual(decide_picture(pixels, kept).type, "animal")

    def test_a_real_person_beside_an_animal_stays(self) -> None:
        dog = Detection("dog", "animal", 0.8, (0.1, 0.5, 0.2, 0.3), "yolo")
        owner = Detection("person", "person", 0.85, (0.5, 0.1, 0.3, 0.8), "yolo")
        self.assertEqual(len(drop_animal_heads([dog, owner])), 2)

    def test_an_edited_description_naming_a_duck_routes_to_the_animal_guide(self) -> None:
        pixels = np.full((60, 80, 3), 0.4, dtype=np.float32)
        portrait = decide_picture(pixels, [BIRD, HEAD])
        self.assertIn(portrait.type, {"portrait", "group", "street"})
        retyped, found, changed = retype_from_text(portrait, [BIRD, HEAD], "A male Common Eider duck on a ledge.")
        self.assertTrue(changed)
        self.assertEqual(retyped.type, "animal")
        self.assertNotIn("person", [item.category for item in found])
        # Only a person box: it becomes the animal the words name.
        _, only, _ = retype_from_text(portrait, [HEAD], "a duck on a ledge")
        self.assertEqual([(item.category, item.source) for item in only], [("animal", "reading")])

    def test_without_yolo_the_subject_box_becomes_the_named_animal(self) -> None:
        from dataclasses import replace

        blob = Detection("subject", "subject", 0.34, (0.33, 0.33, 0.5, 0.5), "heuristic")
        general = replace(decide_picture(np.full((60, 80, 3), 0.4, dtype=np.float32), []), type="general")
        retyped, found, changed = retype_from_text(general, [blob], READING)
        self.assertTrue(changed)
        self.assertEqual(retyped.type, "animal")
        self.assertIn(("animal", "reading"), [(item.category, item.source) for item in found])
        self.assertNotIn("eider", retyped.reason.lower())

    def test_a_person_in_the_words_keeps_the_portrait(self) -> None:
        pixels = np.full((60, 80, 3), 0.4, dtype=np.float32)
        portrait = decide_picture(pixels, [BIRD, HEAD])
        _, _, changed = retype_from_text(portrait, [BIRD, HEAD], "A woman feeding a duck.")
        self.assertFalse(changed)

    def test_stone_foxes_are_not_an_animal_subject(self) -> None:
        self.assertFalse(subject_is_animal("statue", [], "Two stone fox statues with red aprons."))
        self.assertTrue(subject_is_animal("landscape", [], "A duck on a pond."))


class AnimalPlanTests(unittest.TestCase):
    def test_subject_skin_eyes_background_become_animal_steps(self) -> None:
        plan = {
            "step_queue": [
                {"id": "subject", "title": "Make the duck the main theme"},
                {"id": "skin", "title": "Even the skin"},
                {"id": "eyes", "title": "Brighten the eyes"},
                {"id": "background", "title": "Darken everything around the duck"},
            ],
            "guide_stages": ["subject", "skin", "eyes", "background"],
            "reasoning": {"paragraph": "Darken the background keeping the water and ledge untouched; the eider pops."},
        }
        out = animal_plan(plan, steer="Make the duck the main theme", description=READING, ground=True, water=True)
        ids = out["features_path"]
        self.assertEqual(ids, ["mask_animal", "mask_animal_background", "mask_bottom_gradient", "mask_edge_burn"])
        self.assertFalse(PERSON_STEPS & set(ids))
        self.assertNotIn("skin", out["guide_stages"])
        titles = [item["title"] for item in out["step_queue"]]
        self.assertNotIn("Make the duck the main theme", titles)
        self.assertEqual(titles[0], "Lift the duck a little and crisp its feathers")
        paragraph = out["reasoning"]["paragraph"]
        self.assertNotIn("eider", paragraph.lower())
        self.assertNotIn("untouched", paragraph)
        self.assertIn("water", paragraph)

    def test_no_bright_ground_no_gradient(self) -> None:
        out = animal_plan({"step_queue": [{"id": "subject"}, {"id": "background"}]}, steer="", ground=False)
        self.assertEqual(out["features_path"], ["mask_animal", "mask_animal_background", "mask_edge_burn"])

    def test_two_animals_use_the_all_animals_lift(self) -> None:
        out = animal_plan({"step_queue": [{"id": "subject"}]}, steer="make them stand out", animals=2)
        self.assertEqual(out["features_next"], "mask_animals")

    def test_words(self) -> None:
        self.assertEqual(animal_noun("Make the duck the main theme"), "the duck")
        self.assertEqual(animal_noun("", "A male Common Eider on a ledge."), "the bird")
        self.assertEqual(animal_noun("", "A male Common Eider duck on a ledge."), "the duck")
        self.assertEqual(generic_species("the eider stands out", "the duck"), "the duck stands out")
        self.assertIn("eider", generic_species("the eider stands out", "the bird", "lift the eider"))
        fixed = consistent_regions("darken the background keeping the water and ledge untouched.", water=True, ground=True)
        self.assertNotIn("untouched", fixed)

    def test_bright_ground(self) -> None:
        pixels = np.full((100, 100, 3), 0.25, dtype=np.float32)
        self.assertFalse(bright_ground(pixels))
        pixels[85:] = 0.8
        self.assertTrue(bright_ground(pixels))

    def test_the_planner_prompt_lists_the_animal_steps(self) -> None:
        text = catalog.catalog_text()
        for name in ("mask_animal", "mask_animal_background", "mask_bottom_gradient", "mask_edge_burn", "mask_glow"):
            self.assertIn(name, text)
        self.assertIn("never", text.lower())


class AskTests(unittest.TestCase):
    def test_ask_with_a_portrait_style_gemini_plan_gives_animal_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "output.json").write_text(json.dumps(_duck_recipe(bright_ground=True)), encoding="utf-8")
            with patch("ai_dt.reasoner.propose_gemini", return_value=_gemini("subject", "skin", "eyes", "background")) as ask:
                plan_chat(directory, "Make the duck the main theme")
            brief = ask.call_args.kwargs.get("brief", "")
            self.assertIn("mask_animal", brief)
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
        path = saved["plan"]["features_path"]
        self.assertEqual(path, ["mask_animal", "mask_animal_background", "mask_bottom_gradient", "mask_edge_burn"])
        self.assertEqual(saved["features_next"], "mask_animal")
        self.assertNotIn("skin", saved["status_text"].split("Current plan")[0].split("Not on this photo")[0])
        self.assertNotIn("make the duck the main theme", saved["features_next_title"].lower())
        self.assertNotIn("Another lift", saved.get("feature_note", ""))
        self.assertNotIn("eider", json.dumps(saved["plan"]["reasoning"]).lower())
        self.assertIn("Current plan (from request \u201cMake the duck the main theme\u201d + the photo description)", saved["status_text"])


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in ("MISTRAL_API_KEY", "GEMINI_API_KEY", "OLLAMA_HOST")}
        os.environ["OLLAMA_HOST"] = "127.0.0.1:9"
        os.environ["GEMINI_API_KEY"] = ""
        os.environ["MISTRAL_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _image(self, directory: Path) -> Path:
        pixels = np.zeros((120, 160, 3), dtype=np.uint8)
        pixels[:] = (40, 90, 60)  # green water
        pixels[100:] = (215, 210, 200)  # bright ledge
        pixels[40:100, 84:124] = (90, 70, 50)  # the duck
        path = directory / "duck.png"
        Image.fromarray(pixels).save(path)
        return path

    def _mistral(self):
        payload = {
            "mood": "calm", "rationale": "ok", "exposure_ev": 0.0, "contrast": 0.0,
            "temperature_shift": 0.0, "tint": 0.0, "saturation": 0.0, "shadows": 0.0,
            "highlights": 0.0, "vibrance": 0.0,
            "description": READING, "summary": READING,
        }
        body = json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode("utf-8")

        class _Body:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return lambda request, timeout=None: _Body()

    def _run(self, directory: Path, **kwargs) -> dict:
        from ai_dt.pipeline import analyze

        with (
            patch("ai_dt.pipeline._detect", return_value=([BIRD, HEAD], [])),
            patch("ai_dt.advisors.mistral_vision.urllib.request.urlopen", self._mistral()),
            patch("ai_dt.pipeline.OllamaLLaVAAdvisor"),
            patch("ai_dt.reasoner.propose_gemini", return_value=_gemini("subject", "skin", "eyes", "background")),
        ):
            return analyze(self._image(directory), detector="heuristic", advisor="auto", output_dir=directory, **kwargs)

    def test_propose_on_the_duck_routes_to_animal_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recipe = self._run(Path(tmp), steer="Make the duck the main theme")
        self.assertEqual(recipe["picture"]["type"], "animal")
        path = recipe["plan"]["features_path"]
        self.assertEqual(path[0], "mask_animal")
        self.assertIn("mask_animal_background", path)
        self.assertIn("mask_edge_burn", path)
        self.assertFalse(PERSON_STEPS & set(path))
        self.assertNotIn("person", str(recipe.get("status_text", "")).split("Current plan")[0].lower().replace("no person", ""))
        self.assertNotIn("Another lift", recipe.get("status_text", ""))
        self.assertEqual(recipe["plan_request"], "Make the duck the main theme")

    def test_an_edited_description_is_a_new_plan_and_is_labelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = self._run(directory, steer="Make the duck the main theme")
            self.assertEqual(first["plan_description_source"], "mistral")
            edited = "A duck on a bright concrete ledge by green water."
            again = self._run(directory, scene_text=edited)
        self.assertEqual(again["plan_description"], edited)
        self.assertEqual(again["plan_description_source"], "user")
        self.assertEqual(again["picture"]["type"], "animal")
        self.assertIn("Current plan (from your edited description)", again["status_text"])
        self.assertNotIn("Make the duck the main theme", again["status_text"])


class LabelTests(unittest.TestCase):
    def test_origin_names_request_and_description(self) -> None:
        recipe = {"plan_request": "lift it", "plan_description": "A duck.", "plan_description_source": "user"}
        self.assertEqual(plan_origin(recipe), "request \u201clift it\u201d + your edited description")
        self.assertEqual(plan_origin({"plan_description": "A duck.", "plan_description_source": "mistral"}), "the photo description")
        self.assertTrue(description_changed(recipe, "A duck on a ledge."))
        self.assertFalse(description_changed(recipe, " a duck. "))
        self.assertFalse(description_changed({}, "anything"))


class EncodingTests(unittest.TestCase):
    def test_windows_mojibake_is_repaired(self) -> None:
        self.assertEqual(clean_text("the duck\u00d4\u00c7\u00d6s head"), "the duck's head")  # cp850 (Windows OEM console)
        self.assertEqual(clean_text("the duck\u00e2\u20ac\u2122s head"), "the duck's head")  # cp1252
        self.assertEqual(clean_text("the duck\u2019s \u201chead\u201d"), "the duck's \"head\"")
        self.assertEqual(repair_mojibake("caf\u00e9 au lait"), "caf\u00e9 au lait")

    def test_text_files_win_over_argv_and_are_read_as_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arg-description.txt"
            path.write_bytes("\ufeffthe duck\u2019s head".encode("utf-8"))
            self.assertEqual(read_text_arg("the duck\u00d4\u00c7\u00d6s head", str(path)), "the duck's head")
            self.assertEqual(read_text_arg("the duck\u00d4\u00c7\u00d6s head", None), "the duck's head")
            args = build_parser().parse_args(["x.jpg", "--output-dir", tmp, "--scene-file", str(path), "--steer-file", str(path)])
            self.assertEqual(args.scene_file, str(path))

    def test_lua_passes_text_through_utf8_files(self) -> None:
        lua = (ROOT / "ai_editor.lua").read_text(encoding="utf-8")
        for flag in ("arg-vibe.txt", "arg-request.txt", "arg-description.txt", "-file "):
            self.assertIn(flag, lua)


if __name__ == "__main__":
    unittest.main()
