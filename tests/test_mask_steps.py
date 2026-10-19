"""Drawn-mask Accept steps: segmentation -> shapes -> one sidecar module.

No darktable, model, or key is needed. The preview is a synthetic scene,
the segmenter is the heuristic backend, and the sidecar is the small test
document from ``test_drawn_mask``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dt import dt_shapes as shapes
from ai_dt import mask_steps
from ai_dt.catalog import KNOWN_IDS, catalog_text, validate_operations
from ai_dt.drawn_mask import decode_blend, main as drawn_main, read_mask_records
from ai_dt.features import commit_waiting_step, history_steps, plan_chat
from ai_dt.guides.library import BUILDS, load_guides
from ai_dt.mapping import step_history_name
from ai_dt.mask_grammar import mask_operations
from ai_dt.revert import accepted_step_names, is_ai_editor_history_name, strip_ai_history
from tests.synthetic_scenes import landscape, subject_on_foliage
from tests.test_drawn_mask import _document

_ENV = {"AI_DT_SEGMENTER": "heuristic", "AI_DT_SEGMENT_GEMINI": ""}


def _lua_names() -> set[str]:
    text = (ROOT / "ai_editor.lua").read_text(encoding="utf-8")
    block = text.split("local editor_history_names = {", 1)[1].split("}", 1)[0]
    return set(re.findall(r'"([^"]+)"', block))


def _work(directory: Path, step_id: str, *, image=None, subjects=None, action="") -> Path:
    (image or landscape()).save(directory / "preview.jpg", quality=95)
    recipe = {
        "features_next": step_id,
        "features_next_title": mask_steps.STEPS[step_id].title if step_id in mask_steps.STEPS else step_id,
        "features_next_action": action,
        "analysis": {"subjects": subjects or []},
        "plan": {
            "step_queue": [{"id": step_id, "action": action or "lift", "title": "x"}, {"id": "mask_ground", "action": "darken", "title": "y"}],
            "features_done": [],
        },
        "segmenter": "heuristic",
    }
    (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
    return directory / "output.json"


class NamesTests(unittest.TestCase):
    def test_step_names_are_accepted_by_python_and_lua(self) -> None:
        names = mask_steps.step_names()
        self.assertTrue(names <= accepted_step_names())
        self.assertEqual(_lua_names(), set(accepted_step_names()))
        for name in names:
            self.assertTrue(is_ai_editor_history_name(name))

    def test_history_name_and_catalog(self) -> None:
        for ident, step in mask_steps.STEPS.items():
            self.assertEqual(step_history_name(ident), step.name)
            self.assertIn(ident, KNOWN_IDS)
            self.assertIn(ident, catalog_text())
        kept, dropped = validate_operations({"operations": [{"id": "mask_landscape"}, {"id": "mask_unicorn"}]})
        self.assertEqual([item["id"] for item in kept], ["mask_landscape"])
        self.assertEqual(dropped, ["mask_unicorn"])

    def test_guides_accept_the_new_builds(self) -> None:
        for build in ("people", "architecture", "vegetation", "water", "ground"):
            self.assertIn(build, BUILDS)
        loaded, problems = load_guides()
        self.assertEqual(problems, [])
        guides = {guide.id: guide for guide in loaded}
        self.assertIn("vegetation", guides["landscape"].stages)
        self.assertIn("water", guides["landscape"].masks)
        self.assertIn("architecture", guides["architecture"].stages)


class GradeTests(unittest.TestCase):
    def test_caps_hold_for_every_step(self) -> None:
        for step in mask_steps.STEPS.values():
            for action in ("lift", "darken", ""):
                grade = mask_steps.capped_grade(step, action)
                self.assertLessEqual(abs(grade.get("exposure", 0.0)), mask_steps.LOCAL_EV_CAP)
                self.assertLessEqual(abs(grade.get("brilliance_global", 0.0)), mask_steps.LOCAL_BRILLIANCE_CAP)
                params, version = mask_steps.module_blob(step, grade)
                self.assertEqual(len(params), 24 if step.operation == "exposure" else 132)
                self.assertGreater(version, 0)

    def test_example_seven_directions(self) -> None:
        sky = mask_steps.capped_grade(mask_steps.STEPS["mask_sky"], "")
        self.assertLess(sky["brilliance_global"], 0.0)  # darker
        self.assertGreater(sky["highlights_C"], 0.0)  # warmer
        self.assertGreater(mask_steps.capped_grade(mask_steps.STEPS["mask_architecture"], "")["brilliance_global"], 0.0)
        self.assertGreater(mask_steps.capped_grade(mask_steps.STEPS["mask_vegetation"], "")["saturation_global"], 0.0)
        water = mask_steps.capped_grade(mask_steps.STEPS["mask_water"], "")
        self.assertLess(water["brilliance_global"], 0.0)
        self.assertLess(water["contrast"], 0.0)  # softer
        self.assertLess(mask_steps.capped_grade(mask_steps.STEPS["mask_ground"], "")["brilliance_global"], 0.0)

    def test_spec_for_an_empty_mask_is_none(self) -> None:
        self.assertIsNone(mask_steps.build_spec("mask_sky", np.zeros((40, 40), dtype=np.float32)))


class GrammarTests(unittest.TestCase):
    def test_regions_and_directions(self) -> None:
        def ids(sentence):
            return [(item["id"], item["action"]) for item in mask_operations(sentence)]

        self.assertEqual(ids("darken the sky"), [("mask_sky", "darken")])
        self.assertEqual(ids("brighten the sky"), [("mask_sky", "lift")])
        self.assertEqual(ids("make the trees greener, brighten the buildings"),
                         [("mask_vegetation", "lift"), ("mask_architecture", "lift")])
        self.assertEqual(ids("make the water bluer"), [("mask_water", "darken")])
        self.assertEqual(ids("do the landscape masks"), [("mask_landscape", "include")])
        self.assertEqual(ids("lift the people"), [("mask_people", "lift")])
        self.assertEqual(ids("darken the background"), [("mask_background", "darken")])

    def test_descriptions_and_taken_phrases_are_not_requests(self) -> None:
        self.assertEqual(mask_operations("a boat on the water"), [])
        self.assertEqual(mask_operations("a little less on the background"), [])
        self.assertEqual(mask_operations("blur the background"), [])
        self.assertEqual(mask_operations("quiet the path", claimed="quiet the path"), [])

    def test_landscape_expands_in_lesson_order(self) -> None:
        found, _held = history_steps([{"id": "mask_landscape", "action": "include", "reason": "five masks"}])
        self.assertEqual([item["id"] for item in found],
                         ["mask_sky", "mask_architecture", "mask_vegetation", "mask_water", "mask_ground"])
        self.assertEqual(found[0]["action"], "darken")


class AcceptTests(unittest.TestCase):
    def test_accepting_the_sky_writes_a_shapes_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            _work(directory, "mask_sky")
            recipe = commit_waiting_step(directory)
            mask = recipe["step_mask"]
            self.assertEqual(mask["kind"], "shapes")
            self.assertEqual(mask["name"], "Grade the sky")
            self.assertEqual(mask["source"], "heuristic")
            self.assertEqual(recipe["features_next"], "mask_ground")
            self.assertNotIn("step_style", recipe)
            spec = json.loads(Path(mask["file"]).read_text(encoding="utf-8"))
            self.assertEqual(spec["operation"], "colorbalancergb")
            rendered = shapes.render_forms(spec["forms"], 360, 240)
            self.assertGreater(float((rendered[:80] >= 0.5).mean()), 0.9)
            self.assertLess(float((rendered[110:] >= 0.5).mean()), 0.02)
            # Lua then runs drawn_mask.py with that file.
            source = directory / "photo.xmp"
            source.write_text(_document(), encoding="utf-8")
            out = directory / "masked.xmp"
            self.assertEqual(drawn_main(["--xmp", str(source), "--xmp-out", str(out), "--mask-json", mask["file"]]), 0)
            written = out.read_text(encoding="utf-8")
            self.assertIn('darktable:multi_name="Grade the sky"', written)
            group = [r for r in read_mask_records(written) if r["type"] == shapes.GROUP]
            self.assertEqual(len(group), 1)
            stripped, removed = strip_ai_history(written)
            self.assertEqual(removed, 1)
            self.assertNotIn("Grade the sky", stripped)
            log = recipe["plan"]["step_log"]
            self.assertEqual(log[-1]["name"], "Grade the sky")

    def test_background_is_the_inverted_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            image, box = subject_on_foliage()
            _work(directory, "mask_background", image=image,
                  subjects=[{"label": "bird", "category": "animal", "bbox": list(box)}])
            recipe = commit_waiting_step(directory)
            spec = json.loads(Path(recipe["step_mask"]["file"]).read_text(encoding="utf-8"))
            self.assertTrue(spec["invert"])
            source = directory / "photo.xmp"
            source.write_text(_document(), encoding="utf-8")
            out = directory / "masked.xmp"
            drawn_main(["--xmp", str(source), "--xmp-out", str(out), "--mask-json", recipe["step_mask"]["file"]])
            written = out.read_text(encoding="utf-8")
            start = written.find('darktable:multi_name="Quiet the background"')
            tag = written[written.rfind("<rdf:li", 0, start) : written.find("/>", start)]
            blend = decode_blend(tag.split('darktable:blendop_params="')[1].split('"')[0])
            self.assertTrue(blend["inverted"])

    def test_a_missing_region_is_left_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            image, _box = subject_on_foliage()
            _work(directory, "mask_water", image=image)
            recipe = commit_waiting_step(directory)
            self.assertNotIn("step_mask", recipe)
            self.assertIn("No water was found", recipe["feature_note"])
            self.assertIn("mask_water", recipe["plan"]["features_skipped"])
            self.assertEqual(recipe["features_next"], "mask_ground")

    def test_no_preview_is_left_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            _work(directory, "mask_sky")
            (directory / "preview.jpg").unlink()
            recipe = commit_waiting_step(directory)
            self.assertNotIn("step_mask", recipe)

    def test_ask_turns_a_sentence_into_mask_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            recipe = {
                "features_next": "",
                "reading": {"sentence": "A lake under a blue sky with pine trees."},
                "plan": {"features_done": [], "features_path": [], "reasoning": {}},
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            reply = {"ask_ollama": False, "proposal": None}
            with patch("ai_dt.reasoner.propose_gemini", return_value=reply):
                plan_chat(directory, "darken the sky, make the trees greener")
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["features_next"], "mask_sky")
            self.assertEqual(saved["features_next_action"], "darken")
            self.assertIn("mask_vegetation", saved["plan"]["features_path"])


class PaintedStageTests(unittest.TestCase):
    def test_region_stage_paints_only_its_region(self) -> None:
        from ai_dt.separate import _region_mask

        pixels = np.asarray(landscape(), dtype=np.float32) / 255.0
        with patch.dict(os.environ, _ENV):
            water = _region_mask(pixels, "water", [])
            people = _region_mask(pixels, "people", [])
        self.assertIsNotNone(water)
        self.assertGreater(float(water[180:205].mean()), 0.8)
        self.assertLess(float(water[:150].mean()), 0.05)
        self.assertIsNone(people)


if __name__ == "__main__":
    unittest.main()
