"""Phase 3: what the photoserge lessons add (order, caps, lesson steps, looks, crop, prompts)."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ai_dt import dt_shapes as shapes
from ai_dt import lessons, mask_steps
from ai_dt.drawn_mask import decode_blend, main as drawn_main, read_mask_records
from ai_dt.features import commit_waiting_step, history_steps, plan_chat
from ai_dt.mapping import step_history_name, step_style
from ai_dt.mask_grammar import mask_operations
from ai_dt.modules_extra import colorzones_curve, colorzones_keep_params
from ai_dt.revert import accepted_step_names
from tests.synthetic_scenes import animals_in_field, landscape, subject_on_foliage
from tests.test_drawn_mask import _document
from tests.test_mask_steps import _ENV, _lua_names, _work


def _curve_at(nodes, position):
    xs = [x for x, _ in nodes]
    ys = [y for _, y in nodes]
    return float(np.interp(position, xs, ys))


def _style_blob(xml: str) -> bytes:
    return bytes.fromhex(xml.split("<op_params>")[1].split("</op_params>")[0])


class DataTests(unittest.TestCase):
    def test_the_lesson_data_is_trimmed_and_complete(self):
        data = json.loads(lessons.DATA.read_text(encoding="utf-8"))
        self.assertEqual(len(data["examples"]), 11)
        for example in data["examples"]:
            self.assertFalse({"before_image", "after_image", "compare_image", "extra_images"} & set(example))
            self.assertTrue(example["edits"])
        self.assertLess(lessons.DATA.stat().st_size, 40_000)

    def test_few_shot_text_fits_the_picture_and_keeps_the_caps(self):
        text = lessons.few_shot_text("animal")
        self.assertIn("mask_animal", text)
        self.assertIn("half a stop", text)
        self.assertIn("+-0.5 EV", text)
        self.assertNotIn("images", text)
        self.assertIn("Tuscany", text)
        self.assertIn("AI landscape masks", lessons.few_shot_text("landscape"))
        self.assertTrue(lessons.few_shot_text("unknown-type"))

    def test_guess_type(self):
        self.assertEqual(lessons.guess_type("A heron standing in a pond"), "general")
        self.assertEqual(lessons.guess_type("two birds on a fence"), "animal")
        self.assertEqual(lessons.guess_type("sunset over the valley"), "landscape")
        self.assertEqual(lessons.guess_type("", "", "neon street at night"), "night")

    def test_the_planner_prompt_carries_the_lessons(self):
        from ai_dt.reasoner import _fill

        text = _fill("A dog running on a beach", "faithful", "none")
        self.assertIn("Editing lessons", text)
        self.assertIn("mask_animal", text)
        self.assertNotIn("images", text)

    def test_the_vision_context_says_global_first(self):
        from ai_dt.pipeline import _vision_context

        self.assertIn("before white balance", _vision_context("", "", describe=False))


class OrderTests(unittest.TestCase):
    def test_global_first_local_then_finishing(self):
        steps, _held = history_steps(
            [
                {"id": "mask_edge_burn", "action": "darken"},
                {"id": "mask_sky", "action": "darken"},
                {"id": "subject", "action": "lift"},
                {"id": "mask_sky_gradient", "action": "darken"},
                {"id": "split_tone", "action": "include"},
            ]
        )
        self.assertEqual(
            [step["id"] for step in steps],
            ["split_tone", "mask_sky", "subject", "mask_sky_gradient", "mask_edge_burn"],
        )

    def test_existing_steps_keep_their_order(self):
        given = [{"id": "eyes"}, {"id": "subject"}, {"id": "foreground"}, {"id": "colorzones"}]
        self.assertEqual(lessons.order_steps(given), given)

    def test_every_local_exposure_stays_inside_the_cap(self):
        for step in mask_steps.STEPS.values():
            for action in ("lift", "darken", "hold"):
                grade = mask_steps.capped_grade(step, action)
                self.assertLessEqual(abs(grade.get("exposure", 0.0)), mask_steps.LOCAL_EV_CAP, step.id)


class NamesTests(unittest.TestCase):
    def test_lesson_and_animal_names_are_known_to_python_and_lua(self):
        names = {step.name for step in mask_steps.STEPS.values()}
        names |= {step_history_name(ident) for ident in ("split_tone", "palander")}
        names |= {step_history_name("keep_color", title=f"keep only the {hue}") for hue in ("red", "orange", "yellow", "green", "blue")}
        self.assertTrue(names <= accepted_step_names(), names - accepted_step_names())
        self.assertTrue(names <= _lua_names(), names - _lua_names())


class ShapeTests(unittest.TestCase):
    def test_sky_gradient_is_the_sky_intersected_with_the_top(self):
        sky = [{"type": "path", "points": [[0, 0], [1, 0], [1, 0.5], [0, 0.5]], "border": 0.01, "mode": "union"}]
        forms = mask_steps.sky_gradient_forms(sky)
        self.assertEqual(forms[-1]["type"], "gradient")
        self.assertEqual(forms[-1]["mode"], "intersect")
        drawn = shapes.render_forms(forms, 200, 100)
        self.assertGreater(drawn[2:10].mean(), 0.6)
        self.assertEqual(float(drawn[60:].max()), 0.0)
        self.assertGreater(drawn[5].mean(), drawn[45].mean())
        alone = mask_steps.sky_gradient_forms([])
        self.assertEqual(len(alone), 1)

    def test_bottom_gradient_and_edge_burn(self):
        bottom = shapes.render_forms(mask_steps.bottom_gradient_forms(), 200, 100)
        self.assertGreater(bottom[-3:].mean(), 0.6)
        self.assertLess(bottom[:20].mean(), 0.05)
        subject = [{"type": "path", "points": [[0.4, 0.3], [0.6, 0.3], [0.6, 0.98], [0.4, 0.98]], "border": 0.01, "mode": "union"}]
        forms = mask_steps.edge_burn_forms(subject)
        self.assertEqual([form["mode"] for form in forms], ["union"] * 4 + ["difference"])
        burn = shapes.render_forms(forms, 200, 100)
        self.assertGreater(burn[1, 1], 0.6)
        self.assertGreater(burn[50, 1], 0.6)
        self.assertLess(burn[50, 100], 0.05)
        self.assertLess(burn[97, 100], 0.05)  # the subject at the bottom edge is spared

    def test_glow_is_behind_but_not_on_the_subject(self):
        subject = [{"type": "path", "points": [[0.45, 0.4], [0.55, 0.4], [0.55, 0.9], [0.45, 0.9]], "border": 0.01, "mode": "union"}]
        forms = mask_steps.glow_forms((0.45, 0.4, 0.1, 0.5), subject)
        self.assertEqual(forms[0]["type"], "ellipse")
        glow = shapes.render_forms(forms, 200, 200)
        self.assertGreater(glow[90, 80], 0.5)
        self.assertEqual(float(glow[120, 100]), 0.0)


class LookTests(unittest.TestCase):
    def test_keep_only_red(self):
        xml = step_style("keep_color", title="keep only the red")
        self.assertIn("<operation>colorzones</operation>", xml)
        self.assertIn("Keep only the red", xml)
        chroma = colorzones_curve(_style_blob(xml), 1)
        self.assertGreater(_curve_at(chroma, 0.0), 0.6)
        self.assertGreater(_curve_at(chroma, 0.99), 0.6)
        for hue in (0.2, 0.33, 0.5, 0.66, 0.8):
            self.assertLess(_curve_at(chroma, hue), 0.05)
        self.assertLessEqual(len(chroma), 20)

    def test_warm_only_look(self):
        xml = step_style("palander")
        chroma = colorzones_curve(_style_blob(xml), 1)
        for hue in (0.0, 0.08, 0.16):
            self.assertGreater(_curve_at(chroma, hue), 0.6)
        for hue in (0.33, 0.5, 0.66, 0.8):
            self.assertLess(_curve_at(chroma, hue), 0.4)
        lightness = colorzones_curve(_style_blob(xml), 0)
        self.assertEqual({y for _x, y in lightness}, {0.5})

    def test_split_tone(self):
        xml = step_style("split_tone")
        self.assertIn("<operation>colorbalancergb</operation>", xml)
        floats = struct.unpack_from("<32f", _style_blob(xml))
        self.assertTrue(any(value > 0.1 for value in floats))
        gentle = step_style("split_tone", action="hold")
        self.assertNotEqual(gentle, xml)

    def test_keep_params_need_a_hue(self):
        self.assertIsNone(colorzones_keep_params([]))


class GrammarTests(unittest.TestCase):
    def _ids(self, sentence):
        return [(item["id"], item["action"]) for item in mask_operations(sentence)]

    def test_lesson_phrases(self):
        self.assertEqual(self._ids("burn the edges"), [("mask_edge_burn", "darken")])
        self.assertEqual(self._ids("add a gradient to the sky"), [("mask_sky_gradient", "darken")])
        self.assertEqual(self._ids("darken the sky with a gradient"), [("mask_sky_gradient", "darken")])
        self.assertEqual(self._ids("a gradient from the bottom"), [("mask_bottom_gradient", "darken")])
        self.assertEqual(self._ids("brighten the top with a gradient"), [("mask_sky_gradient", "lift")])
        self.assertEqual(self._ids("add a glow behind him"), [("mask_glow", "lift")])
        self.assertEqual(self._ids("give it the palander look"), [("palander", "include")])
        self.assertEqual(self._ids("keep only the red"), [("keep_color", "include")])
        self.assertEqual(self._ids("a subtle split tone"), [("split_tone", "hold")])

    def test_animal_phrases(self):
        self.assertEqual(self._ids("brighten the dog"), [("mask_animal", "lift")])
        self.assertEqual(self._ids("lift the birds"), [("mask_animals", "lift")])
        self.assertEqual(self._ids("darken around the dog"), [("mask_animal_background", "darken")])
        self.assertEqual([item[0] for item in self._ids("the wildlife look")], list(lessons.PRESETS["animal"]))
        self.assertEqual(self._ids("a dog on a beach"), [])


class AcceptTests(unittest.TestCase):
    def _splice(self, directory, spec_file):
        source = directory / "photo.xmp"
        source.write_text(_document(), encoding="utf-8")
        out = directory / "masked.xmp"
        self.assertEqual(drawn_main(["--xmp", str(source), "--xmp-out", str(out), "--mask-json", spec_file]), 0)
        return out.read_text(encoding="utf-8")

    def test_sky_gradient_on_a_landscape(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            _work(directory, "mask_sky_gradient")
            recipe = commit_waiting_step(directory)
            spec = json.loads(Path(recipe["step_mask"]["file"]).read_text(encoding="utf-8"))
            self.assertEqual(spec["operation"], "exposure")
            self.assertEqual(spec["grade"]["exposure"], -0.5)
            self.assertEqual(spec["forms"][-1]["mode"], "intersect")
            drawn = shapes.render_forms(spec["forms"], 360, 240)
            self.assertLess(float((drawn[120:] >= 0.5).mean()), 0.02)
            self.assertGreater(float(drawn[:20].mean()), float(drawn[60:90].mean()))
            written = self._splice(directory, recipe["step_mask"]["file"])
            self.assertIn('darktable:multi_name="Darken the sky gradient"', written)
            kinds = sorted({record["type"] for record in read_mask_records(written)})
            self.assertTrue({shapes.PATH, shapes.GRADIENT, shapes.GROUP} <= set(kinds), kinds)

    def test_bottom_gradient_needs_no_segmenter(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            _work(directory, "mask_bottom_gradient")
            recipe = commit_waiting_step(directory)
            self.assertEqual(recipe["step_mask"]["source"], "gradient")

    def test_glow_without_a_subject_is_left_off(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            _work(directory, "mask_glow")
            recipe = commit_waiting_step(directory)
            self.assertNotIn("step_mask", recipe)
            self.assertIn("No subject was found", recipe["feature_note"])

    def test_glow_and_edge_burn_with_a_subject(self):
        image, box = subject_on_foliage()
        subjects = [{"label": "dog", "category": "animal", "confidence": 0.9, "bbox": list(box)}]
        for step_id in ("mask_glow", "mask_edge_burn"):
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
                directory = Path(tmp)
                _work(directory, step_id, image=image, subjects=subjects)
                recipe = commit_waiting_step(directory)
                spec = json.loads(Path(recipe["step_mask"]["file"]).read_text(encoding="utf-8"))
                self.assertIn("difference", [form["mode"] for form in spec["forms"]], step_id)
                drawn = shapes.render_forms(spec["forms"], image.width, image.height)
                self.assertLess(float(drawn[120:140, 150:190].mean()), 0.1, step_id)  # not on the subject

    def test_animals_get_one_outline_each_and_an_inverse(self):
        image, boxes, _truths = animals_in_field()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            _work(directory, "mask_animals", image=image, subjects=boxes)
            recipe = commit_waiting_step(directory)
            spec = json.loads(Path(recipe["step_mask"]["file"]).read_text(encoding="utf-8"))
            self.assertGreaterEqual(len([f for f in spec["forms"] if f["mode"] == "union"]), 3)
            written = self._splice(directory, recipe["step_mask"]["file"])
            self.assertIn('darktable:multi_name="Lift the animals"', written)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            _work(directory, "mask_animal_background", image=image, subjects=boxes)
            recipe = commit_waiting_step(directory)
            written = self._splice(directory, recipe["step_mask"]["file"])
            start = written.find('darktable:multi_name="Quiet around the animals"')
            tag = written[written.rfind("<rdf:li", 0, start) : written.find("/>", start)]
            blend = decode_blend(tag.split('darktable:blendop_params="')[1].split('"')[0])
            self.assertTrue(blend["inverted"])

    def test_no_animal_is_left_off(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            _work(directory, "mask_animal")
            recipe = commit_waiting_step(directory)
            # A missed animal falls back to the main subject; with no subject
            # either, the step is left off with a plain reason.
            if "step_mask" in recipe:
                self.assertIn("no animal detected", recipe["step_mask"]["source"])
            else:
                self.assertIn("No animal was found", recipe["feature_note"])

    def test_a_global_look_is_a_style_step(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            _work(directory, "keep_color")
            recipe_path = directory / "output.json"
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            recipe["features_next_title"] = "keep only the red"
            recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
            recipe = commit_waiting_step(directory)
            self.assertNotIn("step_mask", recipe)
            xml = Path(recipe["step_style"]["file"]).read_text(encoding="utf-8")
            self.assertIn("Keep only the red", xml)

    def test_ask_queues_a_lesson_look_before_local_steps(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, _ENV):
            directory = Path(tmp)
            recipe = {
                "features_next": "",
                "reading": {"sentence": "A city street at dusk."},
                "plan": {"features_done": [], "features_path": [], "reasoning": {}},
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            reply = {"ask_ollama": False, "proposal": None}
            with patch("ai_dt.reasoner.propose_gemini", return_value=reply):
                plan_chat(directory, "burn the edges, then give it a warm-only look")
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["features_next"], "palander")
            self.assertIn("mask_edge_burn", saved["plan"]["features_path"])


class CropTests(unittest.TestCase):
    def test_horizon_and_subject_land_on_thirds(self):
        crop = lessons.crop_suggestion((3000, 2000), horizon=0.42, subject=(0.55, 0.4, 0.1, 0.2))
        self.assertEqual(crop["aspect"], "16:9")
        self.assertAlmostEqual(crop["w"], 1.0)
        line = (0.42 - crop["y"]) / crop["h"]
        self.assertTrue(min(abs(line - 1 / 3), abs(line - 2 / 3)) < 0.02, line)
        self.assertIn("horizon", crop["why"])

    def test_a_frame_that_already_fits_is_left_alone(self):
        self.assertIsNone(lessons.crop_suggestion((1920, 1080)))

    def test_pipeline_helper_never_raises(self):
        from ai_dt.pipeline import _crop_suggestion
        from ai_dt.schema import Detection

        found = _crop_suggestion(landscape(), [Detection("dog", "animal", 0.9, (0.1, 0.5, 0.1, 0.1), "test")])
        self.assertIsNotNone(found)
        self.assertIsNone(_crop_suggestion(None, []))


if __name__ == "__main__":
    unittest.main()
