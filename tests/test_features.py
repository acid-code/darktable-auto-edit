"""Mask features are chosen per frame. They are not a fixed stage order."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from ai_dt.features import (
    align_plan_to_history,
    body_found,
    cast_operations,
    commit_waiting_step,
    hue_operation,
    plan_chat,
    restore_waiting_step,
    select_mask_features,
)
from ai_dt.mapping import _choose_vignette_target, _vignette_for, step_style
from ai_dt.status_text import panel_status
from ai_dt.guides.decide import Picture
from ai_dt.guides.library import match_guide
from ai_dt.pipeline import analyze, write_outputs
from ai_dt.reasoner import interpret_reply, validate_features
from ai_dt.review import review_feature_path
from ai_dt.schema import Atmosphere, Detection
from ai_dt.separate import render_finish
from ai_dt.steer import empty_plan, parse_steer, revise_plan


def _picture(kind: str, box=None) -> Picture:
    return Picture(
        type=kind,
        reason="test",
        brightness=0.5,
        warmth=0.0,
        saturation=0.1,
        edges=0.04,
        line_fraction=0.01,
        sky=0.2,
        skin=0.0,
        highlight_specks=0.0,
        compact_area=0.1,
        compact_fill=0.5,
        subject_box=box,
    )


def _save(pixels: np.ndarray, path: Path) -> None:
    image = Image.fromarray(np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")
    image.save(path)


class CatalogTests(unittest.TestCase):
    def test_a_window_is_not_a_body(self) -> None:
        window = Detection("window", "subject", 0.8, (0.4, 0.1, 0.2, 0.3), "test")
        person = Detection("person", "person", 0.9, (0.3, 0.2, 0.2, 0.5), "test")
        self.assertFalse(body_found([window]))
        self.assertTrue(body_found([person]))

    def test_the_model_cannot_invent_a_tool(self) -> None:
        payload = {
            "features": [
                {"id": "opening", "reason": "the gap in the trees is too bright"},
                {"id": "dehaze", "reason": "the gap in the trees is too bright"},
                {"id": "foreground", "reason": ""},
            ]
        }
        self.assertEqual([item["id"] for item in validate_features(payload)], ["opening"])

    def test_a_reason_can_name_the_picture_without_the_feature_id(self) -> None:
        kept = review_feature_path(
            [{"id": "opening", "reason": "the gap in the trees is too bright"}],
            "the gap in the trees is too bright",
            "",
        )
        self.assertEqual(kept, ["opening"])

    def test_a_feature_the_sentence_did_not_ask_for_is_dropped(self) -> None:
        kept = review_feature_path(
            [{"id": "shafts", "reason": "the trunks are dark vertical shapes"}],
            "warm the light on the leaves",
            "a bright forest",
        )
        self.assertEqual(kept, [])

    def test_an_empty_sentence_can_rest_on_the_reading(self) -> None:
        kept = review_feature_path(
            [{"id": "highlights", "reason": "the bright leaves need warmth"}],
            "",
            "bright leaves under the canopy",
        )
        self.assertEqual(kept, ["highlights"])

    def test_reply_json_keeps_catalog_ids_only(self) -> None:
        text = (
            '{"reasoning": "The path is the lower band.", "stages": [], '
            '"features": [{"id": "foreground", "reason": "the path under them"}, '
            '{"id": "dehaze", "reason": "the path under them"}]}'
        )
        parsed = interpret_reply(text)
        self.assertEqual([item["id"] for item in parsed["features"]], ["foreground"])


class PanelTests(unittest.TestCase):
    def test_the_panel_shows_the_model_and_the_looks_not_the_stage_list(self) -> None:
        text = panel_status(
            {
                "picture_backend": "Picture: measured locally",
                "plan_backend": "Plan: Gemini gemini-3.5-flash-lite",
                "plan": {
                    "reasoning": {
                        "paragraph": "I see a person against the trees. I would quiet the path under them.",
                    },
                    "label": "Ran grade, subject, eyes.",
                },
                "features_next": "foreground",
                "features_next_title": "Darken the foreground",
            }
        )
        self.assertIn("I see a person against the trees.", text)
        self.assertIn("now — Darken the foreground", text)
        self.assertIn("make a new plan from this photo", text)
        self.assertNotIn("Ran grade", text)

    def test_an_accepted_step_is_one_history_module(self) -> None:
        subject = step_style("subject")
        highlights = step_style("highlights")
        self.assertIsNotNone(subject)
        self.assertIn("Subject lift", subject or "")
        self.assertIn("exposure", subject or "")
        shine = step_style("subject", again=True)
        self.assertIn("Shine the subject", shine or "")
        self.assertIn("<multi_priority>3</multi_priority>", shine or "")
        self.assertIn("<multi_priority>2</multi_priority>", subject or "")
        eyes = step_style("eyes", title="lift the catchlight, then hold the fur")
        self.assertIn("lift the catchlight, then hold the fur", eyes or "")
        self.assertNotIn("Brighten the eyes", eyes or "")
        self.assertIn("<multi_priority>4</multi_priority>", eyes or "")
        self.assertIn("Warm the highlights", highlights or "")
        self.assertIsNone(step_style("dehaze"))

    def test_a_subject_with_a_body_is_a_vignette(self) -> None:
        box = (0.30, 0.40, 0.25, 0.35)
        placed = step_style("subject", box=box, size=(1280, 852))
        self.assertIn("vignette", placed or "")
        self.assertIn("Vignette the subject", placed or "")
        self.assertNotIn("Subject lift", placed or "")
        again = step_style("subject", again=True, box=box, size=(1280, 852))
        self.assertIn("vignette", again or "")
        self.assertNotIn("Shine the subject", again or "")

    def test_two_statues_get_a_hole_around_them_not_a_flat_line(self) -> None:
        box = (0.22, 0.35, 0.42, 0.40)
        style = step_style("subject", box=box, size=(1280, 852))
        blob = bytes.fromhex(style.split("<op_params>")[1].split("</op_params>")[0].strip())
        scale, falloff, brightness, _saturation, _cx, _cy = struct.unpack("<ffffff", blob[:24])
        whratio = struct.unpack_from("<f", blob, 28)[0]
        self.assertGreater(scale, 40.0)
        self.assertGreater(falloff, 15.0)
        self.assertGreater(whratio, 0.7)
        self.assertLess(whratio, 1.35)
        self.assertLess(brightness, -0.15)
        self.assertGreater(brightness, -0.5)

    def test_accept_limits_a_hue_to_its_patches_and_vignettes_around_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image = Image.new("RGB", (200, 160), (16, 16, 24))
            pixels = image.load()
            for x in range(40, 80):
                for y in range(30, 70):
                    pixels[x, y] = (188, 36, 32)
            for x in range(90, 140):
                for y in range(100, 145):
                    pixels[x, y] = (196, 42, 36)
            image.save(directory / "preview.jpg")
            recipe = {
                "chat": True,
                "width": 200,
                "height": 160,
                "features_next": "colorzones",
                "features_next_title": "lift red aprons",
                "features_next_action": "lift",
                "analysis": {
                    "subjects": [
                        {"category": "animal", "bbox": [0.20, 0.55, 0.40, 0.40]}
                    ]
                },
                "plan": {
                    "features_done": [],
                    "features_path": ["colorzones", "subject"],
                    "operations": [
                        {"id": "colorzones", "action": "lift", "reason": "lift red aprons"},
                        {"id": "subject", "action": "lift", "reason": "make foxes more prominent"},
                    ],
                    "step_queue": [
                        {"id": "colorzones", "action": "lift", "title": "lift red aprons"},
                        {"id": "subject", "action": "lift", "title": "make foxes more prominent"},
                    ],
                },
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            colored = commit_waiting_step(directory)
            mask = colored["step_mask"]
            self.assertEqual(mask["kind"], "regions")
            self.assertEqual(mask["name"], "Highlight the red")
            self.assertFalse((directory / "step.dtstyle").exists())
            spec = json.loads(Path(mask["file"]).read_text(encoding="utf-8"))
            self.assertEqual(len(spec["shapes"]), 2)
            self.assertEqual(spec["operation"], "colorzones")
            placed = commit_waiting_step(directory)
            style = (directory / "step.dtstyle").read_text(encoding="utf-8")
            self.assertIn("vignette", style)
            self.assertIn("Vignette the subject", style)
            self.assertNotIn(">exposure<", style)
            self.assertEqual(placed["plan"]["step_log"][-1]["name"], "Vignette the subject")
            restored = restore_waiting_step(directory)
            self.assertEqual(restored["features_next"], "subject")
            self.assertNotIn("subject", restored["plan"]["features_done"])
            self.assertEqual(restored["plan"]["step_log"][-1]["id"], "colorzones")
            blob = bytes.fromhex(style.split("<op_params>")[1].split("</op_params>")[0].strip())
            center_y = struct.unpack("<ffffff", blob[:24])[5]
            self.assertLess(center_y, 0.35)

    def test_a_large_subject_uses_the_object_outline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            Image.new("RGB", (640, 480), (20, 24, 28)).save(directory / "preview.jpg")
            recipe = {
                "chat": True,
                "width": 640,
                "height": 480,
                "features_next": "subject",
                "features_next_title": "lift the fox",
                "features_next_action": "lift",
                "analysis": {
                    "subjects": [
                        {"category": "animal", "bbox": [0.2, 0.2, 0.4, 0.5]}
                    ]
                },
                "plan": {
                    "features_done": [],
                    "features_path": ["subject"],
                    "operations": [
                        {"id": "subject", "action": "lift", "reason": "make the fox more prominent"}
                    ],
                    "step_queue": [
                        {"id": "subject", "action": "lift", "title": "lift the fox"}
                    ],
                },
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            outline = [
                (0.22, 0.24),
                (0.58, 0.22),
                (0.60, 0.68),
                (0.40, 0.72),
                (0.20, 0.50),
            ]
            with patch("ai_dt.object_segment.outline_box", return_value=outline):
                placed = commit_waiting_step(directory)
            mask = placed["step_mask"]
            self.assertEqual(mask["kind"], "path")
            self.assertEqual(mask["name"], "Subject lift")
            self.assertFalse((directory / "step.dtstyle").exists())
            spec = json.loads(Path(mask["file"]).read_text(encoding="utf-8"))
            self.assertEqual(spec["kind"], "path")
            self.assertEqual(len(spec["points"]), 5)
            self.assertAlmostEqual(spec["exposure_ev"], 0.22)

    def test_a_person_in_a_landscape_is_the_vignette(self) -> None:
        box = (0.15, 0.2, 0.22, 0.45)
        person = Detection("person", "person", 0.9, box, "test")
        window = Detection("window", "subject", 0.8, (0.6, 0.1, 0.2, 0.25), "test")
        target = _choose_vignette_target([person, window], _picture("landscape", box))
        self.assertIsNotNone(target)
        self.assertEqual(target.category, "person")
        self.assertEqual(target.bbox, box)
        self.assertIsNone(_choose_vignette_target([], _picture("landscape")))
        self.assertIsNone(_choose_vignette_target([window], _picture("night")))

    def test_a_wide_low_animal_box_is_pulled_up_onto_the_body(self) -> None:
        image = type("Frame", (), {"size": (2817, 1876)})()
        animal = Detection("animal", "animal", 0.74, (0.197, 0.607, 0.75, 0.393), "heuristic")
        params = _vignette_for(animal, image)
        self.assertIsNotNone(params)
        self.assertLessEqual(params["scale"], 24.0)
        self.assertLess(params["center_y"], 0.25)
        self.assertLess(params["falloff_scale"], params["scale"])

    def test_a_model_subject_lift_is_something_you_can_accept(self) -> None:
        plan = empty_plan(["grade", "subject", "background"])
        plan["summary"] = "Ignored: make, the, dog, the, main, theme."
        proposal = {
            "stages": [
                {
                    "stage": "subject",
                    "action": "lift",
                    "reason": "make the dog the main theme and more eye catching",
                }
            ]
        }
        chosen = select_mask_features(
            plan,
            proposal,
            [],
            "make the dog the main theme",
            "two stone foxes",
            "",
        )
        self.assertEqual(chosen["features_next"], "subject")
        self.assertNotIn("could not be mapped", chosen["feature_note"])
        accepted = select_mask_features(
            chosen,
            proposal,
            [],
            "make the dog the main theme",
            "two stone foxes",
            "accept-feature",
        )
        self.assertEqual(accepted["features_next"], "")
        self.assertIn("subject", accepted["features_done"])
        self.assertIn("on this photo", accepted["feature_note"])

    def test_a_new_sentence_lifts_the_subject_again(self) -> None:
        plan = empty_plan(["grade", "subject"])
        plan["features_done"] = ["subject"]
        proposal = {
            "paragraph": (
                "the request asks to pinpoint and outshine the aprons. "
                "we will lift the subject and leave the rest alone."
            ),
            "stages": [
                {
                    "stage": "subject",
                    "action": "lift",
                    "reason": "outshine the aprons",
                }
            ],
        }
        chosen = select_mask_features(
            plan,
            proposal,
            [],
            "pinpoint and outshine the aprons more",
            "two stone foxes",
            "",
        )
        self.assertEqual(chosen["features_next"], "subject")
        self.assertNotIn("subject", chosen["features_done"])
        self.assertIn("Another lift", chosen["feature_note"])

    def test_an_eyes_request_becomes_a_step_after_the_subject_is_done(self) -> None:
        plan = empty_plan(["grade", "subject"])
        plan["features_done"] = ["subject", "foreground"]
        proposal = {
            "paragraph": (
                "the eyes need a localized boost in exposure and saturation "
                "to stand out against the animal's fur."
            ),
            "stages": [
                {
                    "stage": "eyes",
                    "action": "lift",
                    "reason": "lift the catchlight in the cats eyes",
                },
                {
                    "stage": "highlights",
                    "action": "hold",
                    "reason": "hold the fur around the cats eyes",
                },
            ],
            "features": [],
        }
        chosen = select_mask_features(
            plan,
            proposal,
            [],
            "Can you make the cats eyes pop?",
            "a cat in a dark room",
            "",
        )
        self.assertEqual(chosen["features_next"], "eyes")
        self.assertEqual(chosen["features_next_title"], "lift the catchlight in the cats eyes")
        self.assertEqual(chosen["step_queue"][1]["id"], "highlights")
        self.assertIn("subject", chosen["features_done"])
        self.assertNotIn("subject lift is now", chosen["feature_note"])
        self.assertNotIn("Brighten", chosen["features_next_title"])
        text = panel_status(
            {
                "chat": True,
                "features_next": chosen["features_next"],
                "features_next_title": chosen["features_next_title"],
                "plan": chosen,
                "handoff": {"sentence": "This plan starts from the photo you are viewing."},
            }
        )
        self.assertIn("now — lift the catchlight in the cats eyes", text)
        self.assertIn("later — hold the fur around the cats eyes", text)
        self.assertIn("Accept applies this step", text)

    def test_a_finished_plan_does_not_ask_for_accept(self) -> None:
        text = panel_status(
            {
                "chat": True,
                "features_next": "",
                "plan": {"features_done": ["subject", "foreground"], "features_path": []},
                "handoff": {
                    "sentence": "This plan starts from the photo you are viewing. This sentence did not add a step."
                },
            }
        )
        self.assertNotIn("Accept applies this step", text)
        self.assertIn("press Ask", text)
        self.assertNotIn("now —", text)

    def test_committing_the_waiting_step_does_not_touch_the_look(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            recipe = {
                "features_next": "subject",
                "features_next_title": "Lift the subject",
                "feature_note": "Another lift, for this request.",
                "plan": {"features_done": [], "features_path": [], "reasoning": {}},
                "style": {"name": "AI edit", "file": "edit.dtstyle"},
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            (directory / "edit.dtstyle").write_text("<style/>", encoding="utf-8")
            commit_waiting_step(directory)
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["features_next"], "")
            self.assertIn("subject", saved["plan"]["features_done"])
            self.assertIn("plan is finished", saved["feature_note"])
            self.assertTrue(saved["chat"])
            self.assertEqual(saved["choices"], [])
            self.assertIn("Shine the subject", (directory / "step.dtstyle").read_text(encoding="utf-8"))
            self.assertEqual((directory / "edit.dtstyle").read_text(encoding="utf-8"), "<style/>")

    def test_a_new_request_is_steps_and_does_not_rewrite_the_look(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            recipe = {
                "features_next": "",
                "reading": {"sentence": "Two stone foxes in a dark room."},
                "plan": {"features_done": ["subject"], "features_path": [], "reasoning": {}},
                "choices": [{"id": "faithful", "title": "Faithful"}],
                "offer_looks": True,
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            (directory / "edit.dtstyle").write_text("<style/>", encoding="utf-8")
            reply = {
                "ask_ollama": False,
                "proposal": {
                    "status": "gemini",
                    "backend": "gemini-test",
                    "paragraph": "quiet the path and warm the leaves",
                    "stages": [],
                    "features": [
                        {"id": "foreground", "reason": "quiet the path"},
                        {"id": "highlights", "reason": "warm the leaves"},
                    ],
                },
            }
            with patch("ai_dt.reasoner.propose_gemini", return_value=reply):
                plan_chat(directory, "quiet the path and warm the leaves")
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["chat"])
            self.assertFalse(saved["offer_looks"])
            self.assertEqual(saved["choices"], [])
            self.assertEqual(saved["features_next"], "foreground")
            self.assertIn("highlights", saved["plan"]["features_path"])
            self.assertEqual((directory / "edit.dtstyle").read_text(encoding="utf-8"), "<style/>")
            self.assertIn("photo you are viewing", saved["handoff"]["sentence"])

    def test_a_follow_up_keeps_the_next_history_step_and_hides_stages_it_cannot_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            recipe = {
                "features_next": "",
                "reading": {"sentence": "A cat on a wooden railing."},
                "scene": {"summary": "A cat on a wooden railing.", "source": "mistral"},
                "analysis": {"atmosphere": {}},
                "separation": {"stages_ran": ["grade"]},
                "plan": {
                    "guide_stages": ["grade", "subject", "eyes", "reflection", "background"],
                    "skip": [],
                    "features_done": [],
                    "features_path": [],
                    "reasoning": {},
                    "asks": [],
                },
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            reply = {
                "ask_ollama": False,
                "proposal": {
                    "status": "gemini",
                    "backend": "gemini-test",
                    "paragraph": "Highlight the eyes and leave the rest of the scene untouched.",
                    "stages": [],
                    "operations": [
                        {"id": "eyes", "action": "lift", "reason": "highlight the eyes"},
                        {"id": "subject", "action": "lift", "reason": "lift the cat"},
                        {"id": "reflection", "action": "lift", "reason": "the water"},
                        {"id": "background", "action": "hold", "reason": "the city"},
                    ],
                    "features": [],
                },
            }
            with patch("ai_dt.reasoner.propose_gemini", return_value=reply):
                plan_chat(directory, "highlight the eyes and make them pop")
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["features_next"], "eyes")
            self.assertEqual(saved["plan"]["features_path"], ["eyes", "subject"])
            self.assertNotIn("Stages: eyes, subject, reflection, background.", saved["status_text"])
            self.assertIn("Steps: eyes, subject.", saved["status_text"])
            self.assertIn("Not on this photo: reflection, background.", saved["status_text"])
            self.assertIn("stays off this photo", saved["status_text"])
            commit_waiting_step(directory)
            accepted = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(accepted["features_next"], "subject")
            self.assertNotIn("plan is finished", accepted["feature_note"])
            self.assertIn("press Accept", accepted["handoff"]["sentence"])

    def test_foxes_and_red_aprons_are_two_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            recipe = {
                "features_next": "",
                "reading": {"sentence": "Two stone fox statues with red aprons."},
                "scene": {"summary": "Two stone fox statues with red aprons.", "source": "mistral"},
                "analysis": {"atmosphere": {}},
                "separation": {"stages_ran": ["grade"]},
                "plan": {
                    "guide_stages": ["grade", "subject"],
                    "skip": [],
                    "features_done": ["subject"],
                    "features_path": [],
                    "reasoning": {},
                    "asks": [],
                },
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            reply = {
                "ask_ollama": False,
                "proposal": {
                    "status": "gemini",
                    "backend": "gemini-test",
                    "paragraph": (
                        "the request asks to make the two foxes and their red aprons "
                        "prominent, so we lift the subject."
                    ),
                    "stages": [],
                    "operations": [
                        {
                            "id": "subject",
                            "action": "lift",
                            "reason": "make the two foxes prominent",
                        }
                    ],
                    "features": [],
                },
            }
            steer = (
                "the main theme are the two foxes and their red aprons. "
                "make the aprons highlight and the foxes more prominent"
            )
            with patch("ai_dt.reasoner.propose_gemini", return_value=reply):
                plan_chat(directory, steer)
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["plan"]["features_path"], ["subject", "colorzones"])
            self.assertIn("Steps: subject, colorzones.", saved["status_text"])
            image = Image.new("RGB", (200, 160), (16, 16, 24))
            pixels = image.load()
            for x in range(40, 80):
                for y in range(30, 70):
                    pixels[x, y] = (188, 36, 32)
            image.save(directory / "preview.jpg")
            saved["analysis"] = {
                "subjects": [{"category": "animal", "label": "fox", "bbox": [0.2, 0.2, 0.4, 0.5]}]
            }
            (directory / "output.json").write_text(json.dumps(saved), encoding="utf-8")
            commit_waiting_step(directory)
            waiting = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(waiting["features_next"], "colorzones")
            self.assertTrue((directory / "step.dtstyle").is_file())
            commit_waiting_step(directory)
            masked = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(masked["step_mask"]["kind"], "regions")
            self.assertFalse((directory / "step.dtstyle").is_file())


class SeparateTests(unittest.TestCase):
    def test_a_person_in_a_landscape_is_separated_and_a_frame_without_one_is_not(self) -> None:
        pixels = np.full((120, 160, 3), 0.62, dtype=np.float32)
        box = (0.35, 0.25, 0.22, 0.5)
        person = Detection("person", "person", 0.9, box, "test")
        picture = _picture("landscape", box)
        guide = match_guide("landscape").guide
        plain, plain_info = render_finish(
            pixels, Atmosphere(), [person], guide=guide, picture=picture
        )
        strong, info = render_finish(
            pixels,
            Atmosphere(),
            [person],
            guide=guide,
            picture=picture,
            mask_features=["separate"],
        )
        self.assertNotIn("separate", plain_info["stages_ran"])
        self.assertIn("separate", info["stages_ran"])
        self.assertLess(float(strong[4:12, 4:16].mean()), float(plain[4:12, 4:16].mean()) - 0.05)
        body = strong[50:90, 70:100]
        self.assertGreater(float(body.mean()), float(plain[50:90, 70:100].mean()) + 0.02)
        untouched, _untouched_info = render_finish(
            pixels, Atmosphere(), [], guide=guide, picture=_picture("landscape")
        )
        empty, empty_info = render_finish(
            pixels,
            Atmosphere(),
            [],
            guide=guide,
            picture=_picture("landscape"),
            mask_features=["separate"],
        )
        self.assertNotIn("separate", empty_info["stages_ran"])
        self.assertIn("nothing to attach", " ".join(empty_info.get("feature_notes") or []).lower())
        self.assertAlmostEqual(float(empty.mean()), float(untouched.mean()), delta=0.01)

    def test_an_opening_a_shaft_and_a_highlight_need_a_region(self) -> None:
        flat = np.full((80, 80, 3), 0.45, dtype=np.float32)
        _rendered, info = render_finish(
            flat,
            Atmosphere(),
            [],
            picture=_picture("landscape"),
            mask_features=["opening", "shafts", "highlights", "foreground"],
        )
        notes = " ".join(info.get("feature_notes") or []).lower()
        self.assertIn("opening", notes)
        self.assertIn("shafts", notes)
        self.assertIn("highlights", notes)
        self.assertNotIn("opening", info["stages_ran"])
        self.assertNotIn("shafts", info["stages_ran"])

    def test_a_bright_gap_darkens_without_taking_the_sky(self) -> None:
        pixels = np.full((90, 90, 3), 0.4, dtype=np.float32)
        pixels[12:32, 34:54] = 0.92
        _before, info = render_finish(
            pixels,
            Atmosphere(),
            [],
            picture=_picture("landscape"),
            mask_features=["opening"],
        )
        self.assertIn("opening", info["stages_ran"])
        sky = np.full_like(pixels, 0.9)
        _sky, sky_info = render_finish(
            sky,
            Atmosphere(),
            [],
            picture=_picture("landscape"),
            mask_features=["opening"],
        )
        self.assertNotIn("opening", sky_info["stages_ran"])
        self.assertIn("nothing to attach", " ".join(sky_info.get("feature_notes") or []).lower())


class PreviewTests(unittest.TestCase):
    def test_a_named_feature_stays_a_preview_until_it_is_accepted(self) -> None:
        pixels = np.full((180, 240, 3), 0.66, dtype=np.float32)
        pixels[:40] = (0.59, 0.66, 0.71)
        pixels[90:] = (0.46, 0.56, 0.33)
        pixels[20:55, 90:130] = 0.8
        proposal = {
            "status": "skipped",
            "backend": "",
            "note": "",
            "paragraph": "",
            "stages": [],
            "features": [{"id": "opening", "reason": "the gap in the trees is too bright"}],
        }
        steer = "the gap in the trees is too bright"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "frame.png"
            _save(pixels, path)
            with patch("ai_dt.pipeline._detect", return_value=([], [])):
                first = analyze(
                    path,
                    detector="heuristic",
                    advisor="histogram",
                    output_dir=directory,
                    steer=steer,
                    proposal=proposal,
                )
            self.assertEqual(first["features_next"], "opening")
            self.assertNotIn("opening", first["separation"]["stages_ran"])
            self.assertNotIn("separate", first["features_committed"])
            gap = float(np.array(first["_rendered"])[20:55, 90:130].mean())
            preview_gap = float(np.array(first["_next_rendered"])[20:55, 90:130].mean())
            self.assertLess(preview_gap, gap - 0.02)
            write_outputs(first, directory)
            with patch("ai_dt.pipeline._detect", return_value=([], [])):
                second = analyze(
                    path,
                    detector="heuristic",
                    advisor="histogram",
                    output_dir=directory,
                    steer=steer,
                    look="accept-feature",
                    proposal=proposal,
                )
        self.assertIn("opening", second["features_committed"])
        self.assertIn("opening", second["separation"]["stages_ran"])
        self.assertLess(float(np.array(second["_rendered"])[20:55, 90:130].mean()), gap - 0.02)

    def test_an_open_sentence_is_not_guessed_when_no_model_answers(self) -> None:
        plan = revise_plan(empty_plan(["grade", "subject", "background"]), "make the path under them quieter")
        chosen = select_mask_features(plan, {"stages": []}, [], "make the path under them quieter", "a forest path", "")
        self.assertEqual(chosen["features_next"], "")
        self.assertIn("could not be mapped", chosen["feature_note"])

    def test_a_known_phrase_still_scales_and_a_background_tweak_does_not_start_the_next_feature(self) -> None:
        blur = revise_plan(empty_plan(["grade", "subject", "background"]), "less background blur")
        blur = select_mask_features(blur, {"stages": []}, [], "less background blur", "a portrait", "")
        self.assertNotIn("could not be mapped", blur["feature_note"])
        self.assertEqual(parse_steer("less background blur")["scales"][0]["stage"], "background")

        face = Detection("face", "face", 0.9, (0.3, 0.2, 0.3, 0.4), "test")
        plan = revise_plan(empty_plan(["grade", "subject", "eyes", "background"]), "a little less on the background")
        plan = select_mask_features(
            plan,
            {"stages": []},
            [face],
            "a little less on the background",
            "a person",
            "",
        )
        self.assertEqual(plan["features_committed"], ["separate"])
        self.assertAlmostEqual(plan["scales"]["separate"], 0.7)
        self.assertEqual(plan["features_next"], "")
        self.assertNotIn("could not be mapped", plan["feature_note"])

        ignored = revise_plan(empty_plan(["grade", "eyes"]), "less on the eyes")
        ignored = select_mask_features(ignored, {"stages": []}, [], "less on the eyes", "", "")
        self.assertEqual(ignored["scales"], {})
        self.assertIn("could not be mapped", ignored["feature_note"])


class PlanCursorTests(unittest.TestCase):
    def _recipe(self) -> dict:
        return {
            "chat": True,
            "features_next": "subject",
            "features_next_title": "Lift the subject",
            "features_next_action": "lift",
            "plan": {
                "features_done": ["eyes"],
                "features_path": ["subject"],
                "step_queue": [{"id": "subject", "action": "lift", "title": "Lift the subject"}],
                "step_log": [
                    {
                        "id": "eyes",
                        "name": "AI eyes",
                        "title": "Eyes",
                        "action": "lift",
                        "active": True,
                    }
                ],
            },
        }

    def test_back_past_an_editor_step_makes_it_the_waiting_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "output.json").write_text(json.dumps(self._recipe()), encoding="utf-8")
            saved = align_plan_to_history(directory, "AI eyes", "back")
            self.assertEqual(saved["features_next"], "eyes")
            self.assertEqual(saved["plan"]["features_path"], ["eyes", "subject"])
            self.assertFalse(saved["plan"]["step_log"][0]["active"])
            self.assertIn("Next puts that step back", saved["feature_note"])
            restored = align_plan_to_history(directory, "AI eyes", "next")
            self.assertEqual(restored["features_next"], "subject")
            self.assertEqual(restored["plan"]["features_done"], ["eyes"])
            self.assertTrue(restored["plan"]["step_log"][0]["active"])

    def test_a_photographer_module_does_not_change_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "output.json").write_text(json.dumps(self._recipe()), encoding="utf-8")
            saved = align_plan_to_history(directory, "portrait", "back")
            self.assertEqual(saved["features_next"], "subject")
            self.assertTrue(saved["plan"]["step_log"][0]["active"])

    def test_accept_after_back_drops_the_undone_step_and_writes_it_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            recipe = self._recipe()
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            align_plan_to_history(directory, "AI eyes", "back")
            saved = commit_waiting_step(directory)
            self.assertIn("left off", saved["feature_note"])
            self.assertEqual(saved["features_next"], "subject")
            self.assertFalse(
                any(item.get("id") == "eyes" and item.get("active") for item in saved["plan"]["step_log"])
            )


class ExpertMaskTests(unittest.TestCase):
    def test_the_cat_sentence_becomes_subject_and_eye_steps(self) -> None:
        sentence = "make the cat more seen and make his eyes pop"
        plan = revise_plan(empty_plan(["grade", "subject", "eyes"]), sentence)
        chosen = select_mask_features(plan, {"stages": []}, [], sentence, "a cat on a roof", "")
        self.assertEqual(chosen["features_path"], ["subject", "eyes"])
        self.assertNotIn("could not be mapped", chosen["feature_note"])

    def test_too_much_red_is_pulled_back_on_the_whole_frame(self) -> None:
        sentence = "help me make the cats eyes pop and maybe fix the too much red of the picture"
        operations = cast_operations(
            [
                {
                    "id": "colorzones",
                    "action": "include",
                    "reason": "fix the too much red of the picture",
                },
                {"id": "eyes", "action": "lift", "reason": "make the cats eyes pop"},
            ],
            sentence,
        )
        zones = next(item for item in operations if item["id"] == "recolor")
        self.assertEqual(zones["action"], "darken")
        self.assertIn("too much red", zones["reason"])
        self.assertEqual(operations[-1]["id"], "eyes")
        pleasing = cast_operations(
            [
                {
                    "id": "colorbalancergb",
                    "action": "include",
                    "reason": "a cooler balance would be more pleasing",
                }
            ],
            "",
        )
        self.assertEqual(pleasing[0]["id"], "recolor")
        routine = cast_operations(
            [{"id": "colorbalancergb", "action": "include", "reason": "global base"}],
            "make the eyes pop",
        )
        self.assertEqual(routine[0]["id"], "colorbalancergb")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image = Image.new("RGB", (200, 160), (40, 28, 24))
            pixels = image.load()
            for x, y in ((30, 20), (160, 100), (100, 50)):
                for dx in range(28):
                    for dy in range(28):
                        pixels[x + dx, y + dy] = (190, 42, 32)
            image.save(directory / "preview.jpg")
            recipe = {
                "chat": True,
                "features_next": "colorzones",
                "features_next_title": "fix the too much red of the picture",
                "features_next_action": "lift",
                "plan": {
                    "features_done": [],
                    "features_path": ["colorzones", "eyes"],
                    "step_queue": [
                        {
                            "id": "colorzones",
                            "action": "lift",
                            "title": "fix the too much red of the picture",
                        },
                        {"id": "eyes", "action": "lift", "title": "make the cats eyes pop"},
                    ],
                },
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            saved = commit_waiting_step(directory)
            self.assertNotIn("step_mask", saved)
            style = (directory / "step.dtstyle").read_text(encoding="utf-8")
            self.assertIn("Recolor", style)
            self.assertIn("channelmixerrgb", style)
            self.assertNotIn("Highlight the red", style)
            blob = bytes.fromhex(style.split("<op_params>")[1].split("</op_params>")[0].strip())
            red, _green, blue = struct.unpack_from("<fff", blob, 0)[0], struct.unpack_from("<f", blob, 20)[0], struct.unpack_from("<f", blob, 40)[0]
            self.assertLess(red, 1.0)
            self.assertGreater(blue, 1.0)

    def test_lanterns_are_an_orange_highlight(self) -> None:
        found = hue_operation("brighten the lanterns", "a night alley")
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found["id"], "colorzones")
        self.assertIn("orange", found["reason"])

    def test_eyes_without_a_box_are_left_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            recipe = {
                "chat": True,
                "features_next": "eyes",
                "features_next_title": "make his eyes pop",
                "features_next_action": "lift",
                "local_edits": [],
                "analysis": {"subjects": []},
                "plan": {
                    "features_done": [],
                    "features_path": ["eyes", "subject"],
                    "step_queue": [
                        {"id": "eyes", "action": "lift", "title": "make his eyes pop"},
                        {"id": "subject", "action": "lift", "title": "Lift the subject"},
                    ],
                },
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            saved = commit_waiting_step(directory)
            self.assertNotIn("step_style", saved)
            self.assertNotIn("step_mask", saved)
            self.assertFalse((directory / "step.dtstyle").is_file())
            self.assertEqual(saved["features_next"], "subject")
            self.assertIn("left off", saved["feature_note"])
            self.assertNotIn("eyes", saved["plan"]["features_done"])

    def test_a_named_subject_without_a_box_is_left_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            recipe = {
                "chat": True,
                "features_next": "subject",
                "features_next_title": "The fuji is the subject, so it is lifted.",
                "features_next_action": "lift",
                "analysis": {"subjects": []},
                "plan": {
                    "features_done": [],
                    "features_path": ["subject"],
                    "step_queue": [
                        {
                            "id": "subject",
                            "action": "lift",
                            "title": "The fuji is the subject, so it is lifted.",
                        }
                    ],
                },
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            saved = commit_waiting_step(directory)
            self.assertNotIn("step_style", saved)
            self.assertIn("left off", saved["feature_note"])
            self.assertEqual(saved["features_next"], "")

    def test_foreground_is_an_ellipse_along_the_bottom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            recipe = {
                "chat": True,
                "features_next": "foreground",
                "features_next_title": "Darken the foreground",
                "features_next_action": "darken",
                "analysis": {
                    "subjects": [
                        {"category": "animal", "label": "cat", "bbox": [0.62, 0.35, 0.28, 0.55]}
                    ]
                },
                "plan": {
                    "features_done": [],
                    "features_path": ["foreground"],
                    "step_queue": [
                        {"id": "foreground", "action": "darken", "title": "Darken the foreground"}
                    ],
                },
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            saved = commit_waiting_step(directory)
            mask = saved["step_mask"]
            self.assertEqual(mask["name"], "AI foreground")
            self.assertGreater(mask["cy"], 0.7)
            self.assertLess(mask["ry"], 0.25)
            self.assertLess(mask["cx"], 0.5)
            self.assertFalse((directory / "step.dtstyle").is_file())

    def test_a_red_highlight_without_a_patch_is_left_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            Image.new("RGB", (80, 60), (20, 20, 24)).save(directory / "preview.jpg")
            recipe = {
                "chat": True,
                "features_next": "colorzones",
                "features_next_title": "highlight the red",
                "features_next_action": "lift",
                "plan": {
                    "features_done": [],
                    "features_path": ["colorzones"],
                    "step_queue": [
                        {"id": "colorzones", "action": "lift", "title": "highlight the red"}
                    ],
                },
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            saved = commit_waiting_step(directory)
            self.assertNotIn("step_style", saved)
            self.assertNotIn("step_mask", saved)
            self.assertIn("left off", saved["feature_note"])


if __name__ == "__main__":
    unittest.main()
