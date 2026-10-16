"""Mask features are chosen per frame. They are not a fixed stage order."""

from __future__ import annotations

import json
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

from ai_dt.features import body_found, commit_waiting_step, plan_chat, select_mask_features
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


if __name__ == "__main__":
    unittest.main()
