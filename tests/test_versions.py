"""Version records, look choices, and the extra Darktable modules."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dt.choices import apply_named_look, collect_choices
from ai_dt.history_ops import choose_display, history_operations
from ai_dt.masks_xmp import ellipse_sidecar, masks_can_attach, sidecar_attaches_mask
from ai_dt.modules_extra import (
    BLURS_BYTES,
    CHANNELMIXER_BYTES,
    COLORZONES_BYTES,
    CROP_BYTES,
    DIFFUSE_BYTES,
    FILMIC_BYTES,
    GRADUATED_BYTES,
    SIGMOID_BYTES,
    TONE_EQUAL_BYTES,
    append_extra_modules,
    blurs_params,
    channelmixer_params,
    colorzones_params,
    crop_params,
    diffuse_params,
    filmic_params,
    graduated_params,
    retouch_decision,
    sigmoid_params,
    tone_equalizer_params,
)
from ai_dt.pipeline import analyze, apply_expert_requests, write_outputs
from ai_dt.schema import Atmosphere, Detection
from ai_dt.steer import empty_plan, revise_plan
from ai_dt.versions import build_version_record, version_sentence, version_work_dir


class VersionTests(unittest.TestCase):
    def test_work_dir_is_per_duplicate(self) -> None:
        root = Path("/work")
        first = version_work_dir(root, 12, 0)
        second = version_work_dir(root, 12, 1)
        other = version_work_dir(root, 13, 0)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(first.name, "v0")
        self.assertEqual(second.parent.name, "12")

    def test_sentence_names_kept_aside_and_recipe(self) -> None:
        recipe = {
            "features": ["person", "seat"],
            "reasoning": {"set_aside": [{"title": "Velvia"}]},
            "recipe_match": {"title": "Astia"},
            "plan": {"label": "Lift the face."},
        }
        sentence = version_sentence(recipe)
        self.assertIn("Astia", sentence)
        self.assertIn("person", sentence)
        self.assertIn("Velvia", sentence)
        self.assertIn("Lift the face.", sentence)

    def test_a_picked_look_leads_the_sentence(self) -> None:
        recipe = {
            "features": ["train"],
            "look_id": "shadow-lift",
            "reasoning": {"set_aside": [{"title": "Quiet night"}]},
            "recipe_match": {"title": "Faithful correction"},
            "plan": {"label": "Open the dark bands."},
        }
        sentence = version_sentence(recipe)
        self.assertTrue(sentence.startswith("Open shadows."))
        self.assertNotIn("Faithful correction", sentence)
        record = build_version_record(
            recipe,
            Path("/tmp/v2"),
            image_id=1,
            duplicate_index=2,
            parent_id=1,
            parent_index=0,
            look_id="warmer",
        )
        self.assertEqual(record["look_title"], "Warmer")
        self.assertTrue(record["sentence"].startswith("Warmer."))

    def test_record_keeps_the_parent_duplicate(self) -> None:
        record = build_version_record(
            {"features": [], "reasoning": {}, "recipe_match": {}, "plan": {"text": "Measured."}, "modules": []},
            Path("/tmp/v1"),
            image_id=4,
            duplicate_index=1,
            parent_id=4,
            parent_index=0,
            look_id="faithful",
        )
        self.assertEqual(record["parent"], {"image_id": 4, "duplicate_index": 0})
        self.assertEqual(record["look_id"], "faithful")
        self.assertTrue(record["sentence"])


class ChoiceTests(unittest.TestCase):
    def test_faithful_and_neighbors_fit_six_buttons(self) -> None:
        recommendations = [
            {"id": "subject-lift", "title": "Subject lift", "why": "The face is dark."},
            {"id": "fuji-astia", "title": "Astia", "why": "The face is warm."},
        ]
        neighbors = [
            {"id": "portrait-film", "title": "Film", "score": 0.4},
            {"id": "fuji-astia", "title": "Astia", "score": 0.2},
        ]
        choices = collect_choices(recommendations, neighbors, "")
        ids = [item["id"] for item in choices]
        self.assertEqual(ids[0], "faithful")
        self.assertIn("shadow-lift", ids)
        self.assertIn("warmer", ids)
        self.assertIn("sky-gradient", ids)
        self.assertLessEqual(len(choices), 6)
        again = collect_choices(
            [{"id": "shadow-lift", "title": "Open shadows", "why": "Again."}],
            neighbors,
            "",
        )
        self.assertEqual(len([item for item in again if item["id"] == "shadow-lift"]), 1)

    def test_a_named_look_is_a_plan_ask(self) -> None:
        plan = {"asks": [], "skip": ["eyes"], "reflection": "lift"}
        apply_named_look(plan, "subject-lift")
        apply_named_look(plan, "eye-detail")
        apply_named_look(plan, "highlight-hold")
        self.assertEqual(plan["asks"][0]["target"], "subject")
        self.assertNotIn("eyes", plan["skip"])
        self.assertTrue(plan["highlight_hold"])
        apply_named_look(plan, "warmer")
        apply_named_look(plan, "more-color")
        apply_named_look(plan, "sky-gradient")
        self.assertEqual(plan["warmth"], 0.4)
        self.assertEqual(plan["color"], 0.35)
        self.assertTrue(plan["gradient"])


class ModuleTests(unittest.TestCase):
    def test_blob_lengths_match_darktable_5_4_1(self) -> None:
        self.assertEqual(len(tone_equalizer_params(0.2, -0.1, hold=False) or b""), TONE_EQUAL_BYTES)
        self.assertEqual(len(filmic_params(blown=True, contrast=0.1)), FILMIC_BYTES)
        self.assertEqual(len(sigmoid_params(0.1)), SIGMOID_BYTES)
        self.assertEqual(len(colorzones_params(0.2) or b""), COLORZONES_BYTES)
        self.assertEqual(len(crop_params((0.0, 0.05, 1.0, 0.08)) or b""), CROP_BYTES)
        self.assertEqual(len(channelmixer_params(0.2, 0.0) or b""), CHANNELMIXER_BYTES)
        self.assertEqual(len(graduated_params(True) or b""), GRADUATED_BYTES)
        self.assertEqual(len(diffuse_params(sharpness=0.4, iterations=8, radius=8) or b""), DIFFUSE_BYTES)
        self.assertEqual(len(blurs_params(0.8) or b""), BLURS_BYTES)

    def test_zero_grades_add_no_module(self) -> None:
        self.assertIsNone(tone_equalizer_params(0.0, 0.0, hold=False))
        self.assertIsNone(colorzones_params(0.0))
        self.assertIsNone(channelmixer_params(0.0, 0.0))
        self.assertIsNone(graduated_params(False))
        self.assertIsNone(crop_params((0.0, 0.4, 1.0, 0.08)))
        self.assertIsNone(blurs_params(0.0))

    def test_display_transforms_never_stack(self) -> None:
        self.assertIsNone(choose_display(None, look="", vibe="", picture_type="night", problems=[]))
        self.assertIsNone(
            choose_display({"filmicrgb"}, look="sigmoid", vibe="", picture_type="night", problems=["blown-highlights"])
        )
        self.assertIsNone(
            choose_display({"sigmoid"}, look="", vibe="", picture_type="night", problems=[])
        )
        self.assertEqual(
            choose_display(set(), look="sigmoid", vibe="", picture_type="night", problems=[]),
            "sigmoid",
        )
        self.assertEqual(
            choose_display(set(), look="", vibe="", picture_type="night", problems=[]),
            "filmicrgb",
        )
        modules: list[dict] = []
        blobs: list[dict] = []
        atmosphere = Atmosphere(shadows=0.0, highlights=0.0, contrast=0.1, saturation=0.0)
        append_extra_modules(
            modules,
            blobs,
            atmosphere=atmosphere,
            picture_type="night",
            reading={"problems": [], "regions": []},
            plan={},
            separation={"stages_ran": []},
            guide=None,
            history_ops=set(),
            look="",
            vibe="",
            masks=False,
        )
        names = [item["operation"] for item in modules]
        self.assertIn("filmicrgb", names)
        self.assertNotIn("sigmoid", names)

    def test_masked_modules_wait_for_a_mask(self) -> None:
        self.assertFalse(masks_can_attach())
        modules: list[dict] = []
        blobs: list[dict] = []
        atmosphere = Atmosphere()
        append_extra_modules(
            modules,
            blobs,
            atmosphere=atmosphere,
            picture_type="portrait",
            reading={"problems": [], "regions": []},
            plan={},
            separation={"stages_ran": ["subject", "eyes", "skin", "background"], "subject_ev": 0.2},
            guide=None,
            history_ops=None,
            look="",
            vibe="",
            masks=False,
        )
        self.assertEqual(modules, [])
        spot = Detection("spot", "spot", 0.9, (0.4, 0.4, 0.05, 0.05), "test")
        self.assertFalse(retouch_decision([], masks=False)["appended"])
        self.assertIn("No spot", retouch_decision([], masks=True)["reason"])
        self.assertFalse(retouch_decision([spot], masks=True)["appended"])
        self.assertIn("spot", retouch_decision([spot], masks=False)["reason"].lower())

    def test_sidecar_records_the_ellipse_and_does_not_attach(self) -> None:
        document = ellipse_sidecar(
            [{"target": "face", "ellipse": {"cx": 0.5, "cy": 0.4, "rx": 0.2, "ry": 0.3, "feather": 0.2}}]
        )
        self.assertIn("0.5", document)
        self.assertFalse(sidecar_attaches_mask(document))


class PipelineVersionTests(unittest.TestCase):
    def test_a_run_writes_a_version_and_choices_without_applying_a_second_filmic(self) -> None:
        pixels = np.full((64, 64, 3), 0.4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "frame.png"
            Image.fromarray((pixels * 255).astype(np.uint8), mode="RGB").save(path)
            out = folder / "out"
            recipe = analyze(path, detector="heuristic", advisor="histogram", output_dir=out)
            write_outputs(recipe, out)
            record = json.loads((out / "version.json").read_text(encoding="utf-8"))
            self.assertEqual(record["version"], 1)
            self.assertTrue(record["sentence"])
            self.assertTrue((out / "masks.xmp").is_file())
            choices = json.loads((out / "output.json").read_text(encoding="utf-8"))["choices"]
            self.assertEqual(choices[0]["id"], "faithful")
            operations = [item["operation"] for item in recipe["modules"]]
            self.assertNotIn("filmicrgb", operations)
            self.assertNotIn("sigmoid", operations)

    def test_a_picked_look_keeps_the_skip_written_in_this_folder(self) -> None:
        pixels = np.full((64, 64, 3), 0.4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "frame.png"
            Image.fromarray((pixels * 255).astype(np.uint8), mode="RGB").save(path)
            out = folder / "out"
            write_outputs(
                analyze(path, detector="heuristic", advisor="histogram", output_dir=out, steer="skip the eyes"),
                out,
            )
            followed = analyze(
                path,
                detector="heuristic",
                advisor="histogram",
                output_dir=out,
                look="subject-lift",
            )
            self.assertIn("eyes", followed["plan"]["skip"])
            asks = {(item["verb"], item["target"]) for item in followed["plan"]["asks"]}
            self.assertIn(("lift", "subject"), asks)

    def test_history_can_be_read_without_writing_the_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "library.db"
            connection = sqlite3.connect(library)
            connection.execute("CREATE TABLE history (imgid INTEGER, operation TEXT)")
            connection.execute("INSERT INTO history (imgid, operation) VALUES (7, 'filmicrgb')")
            connection.commit()
            connection.close()
            found = history_operations(library, 7)
            self.assertEqual(found, {"filmicrgb"})
            self.assertIsNone(
                choose_display(found, look="sigmoid", vibe="", picture_type="night", problems=["blown-highlights"])
            )


class MaskedModuleTests(unittest.TestCase):
    def test_a_mask_attaches_subject_clarity_skin_and_blurs(self) -> None:
        class _Mask:
            def __init__(self, **kwargs: float) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class _Guide:
            masks = {
                "eyes": _Mask(clarity=0.4),
                "subject": _Mask(clarity=0.1),
                "skin": _Mask(clarity=-0.35),
                "background": _Mask(defocus=0.8),
            }

        modules: list[dict] = []
        blobs: list[dict] = []
        append_extra_modules(
            modules,
            blobs,
            atmosphere=Atmosphere(),
            picture_type="portrait",
            reading={"problems": [], "regions": []},
            plan={},
            separation={"stages_ran": ["subject", "eyes", "skin", "background"], "subject_ev": 0.25},
            guide=_Guide(),
            history_ops=set(),
            look="",
            vibe="",
            masks=True,
        )
        names = [item["multi_name"] for item in modules]
        self.assertEqual(
            names,
            ["AI subject exposure", "AI clarity", "AI skin", "AI blurs"],
        )
        self.assertEqual(len(blobs), 4)
        self.assertIsNone(diffuse_params(sharpness=0.0, iterations=8, radius=8))

    def test_a_sentence_turns_on_the_expert_modules(self) -> None:
        plan = revise_plan(empty_plan([]), "lift the shadows, warmer, more color, gradient")
        plan["falloff"] = {
            "axis": "vertical",
            "bright": "top",
            "position": 0.28,
            "strength": -0.22,
            "gap": 0.45,
        }
        atmosphere = Atmosphere()
        apply_expert_requests(atmosphere, plan)
        modules: list[dict] = []
        blobs: list[dict] = []
        append_extra_modules(
            modules,
            blobs,
            atmosphere=atmosphere,
            picture_type="portrait",
            reading={"problems": [], "regions": []},
            plan=plan,
            separation={"stages_ran": []},
            guide=None,
            history_ops=set(),
            look="",
            vibe="",
            masks=False,
        )
        names = [item["operation"] for item in modules]
        self.assertIn("toneequal", names)
        self.assertIn("channelmixerrgb", names)
        self.assertIn("colorzones", names)
        self.assertIn("graduatednd", names)
        self.assertNotIn("filmicrgb", names)
        self.assertNotIn("sigmoid", names)
        gradient = next(item for item in modules if item["operation"] == "graduatednd")
        self.assertLess(gradient["params"]["offset"], 40.0)
        self.assertGreater(gradient["params"]["hardness"], 20.0)
        self.assertFalse(
            abs(gradient["params"]["rotation"]) < 1.0 and abs(gradient["params"]["offset"] - 50.0) < 2.0
        )

    def test_a_gradient_with_no_direction_is_left_out(self) -> None:
        plan = revise_plan(empty_plan([]), "gradient")
        modules: list[dict] = []
        blobs: list[dict] = []
        append_extra_modules(
            modules,
            blobs,
            atmosphere=Atmosphere(),
            picture_type="night",
            reading={"problems": [], "regions": []},
            plan=plan,
            separation={"stages_ran": [], "gradient": {"skipped": True}},
            guide=None,
            history_ops=set(),
            look="",
            vibe="",
            masks=False,
        )
        self.assertNotIn("graduatednd", [item["operation"] for item in modules])
        self.assertIn("no measured direction", str(plan.get("gradient_note") or "").lower())


if __name__ == "__main__":
    unittest.main()
