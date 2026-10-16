"""The run-through keeps found stages, and a miss does not finish the edit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_dt.guides.decide import Picture
from ai_dt.guides.library import match_guide
from ai_dt.runbook import assess_run, lock_found_stages, original_behind_edit
from ai_dt.schema import Atmosphere, Detection
from ai_dt.separate import render_finish
from ai_dt.steer import parse_steer, revise_plan, empty_plan


def _portrait() -> Picture:
    face = (0.3, 0.2, 0.3, 0.5)
    return Picture(
        type="portrait",
        reason="A face is the clear subject, so this is a portrait.",
        brightness=0.45,
        warmth=0.05,
        saturation=0.1,
        edges=0.04,
        line_fraction=0.01,
        sky=0.0,
        skin=0.4,
        highlight_specks=0.0,
        compact_area=0.15,
        compact_fill=0.7,
        subject_box=face,
    )


class LockTests(unittest.TestCase):
    def test_a_reasoner_skip_of_the_eyes_is_dropped_when_a_face_was_found(self) -> None:
        plan = empty_plan(["grade", "subject", "eyes", "skin", "background"])
        plan["skip"] = ["eyes"]
        plan["skip_by"] = {"eyes": "reasoner"}
        plan["proposed"] = [{"stage": "eyes", "action": "skip", "reason": "No eyes.", "source": "reasoner"}]
        locked = lock_found_stages(
            plan,
            _portrait(),
            [Detection("face", "face", 0.9, (0.3, 0.2, 0.3, 0.5), "test")],
        )
        self.assertNotIn("eyes", locked["skip"])
        self.assertEqual(locked["kept_stages"], ["eyes"])
        self.assertEqual(locked["proposed"], [])

    def test_a_user_skip_of_the_eyes_stays(self) -> None:
        plan = empty_plan(["grade", "subject", "eyes", "skin", "background"])
        plan["skip"] = ["eyes"]
        plan["skip_by"] = {"eyes": "user"}
        locked = lock_found_stages(
            plan,
            _portrait(),
            [Detection("face", "face", 0.9, (0.3, 0.2, 0.3, 0.5), "test")],
        )
        self.assertEqual(locked["skip"], ["eyes"])


class FinishTests(unittest.TestCase):
    def test_a_portrait_missing_the_eyes_is_not_finished(self) -> None:
        report = assess_run(
            "portrait",
            {"stages_ran": ["grade", "subject", "skin", "background"], "skipped": ["eyes"]},
        )
        self.assertFalse(report["finished"])
        self.assertEqual(report["missing"], ["eyes"])
        self.assertIn("Not finished", report["sentence"])

    def test_a_night_with_no_highlights_to_hold_is_finished(self) -> None:
        report = assess_run("night", {"stages_ran": ["grade"], "skipped": ["highlights"]})
        self.assertTrue(report["finished"])

    def test_a_landscape_needs_the_sky_or_the_horizon(self) -> None:
        missing = assess_run("landscape", {"stages_ran": ["grade", "foreground"]})
        found = assess_run("landscape", {"stages_ran": ["grade", "horizon"]})
        self.assertFalse(missing["finished"])
        self.assertTrue(found["finished"])


class FalloffTests(unittest.TestCase):
    def test_the_corner_outside_a_face_gets_darker(self) -> None:
        pixels = np.full((120, 120, 3), 0.55, dtype=np.float32)
        pixels[30:90, 40:80] = (0.62, 0.48, 0.40)
        face = (40 / 120, 30 / 120, 40 / 120, 60 / 120)
        picture = _portrait()
        picture = Picture(
            type="portrait",
            reason=picture.reason,
            brightness=0.5,
            warmth=0.05,
            saturation=0.1,
            edges=0.04,
            line_fraction=0.01,
            sky=0.0,
            skin=0.4,
            highlight_specks=0.0,
            compact_area=0.16,
            compact_fill=0.7,
            subject_box=face,
        )
        rendered, info = render_finish(
            pixels,
            Atmosphere(),
            [Detection("face", "face", 0.9, face, "test")],
            guide=match_guide("portrait").guide,
            picture=picture,
        )
        self.assertIn("falloff", info["stages_ran"])
        self.assertLess(float(rendered[2:8, 2:8].mean()), float(pixels[2:8, 2:8].mean()) - 0.01)


class ScaleTests(unittest.TestCase):
    def test_more_eye_clarity_raises_the_eye_scale(self) -> None:
        parsed = parse_steer("more eye clarity")
        self.assertEqual(parsed["scales"], [{"stage": "eyes", "factor": 1.35}])
        plan = revise_plan(empty_plan(["grade", "eyes"]), "more eye clarity")
        self.assertAlmostEqual(plan["scales"]["eyes"], 1.35)

    def test_less_on_the_eyes_is_not_an_amount(self) -> None:
        parsed = parse_steer("less on the eyes")
        self.assertEqual(parsed["scales"], [])


class FollowBackTests(unittest.TestCase):
    def test_an_opened_copy_rereads_the_camera_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera = root / "camera.png"
            camera.write_bytes(b"not-a-real-png-but-present")
            opened = root / "opened"
            opened.mkdir()
            copy = opened / "faithful-1.tif"
            copy.write_bytes(b"tif")
            recipe = {"input": {"path": str(camera)}}
            (root / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            self.assertEqual(original_behind_edit(copy), camera)
            stray_dir = root / "other"
            stray_dir.mkdir()
            stray = stray_dir / "note.tif"
            stray.write_bytes(b"x")
            self.assertIsNone(original_behind_edit(stray))


if __name__ == "__main__":
    unittest.main()
