"""Offline plans: without a planner the guide and the request still make steps."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from ai_dt.offline_plan import offline_operations, planner_missing
from ai_dt.reasoner import skipped
from ai_dt.status_text import plan_backend_line

ROOT = Path(__file__).resolve().parents[1]


class OfflinePlanTests(unittest.TestCase):
    def test_a_skipped_or_empty_proposal_is_no_planner(self) -> None:
        self.assertTrue(planner_missing(skipped("Reasoning was skipped. No local model and no Gemini key.")))
        self.assertTrue(planner_missing(None))
        self.assertTrue(planner_missing({"status": "gemini", "stages": []}))
        self.assertFalse(planner_missing({"status": "gemini", "stages": [{"stage": "subject", "action": "lift"}]}))

    def test_duck_main_theme_gets_the_animal_steps(self) -> None:
        ops = offline_operations("make the duck the main theme", picture_type="animal", animal=True, ground=True)
        ids = [item["id"] for item in ops]
        for step in ("mask_animal", "mask_animal_background", "mask_bottom_gradient", "mask_edge_burn"):
            self.assertIn(step, ids)

    def test_dark_ground_has_no_bottom_gradient(self) -> None:
        ids = [item["id"] for item in offline_operations("", picture_type="animal", animal=True, ground=False)]
        self.assertNotIn("mask_bottom_gradient", ids)

    def test_water_darker_is_the_water_step_first(self) -> None:
        ops = offline_operations("make the water darker", picture_type="animal", animal=True, ground=True)
        self.assertEqual(ops[0]["id"], "mask_water")
        self.assertEqual(ops[0]["action"], "darken")

    def test_a_specific_follow_up_keeps_only_its_step(self) -> None:
        ops = offline_operations("make the water darker", picture_type="animal", animal=True, first=False)
        self.assertEqual([item["id"] for item in ops], ["mask_water"])

    def test_landscape_uses_the_lesson_preset(self) -> None:
        ids = [item["id"] for item in offline_operations("", picture_type="landscape")]
        self.assertIn("mask_landscape", ids)


class HeaderTests(unittest.TestCase):
    def test_offline_header_says_why_and_what_was_used(self) -> None:
        info = skipped("Reasoning was skipped. No local model and no Gemini key.")
        self.assertEqual(plan_backend_line(info), "Plan: offline (no Gemini key), from the guide and request")
        self.assertIn("description", plan_backend_line(info, description=True))
        self.assertNotIn("not used", plan_backend_line(info))

    def test_gemini_header_unchanged(self) -> None:
        self.assertEqual(plan_backend_line({"status": "gemini", "backend": "gemini-x"}), "Plan: Gemini gemini-x")


class ButtonWordsTests(unittest.TestCase):
    """Status text names only buttons that exist."""

    def test_status_strings_name_real_buttons(self) -> None:
        lua = (ROOT / "ai_editor.lua").read_text(encoding="utf-8")
        labels = set(re.findall(r'^\s*label = "([^"]+)",', lua, re.MULTILINE))
        self.assertIn("Ask", labels)
        self.assertNotIn("Propose", labels)
        sources = [lua] + [(ROOT / "ai_dt" / name).read_text(encoding="utf-8")
                           for name in ("status_text.py", "features.py", "pipeline.py")]
        for text in sources:
            for word in re.findall(r"press ([A-Z][a-z]+)", text):
                self.assertIn(word, labels | {"Accept", "Skip"}, word)
            self.assertNotIn("Propose looks", text)


class DescriptionTests(unittest.TestCase):
    def test_default_description_is_facts_not_advice(self) -> None:
        from types import SimpleNamespace

        from ai_dt.pipeline import _local_description

        picture = SimpleNamespace(type="animal", brightness=0.6, warmth=0.0)
        text = _local_description(picture, [SimpleNamespace(label="bird", category="animal")])
        self.assertIn("animal photo with bird", text)
        self.assertNotIn("Leave", text)


if __name__ == "__main__":
    unittest.main()
