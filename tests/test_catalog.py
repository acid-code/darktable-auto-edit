"""Catalog choice, the technical brief, and dropped operations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dt.catalog import apply_operations, catalog_text, technical_brief, validate_operations
from ai_dt.reasoner import interpret_reply
from ai_dt.status_text import panel_status
from ai_dt.steer import empty_plan


class CatalogTests(unittest.TestCase):
    def test_unknown_operation_is_named_and_a_known_one_reorders_stages(self) -> None:
        kept, dropped = validate_operations(
            {
                "operations": [
                    {"id": "retouch", "action": "include", "reason": "Heal a spot."},
                    {"id": "subject", "action": "lift", "reason": "The face is dark."},
                    {"id": "exposure", "action": "include", "reason": "Open the frame."},
                ]
            }
        )
        self.assertEqual(dropped, ["retouch"])
        self.assertEqual([item["id"] for item in kept], ["subject", "exposure"])
        plan = apply_operations(empty_plan(["grade", "subject", "eyes", "skin", "background"]), kept, dropped)
        self.assertEqual(plan["dropped"], ["retouch"])
        self.assertTrue(plan["catalog_chosen"])
        self.assertEqual(plan["history_chosen"], ["exposure"])
        self.assertIn("subject", plan["guide_stages"])
        self.assertNotIn("eyes", plan["skip"])
        self.assertLess(plan["guide_stages"].index("subject"), plan["guide_stages"].index("eyes"))
        skipped = apply_operations(
            empty_plan(["grade", "subject", "eyes"]),
            [{"id": "eyes", "action": "skip", "reason": "Leave the eyes."}],
        )
        self.assertIn("eyes", skipped["skip"])

    def test_the_brief_names_the_grade_and_the_catalog_names_the_mask_limit(self) -> None:
        text = technical_brief(
            {"exposure_ev": 0.2, "contrast": 0, "temperature_shift": -0.1, "tint": 0, "shadows": 0.1, "highlights": -0.2},
            subject_brightness=0.3,
            frame_brightness=0.55,
            stages=["subject"],
        )
        self.assertIn("+0.20 EV", text)
        self.assertIn("already run: subject", text)
        self.assertIn("subject brightness 0.30", text)
        catalog = catalog_text()
        self.assertIn("colorbalancergb", catalog)
        self.assertIn("edited.tif", catalog)
        self.assertIn("drawn mask", catalog)
        self.assertIn("diffuse", catalog)

    def test_a_reply_keeps_operations_and_the_paragraph(self) -> None:
        reply = interpret_reply(
            '{"reasoning": "Lift the face.", "operations": [{"id": "subject", "action": "lift", "reason": "The face is dark."}, {"id": "filmic", "action": "include", "reason": "No."}]}'
        )
        self.assertEqual(reply["paragraph"], "Lift the face.")
        self.assertEqual(reply["operations"][0]["id"], "subject")
        self.assertEqual(reply["dropped"], ["filmic"])

    def test_the_panel_names_the_stages_that_survived_and_what_was_dropped(self) -> None:
        text = panel_status(
            {
                "plan_backend": "Plan: Gemini gemini-3.5-flash-lite",
                "plan": {
                    "reasoning": {"paragraph": "Lift the face."},
                    "guide_stages": ["grade", "subject", "eyes"],
                    "skip": ["eyes"],
                    "dropped": ["retouch"],
                },
            }
        )
        self.assertIn("Lift the face.", text)
        self.assertIn("Stages: subject.", text)
        self.assertIn("Dropped: retouch.", text)


if __name__ == "__main__":
    unittest.main()
