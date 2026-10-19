"""The New plan button forgets the saved plan only (no darktable history, no sidecar)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LUA = (ROOT / "ai_editor.lua").read_text(encoding="utf-8")


class NewPlanTests(unittest.TestCase):
    def test_label_and_tooltip(self) -> None:
        block = LUA[LUA.index("request_ui.new_plan_button = dt.new_widget(\"button\")"):]
        block = block[: block.index("\n}\n")]
        self.assertIn('label = "New plan",', block)
        self.assertIn(
            "tooltip = \"Forget this photo's saved plan and start fresh. Your darktable edits stay as they are.\"",
            block,
        )
        self.assertIn("request_ui.new_plan_button,", LUA)

    def test_only_plan_files_are_removed(self) -> None:
        listed = LUA[LUA.index("request_ui.plan_files = {"):]
        names = re.findall(r'"([^"]+)"', listed[: listed.index("}")])
        self.assertIn("output.json", names)
        for kept in ("edited.tif", "preview.jpg", ".xmp"):
            self.assertNotIn(kept, names)
        self.assertFalse(any(name.endswith(".jpg.xmp") or name.lower().endswith(".jpg") and name != "look.jpg" for name in names))
        body = LUA[LUA.index("function request_ui.new_plan()"):]
        body = body[: body.index("\nend\n")]
        self.assertNotIn("apply_sidecar", body)
        self.assertNotIn("sidecar_file", body)
        self.assertIn("press Ask", body)
        self.assertNotIn('steer_entry.text = ""', body)


if __name__ == "__main__":
    unittest.main()


class AskAfterNewPlanTests(unittest.TestCase):
    def test_ask_without_saved_reading_plans_from_scratch(self):
        source = (Path(__file__).resolve().parent.parent / "ai_editor.lua").read_text(encoding="utf-8")
        body = source.split("local function propose_image", 1)[1].split("\nend\n", 1)[0]
        self.assertIn('file_size(join_path(work, "output.json")) < 2', body)
        self.assertIn("asking = false", body)

