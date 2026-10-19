"""The panel has no Looks button, vibe box, preset dropdown or Look line."""

from __future__ import annotations

import unittest
from pathlib import Path

LUA = (Path(__file__).resolve().parents[1] / "ai_editor.lua").read_text(encoding="utf-8")


class NoLooksUiTests(unittest.TestCase):
    def test_widgets_are_gone(self) -> None:
        for gone in ('label = "Looks"', 'label = "Vibe"', 'label = "Preset"', '"Look: —"\n}', "looks_button = dt.new_widget",
                     "look_name_label = dt.new_widget", "vibe_entry = dt.new_widget", "combobox"):
            self.assertNotIn(gone, LUA)

    def test_panel_and_status_do_not_name_looks(self) -> None:
        panel = LUA[LUA.index('local panel = dt.new_widget("box") {'):]
        panel = panel[: panel.index("\n}\n")]
        for gone in ("vibe_label", "look_row", "look_name_label"):
            self.assertNotIn(gone, panel)
        reachable = LUA.replace(LUA[LUA.index("local function open_looks()"):LUA.index("\nend\n", LUA.index("local function open_looks()"))], "")
        self.assertNotIn("press Looks", reachable)
        self.assertIn('if vibe_entry == nil and preset_box == nil then\n    return ""', LUA)


if __name__ == "__main__":
    unittest.main()
