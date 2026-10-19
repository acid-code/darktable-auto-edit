"""The status box is a label (no keyboard focus, normal contrast) that wraps itself."""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LUA = shutil.which("lua") or shutil.which("lua5.4") or shutil.which("lua5.3")

SCRIPT = r'''
local src = io.open(arg[1]):read("a")
local body = "local request_ui = {}\n" .. src:match("(request_ui.status_columns = 46.-\nend\n)")
local wrap = load(body .. "\nreturn request_ui.wrap_status")()
local long = string.rep("word ", 40) .. "\nAccept: Lift the parrot a little and crisp its feathers " .. string.rep("x", 60)
local out = wrap(long)
for line in (out .. "\n"):gmatch("(.-)\n") do assert(#line <= 46, "too wide") end
assert(out:gsub("%s+", "") == long:gsub("%s+", ""), "text lost")
print("ok")
'''


class StatusLabelTests(unittest.TestCase):
    def test_status_is_a_label_not_a_text_view(self) -> None:
        lua = (ROOT / "ai_editor.lua").read_text(encoding="utf-8")
        self.assertIn('request_ui.status_widget = dt.new_widget("label")', lua)
        self.assertNotIn('status_label = dt.new_widget("text_view")', lua)
        self.assertNotIn("sensitive = false", lua)

    @unittest.skipUnless(LUA, "no lua interpreter")
    def test_wrap_keeps_every_word_and_the_width(self) -> None:
        done = subprocess.run([LUA, "-", str(ROOT / "ai_editor.lua")], input=SCRIPT, capture_output=True, text=True)
        self.assertEqual(done.stdout.strip(), "ok", done.stderr)


if __name__ == "__main__":
    unittest.main()
