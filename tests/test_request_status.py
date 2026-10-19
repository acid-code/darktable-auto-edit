"""Accept/Skip always name their step; an edited request keeps the plan. No darktable.

Bug 3: rewriting the request text made the plan disappear while Accept and
Skip stayed on, so it was unclear what they applied to. The plan now stays
the current plan until Ask returns a new one, the status says the request
changed, and every Accept/Skip state names its step (or says it is off).
The Python helpers write the lines; ``ai_editor.lua`` swaps them while the
request box is edited, which is checked here with lua5.4 when available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dt.features import commit_waiting_step, plan_chat, skip_waiting_step
from ai_dt.status_text import (
    accept_skip_lines,
    accept_skip_state,
    panel_status,
    request_changed,
)

WAITING = {
    "chat": True,
    "features_next": "mask_background",
    "features_next_title": "Quiet the water around the duck",
    "plan_request": "finish the duck",
    "plan": {
        "features_done": ["subject"],
        "features_path": ["mask_background", "foreground"],
        "step_queue": [
            {"id": "mask_background", "title": "Quiet the water around the duck"},
            {"id": "foreground", "title": "Darken the foreground"},
        ],
    },
}


class AcceptSkipStateTests(unittest.TestCase):
    def test_unchanged_request_names_the_step(self) -> None:
        state = accept_skip_state(WAITING, typed="  Finish   the duck ")
        self.assertTrue(state["enabled"])
        self.assertFalse(state["changed"])
        self.assertEqual(state["line"], "Accept/Skip apply to: Quiet the water around the duck")

    def test_changed_request_keeps_the_step(self) -> None:
        state = accept_skip_state(WAITING, typed="make the sky moodier")
        self.assertTrue(state["enabled"])
        self.assertTrue(state["changed"])
        self.assertEqual(
            state["line"],
            "Request changed — press Ask for a new plan; Accept/Skip still apply to: Quiet the water around the duck",
        )

    def test_no_step_turns_the_buttons_off_and_says_so(self) -> None:
        empty = {**WAITING, "features_next": "", "features_next_title": ""}
        state = accept_skip_state(empty, typed="finish the duck")
        self.assertFalse(state["enabled"])
        self.assertIn("No step is waiting, so Accept and Skip are off", state["line"])
        changed = accept_skip_state(empty, typed="something else")
        self.assertIn("Request changed — press Ask", changed["line"])
        self.assertIn("Accept and Skip are off", changed["line"])

    def test_older_recipes_without_a_recorded_request_count_as_unchanged(self) -> None:
        legacy = {key: value for key, value in WAITING.items() if key != "plan_request"}
        self.assertFalse(request_changed(legacy, "anything"))
        self.assertFalse(request_changed(WAITING, None))

    def test_panel_marks_the_current_plan_and_names_the_step(self) -> None:
        text = panel_status(WAITING)
        self.assertIn("Current plan (from \u201cfinish the duck\u201d)", text)
        self.assertIn("now — Quiet the water around the duck", text)
        self.assertTrue(text.endswith("Accept/Skip apply to: Quiet the water around the duck"))
        lines = accept_skip_lines(WAITING)
        self.assertEqual(lines["line"], "Accept/Skip apply to: Quiet the water around the duck")
        self.assertTrue(lines["changed_line"].startswith("Request changed — press Ask"))


def _duck(directory: Path) -> None:
    recipe = {
        "features_next": "",
        "reading": {"sentence": "A duck on a pond."},
        "scene": {"summary": "A duck on a pond.", "source": "mistral"},
        "analysis": {"atmosphere": {}},
        "separation": {"stages_ran": ["grade"]},
        "plan": {"guide_stages": ["grade"], "skip": [], "features_done": [], "features_path": [], "reasoning": {}, "asks": []},
    }
    (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")


def _reply(*ids: str) -> dict:
    return {
        "ask_ollama": False,
        "proposal": {
            "status": "gemini",
            "backend": "gemini-test",
            "paragraph": "",
            "stages": [],
            "operations": [{"id": ident, "action": "include", "reason": ident} for ident in ids],
            "features": [],
        },
    }


class PlanLifecycleTests(unittest.TestCase):
    def test_ask_records_the_request_and_the_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _duck(directory)
            with patch("ai_dt.reasoner.propose_gemini", return_value=_reply("mask_background", "foreground")):
                plan_chat(directory, "  finish  the duck ")
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["plan_request"], "finish the duck")
            self.assertTrue(saved["accept_skip"]["enabled"])
            self.assertIn(saved["accept_skip"]["step"], saved["accept_skip"]["line"])
            self.assertIn(saved["accept_skip"]["line"], saved["status_text"])

    def test_accept_and_skip_update_the_named_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _duck(directory)
            with patch("ai_dt.reasoner.propose_gemini", return_value=_reply("highlights", "split_tone")):
                plan_chat(directory, "warm it")
            first = json.loads((directory / "output.json").read_text(encoding="utf-8"))["accept_skip"]
            skip_waiting_step(directory)
            second = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertNotEqual(first["step"], second["accept_skip"]["step"])
            self.assertIn(second["accept_skip"]["line"], second["status_text"])
            skip_waiting_step(directory)
            done = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertFalse(done["accept_skip"]["enabled"])
            self.assertIn("Accept and Skip are off", done["status_text"])

    def test_a_request_that_adds_nothing_keeps_the_current_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _duck(directory)
            with patch("ai_dt.reasoner.propose_gemini", return_value=_reply("mask_background", "foreground")):
                plan_chat(directory, "finish the duck")
            before = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            with patch("ai_dt.reasoner.propose_gemini", return_value=_reply()):
                plan_chat(directory, "hmm, not sure")
            after = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(after["features_next"], before["features_next"])
            self.assertEqual(after["plan"]["features_path"], before["plan"]["features_path"])
            self.assertIn("current plan stays", after["feature_note"])
            self.assertTrue(after["accept_skip"]["enabled"])


LUA_HARNESS = r"""
local chunk = io.open(arg[1]):read("a")
local start = chunk:find("-- The request box has no change signal", 1, true)
local finish = chunk:find("local function failure_text", 1, true)
local body = chunk:sub(start, finish - 1)
local env = setmetatable({}, { __index = _G })
env.steer_entry = { text = arg[2] }
env.panel_body = function(recipe) return recipe.status_text or "" end
local loader = assert(load(body .. "\nreturn plan_status, request_ui", "plan_status", "t", env))
local plan_status, request_ui = loader()
local recipe = assert(require_json(arg[3]))
io.write(plan_status(recipe))
"""


class LuaParityTests(unittest.TestCase):
    def test_lua_swaps_the_line_when_the_request_changes(self) -> None:
        lua = shutil.which("lua5.4") or shutil.which("lua")
        if not lua:
            self.skipTest("no Lua interpreter")
        recipe = dict(WAITING)
        recipe["status_text"] = panel_status(recipe)
        recipe["accept_skip"] = accept_skip_lines(recipe)
        with tempfile.TemporaryDirectory() as tmp:
            recipe_path = Path(tmp) / "output.json"
            recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
            harness = Path(tmp) / "harness.lua"
            harness.write_text(
                f"local json = dofile({json.dumps(str(ROOT / 'lib' / 'json.lua'))})\n"
                "function require_json(path) return json.decode(io.open(path):read('a')) end\n" + LUA_HARNESS,
                encoding="utf-8",
            )

            def run(typed: str) -> str:
                done = subprocess.run(
                    [lua, str(harness), str(ROOT / "ai_editor.lua"), typed, str(recipe_path)],
                    capture_output=True, text=True, timeout=30, check=True,
                )
                return done.stdout

            same = run("Finish  the duck")
            self.assertTrue(same.endswith("Accept/Skip apply to: Quiet the water around the duck"))
            self.assertIn("Current plan", same)
            changed = run("make the sky moodier")
            self.assertIn("Current plan", changed)
            self.assertIn("now — Quiet the water around the duck", changed)
            self.assertTrue(
                changed.endswith(
                    "Request changed — press Ask for a new plan; Accept/Skip still apply to: Quiet the water around the duck"
                )
            )
            self.assertNotIn("\nAccept/Skip apply to:", changed)


if __name__ == "__main__":
    unittest.main()
