"""Complete plan orchestration (ai_editor.lua ``complete_plan_run``) with mocked darktable."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

try:
    import lupa
except ImportError:  # pragma: no cover
    lupa = None

ROOT = Path(__file__).resolve().parents[1]

DRIVER = r"""
local run = ...
return function(plan, mode)
  local hist, i, note = 5, 1, nil
  local deps = {
    next_step = function() return plan[i], plan[i] and ("Title " .. plan[i]) end,
    history = function() return hist end,
    note = function() return note end,
    accept = function(id)
      local m = mode[id] or "ok"
      if m == "ok" then hist = hist + 1; i = i + 1; return nil end
      if m == "noop" then i = i + 1; return nil end
      if m == "leftoff" then note = "No animal was found."; i = i + 1; return nil end
      if m == "soft" then i = i + 1; return false, "left off: no mask" end
      if m == "fail" then return false, "worker failed" end
      if m == "stuck" then hist = hist + 1; return nil end
      error("boom in " .. id)
    end,
  }
  local done, text, results = run(deps)
  local states = {}
  for k, r in ipairs(results) do states[k] = r.state end
  return done, text, table.concat(states, ","), hist
end
"""


@unittest.skipIf(lupa is None, "lupa not installed")
class CompletePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (ROOT / "ai_editor.lua").read_text(encoding="utf-8")
        block = re.search(r"-- BEGIN complete_plan_run\n(.*?)-- END complete_plan_run", source, re.S).group(1)
        lua = lupa.LuaRuntime()
        run = lua.execute("local request_ui = {}\n" + block + "\nreturn request_ui.complete_plan_run")
        cls.drive = lua.execute(DRIVER, run)
        cls.lua = lua

    def go(self, plan, mode=None):
        return self.drive(self.lua.table_from(plan), self.lua.table_from(mode or {}))

    def test_happy_path_applies_all_and_checks_history(self):
        done, text, states, hist = self.go(["a", "b", "c"])
        self.assertTrue(done)
        self.assertEqual(states, "applied,applied,applied")
        self.assertEqual(hist, 8)
        self.assertIn("Plan complete: 3 applied", text)
        self.assertIn("history 5 → 6", text)

    def test_no_history_change_is_a_failure(self):
        done, text, states, _ = self.go(["a", "b", "c"], {"b": "noop"})
        self.assertFalse(done)
        self.assertEqual(states, "applied,failed")
        self.assertIn("did not change", text)
        self.assertIn("Title b", text)

    def test_left_off_with_note_is_soft(self):
        done, text, states, _ = self.go(["a", "b", "c"], {"b": "leftoff"})
        self.assertTrue(done)
        self.assertEqual(states, "applied,skipped,applied")
        self.assertIn("No animal was found.", text)

    def test_soft_false_moves_on(self):
        done, _, states, _ = self.go(["a", "b"], {"a": "soft"})
        self.assertTrue(done)
        self.assertEqual(states, "skipped,applied")

    def test_hard_failure_stops(self):
        done, text, states, _ = self.go(["a", "b", "c"], {"b": "fail"})
        self.assertFalse(done)
        self.assertEqual(states, "applied,failed")
        self.assertIn("worker failed", text)

    def test_error_is_recorded_and_stops(self):
        done, text, states, _ = self.go(["a", "b"], {"a": "error"})
        self.assertFalse(done)
        self.assertEqual(states, "failed")
        self.assertIn("boom in a", text)

    def test_step_that_stays_waiting_stops(self):
        done, text, states, _ = self.go(["a"], {"a": "stuck"})
        self.assertFalse(done)
        self.assertIn("still waiting", text)

    def test_nothing_waiting(self):
        done, text, states, _ = self.go([])
        self.assertTrue(done)
        self.assertIn("nothing to complete", text)

    def test_button_sits_with_accept(self):
        source = (ROOT / "ai_editor.lua").read_text(encoding="utf-8")
        self.assertRegex(source, r"accept_button,\s*skip_button,\s*request_ui\.complete_plan_button,")


if __name__ == "__main__":
    unittest.main()
