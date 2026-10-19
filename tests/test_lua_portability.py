"""ai_editor.lua on Linux/macOS: no Windows worker, no ``< NUL``.

The block from ``run_popen`` to ``run_captured`` is run with lua5.4 and stub
darktable functions. Off Windows, ``run_hidden`` and ``run_captured`` are a
plain popen with stdin from /dev/null, and ``ensure_worker`` returns at once
(it used to sleep about 2.5 s waiting for a Windows script host).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r"""
local chunk = io.open(arg[1]):read("a")
local start = chunk:find("local function run_popen(command)", 1, true)
local finish = chunk:find("local function browser_app()", 1, true)
local body = chunk:sub(start, finish - 1)
local sleeps, executes = 0, {}
local env = setmetatable({}, { __index = _G })
env.on_windows = function() return arg[2] == "windows" end
env.dt = { control = { sleep = function(ms) sleeps = sleeps + ms end } }
env.os = setmetatable({ execute = function(c) executes[#executes + 1] = c; return true end }, { __index = os })
env.show_progress = function() end
env.file_size = function() return 0 end
env.read_file = function() return nil end
env.write_file = function() return nil end
env.join_path = function(a, b) return a .. "/" .. b end
env.script_directory = function() return arg[3] end
env.mkdir_p = function() end
env.shell_quote = function(s) return "'" .. s .. "'" end
local loader = assert(load(body .. "\nreturn run_captured, run_hidden", "run_block", "t", env))
local run_captured, run_hidden = loader()
local run = arg[4] == "hidden" and run_hidden or run_captured
local out, code = run("cat; echo hi", arg[3])
out = (out or ""):gsub("\n", "|")
io.write(string.format("out=%s code=%s sleeps=%d executes=%d", out, tostring(code), sleeps, #executes))
"""


class LuaPortabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lua = shutil.which("lua5.4") or shutil.which("lua")
        if not self.lua:
            self.skipTest("no Lua interpreter")

    def _run(self, os_name: str, which: str = "captured") -> str:
        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "harness.lua"
            harness.write_text(HARNESS, encoding="utf-8")
            done = subprocess.run(
                [self.lua, str(harness), str(ROOT / "ai_editor.lua"), os_name, tmp, which],
                capture_output=True, text=True, timeout=30, check=True, stdin=subprocess.DEVNULL,
            )
            return done.stdout

    def test_linux_runs_a_plain_popen_without_waiting_for_a_worker(self) -> None:
        for which in ("captured", "hidden"):
            self.assertEqual(self._run("linux", which), "out=hi| code=0 sleeps=0 executes=0", which)

    def test_the_windows_path_is_unchanged(self) -> None:
        source = (ROOT / "ai_editor.lua").read_text(encoding="utf-8")
        self.assertIn('command .. " < NUL\\r\\n"', source)
        self.assertIn("windows-hidden-worker.vbs", source)
        # On "windows" the stubbed worker never answers, so it waits and
        # falls back to the cmd.exe form; it must still try the worker.
        result = self._run("windows", "hidden")
        self.assertIn("sleeps=", result)
        self.assertNotIn("sleeps=0 ", result)


if __name__ == "__main__":
    unittest.main()
