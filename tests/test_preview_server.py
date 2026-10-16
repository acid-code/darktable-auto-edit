"""The preview page is opened only after the local server answers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import preview_server


class EnsureTests(unittest.TestCase):
    def test_a_live_server_is_left_alone(self) -> None:
        with (
            patch.object(preview_server, "server_responds", return_value=True),
            patch.object(preview_server, "_spawn_server") as spawn,
        ):
            self.assertTrue(preview_server.ensure_running())
        spawn.assert_not_called()

    def test_a_silent_port_is_started_and_then_replaced_once(self) -> None:
        with (
            patch.object(preview_server, "server_responds", return_value=False),
            patch.object(preview_server, "_wait_until_up", side_effect=[False, True]),
            patch.object(preview_server, "_spawn_server") as spawn,
            patch.object(preview_server, "_stop_stuck_preview") as stop,
            patch.object(preview_server.time, "sleep"),
        ):
            self.assertTrue(preview_server.ensure_running())
        self.assertEqual(spawn.call_count, 2)
        stop.assert_called_once()

    def test_only_a_stuck_preview_process_is_stopped(self) -> None:
        def command_line(pid: int) -> str:
            if pid == 42:
                return r"C:\Python\python.exe preview_server.py"
            return r"C:\nginx\nginx.exe"

        with (
            patch.object(preview_server, "_listening_pids", return_value=[42, 99, os_pid()]),
            patch.object(preview_server, "_command_line", side_effect=command_line),
            patch.object(preview_server.subprocess, "run") as run,
        ):
            preview_server._stop_stuck_preview()
        killed = [call.args[0] for call in run.call_args_list]
        self.assertEqual(killed, [["taskkill", "/PID", "42", "/F"]])


def os_pid() -> int:
    import os

    return os.getpid()
