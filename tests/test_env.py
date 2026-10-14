"""Repo-root .env loading. These tests do not call Gemini or Mistral."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_dt.env import load_project_env, project_root


class EnvFileTests(unittest.TestCase):
    def test_project_root_is_the_processor_directory(self) -> None:
        root = project_root()
        self.assertTrue((root / "ai_processor.py").is_file())
        self.assertTrue((root / ".env.example").is_file())

    def test_missing_file_does_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(load_project_env(Path(tmp) / ".env"))

    def test_existing_environment_wins_over_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "OLLAMA_HOST=127.0.0.1:11434\nGEMINI_API_KEY=from-file\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"OLLAMA_HOST": "127.0.0.1:9", "GEMINI_API_KEY": ""},
                clear=False,
            ):
                self.assertTrue(load_project_env(path))
                self.assertEqual(os.environ["OLLAMA_HOST"], "127.0.0.1:9")
                self.assertEqual(os.environ["GEMINI_API_KEY"], "")

    def test_a_second_load_does_not_replace_an_empty_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GEMINI_API_KEY", None)
                self.assertTrue(load_project_env(path))
                self.assertEqual(os.environ["GEMINI_API_KEY"], "from-file")
                with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
                    load_project_env(path)
                    self.assertEqual(os.environ["GEMINI_API_KEY"], "")

    def test_file_fills_a_key_that_is_not_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("AI_EDITOR_REASONER=off\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AI_EDITOR_REASONER", None)
                self.assertTrue(load_project_env(path))
                self.assertEqual(os.environ["AI_EDITOR_REASONER"], "off")
