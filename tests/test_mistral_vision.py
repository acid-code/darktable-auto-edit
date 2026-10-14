"""Mistral vision advisor. The HTTP call is stubbed. No live API."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from ai_dt.advisors.mistral_vision import CHAT_URL, DEFAULT_MODEL, MistralVisionAdvisor
from ai_dt.errors import AdvisorUnavailable
from ai_dt.pipeline import _advise, analyze

PROMPT = (ROOT / "ai_dt" / "prompts" / "llava_lighting.txt").read_text(encoding="utf-8").strip()


def _reply(**overrides) -> dict:
    payload = {
        "mood": "epic strong light",
        "rationale": "The preview is a little dark.",
        "exposure_ev": 0.4,
        "contrast": 0.1,
        "temperature_shift": 0.0,
        "tint": 0.0,
        "saturation": 0.0,
        "shadows": 0.1,
        "highlights": -0.1,
        "vibrance": 0.0,
        "caps_open": True,
        "strong": True,
        "epic": True,
    }
    payload.update(overrides)
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


class _Body:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_Body":
        return self

    def __exit__(self, *args) -> bool:
        return False


class MistralVisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            key: os.environ.get(key)
            for key in ("MISTRAL_API_KEY", "MISTRAL_VISION_MODEL", "GEMINI_API_KEY", "OLLAMA_HOST")
        }
        os.environ["OLLAMA_HOST"] = "127.0.0.1:9"
        os.environ["GEMINI_API_KEY"] = ""
        os.environ.pop("MISTRAL_API_KEY", None)
        os.environ.pop("MISTRAL_VISION_MODEL", None)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _urlopen(self, payload: dict, calls: list):
        def fake_urlopen(request, timeout=None):
            calls.append(request)
            self.assertEqual(request.full_url, CHAT_URL)
            self.assertNotIn("generativelanguage.googleapis.com", request.full_url)
            self.assertIsNotNone(timeout)
            return _Body(payload)

        return fake_urlopen

    def test_auto_sends_the_picture_to_mistral_and_leaves_caps_closed(self) -> None:
        calls: list = []
        os.environ["MISTRAL_API_KEY"] = "test-key"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "plain.png"
            Image.new("RGB", (32, 24), (40, 40, 40)).save(path)
            with (
                patch("ai_dt.advisors.mistral_vision.urllib.request.urlopen", self._urlopen(_reply(), calls)),
                patch("ai_dt.pipeline.OllamaLLaVAAdvisor") as ollama,
            ):
                recipe = analyze(path, detector="heuristic", advisor="auto", output_dir=directory)
        ollama.assert_not_called()
        self.assertEqual(len(calls), 1)
        request = calls[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], DEFAULT_MODEL)
        content = body["messages"][0]["content"]
        self.assertEqual(content[0]["text"], PROMPT)
        self.assertEqual(content[1]["type"], "image_url")
        image_url = content[1]["image_url"]
        self.assertIsInstance(image_url, dict)
        self.assertTrue(str(image_url.get("url") or "").startswith("data:image/jpeg;base64,"))
        self.assertTrue(request.full_url.endswith("/v1/chat/completions"))
        self.assertNotIn("/v1/completions", request.full_url.replace("/v1/chat/completions", ""))
        self.assertNotIn("test-key", request.data.decode("utf-8"))
        self.assertIn(f"mistral:{DEFAULT_MODEL}", recipe["backends"]["advisor"])
        self.assertIn("histogram", recipe["backends"]["advisor"])
        self.assertEqual(recipe["analysis"]["llm"]["exposure_ev"], 0.4)
        self.assertEqual(recipe["analysis"]["llm"]["mood"], "epic strong light")
        self.assertFalse(recipe["plan"]["caps_open"])
        self.assertFalse(recipe["plan"]["strong"])
        self.assertFalse(recipe["plan"].get("strong_word"))

    def test_missing_key_does_not_call_mistral(self) -> None:
        calls: list = []

        def fail_urlopen(request, timeout=None):
            calls.append(request)
            raise AssertionError("Mistral was called without a key")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.png"
            Image.new("RGB", (16, 16), (30, 30, 30)).save(path)
            with patch("ai_dt.advisors.mistral_vision.urllib.request.urlopen", fail_urlopen):
                recipe = analyze(path, detector="heuristic", advisor="auto")
        self.assertEqual(calls, [])
        self.assertIn("histogram", recipe["backends"]["advisor"])
        self.assertNotIn("mistral:", recipe["backends"]["advisor"])
        self.assertIsNone(recipe["analysis"]["llm"])

    def test_http_failure_falls_back_to_histogram_and_ollama(self) -> None:
        os.environ["MISTRAL_API_KEY"] = "test-key"

        def fail_urlopen(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.png"
            Image.new("RGB", (16, 16), (30, 30, 30)).save(path)
            with patch("ai_dt.advisors.mistral_vision.urllib.request.urlopen", fail_urlopen):
                recipe = analyze(path, detector="heuristic", advisor="auto")
        notes = " ".join(recipe["backends"]["advisor_notes"])
        self.assertIn("Mistral request failed", notes)
        self.assertIn("Ollama", notes)
        self.assertIn("histogram", recipe["backends"]["advisor"])
        self.assertNotIn("mistral:", recipe["backends"]["advisor"])
        self.assertFalse(recipe["plan"]["caps_open"])

    def test_histogram_and_ollama_choices_skip_mistral(self) -> None:
        os.environ["MISTRAL_API_KEY"] = "test-key"
        calls: list = []

        def fail_urlopen(request, timeout=None):
            calls.append(request)
            raise AssertionError("Mistral was called for an explicit advisor")

        image = Image.new("RGB", (8, 8), (20, 20, 20))
        path = Path("unused.png")
        with patch("ai_dt.advisors.mistral_vision.urllib.request.urlopen", fail_urlopen):
            measured, suggestion, notes = _advise(image, path, "histogram", "llava", None)
        self.assertEqual(calls, [])
        self.assertIsNone(suggestion)
        self.assertEqual(notes, [])
        self.assertEqual(measured.source, "histogram")
        with patch("ai_dt.advisors.mistral_vision.urllib.request.urlopen", fail_urlopen):
            with self.assertRaises(AdvisorUnavailable) as caught:
                _advise(image, path, "ollama", "llava", None)
        self.assertEqual(calls, [])
        self.assertIn("Ollama", str(caught.exception))

    def test_model_override_and_exposure_clamp(self) -> None:
        os.environ["MISTRAL_API_KEY"] = "test-key"
        os.environ["MISTRAL_VISION_MODEL"] = "ministral-8b-2512"
        calls: list = []
        image = Image.new("RGB", (8, 8), (12, 12, 12))
        with patch(
            "ai_dt.advisors.mistral_vision.urllib.request.urlopen",
            self._urlopen(_reply(exposure_ev=9), calls),
        ):
            atmosphere = MistralVisionAdvisor().advise(image, Path("frame.png"))
        body = json.loads(calls[0].data.decode("utf-8"))
        self.assertEqual(body["model"], "ministral-8b-2512")
        self.assertEqual(atmosphere.exposure_ev, 1.5)
        self.assertEqual(atmosphere.source, "mistral:ministral-8b-2512")
        self.assertNotIn("caps", atmosphere.to_dict())


if __name__ == "__main__":
    unittest.main()
