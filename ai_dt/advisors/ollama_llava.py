"""Ollama vision advisor.

The prompt lives in ``ai_dt/prompts/llava_lighting.txt``. Pass another file
with ``--prompt`` or the Darktable preference to replace it without editing
this module. The model name is likewise a constructor argument, so
``llava`` can be swapped for another local vision model.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

from PIL import Image

from ai_dt.errors import AdvisorUnavailable
from ai_dt.schema import Atmosphere

_PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = _PACKAGE / "prompts" / "llava_lighting.txt"

_LIMITS = {
    "exposure_ev": (-1.5, 1.5),
    "contrast": (-0.4, 0.4),
    "temperature_shift": (-1.0, 1.0),
    "tint": (-1.0, 1.0),
    "saturation": (-0.4, 0.4),
    "shadows": (-0.4, 0.4),
    "highlights": (-0.4, 0.4),
    "vibrance": (-0.3, 0.3),
}


def ollama_reachable(timeout: float = 0.4) -> bool:
    """Return True when a TCP connection to the Ollama host succeeds."""

    hostport = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    hostport = hostport.removeprefix("http://").removeprefix("https://").split("/")[0]
    if hostport.startswith("["):
        host, _, port_text = hostport[1:].partition("]")
        port_text = port_text[1:] if port_text.startswith(":") else "11434"
    elif ":" in hostport:
        host, port_text = hostport.rsplit(":", 1)
    else:
        host, port_text = hostport, "11434"
    try:
        port = int(port_text)
    except ValueError:
        port = 11434
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def _load_prompt(path: Path | None) -> str:
    prompt_path = path or DEFAULT_PROMPT
    try:
        text = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AdvisorUnavailable(f"Cannot read prompt {prompt_path}: {exc}") from exc
    if not text:
        raise AdvisorUnavailable(f"Prompt file is empty: {prompt_path}")
    return text


def _content_of(response) -> str:
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
        if response.get("response"):
            return str(response["response"])
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if content:
        return str(content)
    raise AdvisorUnavailable("Ollama returned a response without text")


def _parse_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise AdvisorUnavailable("Ollama did not return JSON")
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AdvisorUnavailable(f"Ollama JSON could not be parsed: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdvisorUnavailable("Ollama JSON was not an object")
    return payload


def _number(payload: dict, key: str) -> float | None:
    if key not in payload or payload[key] is None:
        return None
    try:
        value = float(payload[key])
    except (TypeError, ValueError):
        return None
    low, high = _LIMITS[key]
    return max(low, min(high, value))


class OllamaLLaVAAdvisor:
    name = "ollama"

    def __init__(self, model: str = "llava", prompt_path: Path | None = None) -> None:
        self.model = model
        self.prompt_path = prompt_path

    def advise(self, image: Image.Image, image_path: Path) -> Atmosphere:
        if not ollama_reachable():
            raise AdvisorUnavailable(
                "Ollama is not running. Start it and pull a vision model, "
                "for example: ollama pull llava"
            )
        try:
            import ollama
        except ImportError as exc:
            raise AdvisorUnavailable(
                "The ollama Python package is not installed. "
                "Install requirements-ai.txt or use the histogram advisor."
            ) from exc

        prompt = _load_prompt(self.prompt_path)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [str(image_path)],
                    }
                ],
                format="json",
            )
        except Exception as exc:
            raise AdvisorUnavailable(f"Ollama request failed: {exc}") from exc

        payload = _parse_json(_content_of(response))
        mood = str(payload.get("mood") or "").strip()
        rationale = str(payload.get("rationale") or "").strip()
        return Atmosphere(
            exposure_ev=_number(payload, "exposure_ev") or 0.0,
            contrast=_number(payload, "contrast") or 0.0,
            temperature_shift=_number(payload, "temperature_shift") or 0.0,
            tint=_number(payload, "tint") or 0.0,
            saturation=_number(payload, "saturation") or 0.0,
            shadows=_number(payload, "shadows") or 0.0,
            highlights=_number(payload, "highlights") or 0.0,
            vibrance=_number(payload, "vibrance") or 0.0,
            mood=mood or "model suggestion",
            rationale=rationale or f"Suggestion from {self.model}.",
            source=f"ollama:{self.model}",
        )
