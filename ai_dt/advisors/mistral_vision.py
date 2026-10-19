"""Mistral vision advisor.

Pixtral 12B (``pixtral-12b-2409``) and Pixtral Large (``pixtral-large-2411``)
are retired on the Mistral API. The documented replacement with vision on
``POST /v1/chat/completions`` is Ministral 3 14B, ``ministral-14b-2512``.
Override that id with ``MISTRAL_VISION_MODEL``.

The call runs only when ``MISTRAL_API_KEY`` is set. The picture is a base64
data URL in the chat message. The same lighting prompt LLaVA uses is sent
as text. The reply is parsed into an ``Atmosphere``. Caps are not a field
this advisor reads or sets.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request

from PIL import Image

from ai_dt.advisors.ollama_llava import DEFAULT_PROMPT, _load_prompt, _number
from ai_dt.errors import AdvisorUnavailable
from ai_dt.schema import Atmosphere

# Pinned id from the Ministral 3 14B model card and the vision model list.
# https://docs.mistral.ai/models/ministral-3-14b-25-12
# https://docs.mistral.ai/capabilities/vision
DEFAULT_MODEL = "ministral-14b-2512"
CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
_MAX_EDGE = 1280


def vision_model() -> str:
    chosen = os.environ.get("MISTRAL_VISION_MODEL", DEFAULT_MODEL).strip()
    return chosen or DEFAULT_MODEL


def _jpeg_data_url(image: Image.Image) -> str:
    frame = image.convert("RGB")
    if max(frame.size) > _MAX_EDGE:
        frame = frame.copy()
        frame.thumbnail((_MAX_EDGE, _MAX_EDGE))
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


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
            raise AdvisorUnavailable("Mistral did not return JSON")
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AdvisorUnavailable(f"Mistral JSON could not be parsed: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdvisorUnavailable("Mistral JSON was not an object")
    return payload


_SCENE_ORDER = ("subject", "light", "background")
_LABELS = frozenset({"mood", "summary", "description", "subject", "light", "background", "rationale"})


def _is_label(text: str) -> bool:
    words = [word.strip(".,!?;:\"'").lower() for word in text.split()]
    words = [word for word in words if word]
    return len(words) == 1 and words[0] in _LABELS


def _as_prose(value: object) -> str:
    """A sentence, or sentences from a subject/light/background record."""

    if isinstance(value, dict):
        parts: list[str] = []
        used: set[str] = set()
        for key in _SCENE_ORDER:
            if key not in value:
                continue
            used.add(key)
            sentence = _as_prose(value.get(key))
            if sentence:
                parts.append(sentence)
        for key, item in value.items():
            if key in used or isinstance(item, (int, float, bool)):
                continue
            sentence = _as_prose(item)
            if sentence:
                parts.append(sentence)
        return " ".join(parts)
    if isinstance(value, (list, tuple)):
        return " ".join(part for part in (_as_prose(item) for item in value) if part)
    if isinstance(value, (int, float, bool)) or value is None:
        return ""
    text = " ".join(str(value).split())
    if not text or text in {"{}", "[]"}:
        return ""
    if text[-1] not in ".!?":
        text += "."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _content_of(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise AdvisorUnavailable("Mistral returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        text = "".join(parts).strip()
        if text:
            return text
    raise AdvisorUnavailable("Mistral returned a response without text")


def _post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except OSError:
            detail = ""
        detail = " ".join(detail.split())[:240]
        raise AdvisorUnavailable(f"Mistral request failed: HTTP {exc.code} {detail}".strip()) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AdvisorUnavailable(f"Mistral request failed: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdvisorUnavailable(f"Mistral JSON could not be parsed: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdvisorUnavailable("Mistral JSON was not an object")
    return payload


class MistralVisionAdvisor:
    name = "mistral"

    def __init__(self, prompt_path=None, model: str | None = None) -> None:
        self.prompt_path = prompt_path
        self.model = model or vision_model()

    def advise(self, image: Image.Image, image_path=None, context: str = "") -> Atmosphere:
        key = os.environ.get("MISTRAL_API_KEY", "").strip()
        if not key:
            raise AdvisorUnavailable("MISTRAL_API_KEY is not set")
        prompt = _load_prompt(self.prompt_path or DEFAULT_PROMPT)
        extra = str(context or "").strip()
        if extra:
            prompt = f"{prompt}\n\n{extra}"
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 900 if context else 400,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _jpeg_data_url(image)}},
                    ],
                }
            ],
        }
        payload = _post_json(
            CHAT_URL,
            body,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            timeout=45.0,
        )
        parsed = _parse_json(_content_of(payload))
        mood = str(parsed.get("mood") or "").strip()
        rationale = str(parsed.get("rationale") or "").strip()
        description = _as_prose(parsed.get("description"))
        summary = _as_prose(parsed.get("summary"))
        if _is_label(summary) or summary.strip(" .!?").lower() == mood.strip(" .!?").lower():
            summary = ""
        if not description:
            description = _as_prose({key: parsed.get(key) for key in _SCENE_ORDER})
        if not summary:
            summary = description
        if len(description) > 1200:
            description = description[:1200].rstrip()
        if len(summary) > 500:
            summary = summary[:500].rstrip()
        return Atmosphere(
            exposure_ev=_number(parsed, "exposure_ev") or 0.0,
            contrast=_number(parsed, "contrast") or 0.0,
            temperature_shift=_number(parsed, "temperature_shift") or 0.0,
            tint=_number(parsed, "tint") or 0.0,
            saturation=_number(parsed, "saturation") or 0.0,
            shadows=_number(parsed, "shadows") or 0.0,
            highlights=_number(parsed, "highlights") or 0.0,
            vibrance=_number(parsed, "vibrance") or 0.0,
            mood=mood or "model suggestion",
            rationale=rationale or f"Suggestion from {self.model}.",
            source=f"mistral:{self.model}",
            scene_description=description,
            scene_summary=summary,
        )
