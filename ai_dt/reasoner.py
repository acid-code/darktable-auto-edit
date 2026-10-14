"""Propose stages from text the pipeline already has.

The inputs are the picture description, the vibe words, and the preset
name. No picture is uploaded. Gemini is tried first when ``GEMINI_API_KEY``
is set. Ollama is the fallback when Gemini is not configured or does not
return a plan. If neither backend answers, reasoning is skipped and the
caller's steer clauses still apply.

The model may only name stages and actions in the steer vocabulary.
It cannot open the caps. ``epic`` and ``strong`` do that, and only when
the user wrote them.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ai_dt.ollama_host import ollama_base_url, ollama_reachable

# Skip targets the renderer already knows. ``surround`` is the user's word.
SKIP_STAGES = frozenset(
    {
        "gradient",
        "subject",
        "surround",
        "background",
        "vignette",
        "theme",
        "reflection",
        "eyes",
        "eye",
        "skin",
        "sky",
    }
)
# darken / lift / hold land on these. ``gradient`` asks for the linear field.
ASK_STAGES = frozenset(
    {
        "subject",
        "foreground",
        "rock",
        "sky",
        "behind",
        "reflection",
        "surround",
        "background",
        "gradient",
    }
)
_BLOCKED = frozenset({"strong", "epic", "caps", "caps_open", "open"})

_PROMPT = """You plan local photo edits. The picture stays on this machine. No photo is uploaded.
Do not include slider numbers. Do not open caps. Never use epic, strong, or caps_open.
Write about this picture and this request. Do not copy an example.

Reply with a short paragraph, then JSON. The paragraph is what a person can answer:
what you see, what you will change, and what you will leave alone.
The JSON may be raw or inside a ```json fence.
{"reasoning": "two or three sentences", "stages": [{"stage": "subject", "action": "lift", "reason": "why"}]}

Stages you may skip: gradient, subject, surround, vignette, theme, reflection, eyes, skin, sky.
Targets you may darken, lift, or hold: subject, foreground, rock, sky, behind, reflection.
foreground is the surround and the water, not the subject.
A request to make a reflection clearer is action lift on stage reflection. Do not skip that stage.
You may request the linear gradient with stage gradient and action darken, lift, or hold.
Leave the gradient out when the request is the subject, the dark water, and the reflection.

Picture: {description}
Vibe: {vibe}
Preset: {preset}
Request: {steer}
"""

# The old prompt's sample reason. A live model sometimes echoes it.
_PLACEHOLDER = "one short sentence"


def skipped(note: str, *, backend: str = "") -> dict:
    return {"status": "skipped", "backend": backend, "note": note, "stages": []}


def validate_proposal(payload: object) -> list[dict]:
    """Keep stages and actions in the steer vocabulary. Drop caps words."""

    if not isinstance(payload, dict):
        return []
    raw = payload.get("stages")
    if not isinstance(raw, list):
        return []
    kept: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or "").strip().lower()
        action = str(item.get("action") or "").strip().lower()
        if stage in _BLOCKED or action in _BLOCKED:
            continue
        if stage == "eye":
            stage = "eyes"
        if action == "light":
            action = "lift"
        if action == "revert":
            action = "skip"
        if action == "skip" and stage in SKIP_STAGES:
            allowed = True
        elif action in {"darken", "lift", "hold"} and stage in ASK_STAGES:
            allowed = True
        else:
            allowed = False
        if not allowed:
            continue
        reason = " ".join(str(item.get("reason") or "").split())
        if _is_placeholder(reason):
            continue
        if len(reason) > 180:
            reason = reason[:180].rstrip()
        kept.append({"stage": stage, "action": action, "reason": reason})
    return kept


def _is_placeholder(text: str) -> bool:
    cleaned = " ".join(str(text or "").lower().replace(".", " ").split())
    return cleaned == _PLACEHOLDER or cleaned.startswith(_PLACEHOLDER + " ")


def _usable_paragraph(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned or _is_placeholder(cleaned):
        return ""
    if len(cleaned) > 600:
        cleaned = cleaned[:600].rstrip()
    return cleaned


def _parse_json(text: str) -> dict | None:
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
            return None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    return payload


def _fill(description: str, vibe: str, preset: str, steer: str = "") -> str:
    return (
        _PROMPT.replace("{description}", description or "No description.")
        .replace("{vibe}", vibe or "faithful")
        .replace("{preset}", preset or "none")
        .replace("{steer}", steer.strip() or "none")
    )


def _prose_paragraph(text: str) -> str:
    """Sentences outside a JSON object, when the model wrote both."""

    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.startswith("```")]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    prose = stripped[:start] if start >= 0 else stripped
    prose = prose.replace("```json", " ").replace("```", " ")
    return _usable_paragraph(prose)


def _stages_from_prose(text: str) -> list[dict]:
    """Stage names in a short reply that never became JSON."""

    low = " ".join(str(text or "").lower().split())
    if not low or _is_placeholder(low):
        return []
    stages: list[dict] = []
    if "reflect" in low and any(word in low for word in ("clear", "lift", "sharp")):
        stages.append({"stage": "reflection", "action": "lift", "reason": ""})
    elif "reflect" in low and "hold" in low:
        stages.append({"stage": "reflection", "action": "hold", "reason": ""})
    water = any(phrase in low for phrase in ("darken the water", "darken the surround", "dark water", "go darker", "darker"))
    if water or ("foreground" in low and "dark" in low):
        stages.append({"stage": "foreground", "action": "darken", "reason": ""})
    if "darken the subject" in low or "darken the bird" in low:
        stages.append({"stage": "subject", "action": "darken", "reason": ""})
    elif any(word in low for word in ("bird", "subject")) and any(word in low for word in ("lift", "subject", "bird")):
        stages.append({"stage": "subject", "action": "lift", "reason": ""})
    return stages


def interpret_reply(text: str) -> dict:
    """A paragraph and a stage list from JSON, a fence, or a short prose reply."""

    payload = _parse_json(text)
    paragraph = ""
    stages: list[dict] = []
    if payload is not None:
        paragraph = _usable_paragraph(str(payload.get("reasoning") or payload.get("paragraph") or ""))
        stages = validate_proposal(payload)
    if not paragraph:
        paragraph = _prose_paragraph(text)
    if not stages:
        stages = validate_proposal({"stages": _stages_from_prose(text)})
    return {"paragraph": paragraph, "stages": stages}


def _text_model() -> str:
    return os.environ.get("OLLAMA_REASON_MODEL", "llama3.2")


def _post_json(url: str, body: dict, timeout: float) -> dict | None:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _ask_ollama(description: str, vibe: str, preset: str, steer: str = "") -> dict | None:
    """One text chat. No image is sent."""

    if not ollama_reachable():
        return None
    payload = _post_json(
        ollama_base_url().rstrip("/") + "/api/chat",
        {
            "model": _text_model(),
            "stream": False,
            "messages": [{"role": "user", "content": _fill(description, vibe, preset, steer)}],
        },
        timeout=30.0,
    )
    if payload is None:
        return None
    message = payload.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else ""
    if not content:
        content = payload.get("response") or ""
    return interpret_reply(str(content)) if str(content).strip() else None


def _ask_gemini(description: str, vibe: str, preset: str, steer: str = "") -> dict | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("GEMINI_REASON_MODEL", "gemini-2.0-flash")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model
        + ":generateContent?key="
        + key
    )
    payload = _post_json(
        url,
        {"contents": [{"parts": [{"text": _fill(description, vibe, preset, steer)}]}]},
        timeout=30.0,
    )
    if payload is None:
        return None
    candidates = payload.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return None
    content = candidates[0].get("content") or {}
    parts = content.get("parts") if isinstance(content, dict) else None
    text = ""
    if isinstance(parts, list) and parts and isinstance(parts[0], dict):
        text = str(parts[0].get("text") or "")
    return interpret_reply(text) if text.strip() else None


def _from_reply(reply: dict | None, *, status: str, backend: str, note: str) -> dict | None:
    if not reply:
        return None
    paragraph = str(reply.get("paragraph") or "")
    stages = list(reply.get("stages") or [])
    if not paragraph and not stages:
        # The model answered, but only with a placeholder. The steer still applies.
        return {"status": status, "backend": backend, "note": "", "paragraph": "", "stages": []}
    return {
        "status": status,
        "backend": backend,
        "note": note if stages or paragraph else "",
        "paragraph": paragraph,
        "stages": stages,
    }


def propose_stages(description: str, vibe: str, preset: str, steer: str = "") -> dict:
    """Ask Gemini when a key is set, then Ollama. Skip when neither answers.

    The returned dict never opens caps. It has ``status``, ``backend``,
    ``note``, ``paragraph``, and ``stages``. A usable paragraph or stage
    list is not reported as a skipped plan.
    """

    if os.environ.get("AI_EDITOR_REASONER") == "off":
        return skipped("Reasoning was skipped.")
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        gemini_plan = _from_reply(
            _ask_gemini(description, vibe, preset, steer),
            status="gemini",
            backend=os.environ.get("GEMINI_REASON_MODEL", "gemini-2.0-flash"),
            note="Stages proposed by Gemini.",
        )
        if gemini_plan and (gemini_plan.get("paragraph") or gemini_plan.get("stages")):
            return gemini_plan
    ollama_reply = _ask_ollama(description, vibe, preset, steer)
    if ollama_reply is not None:
        return _from_reply(
            ollama_reply,
            status="ollama",
            backend=_text_model(),
            note=f"Stages proposed by the local model {_text_model()}.",
        ) or skipped("Ollama is up but the text model did not return a plan. Reasoning was skipped.")
    if gemini_key:
        return skipped("Gemini was configured but did not return a plan. Reasoning was skipped.")
    if ollama_reachable():
        return skipped("Ollama is up but the text model did not return a plan. Reasoning was skipped.")
    return skipped("Reasoning was skipped. No local model and no Gemini key.")
