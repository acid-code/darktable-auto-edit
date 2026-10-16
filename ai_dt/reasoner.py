"""Propose stages from text the pipeline already has.

The inputs are the picture description, the vibe words, and the preset
name. No picture is uploaded. Gemini is tried first when ``GEMINI_API_KEY``
is set. A busy model, a used-up free quota, or a missing model id tries the
next text model. Ollama is the fallback when Gemini is not configured or
none of those models return a plan. If neither backend answers, reasoning
is skipped and the caller's steer clauses still apply.

The model may only name stages and actions in the steer vocabulary.
It cannot open the caps. ``epic`` and ``strong`` do that, and only when
the user wrote them.
"""

from __future__ import annotations

import json
import os
import re
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
        "eyes",
        "eye",
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
Targets you may darken, lift, or hold: subject, foreground, rock, sky, behind, reflection, eyes.
foreground is the surround, not the subject.
Name water, a reflection, eyes, or an animal only when the picture description or the request already says so. Do not invent them.
A request to make a reflection clearer is action lift on stage reflection. Do not skip that stage.
You may request the linear gradient with stage gradient and action darken, lift, or hold.
Leave the gradient out when the request is only about the subject.
The local grade fills these controls from the picture. Do not invent slider numbers:
exposure EV, a small color-calibration warmth on the pipeline white (never a custom illuminant or a daylight Kelvin), tone-equalizer shadow and highlight bands, color-zone saturation, graduated-density position rotation and density from a measured falloff, and a crop window when a horizon is off center.
Diffuse strength is a masked control and is not written into the style.
Do not ask for a center-line gradient. Name sky, highlights, or the bright end, or leave the gradient out.
Caps stay closed. Only the user's own strong or epic word opens them.

Also name mask features the request is asking for. When the request is none, leave the stages array empty and name a short path the picture description supports.
Add this array to the same JSON object:
{"features": [{"id": "foreground", "reason": "words from the request or the picture description"}]}
Feature ids only: foreground, opening, shafts, highlights, eyes.
foreground darkens the lower band and leaves a person or animal alone.
opening darkens one bright compact gap, not the whole sky.
shafts darkens tall dark vertical shapes so the gap beside them stays the light.
highlights warms the bright range and adds a little local contrast, leaving a person or animal out.
eyes is only the eyes of a person or animal the request already named. Do not use it for the whole subject.
Use an id only when the request or the picture description supports it. The reason must point at those words. Do not invent an id outside this list. The id itself does not have to appear in the request.
Return every stage the request needs, in the order they should be accepted. Several stages are allowed. Do not collapse them into one.

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


_FEATURE_IDS = frozenset({"foreground", "opening", "shafts", "highlights", "eyes", "separate"})


def validate_features(payload: object) -> list[dict]:
    """Keep catalog ids. Drop a tool the model invented."""

    if not isinstance(payload, dict):
        return []
    raw = payload.get("features")
    if not isinstance(raw, list):
        return []
    kept: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("id") or item.get("feature") or "").strip().lower()
        if name not in _FEATURE_IDS or name in seen:
            continue
        reason = " ".join(str(item.get("reason") or "").split())
        if _is_placeholder(reason) or not reason:
            continue
        if len(reason) > 180:
            reason = reason[:180].rstrip()
        seen.add(name)
        kept.append({"id": name, "reason": reason})
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
    features: list[dict] = []
    if payload is not None:
        paragraph = _usable_paragraph(str(payload.get("reasoning") or payload.get("paragraph") or ""))
        stages = validate_proposal(payload)
        features = validate_features(payload)
    if not paragraph:
        paragraph = _prose_paragraph(text)
    if not stages:
        stages = validate_proposal({"stages": _stages_from_prose(text)})
    reply = {"paragraph": paragraph, "stages": stages}
    if features:
        reply["features"] = features
    return reply


def _text_model() -> str:
    return os.environ.get("OLLAMA_REASON_MODEL", "llama3.2")


# Highest free-tier daily quota first. These are text generateContent ids,
# not Live API models. The API lists Gemini 3 Flash as gemini-3-flash-preview.
# gemini-2.5-flash can return HTTP 404; that skips to the next id.
REASON_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
)
DEFAULT_REASON_MODEL = REASON_MODELS[0]
GEMINI_TIMEOUT_SECONDS = 30
# A chain summary is already short: "model: reason; model: reason".
_CHAIN_SUMMARY = re.compile(
    r"^(?:[A-Za-z0-9._-]+: .+)(?:; [A-Za-z0-9._-]+: .+)*$"
)


def _reason_models() -> list[str]:
    """The text models to try. ``GEMINI_REASON_MODEL`` is first when it is set."""

    chosen = os.environ.get("GEMINI_REASON_MODEL", "").strip()
    chain = list(REASON_MODELS)
    if not chosen:
        return chain
    return [chosen] + [item for item in chain if item != chosen]


def _reason_model() -> str:
    return _reason_models()[0]


def plain_gemini_failure(detail: str) -> str:
    """A short reason safe to show in the panel. No response body and no key."""

    text = " ".join(str(detail or "").split())
    if _CHAIN_SUMMARY.match(text):
        return text
    lowered = text.lower()
    if "timed out" in lowered:
        return f"timed out after {int(GEMINI_TIMEOUT_SECONDS)}s"
    code_match = re.search(r"HTTP\s+(\d+)", text, re.IGNORECASE)
    code = code_match.group(1) if code_match else ""
    if code == "503" or "high demand" in lowered or "unavailable" in lowered:
        return "HTTP 503, high demand"
    if code == "404" or "no longer available" in lowered or "not found" in lowered:
        return f"HTTP {code or '404'}, model not available"
    if code in {"401", "403"} or (code == "400" and "api key" in lowered):
        return f"HTTP {code}, API key rejected"
    if (
        code == "429"
        or "rate limit" in lowered
        or "quota" in lowered
        or "resource_exhausted" in lowered
        or "resource exhausted" in lowered
        or "free tier" in lowered
        or "free plan" in lowered
    ):
        return f"HTTP {code or '429'}, rate limit"
    if code:
        return f"HTTP {code}"
    if not text or lowered in {"no reply", "empty reply"}:
        return "no reply"
    if len(text) > 80:
        text = text[:80].rstrip()
    return text


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
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except OSError:
            detail = ""
        detail = " ".join(detail.split())
        if len(detail) > 180:
            detail = detail[:180].rstrip()
        return {"_http_error": f"HTTP {exc.code} {detail}".strip()}
    except TimeoutError:
        return {"_http_error": "timed out"}
    except urllib.error.URLError as exc:
        reason = " ".join(str(getattr(exc, "reason", exc)).split())[:120]
        return {"_http_error": f"request failed: {reason}".strip()}
    except (OSError, json.JSONDecodeError):
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


def _gemini_advances(detail: str) -> bool:
    """True when this model should be skipped and the next text model tried.

    A rejected API key stops the chain. The same key would fail every id.
    """

    text = " ".join(str(detail or "").split())
    lowered = text.lower()
    code_match = re.search(r"HTTP\s+(\d+)", text, re.IGNORECASE)
    code = code_match.group(1) if code_match else ""
    if code in {"401", "403"} or (code == "400" and "api key" in lowered):
        return False
    if code in {"503", "429", "404"}:
        return True
    if "unavailable" in lowered or "high demand" in lowered:
        return True
    if any(
        phrase in lowered
        for phrase in (
            "resource_exhausted",
            "resource exhausted",
            "quota",
            "rate limit",
            "free tier",
            "free plan",
        )
    ):
        return True
    if "no longer available" in lowered or "not found" in lowered:
        return True
    return False


def _ask_gemini_model(
    model: str,
    description: str,
    vibe: str,
    preset: str,
    steer: str,
    key: str,
) -> dict:
    """One generateContent call. ``model`` is set when the reply has a plan."""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model
        + ":generateContent?key="
        + key
    )
    body = {"contents": [{"parts": [{"text": _fill(description, vibe, preset, steer)}]}]}
    payload = _post_json(url, body, timeout=GEMINI_TIMEOUT_SECONDS)
    if payload is None:
        return {"paragraph": "", "stages": [], "error": "no reply"}
    http_error = str(payload.get("_http_error") or "").strip()
    if http_error:
        if key:
            http_error = http_error.replace(key, "")
        return {"paragraph": "", "stages": [], "error": http_error}
    candidates = payload.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        feedback = payload.get("promptFeedback") if isinstance(payload.get("promptFeedback"), dict) else {}
        blocked = str((feedback or {}).get("blockReason") or "").strip()
        return {"paragraph": "", "stages": [], "error": blocked or "empty reply"}
    content = candidates[0].get("content") or {}
    parts = content.get("parts") if isinstance(content, dict) else None
    texts: list[str] = []
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict) or part.get("thought") is True:
                continue
            piece = str(part.get("text") or "").strip()
            if piece:
                texts.append(piece)
    text = "\n".join(texts)
    if not text.strip():
        finish = str(candidates[0].get("finishReason") or "").strip()
        return {"paragraph": "", "stages": [], "error": finish or "empty reply"}
    parsed = interpret_reply(text)
    parsed["model"] = model
    return parsed


def _ask_gemini(description: str, vibe: str, preset: str, steer: str = "") -> dict | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    tried: list[str] = []
    for model in _reason_models():
        reply = _ask_gemini_model(model, description, vibe, preset, steer, key)
        error = str(reply.get("error") or "").strip()
        if not error and (reply.get("paragraph") or reply.get("stages") or reply.get("features")):
            return reply
        reason = plain_gemini_failure(error or "no reply")
        tried.append(f"{model}: {reason}")
        if error and _gemini_advances(error):
            continue
        break
    return {"paragraph": "", "stages": [], "error": "; ".join(tried)}


def _from_reply(reply: dict | None, *, status: str, backend: str, note: str) -> dict | None:
    if not reply:
        return None
    paragraph = str(reply.get("paragraph") or "")
    stages = list(reply.get("stages") or [])
    features = list(reply.get("features") or [])
    if not paragraph and not stages and not features:
        # The model answered, but only with a placeholder. The steer still applies.
        return {"status": status, "backend": backend, "note": "", "paragraph": "", "stages": []}
    plan = {
        "status": status,
        "backend": backend,
        "note": note if stages or paragraph or features else "",
        "paragraph": paragraph,
        "stages": stages,
    }
    if features:
        plan["features"] = features
    return plan


def _note_gemini_failure(result: dict, gemini_failed: bool, detail: str = "") -> dict:
    """Say Gemini was tried when the plan actually came from somewhere else."""

    if not gemini_failed:
        return result
    updated = dict(result)
    reason = plain_gemini_failure(detail) if str(detail or "").strip() else ""
    if reason:
        updated["failure"] = reason
    note = str(updated.get("note") or "")
    if "Gemini" in note and "did not" in note:
        return updated
    prefix = "Gemini did not return a plan."
    if reason:
        prefix = f"Gemini did not return a plan: {reason}."
    updated["note"] = f"{prefix} {note}".strip()
    return updated


def propose_gemini(description: str, vibe: str, preset: str, steer: str = "") -> dict:
    """Ask only Gemini. Ollama is a later stage when ``ask_ollama`` is set."""

    if os.environ.get("AI_EDITOR_REASONER") == "off":
        return {
            "ask_ollama": False,
            "gemini_tried": False,
            "failure": "",
            "proposal": skipped("Reasoning was skipped."),
        }
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return {"ask_ollama": True, "gemini_tried": False, "failure": "", "proposal": None}
    reply = _ask_gemini(description, vibe, preset, steer)
    error = str(reply.get("error") or "").strip() if isinstance(reply, dict) else "no reply"
    model = str(reply.get("model") or "").strip() if isinstance(reply, dict) else ""
    plan = _from_reply(
        None if error else reply,
        status="gemini",
        backend=model or _reason_model(),
        note="Stages proposed by Gemini.",
    )
    if plan and (plan.get("paragraph") or plan.get("stages") or plan.get("features")):
        return {"ask_ollama": False, "gemini_tried": True, "failure": "", "proposal": plan}
    reason = plain_gemini_failure(error)
    return {"ask_ollama": True, "gemini_tried": True, "failure": reason, "proposal": None}


def propose_ollama(
    description: str,
    vibe: str,
    preset: str,
    steer: str = "",
    *,
    failure: str = "",
    gemini_tried: bool = False,
) -> dict:
    """Ask only Ollama. A Gemini failure is copied onto the plan."""

    if os.environ.get("AI_EDITOR_REASONER") == "off":
        return {"ask_ollama": False, "proposal": skipped("Reasoning was skipped.")}
    reply = _ask_ollama(description, vibe, preset, steer)
    if reply is not None:
        result = _from_reply(
            reply,
            status="ollama",
            backend=_text_model(),
            note=f"Stages proposed by the local model {_text_model()}.",
        ) or skipped("Ollama is up but the text model did not return a plan. Reasoning was skipped.")
        return {
            "ask_ollama": False,
            "proposal": _note_gemini_failure(result, gemini_tried, failure),
        }
    if gemini_tried:
        result = skipped(
            f"Gemini was configured but did not return a plan. {failure} Reasoning was skipped.".strip()
        )
        return {"ask_ollama": False, "proposal": _note_gemini_failure(result, True, failure)}
    if ollama_reachable():
        return {
            "ask_ollama": False,
            "proposal": skipped("Ollama is up but the text model did not return a plan. Reasoning was skipped."),
        }
    return {
        "ask_ollama": False,
        "proposal": skipped("Reasoning was skipped. No local model and no Gemini key."),
    }


def propose_stages(description: str, vibe: str, preset: str, steer: str = "") -> dict:
    """Ask Gemini when a key is set, then Ollama. Skip when neither answers.

    The returned dict never opens caps. It has ``status``, ``backend``,
    ``note``, ``paragraph``, and ``stages``. A usable paragraph or stage
    list is not reported as a skipped plan.
    """

    if os.environ.get("AI_EDITOR_REASONER") == "off":
        return skipped("Reasoning was skipped.")
    gemini = propose_gemini(description, vibe, preset, steer)
    if not gemini.get("ask_ollama") and isinstance(gemini.get("proposal"), dict):
        return gemini["proposal"]
    ollama = propose_ollama(
        description,
        vibe,
        preset,
        steer,
        failure=str(gemini.get("failure") or ""),
        gemini_tried=bool(gemini.get("gemini_tried")),
    )
    proposal = ollama.get("proposal")
    if isinstance(proposal, dict):
        return proposal
    return skipped("Reasoning was skipped. No local model and no Gemini key.")
