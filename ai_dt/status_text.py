"""Plain status lines for the panel, built only from fields the run recorded.

A vision model is named only when its id is in the advisor source. A text
plan is named only when the reasoner returned that backend. A failed call
is mentioned only when the notes say that call was tried.
"""

from __future__ import annotations

import re

from ai_dt.reasoner import plain_gemini_failure
from ai_dt.versions import version_sentence

_MISTRAL = re.compile(r"mistral:([^\s+]+)")
_OLLAMA = re.compile(r"ollama:([^\s+]+)")


def picture_backend_line(advisor: str, notes: list[str] | None = None) -> str:
    """One line: which picture reader ran, and which one failed first."""

    source = str(advisor or "")
    recorded = [str(item) for item in (notes or []) if str(item).strip()]
    mistral = _MISTRAL.search(source)
    ollama = _OLLAMA.search(source)
    mistral_failed = any("mistral" in item.lower() for item in recorded)
    ollama_failed = any("ollama" in item.lower() for item in recorded)
    if mistral:
        return f"Picture: Mistral {mistral.group(1)}"
    if ollama:
        if mistral_failed:
            return f"Picture: Mistral failed, Ollama {ollama.group(1)}"
        return f"Picture: Ollama {ollama.group(1)}"
    if "histogram" in source:
        failed = []
        if mistral_failed:
            failed.append("Mistral failed")
        if ollama_failed:
            failed.append("Ollama failed")
        if failed:
            return "Picture: " + ", ".join(failed) + ", measured locally"
        return "Picture: measured locally"
    return ""


def gemini_failure_phrase(reasoning: dict | None) -> str:
    """The parenthetical reason, when this run recorded one."""

    info = reasoning if isinstance(reasoning, dict) else {}
    explicit = str(info.get("failure") or "").strip()
    if explicit:
        return explicit
    note = str(info.get("note") or "")
    marker = "did not return a plan:"
    if marker in note:
        return plain_gemini_failure(note.split(marker, 1)[1].strip(" ."))
    if "timed out" in note.lower() or "HTTP" in note:
        return plain_gemini_failure(note)
    return ""


def _gemini_failed_text(reason: str) -> str:
    if reason:
        return f"Gemini failed ({reason})"
    return "Gemini failed"


def plan_backend_line(reasoning: dict | None) -> str:
    """One line: Gemini, Ollama, or that the text plan was not used."""

    info = reasoning if isinstance(reasoning, dict) else {}
    status = str(info.get("status") or "")
    backend = str(info.get("backend") or "").strip()
    note = str(info.get("note") or "")
    gemini_failed = "Gemini" in note and "did not" in note
    ollama_failed = "Ollama" in note and "did not" in note
    reason = gemini_failure_phrase(info) if gemini_failed else ""
    if status == "gemini" and backend:
        return f"Plan: Gemini {backend}"
    if status == "ollama" and backend:
        if gemini_failed:
            return f"Plan: {_gemini_failed_text(reason)}, Ollama {backend}"
        return f"Plan: Ollama {backend}"
    parts = []
    if gemini_failed:
        parts.append(_gemini_failed_text(reason))
    if ollama_failed:
        parts.append("Ollama failed")
    if parts:
        return "Plan: " + ", ".join(parts) + ". Plan: not used"
    return "Plan: not used"


def panel_status(recipe: dict) -> str:
    """The full panel text. It is not cut to a character limit."""

    lines = []
    for key in ("picture_backend", "plan_backend"):
        text = str(recipe.get(key) or "").strip()
        if text:
            lines.append(text)
    subject = str(recipe.get("subject_name") or "").strip()
    if subject:
        lines.append(f"Subject: {subject}.")
    sentence = version_sentence(recipe)
    if sentence:
        lines.append(sentence)
    plan = recipe.get("plan") if isinstance(recipe.get("plan"), dict) else {}
    for key in ("warmth_note", "shadow_note", "gradient_note"):
        note = str(plan.get(key) or "").strip()
        if note and note not in lines:
            lines.append(note)
    return "\n".join(lines)
