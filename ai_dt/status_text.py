"""Plain status lines for the panel, built only from fields the run recorded.

A vision model is named only when its id is in the advisor source. A text
plan is named only when the reasoner returned that backend. A failed call
is mentioned only when the notes say that call was tried.
"""

from __future__ import annotations

import re

from ai_dt.drawn_mask import eye_ellipse
from ai_dt.features import feature_title
from ai_dt.reasoner import plain_gemini_failure
from ai_dt.steer import active_stages
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


def _plan_sheet(recipe: dict) -> str:
    """The steps still ahead, and how a typed sentence rewrites them."""

    plan = recipe.get("plan") if isinstance(recipe.get("plan"), dict) else {}
    done = [str(item) for item in (plan.get("features_done") or []) if str(item).strip()]
    path = [str(item) for item in (plan.get("features_path") or []) if str(item).strip()]
    queue = [item for item in (plan.get("step_queue") or []) if isinstance(item, dict)]
    nxt = str(recipe.get("features_next") or "").strip()
    if not done and not path and not nxt and not queue:
        return ""
    rows = ["Plan"]
    for item in done:
        rows.append(f"done — {feature_title(item)}")
    if queue:
        rows.append(f"now — {str(queue[0].get('title') or '').strip() or feature_title(str(queue[0].get('id') or ''))}")
        for item in queue[1:]:
            label = str(item.get("title") or "").strip() or feature_title(str(item.get("id") or ""))
            rows.append(f"later — {label}")
    elif nxt:
        rows.append(f"now — {str(recipe.get('features_next_title') or '').strip() or feature_title(nxt)}")
        for item in path:
            if item != nxt and item not in done:
                rows.append(f"later — {feature_title(item)}")
    ask = "Ask" if recipe.get("chat") else "Propose looks"
    if nxt:
        rows.append(
            "Accept applies this step. Skip leaves it out. "
            f"Type a change and press {ask} to make a new plan from this photo."
        )
    else:
        rows.append(f"Type a change and press {ask} to make a new plan from this photo.")
    return "\n".join(rows)


def panel_status(recipe: dict) -> str:
    """The full panel text. It is not cut to a character limit."""

    lines = []
    for key in ("picture_backend", "plan_backend"):
        text = str(recipe.get(key) or "").strip()
        if text:
            lines.append(text)
    plan = recipe.get("plan") if isinstance(recipe.get("plan"), dict) else {}
    reasoning = plan.get("reasoning") if isinstance(plan.get("reasoning"), dict) else {}
    paragraph = str(reasoning.get("paragraph") or "").strip()
    if paragraph:
        lines.append(paragraph)
    chat = bool(recipe.get("chat"))
    if chat:
        done = [str(item) for item in (plan.get("features_done") or []) if str(item).strip()]
        waiting = [
            str(item)
            for item in (plan.get("features_path") or [])
            if str(item).strip() and str(item) not in done
        ]
        steps = done + waiting
        if steps:
            lines.append("Steps: " + ", ".join(steps) + ".")
        held = [str(item) for item in (plan.get("not_history") or []) if str(item).strip()]
        if held:
            lines.append(
                "Not on this photo: "
                + ", ".join(held)
                + ". Those grades are not written into the history."
            )
        if "eyes" in steps:
            if eye_ellipse(list(recipe.get("local_edits") or [])):
                lines.append(
                    "Eyes is exposure inside an ellipse on the eyes. "
                    "Accept adds that shape to the history already on this photo."
                )
            else:
                lines.append("No eye was found, so Eyes stays off this photo.")
    else:
        survived = [stage for stage in active_stages(plan) if stage != "grade"]
        if survived and (paragraph or plan.get("catalog_chosen") or plan.get("operations")):
            lines.append("Stages: " + ", ".join(survived) + ".")
    dropped = [str(item) for item in (plan.get("dropped") or []) if str(item).strip()]
    if dropped:
        lines.append("Dropped: " + ", ".join(dropped) + ".")
    subject = str(recipe.get("subject_name") or "").strip()
    if subject:
        lines.append(f"Subject: {subject}.")
    nxt = str(recipe.get("features_next_title") or recipe.get("features_next") or "").strip()
    plan_sheet = _plan_sheet(recipe)
    if plan_sheet:
        lines.append(plan_sheet)
    elif nxt:
        lines.append(
            f"I would try {nxt} next. Accept applies it to this photo. "
            "Type a change in the steer box and press Propose looks to rewrite the plan."
        )
    elif not paragraph:
        sentence = version_sentence(recipe)
        if sentence:
            lines.append(sentence)
    handoff = recipe.get("handoff") if isinstance(recipe.get("handoff"), dict) else {}
    handoff_sentence = str(handoff.get("sentence") or "").strip()
    if handoff_sentence:
        lines.append(handoff_sentence)
    elif not nxt:
        lines.append(
            "Type a change in the steer box, then press Propose looks. "
            "Gemini rewrites the plan from that sentence."
        )
    for key in ("warmth_note", "shadow_note", "gradient_note"):
        note = str(plan.get(key) or "").strip()
        if note and note not in lines:
            lines.append(note)
    feature_note = str(recipe.get("feature_note") or "").strip()
    if feature_note and feature_note not in lines:
        lines.append(feature_note)
    return "\n".join(lines)
