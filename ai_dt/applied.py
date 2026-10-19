"""Steps already on the photo, so a new plan does not repeat them.

A plan proposed after some steps were accepted used to come back with the
same steps again (the duck report: subject, then subject again). The
planner is now told what is applied and what was skipped, and a repeated
step is dropped from the new queue unless the request asks to redo or
strengthen it.

Sources, most trusted first:

* the active rows of the photo's XMP history (``history_end`` prefix) whose
  names this editor writes (``revert.accepted_step_names()``, kept in sync
  with ``editor_history_names`` in ``ai_editor.lua``, plus the names on this
  photo's ``step_log``);
* the plan's ``step_log`` (active entries) and ``features_done``;
* ``features_skipped`` for the steps the photographer skipped.
"""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path

# Steps that change the same thing. Proposing one after the other is a repeat.
_KIND = {
    "mask_animal": "subject",
    "mask_animals": "subject",
    "mask_animal_background": "mask_background",
    "mask_sky_gradient": "mask_sky",
    "mask_bottom_gradient": "foreground",
    "eye": "eyes",
    "surround": "foreground",
}

# "redo" / "again" in the request lets any applied step repeat.
_REDO_ANY = re.compile(r"\b(re-?do|again|repeat|once more|one more time)\b", re.IGNORECASE)
# "more" / "stronger" lets a step repeat only when the request names it.
_STRONGER = re.compile(
    r"\b(more|stronger|strengthen|further|another|extra|increase|boost|intensify|harder|deeper|even)\b",
    re.IGNORECASE,
)
_PLAIN = {"the", "and", "with", "lift", "darken", "make", "grade", "quiet", "shine", "around", "behind", "only", "keep", "this", "that", "from", "into", "more", "less", "little", "photo"}


def step_kind(step_id: str) -> str:
    ident = str(step_id or "").strip().lower()
    return _KIND.get(ident, ident)


def wants_redo(steer: str, step: dict | None = None) -> bool:
    """Whether the request explicitly asks to redo or strengthen ``step`` (any step when ``None``)."""

    typed = str(steer or "")
    if _REDO_ANY.search(typed):
        return True
    if not _STRONGER.search(typed):
        return False
    if step is None:
        return True
    lowered = typed.lower()
    if _named(str(step.get("id") or ""), lowered):
        return True
    words = {
        word
        for word in re.findall(r"[a-z]+", str(step.get("title") or "").lower())
        if len(word) > 3 and word not in _PLAIN
    }
    return any(re.search(rf"\b{re.escape(word)}", lowered) for word in words)


def _name_to_step() -> dict[str, str]:
    from ai_dt.mapping import step_history_name
    from ai_dt.mask_steps import STEPS

    names: dict[str, str] = {}
    for ident, step in STEPS.items():
        names[step.name] = ident
    for ident in ("subject", "foreground", "opening", "shafts", "highlights", "recolor", "split_tone", "palander"):
        name = step_history_name(ident)
        if name:
            names[name] = ident
    names["Shine the subject"] = "subject"
    names["Vignette the subject"] = "subject"
    names["Eyes"] = "eyes"
    names["AI eyes"] = "eyes"
    for hue in ("red", "orange", "yellow", "green", "blue", "color"):
        names[f"Highlight the {hue}"] = "colorzones"
        names[f"Keep only the {hue}"] = "keep_color"
    return names


def history_names_from_xmp(path: str | Path | None) -> list[str] | None:
    """Instance names of the active history rows, oldest first. ``None`` without a readable sidecar."""

    if not path:
        return None
    try:
        document = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    from ai_dt.revert import _HISTORY_ITEM, _attr

    rows = [m.group(0) for m in _HISTORY_ITEM.finditer(document) if _attr(m.group(0), "darktable:operation")]
    end_match = re.search(r'darktable:history_end="(\d+)"', document)
    end = int(end_match.group(1)) if end_match else len(rows)
    return [unescape(_attr(tag, "darktable:multi_name")) for tag in rows[: max(0, min(end, len(rows)))]]


def applied_steps(plan: dict, history_names: list[str] | None = None) -> list[dict]:
    """``[{"id", "name", "title"}]`` for each step on the photo, oldest first.

    ``history_names`` are the active XMP rows when the sidecar was
    readable. The plan's active ``step_log`` entries are added too, because
    the sidecar can lag the darkroom (autosave) right after an Accept; Back
    and Revert already mark their log entries inactive. ``features_done``
    counts only when neither source names a step.
    """

    from ai_dt.revert import is_ai_editor_history_name

    log = [item for item in (plan.get("step_log") or []) if isinstance(item, dict) and item.get("active")]
    logged_names = {str(item.get("name") or "") for item in log if item.get("name")}
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(ident: str, name: str, title: str) -> None:
        key = (ident, name)
        if not ident or key in seen:
            return
        seen.add(key)
        found.append({"id": ident, "name": name, "title": title or name or ident})

    if history_names is not None:
        by_name = _name_to_step()
        log_by_name = {str(item.get("name") or ""): item for item in log}
        for name in history_names:
            if not is_ai_editor_history_name(name, logged_names):
                continue
            item = log_by_name.get(name)
            if item:
                add(str(item.get("id") or ""), name, str(item.get("title") or ""))
                continue
            ident = by_name.get(name, "")
            if not ident and name.startswith("AI "):
                ident = name[3:].strip().lower().replace(" ", "_")
            add(ident, name, name)
    # With a readable sidecar, the sidecar is the truth: a logged step whose
    # row is not in the history did not land (a failed or no-op Accept on a
    # freshly opened JPG) and is not done. The log only fills in when there
    # is no sidecar to read.
    if history_names is None:
        for item in log:
            add(str(item.get("id") or ""), str(item.get("name") or ""), str(item.get("title") or ""))
    if history_names is None:
        logged_ids = {item["id"] for item in found}
        for ident in plan.get("features_done") or []:
            ident = str(ident or "").strip()
            if ident and ident not in logged_ids:
                add(ident, "", ident)
    return found


def skipped_steps(plan: dict) -> list[str]:
    return [str(item) for item in (plan.get("features_skipped") or []) if str(item).strip()]


def applied_prompt_text(applied: list[dict], skipped: list[str]) -> str:
    """The prompt block that tells the planner what is already done."""

    if not applied and not skipped:
        return ""
    lines = []
    if applied:
        listed = "; ".join(
            f"{item['id']} ({item['name']})" if item.get("name") else item["id"] for item in applied
        )
        lines.append(
            "Already applied to this photo, in order: "
            f"{listed}. Do not propose these operations again unless the request "
            "explicitly asks to redo or strengthen them; propose only what is still missing."
        )
    if skipped:
        lines.append(
            "Skipped by the photographer in this session: "
            f"{', '.join(skipped)}. Do not propose them again unless the request asks for them."
        )
    return "Edit state:\n" + "\n".join(lines)


def drop_repeated(queue: list[dict], applied: list[dict], skipped: list[str], steer: str) -> tuple[list[dict], list[str]]:
    """The queue without steps whose id or kind is already applied (or skipped).

    A request that asks to redo or strengthen keeps applied steps; a skipped
    step stays only when the request names it.
    """

    typed = str(steer or "").lower()
    applied_by_kind = {step_kind(item["id"]): item for item in applied}
    applied_kinds = set(applied_by_kind)
    skipped_kinds = {step_kind(item) for item in skipped}
    kept: list[dict] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for item in queue:
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or "")
        kind = step_kind(ident)
        if kind in seen:
            dropped.append(ident)
            continue
        if kind in applied_kinds and not (
            wants_redo(steer, applied_by_kind[kind]) or wants_redo(steer, item)
        ):
            dropped.append(ident)
            continue
        if kind in skipped_kinds and kind not in applied_kinds and not _named(ident, typed):
            dropped.append(ident)
            continue
        seen.add(kind)
        kept.append(item)
    return kept, dropped


def _named(ident: str, typed: str) -> bool:
    words = [word for word in re.split(r"[_\s]+", ident.replace("mask_", "")) if len(word) > 2]
    return bool(words) and all(re.search(rf"\b{re.escape(word)}", typed) for word in words)
