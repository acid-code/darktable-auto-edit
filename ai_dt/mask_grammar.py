"""Requests that name a region, turned into drawn-mask operations.

This is a small grammar, like the hue grammar in ``features.py``. It only
fires when the sentence names a region *and* an edit, so a description
such as "a boat on the water" is not a request. The result uses the
catalog's operation shape, ``{"id", "action", "reason"}``.
"""

from __future__ import annotations

import re

_REGIONS = (
    ("mask_landscape", r"\b(?:landscape masks?|every (?:part|region)|each (?:part|region)|all (?:the )?(?:regions|parts)|five masks)\b"),
    ("mask_sky", r"\b(?:sky|skies|clouds?)\b"),
    ("mask_water", r"\b(?:water|sea|ocean|lake|river|pond|waves?)\b"),
    ("mask_vegetation", r"\b(?:trees?|grass|foliage|vegetation|greenery|forest|leaves|bushes|plants?)\b"),
    ("mask_architecture", r"\b(?:buildings?|architecture|houses?|church|castle|facades?|towers?|bridge)\b"),
    ("mask_ground", r"\b(?:ground|road|sand|beach|path|rocks?|dirt|pavement)\b"),
    ("mask_people", r"\b(?:people|persons?|crowd|tourists?|figures?)\b"),
    ("mask_background", r"\bbackground\b"),
)

_DARKEN = r"\b(?:darken|darker|deepen|deeper|dim|burn|moody|dramatic|drama|quiet|mute|calm|tone down|less bright)\b"
_LIFT = r"\b(?:lift|brighten|brighter|lighter|lighten|dodge|pop|glow|clarity|crisp|crisper|greener|bluer|warmer|cooler|vivid|richer|more color)\b"
_EDIT = rf"(?:{_DARKEN}|{_LIFT}|\b(?:grade|masks?|select|edit|adjust|fix|enhance|warm|cool|soften|softer|saturate)\b)"

# Phrases that already mean something else in this editor.
_TAKEN = (
    r"\b(?:less|more) on the background\b",
    r"\bblur (?:the )?background\b",
    r"\bbackground blur\b",
)


def _clauses(sentence: str) -> list[str]:
    text = " ".join(str(sentence or "").lower().split())
    return [part.strip() for part in re.split(r"[,.;!?]|\band then\b|\bthen\b", text) if part.strip()]


def _action(clause: str, step_id: str) -> str:
    from ai_dt.mask_steps import DEFAULT_ACTION

    darken = re.search(_DARKEN, clause)
    lift = re.search(_LIFT, clause)
    if darken and not lift:
        return "darken"
    if lift and not darken:
        # "bluer water" is the water grade, which darkens; the lesson's grade has both.
        if step_id == "mask_water" and re.search(r"\bbluer\b", clause):
            return "darken"
        return "lift"
    return DEFAULT_ACTION.get(step_id, "lift")


def mask_operations(sentence: str, claimed: str = "") -> list[dict]:
    """Drawn-mask operations for each region the sentence asks to change, in order.

    ``claimed`` is the text of the model's own features and stages. A
    region word the model already turned into another step (``the path``
    as the foreground band, say) is left to that step.
    """

    taken_words = set(re.findall(r"[a-z]+", str(claimed or "").lower()))
    found: list[dict] = []
    seen: set[str] = set()
    for clause in _clauses(sentence):
        if any(re.search(pattern, clause) for pattern in _TAKEN):
            continue
        landscape = re.search(_REGIONS[0][1], clause)
        if not landscape and not re.search(_EDIT, clause):
            continue
        hits = []
        for step_id, pattern in _REGIONS:
            match = re.search(pattern, clause)
            if match and match.group(0).split()[-1] not in taken_words:
                hits.append((match.start(), step_id))
        if any(step == "mask_landscape" for _start, step in hits):
            hits = [(0, "mask_landscape")]
        for _start, step_id in sorted(hits):
            if step_id in seen:
                continue
            seen.add(step_id)
            found.append(
                {
                    "id": step_id,
                    "action": "include" if step_id == "mask_landscape" else _action(clause, step_id),
                    "reason": clause[:120],
                }
            )
    return found
