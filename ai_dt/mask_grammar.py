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
    ("mask_animal_background", r"\b(?:around|behind) the (?:animals?|birds?|dogs?|cats?|horses?|deer|wildlife)\b"),
    ("mask_animals", r"\b(?:all (?:the )?animals|animals|birds|herd|flock)\b"),
    ("mask_animal", r"\b(?:animal|bird|dog|cat|horse|deer|fox|owl|eagle|cow|sheep|bear|elephant|zebra|giraffe|puppy|kitten)\b"),
    ("mask_background", r"\bbackground\b"),
)

# Lesson phrases (photoserge examples 4-11). Explicit enough to fire without
# an edit verb; a lesson match takes the clause, so "gradient on the sky"
# is the sky gradient and not also the sky grade.
_LESSONS = (
    ("mask_sky_gradient", r"\b(?:gradient|graduated|grad)\b.*\b(?:sky|top)\b|\b(?:sky|top)\b.*\bgradient\b|\bdarken the top\b"),
    ("mask_bottom_gradient", r"\b(?:gradient|graduated|grad)\b.*\b(?:bottom|foreground)\b|\b(?:bottom|foreground)\b.*\bgradient\b|\bdarken the bottom\b"),
    ("mask_edge_burn", r"\bburn (?:the )?(?:edges|corners|sides)\b|\bedge burn\b|\bdodge and burn\b|\bdarken the (?:edges|corners)\b"),
    ("mask_glow", r"\bglow (?:behind|around)\b|\bbackground glow\b|\bhalo\b|\badd (?:a )?glow\b"),
    ("split_tone", r"\bsplit[- ]?ton(?:e|ed|ing)\b"),
    ("palander", r"\bpalander\b|\bwarm[- ]only\b|\bonly (?:the )?warm (?:colou?rs|tones|hues)\b"),
    ("keep_color", r"\b(?:keep|leave) only (?:the )?(?:red|orange|yellow|green|blue)\b|\bonly (?:the )?(?:red|orange|yellow|green|blue) (?:stays|remains)\b|\bselective colou?r\b"),
)
_PRESET = r"\b(?:animal|wildlife|landscape|portrait|street|night|architecture) (?:look|edit|preset|treatment|lesson)\b"

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


def _lesson_hits(clause: str) -> list[tuple[str, str]]:
    from ai_dt.lessons import PRESETS

    preset = re.search(_PRESET, clause)
    if preset:
        kind = preset.group(0).split()[0]
        kind = "animal" if kind == "wildlife" else kind
        return [(step_id, "include") for step_id in PRESETS.get(kind, ())]
    hits = []
    for step_id, pattern in _LESSONS:
        match = re.search(pattern, clause)
        if match:
            hits.append((match.start(), step_id))
    out = []
    for _start, step_id in sorted(hits):
        if step_id in {"split_tone", "palander", "keep_color"}:
            action = "hold" if re.search(r"\b(?:subtle|gentle|light|a little|slight)\b", clause) else "include"
        else:
            action = _action(clause, step_id)
        out.append((step_id, action))
    return out


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
        lesson_hits = _lesson_hits(clause)
        if lesson_hits:
            for step_id, action in lesson_hits:
                if step_id in seen:
                    continue
                seen.add(step_id)
                found.append({"id": step_id, "action": action, "reason": clause[:120]})
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
        if any(step == "mask_animal_background" for _start, step in hits):
            hits = [(start, step) for start, step in hits if step not in {"mask_animal", "mask_animals"}]
        if any(step == "mask_animals" for _start, step in hits):
            hits = [(start, step) for start, step in hits if step != "mask_animal"]
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
