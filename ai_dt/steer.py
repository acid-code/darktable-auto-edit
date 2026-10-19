"""Turn a steer line into stage actions the renderer already knows.

The line is not free prose. It is split on commas and on the word "and".
Each clause has to match the grammar below. Anything else is ignored and
named in the summary.

Grammar, one clause at a time, case insensitive:

- ``strong`` or ``epic`` opens the caps for this run
- ``revert <stage>`` or ``skip <stage>``
  stages: gradient, subject, surround, vignette, theme, reflection, eyes, skin, sky
- ``darken <target>``, ``lift <target>``, ``light <target>``, ``hold <target>``
  targets: subject, foreground, rock, sky, behind, reflection
  ``light`` is the same verb as ``lift``. ``light behind the subject`` is lift behind.
- ``gradient`` asks for the linear gradient. It stays modest unless ``strong`` or ``epic`` is also there.
- Expert modules, said directly: ``lift the shadows``, ``hold the highlights``,
  ``warmer`` or ``cooler``, ``more color`` or ``mute``, ``more contrast``,
  ``crop``. These set tone equalizer, color calibration, color zones, and crop.

A model cannot open the caps. Only these words in the user's own line can.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_FILLER = frozenset({"the", "a", "an"})
_OPEN = frozenset({"strong", "epic"})
_SKIP_VERBS = frozenset({"revert", "skip"})
_ASK_VERBS = frozenset({"darken", "lift", "light", "hold"})

# Steer names the user can say, and the stage id the renderer uses.
_STAGE_WORDS = {
    "gradient": "gradient",
    "subject": "subject",
    "surround": "background",
    "background": "background",
    "vignette": "vignette",
    "theme": "theme",
    "reflection": "reflection",
    "eyes": "eyes",
    "eye": "eyes",
    "skin": "skin",
    "sky": "sky",
}

_TARGETS = frozenset({"subject", "foreground", "rock", "sky", "behind", "reflection"})
_EXPERT_WORDS = {
    "warmer": ("warmth", 0.4),
    "warm": ("warmth", 0.4),
    "cooler": ("warmth", -0.4),
    "cool": ("warmth", -0.4),
    "saturate": ("color", 0.35),
    "mute": ("color", -0.35),
    "desaturate": ("color", -0.35),
    "crop": ("crop_horizon", True),
    "contrast": ("contrast_push", 0.25),
}

# How a stage is named in the list the user reads.
_DISPLAY = {
    "background": "surround",
    "grade": "grade",
}


def empty_plan(guide_stages: list[str], *, reflection: str = "lift", theme: str = "") -> dict:
    return {
        "guide_stages": list(guide_stages),
        "skip": [],
        "scales": {},
        "pinned": "",
        "theme": theme,
        "reflection": reflection if reflection in {"lift", "hold", "quiet"} else "lift",
        "strong": False,
        "caps_open": False,
        "gradient": False,
        "gradient_by": "",
        "asks": [],
        "proposed": [],
        "ignored": [],
        "summary": "",
        "skip_by": {},
        "reasoning": {
            "status": "skipped",
            "backend": "",
            "note": "Reasoning was skipped.",
        },
        "lines": [],
    }


def active_stages(plan: dict) -> list[str]:
    skipped = set(plan.get("skip") or [])
    # surround is stored as the renderer name "background"
    return [stage for stage in plan.get("guide_stages") or [] if stage not in skipped]


def load_stored(output_dir: Path) -> dict | None:
    """The reading and plan written by the previous run, if this folder has one."""

    path = Path(output_dir) / "output.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    plan = payload.get("plan")
    reading = payload.get("reading")
    if not isinstance(plan, dict) or not isinstance(reading, dict):
        return None
    return {"plan": plan, "reading": reading}


def _plain_word(word: str) -> str:
    return word.strip(".,;:!?\"'`()[]{}")


def _clauses(sentence: str) -> list[str]:
    text = " ".join(sentence.lower().split())
    if not text:
        return []
    parts = re.split(r"[,.;!?]|\band\b", text)
    cleaned = []
    for part in parts:
        words = [_plain_word(word) for word in part.split()]
        words = [word for word in words if word]
        if words:
            cleaned.append(" ".join(words))
    return cleaned


def _tokens(clause: str) -> list[str]:
    return [word for word in clause.split() if word not in _FILLER]


def _ignored_words(clause: str) -> list[str]:
    words = []
    for word in clause.split():
        if word in _FILLER or word == "and":
            continue
        words.append(word)
    return words


# Misspellings that still mean the reflection stage.
_REFLECTION_WORDS = {
    "reflection": "reflection",
    "refelction": "reflection",
    "refleciton": "reflection",
    "refection": "reflection",
    "reflections": "reflection",
}
_CLARITY = frozenset({"clearer", "clear", "clarify", "clarified", "clarity", "sharper", "sharpen", "sharp"})
_CREATURE = frozenset({"bird", "birds", "animal", "creature", "wildlife"})
# A sentence may name the thing. These words are the subject, not leftovers.
NAMED_SUBJECTS = frozenset(
    {
        "cat",
        "cats",
        "dog",
        "dogs",
        "fox",
        "foxes",
        "heron",
        "bird",
        "birds",
        "animal",
        "creature",
        "wildlife",
        "person",
        "man",
        "bike",
        "bicycle",
        "cyclist",
        "train",
        "fuji",
        "mountain",
    }
)
_LIFT_CUES = frozenset(
    {
        "lift",
        "light",
        "seen",
        "prominent",
        "theme",
        "pop",
        "brighten",
        "brighter",
        "highlight",
        "bright",
        "catchlight",
    }
)
# Glue inside a request. These are not a separate ignored list.
_GLUE = frozenset(
    {"with", "as", "to", "of", "for", "main", "make", "it", "be", "is", "on", "in", "so", "his", "her", "their", "mount"}
)

_BIRD_REASON = "The bird is the subject, so it is lifted and not crushed."
_WATER_REASON = "The water around the bird should go darker."
_REFLECTION_REASON = "The reflection under the bird should get clearer."


def parse_clause(clause: str) -> dict | None:
    """One grammar clause, or None when the words are not a request."""

    tokens = _tokens(clause)
    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0] in _OPEN:
        return {"kind": "strong", "word": tokens[0]}
    if tokens == ["gradient"]:
        return {"kind": "gradient"}
    tool = _expert_clause(tokens)
    if tool is not None:
        return tool
    scale = _scale_clause(tokens)
    if scale is not None:
        return scale
    if tokens[0] in _SKIP_VERBS and len(tokens) == 2 and tokens[1] in _STAGE_WORDS:
        return {"kind": "skip", "stage": _STAGE_WORDS[tokens[1]], "said": tokens[1]}
    if tokens[0] in _ASK_VERBS:
        verb = "lift" if tokens[0] in {"lift", "light"} else tokens[0]
        rest = tokens[1:]
        # "light behind the subject" names the behind region, not the subject.
        if "behind" in rest:
            return {"kind": "ask", "verb": verb, "target": "behind"}
        found = [word for word in rest if word in _TARGETS]
        if len(found) == 1 and len(rest) == 1:
            return {"kind": "ask", "verb": verb, "target": found[0]}
    return None


_SCALE_PHRASES = {
    ("more", "eye", "clarity"): ("eyes", 1.35),
    ("less", "eye", "clarity"): ("eyes", 0.7),
    ("more", "eyes"): ("eyes", 1.35),
    ("less", "eyes"): ("eyes", 0.7),
    ("more", "background", "blur"): ("background", 1.35),
    ("less", "background", "blur"): ("background", 0.7),
    ("more", "subject", "lift"): ("subject", 1.35),
    ("less", "subject", "lift"): ("subject", 0.7),
}


def _scale_clause(tokens: list[str]) -> dict | None:
    """A follow-up amount: more eye clarity, less background blur, less subject lift."""

    found = _SCALE_PHRASES.get(tuple(tokens))
    if found is None:
        return None
    stage, factor = found
    return {"kind": "scale", "stage": stage, "factor": factor}


def _expert_clause(tokens: list[str]) -> dict | None:
    """A direct module request: shadows, highlights, warmth, color, contrast, crop."""

    if tokens in (["lift", "shadows"], ["open", "shadows"], ["raise", "shadows"], ["shadows"]):
        return {"kind": "tool", "name": "shadow_lift", "value": 0.5}
    if tokens in (["hold", "highlights"], ["recover", "highlights"], ["highlights"]):
        return {"kind": "tool", "name": "highlight_hold", "value": True}
    if tokens in (["more", "color"], ["more", "saturation"]):
        return {"kind": "tool", "name": "color", "value": 0.35}
    if tokens in (["less", "color"], ["less", "saturation"]):
        return {"kind": "tool", "name": "color", "value": -0.35}
    if tokens in (["more", "contrast"],):
        return {"kind": "tool", "name": "contrast_push", "value": 0.25}
    if len(tokens) == 1 and tokens[0] in _EXPERT_WORDS:
        name, value = _EXPERT_WORDS[tokens[0]]
        return {"kind": "tool", "name": name, "value": value}
    return None


def _consume_expert(tokens: list[str], consumed: set[int]) -> list[dict]:
    """Mark expert words inside a looser sentence and return their tools."""

    items: list[dict] = []
    if "shadows" in tokens and any(word in tokens for word in ("lift", "open", "raise", "brighten")):
        items.append({"kind": "tool", "name": "shadow_lift", "value": 0.5})
        for index, word in enumerate(tokens):
            if word in {"shadows", "lift", "open", "raise", "brighten"}:
                consumed.add(index)
    if "highlights" in tokens and any(word in tokens for word in ("hold", "recover", "pull")):
        items.append({"kind": "tool", "name": "highlight_hold", "value": True})
        for index, word in enumerate(tokens):
            if word in {"highlights", "hold", "recover", "pull"}:
                consumed.add(index)
    for index, word in enumerate(tokens):
        if index in consumed or word not in _EXPERT_WORDS:
            continue
        name, value = _EXPERT_WORDS[word]
        items.append({"kind": "tool", "name": name, "value": value})
        consumed.add(index)
        if word in {"warm", "cool"} and index + 1 < len(tokens) and tokens[index + 1] in {"it", "up", "down"}:
            consumed.add(index + 1)
    if "color" in tokens and any(word in tokens for word in ("more", "less")):
        value = 0.35 if "more" in tokens else -0.35
        items.append({"kind": "tool", "name": "color", "value": value})
        for index, word in enumerate(tokens):
            if word in {"color", "more", "less"}:
                consumed.add(index)
    return items


def _loose_clause(clause: str) -> tuple[list[dict], list[str]]:
    """Read a sentence that is not in the strict grammar.

    ``dark end result`` darkens the surround. A bird named as the subject
    is lifted, not crushed. A reflection, including the usual typos, is
    lifted when the user wants it clearer.
    """

    tokens = _tokens(clause)
    if not tokens:
        return [], []
    items: list[dict] = []
    consumed: set[int] = set()
    items.extend(_consume_expert(tokens, consumed))

    for index, word in enumerate(tokens):
        if word in _OPEN:
            items.append({"kind": "strong", "word": word})
            consumed.add(index)

    reflection_at = [index for index, word in enumerate(tokens) if word in _REFLECTION_WORDS]
    if reflection_at:
        clarity = any(word in _CLARITY or word in {"lift", "light", "make"} for word in tokens)
        holding = "hold" in tokens and not any(word in _CLARITY for word in tokens)
        darkening = (
            "darken" in tokens
            and not clarity
            and "end" not in tokens
            and "result" not in tokens
        )
        if holding:
            verb = "hold"
            reason = ""
        elif darkening:
            verb = "darken"
            reason = ""
        else:
            verb = "lift"
            reason = _REFLECTION_REASON
        items.append({"kind": "ask", "verb": verb, "target": "reflection", "reason": reason})
        consumed.update(reflection_at)
        for index, word in enumerate(tokens):
            if word in _CLARITY or word in {"make", "lift", "light"}:
                consumed.add(index)
            if verb == "hold" and word == "hold":
                consumed.add(index)
            if verb == "darken" and word == "darken":
                consumed.add(index)

    if any(word in {"dark", "darken"} for word in tokens) and any(word in {"end", "result"} for word in tokens):
        items.append({"kind": "ask", "verb": "darken", "target": "foreground", "reason": _WATER_REASON})
        for index, word in enumerate(tokens):
            if word in {"dark", "darken", "end", "result"}:
                consumed.add(index)

    creature = any(word in _CREATURE for word in tokens)
    if creature and "subject" in tokens:
        items.append({"kind": "ask", "verb": "lift", "target": "subject", "reason": _BIRD_REASON})
        for index, word in enumerate(tokens):
            if word in _CREATURE or word in {"subject", "main"}:
                consumed.add(index)
    elif "main" in tokens and "subject" in tokens and not any(
        item.get("target") == "subject" for item in items
    ):
        items.append({"kind": "ask", "verb": "lift", "target": "subject", "reason": _BIRD_REASON})
        for index, word in enumerate(tokens):
            if word in {"subject", "main"}:
                consumed.add(index)

    named_at = [index for index, word in enumerate(tokens) if word in NAMED_SUBJECTS]
    lifting = any(word in _LIFT_CUES or word == "subject" for word in tokens)
    if named_at and lifting and not any(item.get("target") == "subject" for item in items):
        word = tokens[named_at[0]]
        items.append(
            {
                "kind": "ask",
                "verb": "lift",
                "target": "subject",
                "reason": f"The {word} is the subject, so it is lifted.",
            }
        )
    if named_at and lifting:
        consumed.update(named_at)
        for index, word in enumerate(tokens):
            if word in _LIFT_CUES or word in {"subject", "main", "theme", "mount"}:
                consumed.add(index)

    eye_at = [index for index, word in enumerate(tokens) if word in {"eye", "eyes"}]
    if eye_at and any(word in _LIFT_CUES or word in _CLARITY for word in tokens):
        if not any(item.get("target") == "eyes" for item in items):
            items.append(
                {
                    "kind": "ask",
                    "verb": "lift",
                    "target": "eyes",
                    "reason": "The eyes are lifted.",
                }
            )
        consumed.update(eye_at)
        for index, word in enumerate(tokens):
            if word in _LIFT_CUES or word in _CLARITY:
                consumed.add(index)

    if any(word in {"lantern", "lanterns"} for word in tokens) and any(word in _LIFT_CUES for word in tokens):
        items.append({"kind": "hue", "name": "orange"})
        for index, word in enumerate(tokens):
            if word in {"lantern", "lanterns"} or word in _LIFT_CUES:
                consumed.add(index)

    if items:
        for index, word in enumerate(tokens):
            if word in _GLUE:
                consumed.add(index)
    ignored = [tokens[index] for index in range(len(tokens)) if index not in consumed]
    return items, ignored


def _interpret(clause: str) -> tuple[list[dict], list[str]]:
    parsed = parse_clause(clause)
    if parsed is not None:
        return [parsed], []
    return _loose_clause(clause)


def parse_steer(sentence: str) -> dict:
    """Split a steer line. Caps open only when the user wrote strong or epic."""

    strong = False
    strong_word = ""
    gradient = False
    skip: list[str] = []
    asks: list[dict] = []
    tools: dict[str, object] = {}
    scales: list[dict] = []
    ignored: list[str] = []
    for clause in _clauses(sentence):
        items, leftover = _interpret(clause)
        if not items:
            ignored.extend(leftover or _ignored_words(clause))
            continue
        ignored.extend(leftover)
        for parsed in items:
            if parsed["kind"] == "strong":
                strong = True
                strong_word = str(parsed["word"])
            elif parsed["kind"] == "gradient":
                gradient = True
            elif parsed["kind"] == "skip":
                stage = str(parsed["stage"])
                if stage not in skip:
                    skip.append(stage)
            elif parsed["kind"] == "tool":
                tools[str(parsed["name"])] = parsed["value"]
            elif parsed["kind"] == "scale":
                scales.append({"stage": str(parsed["stage"]), "factor": float(parsed["factor"])})
            elif parsed["kind"] == "hue":
                continue
            else:
                ask = {
                    "verb": parsed["verb"],
                    "target": parsed["target"],
                    "source": "user",
                }
                reason = str(parsed.get("reason") or "").strip()
                if reason:
                    ask["reason"] = reason
                asks.append(ask)
    summary = ""
    if ignored:
        summary = "Ignored: " + ", ".join(ignored) + "."
    return {
        "strong": strong,
        "caps_open": strong,
        "strong_word": strong_word,
        "gradient": gradient,
        "skip": skip,
        "asks": asks,
        "tools": tools,
        "scales": scales,
        "ignored": ignored,
        "summary": summary,
    }


def _same_ask(item: dict, other: dict) -> bool:
    return item.get("verb") == other.get("verb") and item.get("target") == other.get("target")


_BLOCKED = frozenset({"strong", "epic", "caps", "caps_open", "open"})
_ASK_TARGET = {
    "surround": "foreground",
    "background": "foreground",
    "subject": "subject",
    "sky": "sky",
    "reflection": "reflection",
    "behind": "behind",
    "foreground": "foreground",
    "rock": "rock",
    "eyes": "eyes",
    "eye": "eyes",
}


def _sync_caps(plan: dict) -> dict:
    """Caps follow the user's strong or epic word. A proposal cannot set them."""

    plan["caps_open"] = bool(plan.get("strong"))
    return plan


def apply_proposal(plan: dict, proposal: dict) -> dict:
    """Copy a validated reasoner plan onto the renderer plan. Caps stay shut."""

    updated = _copy_plan(plan)
    status = str(proposal.get("status") or "skipped")
    backend = str(proposal.get("backend") or "")
    note = str(proposal.get("note") or "")
    if not note and status == "skipped":
        note = "Reasoning was skipped."
    paragraph = _usable_paragraph(str(proposal.get("paragraph") or ""))
    updated["reasoning"] = {"status": status, "backend": backend, "note": note}
    failure = str(proposal.get("failure") or "").strip()
    if failure:
        updated["reasoning"]["failure"] = failure
    if paragraph:
        updated["reasoning"]["paragraph"] = paragraph
    proposed: list[dict] = []
    skip_by = dict(updated.get("skip_by") or {})
    for item in proposal.get("stages") or []:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or "").strip().lower()
        action = str(item.get("action") or "").strip().lower()
        reason = str(item.get("reason") or "").strip()
        if _is_placeholder(reason):
            continue
        if stage in _BLOCKED or action in _BLOCKED:
            continue
        if action == "light":
            action = "lift"
        if action == "revert":
            action = "skip"
        if action == "skip":
            stored = _STAGE_WORDS.get(stage, "")
            if not stored:
                continue
            if stored not in updated["skip"]:
                updated["skip"].append(stored)
            if skip_by.get(stored) != "user":
                skip_by[stored] = "reasoner"
            proposed.append(
                {
                    "stage": "surround" if stored == "background" else stored,
                    "action": "skip",
                    "reason": reason,
                    "source": "reasoner",
                }
            )
            continue
        target = _ASK_TARGET.get(stage, "")
        if action in {"darken", "lift", "hold"} and target:
            ask = {"verb": action, "target": target, "source": "reasoner", "reason": reason}
            proposed.append({"stage": stage, "action": action, "reason": reason, "source": "reasoner"})
            if not any(_same_ask(ask, old) for old in updated["asks"]):
                updated["asks"].append(ask)
        if stage == "gradient" and action in {"lift", "hold", "darken"}:
            updated["gradient"] = True
            if updated.get("gradient_by") != "user":
                updated["gradient_by"] = "reasoner"
            proposed.append({"stage": "gradient", "action": action, "reason": reason, "source": "reasoner"})
    updated["proposed"] = proposed
    updated["skip_by"] = skip_by
    if proposal.get("operations"):
        updated["operations"] = [
            dict(item) for item in proposal["operations"] if isinstance(item, dict)
        ]
    if proposal.get("dropped"):
        updated["dropped"] = [str(item) for item in proposal["dropped"] if str(item).strip()]
    # A reasoner never opens the caps. Leave a strong flag the user already set.
    return _sync_caps(updated)


def revise_plan(plan: dict, sentence: str) -> dict:
    """Apply the user's clauses on top of a plan. These win over the reasoner."""

    updated = _copy_plan(plan)
    parsed = parse_steer(sentence)
    skip_by = dict(updated.get("skip_by") or {})
    if parsed["strong"]:
        updated["strong"] = True
        updated["strong_word"] = parsed["strong_word"]
    if parsed["gradient"]:
        updated["gradient"] = True
        updated["gradient_by"] = "user"
    for stage in parsed["skip"]:
        if stage not in updated["skip"]:
            updated["skip"].append(stage)
        skip_by[stage] = "user"
    for ask in parsed["asks"]:
        updated["asks"] = [
            item
            for item in updated["asks"]
            if not (item.get("target") == ask["target"] and item.get("source") != "user")
        ]
        # A second user clause for the same target replaces the earlier one.
        updated["asks"] = [item for item in updated["asks"] if not _same_ask(item, ask)]
        updated["asks"].append(ask)
    for name, value in (parsed.get("tools") or {}).items():
        updated[str(name)] = value
    scales = {str(key): float(value) for key, value in (updated.get("scales") or {}).items()}
    for item in parsed.get("scales") or []:
        stage = str(item.get("stage") or "")
        if not stage:
            continue
        current = float(scales.get(stage) or 1.0)
        scales[stage] = max(0.25, min(1.5, current * float(item.get("factor") or 1.0)))
    updated["scales"] = scales
    updated["ignored"] = list(parsed["ignored"])
    updated["summary"] = parsed["summary"]
    updated["skip_by"] = skip_by
    if (
        parsed["asks"]
        or parsed["skip"]
        or parsed["strong"]
        or parsed["gradient"]
        or parsed.get("tools")
        or parsed.get("scales")
    ):
        updated["user"] = True
    updated = _sync_caps(updated)
    updated = _ensure_paragraph(updated)
    if any(item.get("target") == "reflection" for item in parsed["asks"]) and "reflection" not in parsed["skip"]:
        # A guide or reasoner skip loses to a request to change the mirror.
        updated["skip"] = [stage for stage in updated["skip"] if stage != "reflection"]
        skip_by = dict(updated.get("skip_by") or {})
        skip_by.pop("reflection", None)
        updated["skip_by"] = skip_by
    return updated


def _copy_plan(plan: dict) -> dict:
    return {
        "guide_stages": list(plan.get("guide_stages") or []),
        "skip": list(plan.get("skip") or []),
        "scales": {str(key): float(value) for key, value in (plan.get("scales") or {}).items()},
        "pinned": str(plan.get("pinned") or ""),
        "theme": str(plan.get("theme") or ""),
        "reflection": str(plan.get("reflection") or "lift"),
        "strong": bool(plan.get("strong")),
        "caps_open": bool(plan.get("strong")),
        "strong_word": str(plan.get("strong_word") or ""),
        "gradient": bool(plan.get("gradient")),
        "gradient_by": str(plan.get("gradient_by") or "")
        if plan.get("gradient_by") in {"user", "reasoner"}
        else "",
        "shadow_lift": float(plan.get("shadow_lift") or 0),
        "highlight_hold": bool(plan.get("highlight_hold")),
        "warmth": float(plan.get("warmth") or 0),
        "color": float(plan.get("color") or 0),
        "contrast_push": float(plan.get("contrast_push") or 0),
        "crop_horizon": bool(plan.get("crop_horizon")),
        "skip_by": {
            str(key): str(value)
            for key, value in (plan.get("skip_by") or {}).items()
            if value in {"user", "reasoner"}
        },
        "asks": [dict(item) for item in (plan.get("asks") or []) if isinstance(item, dict)],
        "proposed": [dict(item) for item in (plan.get("proposed") or []) if isinstance(item, dict)],
        "ignored": list(plan.get("ignored") or []),
        "summary": str(plan.get("summary") or ""),
        "reasoning": dict(plan.get("reasoning") or {}),
        "lines": [],
        "user": bool(plan.get("user")),
        "features_committed": [str(item) for item in (plan.get("features_committed") or [])],
        "features_skipped": [str(item) for item in (plan.get("features_skipped") or [])],
        "features_done": [str(item) for item in (plan.get("features_done") or [])],
        "features_next": str(plan.get("features_next") or ""),
        "features_path": [str(item) for item in (plan.get("features_path") or [])],
        "operations": [dict(item) for item in (plan.get("operations") or []) if isinstance(item, dict)],
        "dropped": [str(item) for item in (plan.get("dropped") or []) if str(item).strip()],
        "catalog_chosen": bool(plan.get("catalog_chosen")),
        "history_chosen": [str(item) for item in (plan.get("history_chosen") or [])],
        "history_skipped": [str(item) for item in (plan.get("history_skipped") or [])],
        "display": str(plan.get("display") or ""),
    }


def _display(stage: str) -> str:
    return _DISPLAY.get(stage, stage)


def _ask_for(plan: dict, target: str, verbs: set[str] | None = None) -> dict | None:
    for item in plan.get("asks") or []:
        if item.get("target") != target:
            continue
        if verbs is not None and item.get("verb") not in verbs:
            continue
        return item
    return None


_ASK_LINE = {
    "subject": "subject",
    "foreground": "surround",
    "rock": "surround",
    "sky": "sky",
    "behind": "behind",
    "reflection": "reflection",
}


def _proposed_for(plan: dict, stage: str) -> dict | None:
    for item in plan.get("proposed") or []:
        if item.get("stage") == stage:
            return item
    return None


def build_run_lines(plan: dict, separation: dict) -> list[dict]:
    """The ordered stages this run will use, taken from the plan the renderer used."""

    ran = set(separation.get("stages_ran") or [])
    render_skipped = set(separation.get("skipped") or [])
    user_skip = set(plan.get("skip") or [])
    order: list[str] = []
    for stage in list(plan.get("guide_stages") or []):
        if stage not in order:
            order.append(stage)
    for extra in ("gradient", "theme", "vignette"):
        if extra not in order:
            order.append(extra)
    for stage in list(plan.get("skip") or []):
        if stage not in order:
            order.append(stage)
    gradient_info = separation.get("gradient") or {}
    theme_ran = "theme" in ran
    vignette_on = bool(separation.get("vignette"))
    lines: list[dict] = []
    for stage in order:
        if stage == "gradient":
            considered = bool(gradient_info) or stage in user_skip or bool(plan.get("gradient"))
        elif stage == "theme":
            considered = theme_ran or stage in user_skip or bool(plan.get("theme"))
        elif stage == "vignette":
            considered = vignette_on or stage in user_skip
        else:
            considered = stage in set(plan.get("guide_stages") or []) or stage in user_skip or stage in ran
        if not considered:
            continue
        if stage == "vignette":
            skipped = stage in user_skip or not vignette_on
        elif stage in {"gradient", "theme"}:
            skipped = stage in user_skip or stage not in ran
        else:
            # A stage that never landed in the render did not run. An eye
            # that overshot used to sit on neither list and still show up.
            skipped = stage in user_skip or stage not in ran or stage in render_skipped
        reason, source, override = _line_reason(plan, separation, stage, skipped)
        failed = False
        if (
            stage == "reflection"
            and skipped
            and stage not in user_skip
            and _user_ask(plan, "reflection", {"lift"})
        ):
            # The user asked to clarify the mirror. A miss is a failure, not a
            # skip that still claims the reflection will get clearer.
            reason = str(separation.get("reflection_failure") or "").strip()
            if not reason:
                reason = "The reflection under the bird could not be found, so it was not clarified."
            source = "guide"
            override = False
            failed = True
        lines.append(
            {
                "stage": _display(stage),
                "reason": reason,
                "skipped": False if failed else skipped,
                "failed": failed,
                "source": source,
                "override": override,
                "strong": bool(plan.get("caps_open")) and not skipped and not failed,
            }
        )
    shown = {str(line["stage"]) for line in lines}
    for item in plan.get("asks") or []:
        if not isinstance(item, dict):
            continue
        name = _ASK_LINE.get(str(item.get("target") or ""))
        if not name or name in shown:
            continue
        source = str(item.get("source") or "guide")
        verb = str(item.get("verb") or "")
        reason = str(item.get("reason") or "").strip()
        if not reason:
            who = "You" if source == "user" else "The reasoner" if source == "reasoner" else "The guide"
            if name == "behind":
                reason = f"{who} asked to {verb} behind the subject."
            else:
                reason = f"{who} asked to {verb} the {name}."
        ran_names = set(separation.get("stages_ran") or [])
        if name == "surround":
            ran_now = "foreground" in ran_names or "background" in ran_names
        else:
            ran_now = name in ran_names
        lines.append(
            {
                "stage": name,
                "reason": reason,
                "skipped": not ran_now,
                "source": source,
                "override": False,
                "strong": bool(plan.get("caps_open")) and ran_now,
            }
        )
        shown.add(name)
    return lines


def _proposed_stage(plan: dict, stage: str) -> dict | None:
    if stage == "background":
        return (
            _proposed_for(plan, "surround")
            or _proposed_for(plan, "background")
            or _proposed_for(plan, "foreground")
            or _proposed_for(plan, "rock")
        )
    return _proposed_for(plan, stage)


def _skip_source(plan: dict, stage: str) -> str:
    named = (plan.get("skip_by") or {}).get(stage)
    if named in {"user", "reasoner"}:
        return str(named)
    proposed = _proposed_stage(plan, stage)
    if proposed is not None and proposed.get("action") == "skip":
        return "reasoner"
    return "user"


def _is_placeholder(text: str) -> bool:
    cleaned = " ".join(str(text or "").lower().replace(".", " ").split())
    return cleaned == "one short sentence" or cleaned.startswith("one short sentence ")


def _usable_paragraph(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned or _is_placeholder(cleaned):
        return ""
    if len(cleaned) > 600:
        cleaned = cleaned[:600].rstrip()
    return cleaned


def _user_ask(plan: dict, target: str, verbs: set[str]) -> bool:
    for item in plan.get("asks") or []:
        if item.get("source") != "user":
            continue
        if item.get("target") == target and item.get("verb") in verbs:
            return True
    return False


def _ensure_paragraph(plan: dict) -> dict:
    """Keep the paragraph the model wrote. Do not replace it with a stock story."""

    reasoning = dict(plan.get("reasoning") or {})
    current = _usable_paragraph(str(reasoning.get("paragraph") or ""))
    if current:
        reasoning["paragraph"] = current
    plan["reasoning"] = reasoning
    return plan


def _who(source: str) -> str:
    if source == "user":
        return "You"
    if source == "reasoner":
        return "The reasoner"
    return "The guide"


def _spoken(item: dict, fallback: str) -> str:
    reason = str(item.get("reason") or "").strip()
    return reason or fallback


def _line_reason(plan: dict, separation: dict, stage: str, skipped: bool) -> tuple[str, str, bool]:
    """Short reason, where it came from, and whether the user overrode the reasoner."""

    if skipped and stage in set(plan.get("skip") or []):
        source = _skip_source(plan, stage)
        proposed = _proposed_stage(plan, stage)
        override = source == "user" and proposed is not None and proposed.get("action") != "skip"
        reason = ""
        if source == "reasoner" and proposed is not None:
            reason = str(proposed.get("reason") or "").strip()
        if not reason:
            reason = f"{_display(stage)} was reverted."
        return reason, source, override
    proposed = _proposed_stage(plan, stage)
    if stage == "grade":
        return "The measured frame grade.", "guide", False
    if stage == "subject":
        item = _ask_for(plan, "subject")
        if item is not None:
            source = str(item.get("source") or "guide")
            override = source == "user" and proposed is not None and proposed.get("action") != item.get("verb")
            fallback = f"{_who(source)} asked to {item.get('verb')} the subject."
            return _spoken(item, fallback), source, override
        return "The subject is lifted a little.", "guide", False
    if stage == "background":
        item = _ask_for(plan, "foreground") or _ask_for(plan, "rock")
        if item is not None:
            source = str(item.get("source") or "guide")
            override = source == "user" and proposed is not None and proposed.get("action") != item.get("verb")
            fallback = f"{_who(source)} asked to {item.get('verb')} the foreground."
            return _spoken(item, fallback), source, override
        return "The surround stays behind the subject.", "guide", False
    if stage == "gradient":
        info = separation.get("gradient") or {}
        reason = str(info.get("reason") or "").strip()
        if not reason and proposed is not None:
            reason = str(proposed.get("reason") or "").strip()
        if not reason:
            reason = "A soft linear gradient darkens the bright end."
        source = str(plan.get("gradient_by") or "")
        if source not in {"user", "reasoner"}:
            source = "guide"
        return reason, source, False
    if stage == "theme":
        kind = str(plan.get("theme") or separation.get("theme") or "theme")
        return f"The {kind} stays out of the surround.", "guide", False
    if stage == "vignette":
        return "The history vignette darkens around the subject.", "guide", False
    if stage == "reflection":
        item = _ask_for(plan, "reflection")
        if item is not None:
            source = str(item.get("source") or "guide")
            fallback = f"{_who(source)} asked to {item.get('verb')} the reflection."
            return _spoken(item, fallback), source, False
        intention = str(plan.get("reflection") or "lift")
        return f"The reflection is {intention}.", "guide", False
    if stage == "eyes":
        return "The eye is clarified.", "guide", False
    if stage == "skin":
        return "Skin is graded on its own.", "guide", False
    if stage == "sky":
        item = _ask_for(plan, "sky")
        if item is not None:
            source = str(item.get("source") or "guide")
            fallback = f"{_who(source)} asked to {item.get('verb')} the sky."
            return _spoken(item, fallback), source, False
        return "The sky is graded on its own.", "guide", False
    if stage == "horizon":
        return "The sunset line is warmed on its own.", "guide", False
    if stage == "foreground":
        return "The land is graded on its own.", "guide", False
    if stage == "highlights":
        return "Highlights are held.", "guide", False
    return "This stage is part of the guide.", "guide", False


def _visible_lines(plan: dict) -> list[dict]:
    """Stages a person can answer. Guide bookkeeping stays out of the list."""

    visible = []
    for line in plan.get("lines") or []:
        if not isinstance(line, dict):
            continue
        source = str(line.get("source") or "guide")
        stage = str(line.get("stage") or "")
        if line.get("skipped") and source not in {"user", "reasoner"}:
            continue
        if not line.get("skipped") and stage in {"grade", "vignette"}:
            continue
        visible.append(line)
    return visible


def _split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [part.strip() for part in parts if part.strip()]


def _paragraph_with_reflection_failure(plan: dict) -> str:
    """Replace a promise to clarify the mirror when that stage did not run."""

    reasoning = plan.get("reasoning") or {}
    paragraph = str(reasoning.get("paragraph") or "").strip()
    failure = ""
    for line in _visible_lines(plan):
        if line.get("failed"):
            failure = str(line.get("reason") or "").strip()
            break
    if not failure:
        return paragraph
    success = "The reflection under the bird should get clearer."
    if success in paragraph:
        paragraph = paragraph.replace(success, failure)
    elif failure not in paragraph:
        paragraph = (paragraph + " " + failure).strip()
    if paragraph != str(reasoning.get("paragraph") or "").strip():
        updated = dict(reasoning)
        updated["paragraph"] = paragraph
        plan["reasoning"] = updated
    return paragraph


def format_plan_text(plan: dict) -> str:
    """A paragraph, then the stages that will run, each with a because-clause."""

    rows = []
    reasoning = plan.get("reasoning") or {}
    paragraph = _paragraph_with_reflection_failure(plan)
    note = str(reasoning.get("note") or "").strip()
    if paragraph:
        rows.append(paragraph)
    elif note:
        rows.append(note)
    if plan.get("caps_open") or plan.get("strong"):
        word = str(plan.get("strong_word") or "strong")
        caps_line = f"Caps open because you said {word}."
    else:
        caps_line = "Caps closed."
    if caps_line not in paragraph and not (paragraph and caps_line == "Caps closed."):
        rows.append(caps_line)
    skipped_names: list[str] = []
    for line in _visible_lines(plan):
        name = str(line.get("stage") or "")
        reason = str(line.get("reason") or "").strip()
        if _is_placeholder(reason):
            reason = ""
        source = str(line.get("source") or "guide")
        if source not in {"user", "reasoner", "guide"}:
            source = "guide"
        if line.get("failed"):
            rows.append(f"{name} — {reason} — {source}")
            continue
        mark = "skipped — " if line.get("skipped") else ""
        override = " — override" if line.get("override") else ""
        rows.append(f"{name} — {mark}{reason} — {source}{override}")
        if line.get("skipped") and name and name not in skipped_names:
            skipped_names.append(name)
    if skipped_names:
        rows.append("Skipped: " + ", ".join(skipped_names) + ".")
    tools = _tool_line(plan)
    if tools:
        rows.append(tools)
    summary = str(plan.get("summary") or "").strip()
    if summary:
        rows.append(summary)
    elif plan.get("ignored"):
        rows.append("Ignored: " + ", ".join(str(word) for word in plan["ignored"]) + ".")
    for key in ("warmth_note", "gradient_note"):
        note = str(plan.get(key) or "").strip()
        if note and note not in rows:
            rows.append(note)
    return "\n".join(rows)


def format_plan_label(plan: dict) -> str:
    """First two sentences of the paragraph, then the stage names.

    Without a paragraph the label is the full plan text.
    """

    paragraph = str((plan.get("reasoning") or {}).get("paragraph") or "").strip()
    if not paragraph:
        return format_plan_text(plan)
    head = " ".join(_split_sentences(paragraph)[:2]).strip()
    names: list[str] = []
    for line in _visible_lines(plan):
        if line.get("skipped"):
            continue
        name = str(line.get("stage") or "")
        if name and name not in names:
            names.append(name)
    if names:
        label = f"{head} Stages: {', '.join(names)}."
    else:
        label = head
    tools = _tool_line(plan)
    if tools:
        return f"{label} {tools}"
    return label


def _tool_line(plan: dict) -> str:
    """Name the expert modules a sentence or a picked look turned on."""

    names: list[str] = []
    if float(plan.get("shadow_lift") or 0) > 0 or plan.get("highlight_hold"):
        names.append("tone equalizer")
    if abs(float(plan.get("warmth") or 0)) > 0:
        names.append("color calibration")
    if abs(float(plan.get("color") or 0)) > 0:
        names.append("color zones")
    if plan.get("gradient") and not plan.get("gradient_note"):
        names.append("graduated density")
    if plan.get("crop_horizon"):
        names.append("crop")
    if abs(float(plan.get("contrast_push") or 0)) > 0:
        names.append("contrast")
    if not names:
        return ""
    return "Expert tools: " + ", ".join(names) + "."


def plan_stages(
    steer: str = "",
    proposal: dict | None = None,
    *,
    guide_stages: list[str] | None = None,
    separation: dict | None = None,
    reflection: str = "lift",
    theme: str = "",
) -> dict:
    """Apply a reasoner proposal, then the user's steer. Caps stay shut unless the user opens them."""

    plan = empty_plan(list(guide_stages or []), reflection=reflection, theme=theme)
    if proposal is not None:
        plan = apply_proposal(plan, proposal)
    if str(steer or "").strip():
        plan = revise_plan(plan, steer)
    else:
        plan = _ensure_paragraph(_sync_caps(plan))
    if separation is not None:
        plan["lines"] = build_run_lines(plan, separation)
    plan["text"] = format_plan_text(plan)
    plan["label"] = format_plan_label(plan)
    return plan


def stage_intents(plan: dict) -> dict:
    """Caps flag and stage requests to pass into the renderer.

    ``caps_open`` is true only when the user wrote strong or epic.
    """

    asks = [dict(item) for item in (plan.get("asks") or []) if isinstance(item, dict)]
    if plan.get("gradient") and not any(item.get("verb") == "gradient" for item in asks):
        source = str(plan.get("gradient_by") or "")
        if source not in {"user", "reasoner"}:
            source = "user" if plan.get("user") else "reasoner"
        asks.append({"verb": "gradient", "target": "", "source": source})
    return {
        "caps_open": bool(plan.get("caps_open") or plan.get("strong")),
        "strong": bool(plan.get("strong")),
        "asks": asks,
        "skip": list(plan.get("skip") or []),
        "stages": active_stages(plan),
    }
