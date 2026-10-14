"""Check a plan against the picture before any look is offered.

The reasoner sometimes names water, a reflection, eyes, or an animal that
the frame and the user's words do not contain. Those claims are removed
here. A faithful grade also stays with the measurement, so a model that
asks for warmth cannot install a red cast before the choice buttons exist.

This is a local check. It does not call another model.
"""

from __future__ import annotations

import re

from ai_dt.schema import Atmosphere

# Words that must already be in the picture reading or the user's steer.
_CLAIMS = (
    ("water",),
    ("reflection", "reflections"),
    ("eye", "eyes"),
    ("animal", "animals", "bird", "birds", "dog", "cat", "wildlife"),
)



def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text or "").lower()))


def _mentions(text: str, words: tuple[str, ...]) -> bool:
    found = _tokens(text)
    return any(word in found for word in words)


def _evidence(picture: object, reading: object, steer: str) -> str:
    chunks = [
        steer or "",
        str(getattr(picture, "reason", "") or ""),
        str(getattr(picture, "type", "") or ""),
    ]
    if reading is not None:
        chunks.append(str(getattr(reading, "sentence", "") or ""))
        for region in getattr(reading, "regions", None) or []:
            if isinstance(region, dict):
                chunks.extend(str(region.get(key) or "") for key in ("role", "kind", "note"))
        for problem in getattr(reading, "problems", None) or []:
            if isinstance(problem, dict):
                chunks.append(str(problem.get("sentence") or problem.get("kind") or ""))
            else:
                chunks.append(str(problem))
        for line in getattr(reading, "leave_alone", None) or []:
            chunks.append(str(line))
    return " ".join(chunks)


def ungrounded_words(picture: object, reading: object, steer: str) -> tuple[str, ...]:
    """Claim words that neither the picture reading nor the steer contains."""

    have = _tokens(_evidence(picture, reading, steer))
    banned: list[str] = []
    for group in _CLAIMS:
        if any(word in have for word in group):
            continue
        banned.extend(group)
    return tuple(banned)


def _strip_text(text: str, banned: tuple[str, ...]) -> str:
    """Drop clauses that introduce a claim the picture did not support."""

    cleaned = " ".join(str(text or "").split())
    if not cleaned or not banned:
        return cleaned
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    kept: list[str] = []
    for sentence in sentences:
        parts = re.split(r"\s+\b(?:while|whilst)\b\s+", sentence, maxsplit=1, flags=re.IGNORECASE)
        chosen: list[str] = []
        for part in parts:
            clauses = re.split(r",\s*", part)
            good = [clause.strip() for clause in clauses if clause.strip() and not _mentions(clause, banned)]
            if good:
                chosen.append(", ".join(good))
        if not chosen:
            continue
        line = " ".join(chosen).strip(" ,")
        if line and line[-1] not in ".!?":
            line += "."
        kept.append(line)
    return " ".join(kept).strip()


def review_plan(plan: dict, picture: object, reading: object, steer: str) -> dict:
    """Remove invented water, reflection, eye, and animal claims.

    Asks the user wrote, including a look they picked, stay. Caps stay as
    the steer left them: this function never opens them.
    """

    banned = ungrounded_words(picture, reading, steer)
    reasoning = dict(plan.get("reasoning") or {})
    paragraph = _strip_text(str(reasoning.get("paragraph") or ""), banned)
    if paragraph != str(reasoning.get("paragraph") or "").strip():
        reasoning["paragraph"] = paragraph
        plan["reasoning"] = reasoning

    def _keep(item: dict) -> bool:
        if not isinstance(item, dict):
            return False
        if str(item.get("source") or "") == "user":
            return True
        target = str(item.get("target") or item.get("stage") or "")
        if _mentions(target, banned):
            return False
        reason = _strip_text(str(item.get("reason") or ""), banned)
        item["reason"] = reason
        return True

    plan["asks"] = [item for item in (plan.get("asks") or []) if _keep(item)]
    plan["proposed"] = [item for item in (plan.get("proposed") or []) if _keep(item)]
    if _mentions("reflection", banned) and not any(
        isinstance(item, dict) and item.get("target") == "reflection" and item.get("source") == "user"
        for item in plan["asks"]
    ):
        plan["reflection"] = "quiet"
    # A proposal cannot open the caps. Only the user's strong or epic word does.
    plan["caps_open"] = bool(plan.get("strong"))
    return plan


def guard_faithful_grade(
    atmosphere: Atmosphere,
    measured: Atmosphere,
    _picture: object,
    *,
    faithful: bool,
) -> Atmosphere:
    """Keep a faithful grade on the measurement.

    The blended model warmth is what asked a night city to go warmer. That
    number was then packed as a color-calibration illuminant, which replaces
    the camera white balance and turns the frame red. Faithful keeps the
    measured warmth and tint. The histogram already limits those numbers.
    """

    if not faithful:
        return atmosphere
    # The histogram already limits these. Putting the model blend back
    # would pack a warmer illuminant than the frame measured.
    atmosphere.temperature_shift = float(measured.temperature_shift)
    atmosphere.tint = float(measured.tint)
    return atmosphere


def guard_warm_grade(atmosphere: Atmosphere, picture: object, plan: dict, *, faithful: bool) -> Atmosphere:
    """Hold a Warmer look to a small gain on the pipeline white.

    The look adds warmth on purpose. On a night frame that warmth is only a
    channel gain. A color-balance hue near 40 degrees paints the picture red,
    so the grade does not write one. Closed caps keep a visible step.
    ``strong`` or ``epic`` is the only way past that.
    """

    if faithful:
        return atmosphere
    if abs(float(plan.get("warmth") or 0)) <= 0 and str(plan.get("look") or "") != "warmer":
        return atmosphere
    if bool(plan.get("strong") or plan.get("caps_open")):
        return atmosphere
    night = str(getattr(picture, "type", "") or "") == "night"
    ceiling = 0.16 if night else 0.28
    shift = float(atmosphere.temperature_shift)
    if abs(shift) <= ceiling:
        return atmosphere
    atmosphere.temperature_shift = ceiling if shift > 0 else -ceiling
    plan["warmth_note"] = (
        "Warmer stays a small gain on the pipeline white so the frame does not go red."
    )
    return atmosphere
