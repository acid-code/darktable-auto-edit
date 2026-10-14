"""Looks the panel can offer before anything is applied.

Recommendations already say why. Recipe neighbors are the other three
nearby files. Faithful is always first, so the measured grade can be
kept on its own duplicate.
"""

from __future__ import annotations

NAMED_LOOKS = frozenset(
    {
        "subject-lift",
        "highlight-hold",
        "hold-reflection",
        "eye-detail",
        "shadow-lift",
        "warmer",
        "more-color",
        "sky-gradient",
    }
)

# These are the expert modules a duplicate can show on its own.
EXPERT_CHOICES = (
    {
        "id": "shadow-lift",
        "title": "Open shadows",
        "why": "Tone equalizer lifts the dark bands.",
        "source": "expert",
    },
    {
        "id": "highlight-hold",
        "title": "Hold highlights",
        "why": "Tone equalizer holds the bright bands.",
        "source": "expert",
    },
    {
        "id": "warmer",
        "title": "Warmer",
        "why": "Color calibration adds a little warmth on the pipeline white.",
        "source": "expert",
    },
    {
        "id": "more-color",
        "title": "More color",
        "why": "Color zones raise saturation.",
        "source": "expert",
    },
    {
        "id": "sky-gradient",
        "title": "Sky gradient",
        "why": "Graduated density darkens across the frame.",
        "source": "expert",
    },
)

_EXTRA_TITLES = {
    "faithful": "Faithful",
    "subject-lift": "Subject lift",
    "eye-detail": "Eye detail",
    "hold-reflection": "Hold the reflection",
}


def title_for(look_id: str) -> str:
    """The button name for a picked look. Empty when the id is a film recipe."""

    name = (look_id or "").strip()
    if not name:
        return ""
    for item in EXPERT_CHOICES:
        if item["id"] == name:
            return str(item["title"])
    return _EXTRA_TITLES.get(name, "")


def look_sentence(identity: str, why: str = "", subject: str = "") -> str:
    """A tooltip: the modules this look moves, and which way.

    ``subject`` is a word the user named, such as train. An empty subject
    stays general and does not invent a subject.
    """

    name = " ".join(str(subject or "").split())
    named = f"the {name}" if name else ""
    around = f" around {named}" if named else ""
    warms = f", which warms {named} and the lights around it" if named else ""
    color = f" on {named} and the rest of the frame" if named else ""
    stays = f"The {name} stays as measured, and no film recipe is added." if name else "No film recipe is added."
    sentences = {
        "faithful": (
            "Keeps the measured exposure and color balance on the whole frame. " + stays
        ),
        "shadow-lift": (
            "The tone equalizer lifts the dark bands so the shadows open"
            + around
            + ", without raising the highlights."
        ),
        "highlight-hold": (
            "The tone equalizer holds the bright bands down so highlights, "
            "including lamps and windows, are not pushed brighter."
        ),
        "warmer": (
            "Color calibration adds a little warmth on the pipeline white"
            + warms
            + ". The camera white balance stays in place."
        ),
        "more-color": "Color zones raise saturation so the color reads stronger" + color + ".",
        "sky-gradient": "Graduated density lays a gradient across the frame and darkens the brighter end.",
        "subject-lift": (
            f"Exposure on {named or 'the subject'} is lifted on its own mask, "
            "without a frame exposure bump that would lift the ground with it."
        ),
        "eye-detail": (
            "Local contrast clarifies the eyes on their own mask. "
            "This is only offered when a face or an eye was found."
        ),
        "hold-reflection": "The reflection mask is held instead of lifted, so a weak mirror is not brightened.",
    }
    sentence = sentences.get(identity)
    if sentence:
        return sentence
    text = " ".join(str(why or "").split())
    if name and text and name not in text.lower():
        return f"{text} This is balanced around the {name}."
    return text


def collect_choices(
    recommendations: list[dict],
    neighbors: list[dict],
    look: str = "",
    subject: str = "",
) -> list[dict]:
    """Up to six choices. None of them is applied until ``look`` names one."""

    chosen = (look or "").strip()
    choices: list[dict] = []
    seen: set[str] = set()

    def add(item: dict) -> None:
        identity = str(item.get("id") or "")
        if not identity or identity in seen or len(choices) >= 6:
            return
        seen.add(identity)
        title = str(item.get("title") or identity)
        choices.append(
            {
                "id": identity,
                "title": title,
                "why": look_sentence(identity, str(item.get("why") or ""), subject),
                "source": str(item.get("source") or ""),
                "applied": identity == chosen or (identity == "faithful" and chosen == ""),
            }
        )

    add(
        {
            "id": "faithful",
            "title": "Faithful",
            "why": "The measured grade. No creative recipe is added.",
            "source": "faithful",
        }
    )
    for item in EXPERT_CHOICES:
        add(item)
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        add(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "why": item.get("why") or "",
                "source": "recommendation",
            }
        )
    for item in neighbors:
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        why = item.get("why") or ""
        if not why and isinstance(score, (int, float)):
            why = f"Nearby recipe, score {float(score):.2f}."
        add(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "why": why,
                "source": "neighbor",
            }
        )
    return choices


def apply_named_look(plan: dict, look: str) -> dict:
    """Fold a non-recipe choice into the plan the renderer already understands."""

    name = (look or "").strip()
    if name not in NAMED_LOOKS:
        return plan
    asks = [dict(item) for item in (plan.get("asks") or []) if isinstance(item, dict)]
    if name == "subject-lift":
        if not any(item.get("verb") == "lift" and item.get("target") == "subject" for item in asks):
            asks.append(
                {
                    "verb": "lift",
                    "target": "subject",
                    "source": "user",
                    "reason": "You picked subject lift.",
                }
            )
    elif name == "hold-reflection":
        plan["reflection"] = "hold"
        if not any(item.get("verb") == "hold" and item.get("target") == "reflection" for item in asks):
            asks.append(
                {
                    "verb": "hold",
                    "target": "reflection",
                    "source": "user",
                    "reason": "You picked hold the reflection.",
                }
            )
    elif name == "eye-detail":
        plan["skip"] = [stage for stage in (plan.get("skip") or []) if stage != "eyes"]
    elif name == "highlight-hold":
        plan["highlight_hold"] = True
    elif name == "shadow-lift":
        plan["shadow_lift"] = 0.5
    elif name == "warmer":
        plan["warmth"] = 0.4
    elif name == "more-color":
        plan["color"] = 0.35
    elif name == "sky-gradient":
        plan["gradient"] = True
        plan["gradient_by"] = "user"
    plan["asks"] = asks
    return plan


def mark_recommendations(recommendations: list[dict], look: str) -> list[dict]:
    """Copy the recommendation list and set ``applied`` on the picked id."""

    chosen = (look or "").strip()
    marked = []
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        copy = dict(item)
        copy["applied"] = str(copy.get("id") or "") == chosen
        marked.append(copy)
    return marked
