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
        "why": "Color calibration shifts the white balance warmer.",
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


def collect_choices(
    recommendations: list[dict],
    neighbors: list[dict],
    look: str = "",
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
        choices.append(
            {
                "id": identity,
                "title": str(item.get("title") or identity),
                "why": str(item.get("why") or ""),
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
