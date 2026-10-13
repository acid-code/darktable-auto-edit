"""Name two or three looks for this reading. None of them is applied.

Faithful stays the empty choice. A creative look waits until someone
picks it. The scene rules are the same ones ``_auto_fuji`` already uses.
"""

from __future__ import annotations

from ai_dt.guides.decide import Picture
from ai_dt.reading import Reading
from ai_dt.recipes.reason import _SATURATED_LAND, _auto_fuji

_TITLES = {
    "fuji-provia": "Provia",
    "fuji-velvia": "Velvia",
    "fuji-astia": "Astia",
    "fuji-eterna": "Eterna",
    "fuji-classic-chrome": "Classic Chrome",
    "fuji-pro-neg-std": "Pro Neg Std",
    "fuji-pro-neg-high": "Pro Neg High",
    "fuji-classic-neg": "Classic Neg",
    "fuji-nostalgic-neg": "Nostalgic Neg",
    "fuji-bleach-bypass": "Bleach Bypass",
    "fuji-reala-ace": "Reala Ace",
    "fuji-acros": "Acros",
}


def recommend_looks(picture: Picture, reading: Reading) -> list[dict]:
    """Up to three looks. Each one says why. None is marked applied."""

    kinds = [item.get("kind") for item in reading.problems]
    looks: list[dict] = []
    if "dark-subject" in kinds:
        looks.append(
            {
                "id": "subject-lift",
                "title": "Subject lift",
                "why": (
                    "The face is darker than the bright ground, so lift the subject. "
                    "A frame exposure bump would lift the ground with it."
                ),
                "applied": False,
            }
        )

    recipe_id, why = _auto_fuji(picture)
    if picture.type == "landscape" and picture.saturation >= _SATURATED_LAND:
        recipe_id = "fuji-provia"
        why = "Provia, because the land is already saturated."
    elif picture.type in {"portrait", "group", "street"} and picture.warmth > 0.05:
        why = "Astia, because the face is warm and a vivid sim would push it."
    looks.append(
        {
            "id": recipe_id,
            "title": _TITLES.get(recipe_id, recipe_id),
            "why": why,
            "applied": False,
        }
    )

    if "blown-highlights" in kinds:
        looks.append(
            {
                "id": "highlight-hold",
                "title": "Highlight hold",
                "why": "Highlights are blown, so hold them back instead of adding exposure.",
                "applied": False,
            }
        )
    elif "weak-reflection" in kinds or "muddy-reflection" in kinds:
        looks.append(
            {
                "id": "hold-reflection",
                "title": "Hold the reflection",
                "why": "The reflection is weak, so hold it instead of lifting it.",
                "applied": False,
            }
        )
    elif picture.type in {"portrait", "group", "street", "animal"}:
        looks.append(
            {
                "id": "eye-detail",
                "title": "Eye detail",
                "why": "The eyes can take a little detail on their own mask.",
                "applied": False,
            }
        )

    chosen: list[dict] = []
    seen: set[str] = set()
    for look in looks:
        if look["id"] in seen or look["id"] == "frame-exposure":
            continue
        if picture.type == "landscape" and picture.saturation >= _SATURATED_LAND and look["id"] == "fuji-velvia":
            continue
        seen.add(look["id"])
        chosen.append(look)
        if len(chosen) >= 3:
            break
    return chosen
