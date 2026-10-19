"""What the photoserge lesson videos teach, as data the editor can use.

* ``photoserge_examples.json`` is a trimmed copy of the lesson notes
  (eleven before/after examples, no images). ``few_shot_text`` turns the
  examples that fit a photo into a short block for the planner prompt.
* ``STEP_RANK`` / ``order_steps``: global first. The lessons fix exposure,
  then black and white points, white balance, tone, color, and only then
  local work. Propose builds the global look; among Accept steps a global
  color look comes before regional grades, gradients after them, and the
  finishing edge burn and glow last. Steps with the same rank keep the
  order they were given in.
* ``PRESETS``: the lesson steps that suit each picture type.
* ``crop_suggestion``: a rule-of-thirds crop (example 1 and 8). Advisory
  only; it is reported in ``output.json`` and never applied.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).with_name("photoserge_examples.json")

# 0 is the Propose look (exposure, black/white points, white balance, tone).
STEP_RANK = {
    "split_tone": 1,
    "palander": 1,
    "keep_color": 1,
    "mask_sky_gradient": 3,
    "mask_bottom_gradient": 3,
    "mask_edge_burn": 4,
    "mask_glow": 4,
}
DEFAULT_RANK = 2

# Which examples speak to which picture type, and which steps they suggest.
_EXAMPLES_FOR = {
    "landscape": (7, 9, 2, 11, 1),
    "architecture": (9, 4, 5, 8),
    "street": (4, 5, 9, 8),
    "night": (5, 4),
    "animal": (11, 10, 8),
    "portrait": (10, 3),
    "group": (10, 3),
    "general": (3, 2, 7),
}
PRESETS = {
    "landscape": ("mask_landscape", "mask_sky_gradient", "mask_edge_burn"),
    "architecture": ("mask_architecture", "mask_sky_gradient", "mask_bottom_gradient"),
    "street": ("palander", "mask_bottom_gradient"),
    "night": ("palander",),
    # Animals: crisp the animal a little (fur, eyes), quiet and darken around
    # it, burn the edges, all inside the +-0.5 EV local cap.
    "animal": ("mask_animal", "mask_animal_background", "mask_edge_burn"),
    "portrait": ("mask_glow", "mask_background"),
}
ANIMAL_HINTS = (
    "Animals: one outline per animal (mask_animal = the main one, mask_animals = all). "
    "Keep it subtle: a little local contrast for fur and eyes, never more than half a stop; "
    "quiet the surroundings with mask_animal_background, then mask_edge_burn to hold the eye."
)


@lru_cache(maxsize=1)
def load() -> dict:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"examples": [], "general_tips": []}


def examples() -> list[dict]:
    return [item for item in load().get("examples") or [] if isinstance(item, dict)]


def general_tips() -> list[str]:
    # "images" reads like an upload in a prompt that promises none; say photos.
    return [str(item).replace("images", "photos") for item in load().get("general_tips") or []]


_TYPE_WORDS = {
    "animal": r"\b(animal|bird|dog|cat|horse|deer|wildlife|sheep|cow|fox|owl|eagle)s?\b",
    "landscape": r"\b(landscape|mountain|valley|river|lake|sea|coast|field|forest|hills?|view|sunset|sunrise)\b",
    "architecture": r"\b(building|architecture|tower|bridge|church|facade|skyline)s?\b",
    "night": r"\b(night|neon|street ?lights?)\b",
    "street": r"\b(street|city|urban|traffic)\b",
    "portrait": r"\b(portrait|face|person|woman|man|model)\b",
}


def guess_type(*texts: str) -> str:
    """A picture type from free text, or ``general``."""

    joined = " ".join(str(text or "") for text in texts).lower()
    for kind, pattern in _TYPE_WORDS.items():
        if re.search(pattern, joined):
            return kind
    return "general"


def _example_line(example: dict) -> str:
    edits = "; ".join(
        f"{edit.get('tool')} {edit.get('setting')} {edit.get('value')}".strip()
        for edit in (example.get("edits") or [])[:4]
        if isinstance(edit, dict)
    )
    equivalent = " ".join(str(example.get("darktable_equivalent") or "").split())
    if len(equivalent) > 160:
        equivalent = equivalent[:160].rsplit(" ", 1)[0] + "..."
    return f"- {example.get('title')}: {edits}. In darktable: {equivalent}"


def few_shot_text(picture_type: str = "", *, limit: int = 2) -> str:
    """A short "how a pro edits this kind of photo" block for one prompt."""

    kind = picture_type if picture_type in _EXAMPLES_FOR else "general"
    by_id = {item.get("id"): item for item in examples()}
    chosen = [by_id[number] for number in _EXAMPLES_FOR[kind] if number in by_id][:limit]
    if not chosen:
        return ""
    lines = ["Editing lessons (worked examples; borrow the order and the direction, not the numbers):"]
    lines.extend(f"- {tip}" for tip in general_tips())
    lines.extend(_example_line(item) for item in chosen)
    steps = PRESETS.get(kind)
    if steps:
        lines.append("Steps that fit this kind of photo: " + ", ".join(steps) + ".")
    if kind == "animal":
        lines.append(ANIMAL_HINTS)
    lines.append("Local exposure moves stay within +-0.5 EV; stack a step instead of pushing one.")
    return "\n".join(lines)


def step_rank(step_id: str) -> int:
    return STEP_RANK.get(str(step_id or ""), DEFAULT_RANK)


def order_steps(steps: list[dict]) -> list[dict]:
    """Global first, finishing last; a stable sort, so equal ranks keep their order."""

    return sorted(steps, key=lambda item: step_rank(str(item.get("id") or "")))


def crop_suggestion(
    size: tuple[int, int],
    *,
    subject: tuple[float, float, float, float] | None = None,
    horizon: float | None = None,
    aspect: tuple[int, int] = (16, 9),
) -> dict | None:
    """The largest ``aspect`` crop that puts the subject on a third and the horizon on a third.

    ``subject`` is ``(x, y, w, h)``, ``horizon`` a 0..1 row. Returns
    normalized ``{x, y, w, h, aspect, why}`` or None when the photo already
    fits and nothing would move.
    """

    width, height = size
    if width <= 0 or height <= 0:
        return None
    target = aspect[0] / float(aspect[1])
    ratio = width / float(height)
    if ratio > target:
        cw, ch = target / ratio, 1.0
    else:
        cw, ch = 1.0, ratio / target
    why = []
    x = (1.0 - cw) / 2.0
    y = (1.0 - ch) / 2.0
    if subject is not None:
        sx = subject[0] + subject[2] / 2.0
        sy = subject[1] + subject[3] / 2.0
        thirds_x = [x_line for x_line in (1 / 3, 2 / 3)]
        best = min(thirds_x, key=lambda line: abs(sx - (line * cw + (1 - cw) / 2)))
        x = sx - best * cw
        why.append("subject on a vertical third")
        if horizon is None:
            best_y = min((1 / 3, 2 / 3), key=lambda line: abs(sy - (line * ch + (1 - ch) / 2)))
            y = sy - best_y * ch
    if horizon is not None:
        # Lower third for a strong sky or reflection; the lessons pick the line nearest the horizon.
        best_y = min((1 / 3, 2 / 3), key=lambda line: abs(horizon - (line * ch + (1 - ch) / 2)))
        y = horizon - best_y * ch
        why.append("horizon on a horizontal third")
    x = min(max(x, 0.0), 1.0 - cw)
    y = min(max(y, 0.0), 1.0 - ch)
    if cw > 0.995 and ch > 0.995:
        return None
    return {
        "x": round(x, 4),
        "y": round(y, 4),
        "w": round(cw, 4),
        "h": round(ch, 4),
        "aspect": f"{aspect[0]}:{aspect[1]}",
        "why": ", ".join(why) or "centered",
    }
