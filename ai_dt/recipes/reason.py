"""Describe the picture, then join the recipes that belong on it.

Each recipe carries an explanation of what it does and a place: the whole
frame, or a mask such as the eyes, a sunset line, or the lines of a
building. The description is written from the picture type and the
measurements. A recipe is joined only when its place exists on this
picture and its explanation agrees with that description.

Frame grades are not stacked. A second global grade would repaint color
the measurement already settled. Mask grades can be joined together when
they push the same way. Nothing here calls a model or reads the web.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ai_dt.guides.decide import Picture
from ai_dt.guides.library import Guide
from ai_dt.recipes.library import (
    STAGE_KEYS,
    Match,
    Recipe,
    _rank,
    cosine,
    load_library,
    term_frequency,
    tokenize,
)
from ai_dt.schema import Detection

# A recipe stage is turned into a small mask offset. Color stays tiny:
# a mask is there to shape a subject or a line, not to repaint it.
_MASK_EXPOSURE = 0.22
_MASK_CONTRAST = 0.28
_MASK_SATURATION = 0.08
_HORIZON_WARMTH = 0.04
_AUTO_WEIGHT = 0.5

_PLACE_STAGE = {
    "eyes": "eyes",
    "horizon": "horizon",
    "subject": "subject",
    "sky": "sky",
    "lines": "foreground",
    "background": "background",
}

_EYE_TYPES = frozenset({"portrait", "group", "street", "animal", "macro"})
_SUBJECT_TYPES = frozenset({"portrait", "group", "street", "animal", "macro", "still-life", "food"})
_BACKGROUND_TYPES = frozenset({"portrait", "group", "street", "animal", "macro"})

_TYPE_SCENE = {
    "portrait": "portrait",
    "group": "portrait",
    "street": "portrait",
    "animal": "animal",
    "landscape": "landscape",
    "architecture": "architecture",
    "night": "night",
    "general": "general",
    "macro": "general",
    "still-life": "general",
    "food": "general",
}

_COLOR_KEYS = ("exposure_ev", "temperature_shift", "tint", "saturation")

# A named film simulation, longest phrase first so "pro neg high" wins
# over "pro neg" and "classic chrome" does not collapse into "classic neg".
# These are character approximations of the camera looks, not LUT files.
_NAMED_FUJI: tuple[tuple[str, str, str], ...] = (
    ("bleach bypass", "fuji-bleach-bypass", "Bleach Bypass"),
    ("nostalgic neg", "fuji-nostalgic-neg", "Nostalgic Neg"),
    ("classic chrome", "fuji-classic-chrome", "Classic Chrome"),
    ("pro neg high", "fuji-pro-neg-high", "Pro Neg High"),
    ("pro neg std", "fuji-pro-neg-std", "Pro Neg Std"),
    ("classic neg", "fuji-classic-neg", "Classic Neg"),
    ("reala ace", "fuji-reala-ace", "Reala Ace"),
    ("pro neg", "fuji-pro-neg-std", "Pro Neg Std"),
    ("velvia", "fuji-velvia", "Velvia"),
    ("provia", "fuji-provia", "Provia"),
    ("astia", "fuji-astia", "Astia"),
    ("eterna", "fuji-eterna", "Eterna"),
    ("acros", "fuji-acros", "Acros"),
    ("reala", "fuji-reala-ace", "Reala Ace"),
)
_GENERAL_FUJI = ("fuji", "fujifilm", "fujicam")
# Mean chroma at which open land is already colorful enough that Velvia
# would push it. Provia is the controlled landscape look past this point.
# 0.18 matches the histogram advisor's colorfulness pivot.
_SATURATED_LAND = 0.18
# Same window the histogram advisor uses for a cast correction.
_TEMPERATURE_CAP = 0.28
_TINT_CAP = 0.2


@dataclass(frozen=True)
class MaskNudge:
    """A small change added on top of one guide mask."""

    exposure: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    clarity: float | None = None
    warmth: float = 0.0


@dataclass
class Reasoning:
    """The description, the recipes that were joined, and the ones left off."""

    description: str
    decision: str
    used: list[dict] = field(default_factory=list)
    set_aside: list[dict] = field(default_factory=list)
    frame_stages: dict[str, float] | None = None
    nudges: dict[str, MaskNudge] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "decision": self.decision,
            "used": list(self.used),
            "set_aside": list(self.set_aside),
        }


def _vibe_text(vibe: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", vibe.lower()).split())


def _has_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _recipe_by_id(recipes: list[Recipe], recipe_id: str) -> Recipe | None:
    for recipe in recipes:
        if recipe.id == recipe_id and recipe.place == "frame":
            return recipe
    return None


def _auto_fuji(picture: Picture) -> tuple[str, str]:
    """One scene-appropriate Fuji simulation. Not a nearest-neighbor guess.

    Landscape that is already very saturated stays on Provia so Velvia
    does not blow the color. People get Astia (soft, no green, no cool
    cast on a warm face). Night gets Eterna rather than a blue-green
    grade. An animal gets Provia. Architecture gets Classic Chrome.
    Everything else gets Provia, the standard simulation.
    """

    kind = picture.type
    if kind == "landscape":
        if picture.saturation >= _SATURATED_LAND:
            return (
                "fuji-provia",
                "Fuji Provia is the frame look. The land is already very saturated, "
                "so a general Fuji request uses Provia instead of Velvia and does not blow the color.",
            )
        return (
            "fuji-velvia",
            "Fuji Velvia is the frame look. Open land gets Velvia, one vivid simulation, "
            "because the frame is not already oversaturated.",
        )
    if kind in {"portrait", "group", "street"}:
        return (
            "fuji-astia",
            "Fuji Astia is the frame look. A portrait, group, or street picture gets Astia: "
            "softer and skin-safe, with no added green and without cooling a warm face.",
        )
    if kind == "night":
        return (
            "fuji-eterna",
            "Fuji Eterna is the frame look. Night gets Eterna: muted and flat, without a blue-green cast.",
        )
    if kind == "animal":
        return (
            "fuji-provia",
            "Fuji Provia is the frame look. An animal gets Provia, the standard Fuji color.",
        )
    if kind == "architecture":
        return (
            "fuji-classic-chrome",
            "Fuji Classic Chrome is the frame look. Architecture gets Classic Chrome, muted, as the one frame simulation.",
        )
    return (
        "fuji-provia",
        "Fuji Provia is the frame look. A general Fuji request uses Provia, the standard simulation, for this picture.",
    )


def preset_phrase(recipe_id: str) -> str:
    """Vibe text for one preset. Hyphens in the id become spaces.

    The lighttable dropdown sends this string and nothing else, so
    ``kodak-portra-400`` arrives as ``kodak portra 400`` and cannot be
    mixed with whatever was typed in the vibe box.
    """

    return " ".join(recipe_id.replace("-", " ").split())


def _is_preset(recipe: Recipe) -> bool:
    return "/presets/" in recipe.path.replace("\\", "/")


def _resolve_pinned_preset(vibe: str, recipes: list[Recipe]) -> tuple[Recipe, str] | None:
    """Select one preset when the vibe is exactly that preset's phrase.

    A blank vibe and a free-text sentence do not match. The dropdown
    uses this so two frame looks are never applied together.
    """

    text = _vibe_text(vibe)
    if not text:
        return None
    for recipe in recipes:
        if recipe.place != "frame" or not _is_preset(recipe):
            continue
        if preset_phrase(recipe.id) != text:
            continue
        why = (
            f"{recipe.title} is the frame look because that preset was chosen. "
            "One frame look is applied."
        )
        if recipe.id.startswith("fuji-"):
            why += " It is a character approximation, not a camera LUT."
        return recipe, why
    return None


def _resolve_fuji(vibe: str, picture: Picture, recipes: list[Recipe]) -> tuple[Recipe, str] | None:
    """Pick one Fuji frame recipe when the vibe asks for one.

    A blank vibe returns nothing, so the faithful correction stays.
    A sim name selects that recipe. fuji, fujifilm, or fujicam, with no
    sim name, selects the scene look from ``_auto_fuji``.
    """

    text = _vibe_text(vibe)
    if not text:
        return None
    for phrase, recipe_id, label in _NAMED_FUJI:
        if _has_phrase(text, phrase):
            recipe = _recipe_by_id(recipes, recipe_id)
            if recipe is None:
                return None
            why = (
                f"Fuji {label} is the frame look because the vibe names {label}. "
                "One simulation covers the whole frame. It is a character approximation, not a camera LUT."
            )
            return recipe, why
    if not any(_has_phrase(text, word) for word in _GENERAL_FUJI):
        return None
    recipe_id, why = _auto_fuji(picture)
    recipe = _recipe_by_id(recipes, recipe_id)
    if recipe is None:
        return None
    return recipe, why + " It is a character approximation, not a camera LUT."


def _hold_fuji_stages(stages: dict[str, float]) -> dict[str, float]:
    """Keep a Fuji frame grade inside the histogram color window.

    Exposure stays at zero so the simulation cannot lift or crush the
    measurement. Temperature and tint stay inside ±0.28 and ±0.2.
    """

    held = dict(stages)
    held["exposure_ev"] = 0.0
    held["temperature_shift"] = min(max(float(held.get("temperature_shift", 0.0)), -_TEMPERATURE_CAP), _TEMPERATURE_CAP)
    held["tint"] = min(max(float(held.get("tint", 0.0)), -_TINT_CAP), _TINT_CAP)
    return held


def describe_picture(picture: Picture, detections: list[Detection]) -> str:
    """Write a short factual description. It is not a grade and not a vibe."""

    sentences = [picture.reason.strip()]
    if picture.brightness < 0.28:
        sentences.append(
            "The frame is dark, like night, and that darkness is part of the picture."
        )
    elif picture.brightness < 0.45:
        sentences.append("The light is dim.")
    else:
        sentences.append("The light is open.")
    if picture.warmth > 0.05:
        sentences.append("The color already reads warm.")
    elif picture.warmth < -0.04:
        sentences.append("The color already reads cool.")
    else:
        sentences.append("The color is near neutral, so it should stay with the measurement.")
    if picture.type in _EYE_TYPES:
        sentences.append(
            "The eyes can take detail on their own mask, without recoloring the face or the night."
        )
    if picture.type == "landscape":
        sentences.append(
            "A warm sunset line, if one crosses the frame, is a separate mask from the land and the sky."
        )
    if picture.type == "architecture":
        sentences.append(
            "Contrast belongs on the lines of the facade. A cool grade does not belong on the whole frame."
        )
    labels = []
    for item in detections[:4]:
        label = item.label.strip()
        if label and label not in labels:
            labels.append(label)
    if labels:
        sentences.append("Detected: " + ", ".join(labels) + ".")
    return " ".join(sentence if sentence.endswith(".") else sentence + "." for sentence in sentences if sentence)


def reason_about(
    picture: Picture,
    detections: list[Detection],
    vibe: str,
    guide: Guide,
    match: Match,
    directory: Path | None = None,
    recipes: list[Recipe] | None = None,
) -> Reasoning:
    """Choose which recipes join, and which of them become a mask."""

    if recipes is None:
        recipes, _warnings = load_library(directory)
    description = describe_picture(picture, detections)
    pinned = _resolve_pinned_preset(vibe, recipes)
    fuji = None if pinned is not None else _resolve_fuji(vibe, picture, recipes)
    choice = pinned if pinned is not None else fuji
    fuji_why = ""
    if choice is not None:
        recipe, fuji_why = choice
        # One frame look. A preset, or one Fuji sim, is not averaged with another grade.
        match.recipe = recipe
        match.applied = True
        match.reason = fuji_why
        match.method = "preset" if pinned is not None else "fuji-simulation"
        match.score = 1.0
        match.cosine = 1.0
        frame_group = [recipe]
    else:
        frame_group = _frame_group(match, recipes, vibe)
    requested = match.recipe if match.applied and match.recipe is not None else None
    frame_stages = _average_stages(frame_group) if frame_group else None
    if frame_stages is not None and frame_group and frame_group[0].id.startswith("fuji-"):
        frame_stages = _hold_fuji_stages(frame_stages)
        # An automatic Fuji pick for a person must not cool the frame or add green.
        # A named sim such as Classic Chrome keeps its own temperature.
        named = any(_has_phrase(_vibe_text(vibe), phrase) for phrase, _recipe_id, _label in _NAMED_FUJI)
        if not named and picture.type in {"portrait", "group", "street"}:
            frame_stages["temperature_shift"] = max(float(frame_stages["temperature_shift"]), 0.0)
            frame_stages["tint"] = max(float(frame_stages["tint"]), 0.0)

    used: list[dict] = []
    if frame_group:
        lead = frame_group[0]
        joined = [item.title for item in frame_group[1:]]
        if fuji_why:
            why = fuji_why
        elif joined:
            why = (
                f"{lead.title} leads the frame. "
                + ", ".join(joined)
                + " agree in direction, so their offsets are averaged and then held inside the lead recipe."
            )
        else:
            why = _sentence(lead) + " It stays on the whole frame."
        used.append(_record(lead, "frame", 1.0, why, joined=[item.id for item in frame_group[1:]]))

    by_stage: dict[str, list[tuple[Recipe, float]]] = {}
    considered: list[tuple[Recipe, str]] = []
    for recipe in recipes:
        if recipe.place == "frame":
            if frame_group and any(item.id == recipe.id for item in frame_group):
                continue
            if _related(recipe, picture) and not _is_faithful(recipe):
                considered.append((recipe, _frame_aside(recipe, vibe)))
            continue
        if recipe.place not in _PLACE_STAGE:
            continue
        if not _place_fits(recipe, picture, guide):
            if requested is not None and recipe.id == requested.id:
                considered.append((recipe, _missing_mask(recipe, picture)))
            elif _related(recipe, picture):
                considered.append((recipe, _missing_mask(recipe, picture)))
            continue
        weight = 1.0 if requested is not None and recipe.id == requested.id else _AUTO_WEIGHT
        stage = _PLACE_STAGE[recipe.place]
        by_stage.setdefault(stage, []).append((recipe, weight))

    nudges: dict[str, MaskNudge] = {}
    for stage, group in by_stage.items():
        group.sort(key=lambda item: (-_mask_score(item[0], description, vibe), item[0].id))
        lead, lead_weight = group[0]
        kept = [(lead, lead_weight)]
        for recipe, weight in group[1:]:
            if _agrees(lead, recipe):
                kept.append((recipe, weight))
            else:
                considered.append((recipe, _conflict(recipe, lead)))
        nudge = _combine(kept)
        nudges[stage] = nudge
        names = [item.title for item, _weight in kept]
        if len(names) == 1:
            why = _mask_why(lead, nudge)
        else:
            why = (
                ", ".join(names)
                + f" are joined onto the {lead.place}. They push the same way, so their mask offsets are averaged."
            )
        for recipe, weight in kept:
            used.append(_record(recipe, recipe.place, weight, why if recipe is lead else _mask_why(recipe, nudge)))

    considered.sort(key=lambda item: (-_harm(item[0]), item[0].id))
    set_aside = []
    seen: set[str] = set()
    for recipe, why in considered:
        if recipe.id in seen:
            continue
        seen.add(recipe.id)
        set_aside.append(_record(recipe, recipe.place, 0.0, why))
        if len(set_aside) >= 4:
            break

    decision = _decision(picture, frame_group, used, set_aside, fuji_line=fuji_why)
    return Reasoning(
        description=description,
        decision=decision,
        used=used,
        set_aside=set_aside,
        frame_stages=frame_stages,
        nudges=nudges,
    )


def _frame_group(match: Match, recipes: list[Recipe], vibe: str) -> list[Recipe]:
    """The frame grade is one recipe, plus others that agree and were asked for.

    A blank vibe does not recruit extra frame recipes. Their color would
    move the measurement just because the description and a recipe share
    the word night or portrait.
    """

    if not match.applied or match.recipe is None or match.recipe.place != "frame":
        return []
    lead = match.recipe
    if not tokenize(vibe):
        return [lead]
    user_vector = term_frequency(vibe)
    group = [lead]
    ranked = _rank(recipes, match.scene, f"{match.scene} {vibe}")
    for recipe, _neighbor in ranked:
        if recipe.id == lead.id or recipe.place != "frame" or _is_faithful(recipe):
            continue
        # A second Fuji simulation is never stacked on the first.
        if lead.id.startswith("fuji-") and recipe.id.startswith("fuji-"):
            continue
        similarity = cosine(user_vector, term_frequency(recipe.explains + " " + recipe.document))
        if similarity < 0.18 or not _agrees(lead, recipe):
            continue
        group.append(recipe)
        if len(group) >= 3:
            break
    return group


def _average_stages(group: list[Recipe]) -> dict[str, float]:
    """Average agreeing frame recipes, and never past the lead recipe."""

    lead = group[0]
    if len(group) == 1:
        return dict(lead.stages)
    averaged: dict[str, float] = {}
    for key in STAGE_KEYS:
        values = [float(item.stages.get(key, 0.0)) for item in group]
        mean = sum(values) / float(len(values))
        anchor = float(lead.stages.get(key, 0.0))
        if key in _COLOR_KEYS:
            if anchor >= 0.0:
                mean = min(max(mean, 0.0), anchor)
            else:
                mean = max(min(mean, 0.0), anchor)
        if key == "vignette_strength":
            mean = anchor
        averaged[key] = mean
    return averaged


def _place_fits(recipe: Recipe, picture: Picture, guide: Guide) -> bool:
    stage = _PLACE_STAGE.get(recipe.place)
    if stage is None or stage not in guide.stages:
        return False
    plan = guide.masks.get(stage)
    if plan is None or plan.build == "never":
        return False
    if recipe.place == "eyes":
        return picture.type in _EYE_TYPES
    if recipe.place == "horizon":
        return picture.type == "landscape"
    if recipe.place == "lines":
        return picture.type == "architecture"
    if recipe.place == "sky":
        return picture.type == "landscape" and picture.sky >= 0.5
    if recipe.place == "subject":
        return picture.type in _SUBJECT_TYPES
    if recipe.place == "background":
        return picture.type in _BACKGROUND_TYPES
    return False


def _related(recipe: Recipe, picture: Picture) -> bool:
    scene = _TYPE_SCENE.get(picture.type, "general")
    if scene in recipe.scenes:
        return True
    if picture.type in {"portrait", "group", "street"} and "portrait" in recipe.scenes:
        return True
    if picture.brightness < 0.34 and "night" in recipe.scenes:
        return True
    return False


def _agrees(left: Recipe, right: Recipe) -> bool:
    for key in _COLOR_KEYS:
        a = float(left.stages.get(key, 0.0))
        b = float(right.stages.get(key, 0.0))
        if abs(a) < 0.05 or abs(b) < 0.05:
            continue
        if a * b < 0.0:
            return False
    return True


def _is_faithful(recipe: Recipe) -> bool:
    return all(abs(float(recipe.stages.get(key, 0.0))) < 1e-6 for key in STAGE_KEYS if key != "vignette_strength")


def _harm(recipe: Recipe) -> float:
    stages = recipe.stages
    return abs(float(stages.get("temperature_shift", 0.0))) + abs(float(stages.get("saturation", 0.0)))


def _mask_score(recipe: Recipe, description: str, vibe: str) -> float:
    text = " ".join((recipe.explains, recipe.description, recipe.title))
    score = cosine(term_frequency(description), term_frequency(text))
    if tokenize(vibe):
        score = max(score, cosine(term_frequency(vibe), term_frequency(recipe.document)))
    return score


def _map_stages(stages: dict[str, float], place: str) -> MaskNudge:
    exposure = float(stages.get("exposure_ev", 0.0)) * _MASK_EXPOSURE
    contrast = float(stages.get("contrast", 0.0)) * _MASK_CONTRAST
    saturation = float(stages.get("saturation", 0.0)) * _MASK_SATURATION
    warmth = 0.0
    if place == "eyes":
        saturation = min(max(saturation, -0.02), 0.02)
    elif place == "horizon":
        saturation = min(saturation, 0.0)
        exposure = max(0.0, exposure)
        warmth = max(0.0, float(stages.get("temperature_shift", 0.0))) * _HORIZON_WARMTH
    elif place == "lines":
        saturation = min(max(saturation, -0.03), 0.02)
    return MaskNudge(exposure=exposure, contrast=contrast, saturation=saturation, warmth=warmth)


def _combine(group: list[tuple[Recipe, float]]) -> MaskNudge:
    count = float(len(group))
    exposure = contrast = saturation = warmth = 0.0
    for recipe, weight in group:
        part = _map_stages(recipe.stages, recipe.place)
        exposure += part.exposure * weight
        contrast += part.contrast * weight
        saturation += part.saturation * weight
        warmth += part.warmth * weight
    return MaskNudge(
        exposure=exposure / count,
        contrast=contrast / count,
        saturation=saturation / count,
        warmth=warmth / count,
    )


def _sentence(recipe: Recipe) -> str:
    text = recipe.explains or recipe.description
    piece = text.strip()
    if not piece:
        return recipe.title + "."
    if piece[-1] not in ".!?":
        piece += "."
    return piece


def _record(recipe: Recipe, place: str, weight: float, why: str, joined: list[str] | None = None) -> dict:
    record = {
        "id": recipe.id,
        "title": recipe.title,
        "place": place,
        "weight": round(weight, 2),
        "explains": recipe.explains or recipe.description,
        "why": " ".join(why.split()),
    }
    if joined:
        record["joined"] = list(joined)
    return record


def _frame_aside(recipe: Recipe, vibe: str) -> str:
    if tokenize(vibe):
        return (
            f"{recipe.title} stays off the frame. {_sentence(recipe)} "
            "It disagrees with the requested look, or it is not close enough to join."
        )
    return (
        f"{recipe.title} stays off the frame. {_sentence(recipe)} "
        "It was not asked for, so the measured color stays."
    )


def _missing_mask(recipe: Recipe, picture: Picture) -> str:
    return (
        f"{recipe.title} is for the {recipe.place}. {_sentence(recipe)} "
        f"This {picture.type} picture has no {recipe.place} mask, so it is not added."
    )


def _conflict(recipe: Recipe, lead: Recipe) -> str:
    return (
        f"{recipe.title} is not joined onto the {lead.place}. {_sentence(recipe)} "
        f"It pushes color the other way from {lead.title}."
    )


def _mask_why(recipe: Recipe, nudge: MaskNudge) -> str:
    if recipe.place == "eyes":
        return (
            f"{recipe.title} is a mask on the eyes: a little exposure and contrast, "
            "and almost no color, so the face and the night are not repainted."
        )
    if recipe.place == "horizon":
        return (
            f"{recipe.title} is a mask on the sunset line. "
            "Its warmth stays on that band, and the land is not given this grade."
        )
    if recipe.place == "lines":
        return (
            f"{recipe.title} is a mask on the building lines. "
            "Contrast goes there, and the frame is not cooled."
        )
    return (
        f"{recipe.title} is a mask on the {recipe.place}. {_sentence(recipe)} "
        f"Exposure {nudge.exposure:+.2f}, contrast {nudge.contrast:+.2f}."
    )


def _decision(
    picture: Picture,
    frame_group: list[Recipe],
    used: list[dict],
    set_aside: list[dict],
    fuji_line: str = "",
) -> str:
    if fuji_line:
        frame = fuji_line
    elif frame_group:
        frame = f"The frame uses {frame_group[0].title}, so the measured color stays in charge."
    else:
        frame = "No frame recipe is applied, so the measured color stays on the whole picture."
    masks = [item["title"] for item in used if item["place"] != "frame"]
    # The lead mask line is repeated once per joined recipe. Keep unique titles.
    seen: list[str] = []
    for title in masks:
        if title not in seen:
            seen.append(title)
    if seen:
        mask_line = "Joined onto a mask: " + ", ".join(seen) + "."
    elif picture.type in {"portrait", "group", "street", "landscape", "architecture"}:
        mask_line = "No extra mask recipe was joined."
    else:
        mask_line = "No mask is invented for this picture."
    aside = [item["title"] for item in set_aside[:3]]
    if aside:
        aside_line = "Set aside: " + ", ".join(aside) + "."
    else:
        aside_line = ""
    return " ".join(part for part in (frame, mask_line, aside_line) if part)
