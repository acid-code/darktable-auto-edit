"""What this run can apply, and the technical brief sent with each plan.

The photo description is written once. This module is the part that changes
between sends: measured numbers, stages already run, and the operations
Darktable can actually receive from a style or from the painted TIFF.
"""

from __future__ import annotations

HISTORY = (
    ("exposure", "Global exposure in the history style."),
    ("colorbalancergb", "Color balance RGB in the history style."),
    ("vignette", "A falloff around the subject in the history style."),
    ("toneequal", "Tone equalizer for shadows and highlights, when the amount is off zero."),
    ("filmicrgb", "Filmic display transform. Use this or sigmoid, not both."),
    ("sigmoid", "Sigmoid display transform. Use this or filmic, not both."),
    ("colorzones", "Color zones for one hue, such as the red of an apron. Its own accept step, not a subject exposure."),
    ("recolor", "One whole-frame color change when the picture would look better. Not a mask, and not a highlight of one object."),
    ("crop", "Crop when a horizon is off center."),
    ("channelmixerrgb", "A small channel gain on the pipeline white. Not a custom illuminant."),
    ("graduatednd", "Graduated density, only when a measured direction exists."),
)

LOCAL = (
    ("subject", "Subject grade painted into edited.tif."),
    ("skin", "Skin grade painted into edited.tif."),
    ("eyes", "Eye grade painted into edited.tif."),
    ("background", "Background grade painted into edited.tif."),
    ("sky", "Sky grade painted into edited.tif."),
    ("reflection", "Reflection grade painted into edited.tif."),
    ("separate", "Keep a person or animal out of the surround. Painted into edited.tif."),
    ("foreground", "Darken the lower band and leave a person or animal alone. Painted into edited.tif."),
    ("opening", "Darken one bright compact gap. Painted into edited.tif."),
    ("shafts", "Darken tall dark vertical shapes. Painted into edited.tif."),
    ("highlights", "Warm the bright range. An accepted step is a global color-balance change, not a brushed mask."),
)

# Drawn-mask steps. Accept writes each one into the sidecar as one module
# with a group of path / gradient shapes (ai_dt/mask_steps.py).
MASKS = (
    ("mask_sky", "Grade the sky through a sky mask. darken = darker and a little warmer."),
    ("mask_architecture", "Grade the buildings through a building mask. lift = brighter and crisper."),
    ("mask_vegetation", "Grade trees and grass through a vegetation mask. lift = greener."),
    ("mask_water", "Grade the water through a water mask. darken = darker, bluer, softer."),
    ("mask_ground", "Grade the ground through a ground mask. darken = darker."),
    ("mask_people", "Lift the people through a people mask, within half a stop."),
    ("mask_background", "Darken and mute everything except the subject (inverse subject mask)."),
    ("mask_landscape", "All five landscape masks in order: sky, buildings, vegetation, water, ground."),
    ("mask_animal", "Lift the main animal through its outline: a little brilliance and local contrast for fur and eyes."),
    ("mask_animals", "Lift every animal (one outline per animal, joined)."),
    ("mask_animal_background", "Darken, mute and soften everything around the animals (inverse animal mask)."),
    ("mask_sky_gradient", "Exposure through a top gradient intersected with the sky outline. darken = moodier sky."),
    ("mask_bottom_gradient", "Exposure through a gradient from the bottom. darken = leads the eye up."),
    ("mask_edge_burn", "Burn the frame edges with four gradients, minus the subject."),
    ("mask_glow", "A soft glow (ellipse) behind the subject, minus the subject."),
    ("split_tone", "Global look: warm highlights, cool shadows, more contrast, less vibrance."),
    ("palander", "Global warm-only look: reds, oranges and yellows up, every other hue down."),
    ("keep_color", "Global selective color: keep one hue (name it in the reason), gray the rest."),
)

_HISTORY_IDS = frozenset(item[0] for item in HISTORY)
_MASK_IDS = frozenset(item[0] for item in MASKS)
_LOCAL_IDS = frozenset(item[0] for item in LOCAL)
_FEATURE_IDS = frozenset({"separate", "foreground", "opening", "shafts", "highlights", "eyes"})
_STAGE_IDS = frozenset({"subject", "skin", "eyes", "background", "sky", "reflection"})
_ASK_TARGETS = frozenset({"subject", "foreground", "sky", "reflection", "eyes"})
_ACTIONS = frozenset({"lift", "darken", "hold", "skip", "include"})
KNOWN_IDS = _HISTORY_IDS | _LOCAL_IDS | _MASK_IDS


def catalog_text() -> str:
    """The supported operations, and the limits, for one planner prompt."""

    lines = [
        "Supported Darktable operations for this run.",
        "Choose an ordered subset. An id outside this list is dropped.",
        "History modules a style can add:",
    ]
    lines.extend(f"- {name}: {note}" for name, note in HISTORY)
    lines.append("Local steps painted into edited.tif, because a style cannot store a drawn mask:")
    lines.extend(f"- {name}: {note}" for name, note in LOCAL)
    lines.append("Drawn-mask steps written into the photo's history on Accept (one module each, its own mask):")
    lines.extend(f"- {name}: {note}" for name, note in MASKS)
    lines.append(
        "Leave these out: diffuse, blurs, and retouch, while a drawn mask cannot attach; "
        "a custom white-balance illuminant; opening caps unless the user wrote strong or epic."
    )
    lines.append(
        "Accepted feature steps that land in history are global modules with a fixed amount, "
        "not a brushed mask, except the mask_ steps above, which carry their own drawn mask."
    )
    return "\n".join(lines)


def technical_brief(
    values: object,
    *,
    subject_brightness: float | None = None,
    frame_brightness: float | None = None,
    stages: list[str] | None = None,
) -> str:
    """The numbers that move after each edit. This is not a second description."""

    def _num(name: str) -> float:
        if isinstance(values, dict):
            return float(values.get(name) or 0)
        return float(getattr(values, name, 0) or 0)

    lines = [
        f"exposure { _num('exposure_ev'):+.2f} EV",
        f"contrast {_num('contrast'):+.2f}",
        f"warmth {_num('temperature_shift'):+.2f}",
        f"tint {_num('tint'):+.2f}",
        f"shadows {_num('shadows'):+.2f}",
        f"highlights {_num('highlights'):+.2f}",
    ]
    if subject_brightness is not None and frame_brightness is not None:
        lines.append(
            f"subject brightness {subject_brightness:.2f} against the frame {frame_brightness:.2f}"
        )
    ran = [str(stage) for stage in (stages or []) if str(stage).strip()]
    if ran:
        lines.append("already run: " + ", ".join(ran))
    return "\n".join(lines)


def validate_operations(payload: object) -> tuple[list[dict], list[str]]:
    """Keep catalog ids. Unknown ids are returned so the panel can name them."""

    if not isinstance(payload, dict):
        return [], []
    raw = payload.get("operations")
    if not isinstance(raw, list):
        return [], []
    kept: list[dict] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            name = item.strip().lower()
            action = "include"
            reason = ""
        elif isinstance(item, dict):
            name = str(item.get("id") or item.get("operation") or "").strip().lower()
            action = str(item.get("action") or "include").strip().lower()
            reason = " ".join(str(item.get("reason") or "").split())
        else:
            continue
        if action == "light":
            action = "lift"
        if action not in _ACTIONS:
            action = "include"
        if not name or name in seen:
            continue
        seen.add(name)
        if name not in KNOWN_IDS:
            dropped.append(name)
            continue
        if len(reason) > 180:
            reason = reason[:180].rstrip()
        kept.append({"id": name, "action": action, "reason": reason})
    return kept, dropped


def operation_features(operations: list[dict]) -> list[dict]:
    """Mask features named in a catalog choice, for the accept queue."""

    features: list[dict] = []
    seen: set[str] = set()
    for item in operations:
        name = str(item.get("id") or "")
        if name not in _FEATURE_IDS or name in seen or item.get("action") == "skip":
            continue
        seen.add(name)
        reason = str(item.get("reason") or "").strip() or name
        features.append({"id": name, "reason": reason})
    return features


def apply_operations(plan: dict, operations: list[dict] | None, dropped: list[str] | None = None) -> dict:
    """Let a model answer replace the guide order. An empty list leaves the guide."""

    chosen = [item for item in (operations or []) if isinstance(item, dict)]
    named = [str(item) for item in (dropped or []) if str(item).strip()]
    if named:
        plan["dropped"] = list(dict.fromkeys([*(plan.get("dropped") or []), *named]))
    if not chosen:
        return plan

    plan["catalog_chosen"] = True
    plan["operations"] = [dict(item) for item in chosen]
    history: list[str] = []
    local_order: list[str] = []
    skip = list(plan.get("skip") or [])
    skip_by = dict(plan.get("skip_by") or {})
    asks = [dict(item) for item in (plan.get("asks") or []) if isinstance(item, dict)]

    for item in chosen:
        name = str(item.get("id") or "")
        action = str(item.get("action") or "include")
        reason = str(item.get("reason") or "")
        if action == "skip":
            if name in _STAGE_IDS and name not in skip and skip_by.get(name) != "user":
                skip.append(name)
                skip_by[name] = "reasoner"
            if name == "vignette" and "vignette" not in skip and skip_by.get("vignette") != "user":
                skip.append("vignette")
                skip_by["vignette"] = "reasoner"
            if name in _HISTORY_IDS:
                skipped = list(plan.get("history_skipped") or [])
                if name not in skipped:
                    skipped.append(name)
                plan["history_skipped"] = skipped
            continue
        if name in _HISTORY_IDS and name not in history:
            history.append(name)
            _history_flag(plan, name, action)
        if name in _STAGE_IDS and name not in local_order:
            local_order.append(name)
        if action in {"lift", "darken", "hold"} and name in _ASK_TARGETS:
            ask = {"verb": action, "target": name, "source": "reasoner", "reason": reason}
            if not any(
                old.get("verb") == ask["verb"] and old.get("target") == ask["target"] for old in asks
            ):
                asks.append(ask)

    previous = [str(stage) for stage in (plan.get("guide_stages") or [])]
    ordered = ["grade"] if "grade" in previous or not previous else []
    for stage in local_order:
        if stage not in ordered:
            ordered.append(stage)
    for stage in previous:
        if stage not in ordered:
            ordered.append(stage)
    plan["guide_stages"] = ordered
    plan["skip"] = skip
    plan["skip_by"] = skip_by
    plan["asks"] = asks
    plan["history_chosen"] = history
    return plan


def _history_flag(plan: dict, name: str, action: str) -> None:
    if name == "crop":
        plan["crop_horizon"] = True
    elif name == "graduatednd":
        plan["gradient"] = True
        if plan.get("gradient_by") != "user":
            plan["gradient_by"] = "reasoner"
    elif name == "toneequal" and action == "lift":
        plan["shadow_lift"] = max(float(plan.get("shadow_lift") or 0), 0.5)
    elif name == "toneequal" and action in {"darken", "hold"}:
        plan["highlight_hold"] = True
    elif name == "channelmixerrgb" and action == "lift":
        plan["warmth"] = float(plan.get("warmth") or 0) or 0.16
    elif name == "channelmixerrgb" and action == "darken":
        plan["warmth"] = float(plan.get("warmth") or 0) or -0.16
    elif name == "colorzones" and action == "lift":
        plan["color"] = float(plan.get("color") or 0) or 0.2
    elif name == "colorzones" and action == "darken":
        plan["color"] = float(plan.get("color") or 0) or -0.2
    elif name == "filmicrgb":
        plan["display"] = "filmicrgb"
    elif name == "sigmoid":
        plan["display"] = "sigmoid"
