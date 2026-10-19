"""Accept steps whose region is a semantic drawn mask written into the sidecar.

One step is one history module, as everywhere else in this editor. Its
drawn mask is a darktable group of paths (and gradients or ellipses),
built from a segmentation of ``preview.jpg``:

| step id | history name | region | module |
| --- | --- | --- | --- |
| mask_sky | Grade the sky | sky | color balance rgb |
| mask_architecture | Grade the buildings | architecture | color balance rgb |
| mask_vegetation | Grade the vegetation | vegetation | color balance rgb |
| mask_water | Grade the water | water | color balance rgb |
| mask_ground | Grade the ground | ground | color balance rgb |
| mask_people | Lift the people | people | exposure |
| mask_background | Quiet the background | inverse of the subject | color balance rgb |

``mask_landscape`` is not a module. It expands into the five landscape
class steps in the order the video lesson uses (sky, architecture,
vegetation, water, ground). A class that is not in the photo leaves its
step off with a sentence, like a missing eye does.

The grades are the direction and rough size of the Lightroom moves in the
photoserge example 7 ("five landscape masks"), scaled for darktable and
kept inside the local-edit caps (``LOCAL_EV_CAP``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ai_dt.mapping import COLORBALANCE_VERSION, EXPOSURE_VERSION, clamp, colorbalance_params, exposure_params

# Lesson: keep a local brush within about half a stop.
LOCAL_EV_CAP = 0.5
# color balance rgb brilliance is a gain; +-0.25 is already a strong local move.
LOCAL_BRILLIANCE_CAP = 0.25

# Hue angles in color balance rgb (degrees, darktable UCS). Approximate:
# check them against the hue slider in darktable 5.6 (see HANDOFF.md).
HUE_WARM = 65.0
HUE_GREEN = 140.0
HUE_BLUE = 225.0

LANDSCAPE_ORDER = ("sky", "architecture", "vegetation", "water", "ground")


@dataclass(frozen=True)
class MaskStep:
    id: str
    name: str
    title: str
    region: str
    operation: str
    # Grades per action. "lift" and "darken" are the two directions a request can take.
    grades: dict = field(default_factory=dict)
    invert: bool = False
    what: str = "drawn mask"


def _cb(**values: float) -> dict:
    return dict(values)


STEPS: dict[str, MaskStep] = {
    "mask_sky": MaskStep(
        "mask_sky", "Grade the sky", "Darken and warm the sky", "sky", "colorbalancergb",
        {
            "darken": _cb(brilliance_global=-0.15, saturation_global=0.08, highlights_C=0.05, highlights_H=HUE_WARM),
            "lift": _cb(brilliance_global=0.1, saturation_global=0.05),
        },
        what="sky mask",
    ),
    "mask_architecture": MaskStep(
        "mask_architecture", "Grade the buildings", "Brighten the buildings and add crispness", "architecture",
        "colorbalancergb",
        {
            "lift": _cb(brilliance_global=0.12, contrast=0.12, saturation_global=-0.03),
            "darken": _cb(brilliance_global=-0.1, contrast=0.08),
        },
        what="building mask",
    ),
    "mask_vegetation": MaskStep(
        "mask_vegetation", "Grade the vegetation", "Make the vegetation greener", "vegetation", "colorbalancergb",
        {
            "lift": _cb(saturation_global=0.12, vibrance=0.1, midtones_C=0.03, midtones_H=HUE_GREEN),
            "darken": _cb(brilliance_global=-0.1, saturation_global=0.05),
        },
        what="vegetation mask",
    ),
    "mask_water": MaskStep(
        "mask_water", "Grade the water", "Darken the water, make it bluer and softer", "water", "colorbalancergb",
        {
            "darken": _cb(brilliance_global=-0.12, contrast=-0.1, shadows_C=0.06, shadows_H=HUE_BLUE, saturation_global=0.05),
            "lift": _cb(brilliance_global=0.08, shadows_C=0.04, shadows_H=HUE_BLUE),
        },
        what="water mask",
    ),
    "mask_ground": MaskStep(
        "mask_ground", "Grade the ground", "Darken the ground", "ground", "colorbalancergb",
        {
            "darken": _cb(brilliance_global=-0.15, saturation_global=-0.03),
            "lift": _cb(brilliance_global=0.1),
        },
        what="ground mask",
    ),
    "mask_people": MaskStep(
        "mask_people", "Lift the people", "Lift the people a little", "people", "exposure",
        {"lift": {"exposure": 0.25}, "darken": {"exposure": -0.2}},
        what="people mask",
    ),
    "mask_background": MaskStep(
        "mask_background", "Quiet the background", "Darken and mute the background", "subject",
        "colorbalancergb",
        {
            "darken": _cb(brilliance_global=-0.12, saturation_global=-0.15),
            "lift": _cb(brilliance_global=0.08),
        },
        invert=True,
        what="background mask",
    ),
}

DEFAULT_ACTION = {
    "mask_sky": "darken",
    "mask_architecture": "lift",
    "mask_vegetation": "lift",
    "mask_water": "darken",
    "mask_ground": "darken",
    "mask_people": "lift",
    "mask_background": "darken",
}

EXPANDS = {"mask_landscape": tuple(f"mask_{name}" for name in LANDSCAPE_ORDER)}

STEP_IDS = frozenset(STEPS)


def step_names() -> frozenset[str]:
    return frozenset(step.name for step in STEPS.values())


def is_mask_step(step_id: str) -> bool:
    return str(step_id or "") in STEPS


def expand(step_id: str) -> tuple[str, ...]:
    return EXPANDS.get(str(step_id or ""), (str(step_id or ""),))


def capped_grade(step: MaskStep, action: str) -> dict:
    """The step's grade for this action, inside the local caps."""

    direction = action if action in step.grades else DEFAULT_ACTION.get(step.id, "lift")
    if direction not in step.grades:
        direction = next(iter(step.grades))
    grade = dict(step.grades[direction])
    if "exposure" in grade:
        grade["exposure"] = clamp(float(grade["exposure"]), -LOCAL_EV_CAP, LOCAL_EV_CAP)
    for key in ("brilliance_global", "brilliance_shadows", "brilliance_highlights", "brilliance_midtones"):
        if key in grade:
            grade[key] = clamp(float(grade[key]), -LOCAL_BRILLIANCE_CAP, LOCAL_BRILLIANCE_CAP)
    for key in ("shadows_C", "midtones_C", "highlights_C", "global_C"):
        if key in grade:
            grade[key] = clamp(float(grade[key]), 0.0, 0.12)
    for key in ("saturation_global", "contrast", "vibrance"):
        if key in grade:
            grade[key] = clamp(float(grade[key]), -0.3, 0.3)
    return grade


def module_blob(step: MaskStep, grade: dict) -> tuple[bytes, int]:
    if step.operation == "exposure":
        return exposure_params(float(grade.get("exposure", 0.0))), EXPOSURE_VERSION
    return colorbalance_params(grade), COLORBALANCE_VERSION


def build_spec(step_id: str, mask, *, action: str = "", border: float = 0.015, source: str = "") -> dict | None:
    """The ``mask.json`` body for one step, or None when the mask has no outline."""

    from ai_dt.dt_shapes import mask_to_forms

    step = STEPS[step_id]
    forms = mask_to_forms(mask, border=border, max_paths=6, max_points=48, min_area=0.004)
    if not forms:
        return None
    grade = capped_grade(step, action)
    params, version = module_blob(step, grade)
    return {
        "kind": "shapes",
        "step": step.id,
        "name": step.name,
        "what": step.what,
        "operation": step.operation,
        "modversion": version,
        "params": params.hex(),
        "grade": grade,
        "invert": step.invert,
        "forms": forms,
        "source": source,
    }


def subject_boxes(recipe: dict) -> list[dict]:
    analysis = recipe.get("analysis") if isinstance(recipe.get("analysis"), dict) else {}
    return [item for item in (analysis.get("subjects") or []) if isinstance(item, dict)]


def write_step_mask(recipe: dict, directory, step_id: str, *, action: str = "", image=None, backend: str = "") -> dict | None:
    """Segment the preview, write ``mask.json``, and return the ``step_mask`` entry.

    None when there is no preview or the region is not in the photo.
    """

    from ai_dt.segment import segment

    step = STEPS[step_id]
    if image is None:
        path = directory / "preview.jpg"
        if not path.is_file():
            return None
        from PIL import Image

        with Image.open(path) as opened:
            image = opened.convert("RGB")
    chosen_backend = backend or str(recipe.get("segmenter") or "") or "auto"
    found = segment(image, [step.region], boxes=subject_boxes(recipe), backend=chosen_backend)
    mask = found.get(step.region)
    if mask is None:
        return None
    spec = build_spec(step_id, mask, action=action, source=found.sources.get(step.region, ""))
    if spec is None:
        return None
    spec_path = directory / "mask.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return {
        "kind": "shapes",
        "file": str(spec_path),
        "name": step.name,
        "what": step.what,
        "source": spec["source"],
    }


def missing_sentence(step_id: str) -> str:
    step = STEPS[step_id]
    region = "subject" if step.id == "mask_background" else step.region
    if region == "architecture":
        region = "building"
    return f"No {region} was found in this photo, so this step was left off."
