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

Lesson steps (photoserge examples 8 to 11) use the same machinery, but
their group is built from gradients and ellipses, not only outlines:

| step id | history name | shapes | module |
| --- | --- | --- | --- |
| mask_sky_gradient | Darken the sky gradient | sky outline intersected with a top gradient (ex. 9) | exposure |
| mask_bottom_gradient | Darken the bottom | bottom gradient (ex. 8) | exposure |
| mask_edge_burn | Burn the edges | four edge gradients, minus the subject (ex. 11) | exposure |
| mask_glow | Glow behind the subject | wide ellipse behind the subject, minus the subject (ex. 10) | exposure |

The lessons push a gradient to -0.85 EV and a glow past +1 EV. Here every
local exposure stays inside ``LOCAL_EV_CAP`` (the brush rule of the same
lessons); stack the step again for more.

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
    # How the group is built: "segment" (outline of ``region``), "sky_gradient",
    # "bottom_gradient", "edge_burn" or "glow".
    builder: str = "segment"


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

STEPS.update(
    {
        "mask_sky_gradient": MaskStep(
            "mask_sky_gradient", "Darken the sky gradient", "Darken the sky with a gradient from the top", "sky",
            "exposure", {"darken": {"exposure": -0.5}, "lift": {"exposure": 0.3}},
            what="sky gradient", builder="sky_gradient",
        ),
        "mask_bottom_gradient": MaskStep(
            "mask_bottom_gradient", "Darken the bottom", "Darken the bottom of the frame with a gradient", "",
            "exposure", {"darken": {"exposure": -0.5}, "lift": {"exposure": 0.3}},
            what="bottom gradient", builder="bottom_gradient",
        ),
        "mask_edge_burn": MaskStep(
            "mask_edge_burn", "Burn the edges", "Burn the edges to lead the eye in", "subject",
            "exposure", {"darken": {"exposure": -0.4}, "lift": {"exposure": 0.25}},
            what="edge gradients", builder="edge_burn",
        ),
        "mask_glow": MaskStep(
            "mask_glow", "Glow behind the subject", "Add a glow behind the subject", "subject",
            "exposure", {"lift": {"exposure": 0.5}, "darken": {"exposure": -0.3}},
            what="glow ellipse", builder="glow",
        ),
    }
)

# Animals (user priority after landscapes). One outline per animal, joined
# by union; the main animal is the largest. Fur and eye crispness come from
# local contrast, not exposure, and stay subtle.
STEPS.update(
    {
        "mask_animal": MaskStep(
            "mask_animal", "Lift the animal", "Lift the main animal and crisp its fur", "animal",
            "colorbalancergb",
            {
                "lift": _cb(brilliance_global=0.1, contrast=0.12, vibrance=0.05),
                "darken": _cb(brilliance_global=-0.08, contrast=0.06),
            },
            what="animal mask", builder="animal_main",
        ),
        "mask_animals": MaskStep(
            "mask_animals", "Lift the animals", "Lift every animal a little", "animal",
            "colorbalancergb",
            {
                "lift": _cb(brilliance_global=0.08, contrast=0.1, vibrance=0.04),
                "darken": _cb(brilliance_global=-0.08),
            },
            what="animal masks", builder="animal_all",
        ),
        "mask_animal_background": MaskStep(
            "mask_animal_background", "Quiet around the animals", "Darken and soften around the animals",
            "animal", "colorbalancergb",
            {
                # Darker and a little muted, not milky: lower contrast and a
                # -0.1 saturation cut squeezed the water flat and grey (duck
                # render: L p5-p95 range 38.7 -> 31.2). Deeper shadows and a
                # touch more contrast keep the range (35.9) at the same depth.
                "darken": _cb(brilliance_global=-0.12, brilliance_shadows=-0.05, saturation_global=-0.03, contrast=0.04),
                "lift": _cb(brilliance_global=0.06),
            },
            invert=True, what="inverse animal mask", builder="animal_all",
        ),
    }
)

# Default editor step: sharpen only the subject (see ai_dt/default_steps.py).
# "lift" is the normal amount, "hold" the reduced one for noisy photos. The
# threshold keeps flat (noisy) areas out.
STEPS["mask_sharpen_subject"] = MaskStep(
    "mask_sharpen_subject", "Sharpen the subject", "Sharpen the main subject", "subject", "sharpen",
    {
        "lift": {"radius": 1.5, "amount": 0.6, "threshold": 0.9},
        "hold": {"radius": 1.2, "amount": 0.35, "threshold": 1.5},
    },
    what="subject mask",
)

LESSON_STEP_IDS = ("mask_sky_gradient", "mask_bottom_gradient", "mask_edge_burn", "mask_glow")

DEFAULT_ACTION = {
    "mask_sky": "darken",
    "mask_architecture": "lift",
    "mask_vegetation": "lift",
    "mask_water": "darken",
    "mask_ground": "darken",
    "mask_people": "lift",
    "mask_background": "darken",
    "mask_sky_gradient": "darken",
    "mask_bottom_gradient": "darken",
    "mask_edge_burn": "darken",
    "mask_glow": "lift",
    "mask_animal": "lift",
    "mask_animals": "lift",
    "mask_animal_background": "darken",
    "mask_sharpen_subject": "lift",
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
    if step.operation == "sharpen":
        from ai_dt.default_steps import SHARPEN_VERSION, sharpen_blob

        return (
            sharpen_blob(float(grade.get("radius", 1.5)), float(grade.get("amount", 0.5)), float(grade.get("threshold", 0.9))),
            SHARPEN_VERSION,
        )
    if step.operation == "exposure":
        return exposure_params(float(grade.get("exposure", 0.0))), EXPOSURE_VERSION
    return colorbalance_params(grade), COLORBALANCE_VERSION


def build_spec(step_id: str, mask, *, action: str = "", border: float = 0.015, source: str = "") -> dict | None:
    """The ``mask.json`` body for one step, or None when the mask has no outline."""

    from ai_dt.dt_shapes import mask_to_forms

    forms = mask_to_forms(mask, border=border, max_paths=6, max_points=48, min_area=0.004)
    if not forms:
        return None
    return spec_from_forms(step_id, forms, action=action, source=source)


def spec_from_forms(step_id: str, forms: list[dict], *, action: str = "", source: str = "") -> dict:
    step = STEPS[step_id]
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


def _outer(forms: list[dict]) -> list[dict]:
    return [form for form in forms if form.get("mode", "union") != "difference"]


def _subtract(forms: list[dict], cutters: list[dict]) -> list[dict]:
    """Append each outer outline of ``cutters`` as a difference member.

    A cutter's holes (the water between a duck's legs) are not part of the
    subject, so they get the base mask back. A group is a left fold, so for a
    single base form that is ``((base - outer) | holes) & base``; for several
    base forms each hole is added back at the base mask's mean strength
    inside it.
    """

    out = forms + [dict(form, mode="difference") for form in _outer(cutters)]
    holes = [form for form in cutters if form.get("mode") == "difference"]
    if not holes or not forms:
        return out
    if len(forms) == 1:
        return out + [dict(hole, mode="union") for hole in holes] + [dict(forms[0], mode="intersect")]
    from ai_dt.dt_shapes import render_form, render_forms

    width, height = 384, 256
    base = render_forms(forms, width, height)
    for hole in holes:
        inside = render_form(dict(hole, mode="union"), width, height) >= 0.5
        strength = float(base[inside].mean()) if inside.any() else 0.0
        if strength >= 0.02:
            out.append(dict(hole, mode="union", opacity=round(min(strength, 1.0), 3)))
    return out


def sky_gradient_forms(sky_forms: list[dict], *, extent: float = 0.55) -> list[dict]:
    """Sky outline(s) intersected with a gradient from the top (example 9).

    The sky outlines go first so the group's left fold is
    ``(sky1 | sky2 - holes) & gradient``. Without a sky the gradient alone
    is the mask.
    """

    from ai_dt.dt_shapes import edge_gradient

    gradient = edge_gradient("top", extent, compression=0.3)
    if not sky_forms:
        return [dict(gradient, mode="union")]
    return list(sky_forms) + [dict(gradient, mode="intersect")]


def bottom_gradient_forms(*, extent: float = 0.3) -> list[dict]:
    from ai_dt.dt_shapes import edge_gradient

    return [dict(edge_gradient("bottom", extent, compression=0.2), mode="union")]


def edge_burn_forms(subject_forms: list[dict] | None = None, *, extent: float = 0.14) -> list[dict]:
    """Union of four edge gradients, minus the subject (example 11)."""

    from ai_dt.dt_shapes import edge_gradient

    forms = [
        dict(edge_gradient(edge, extent, compression=0.12), mode="union")
        for edge in ("top", "bottom", "left", "right")
    ]
    return _subtract(forms, subject_forms or [])


def glow_forms(box: tuple[float, float, float, float], subject_forms: list[dict] | None = None) -> list[dict]:
    """A wide soft ellipse behind the subject, minus the subject (example 10)."""

    x, y, w, h = box
    cx = min(max(x + w / 2.0, 0.02), 0.98)
    cy = min(max(y + h * 0.4, 0.02), 0.98)
    # Wide enough to read as light behind the subject, not a halo on its edge.
    rx = min(max(w * 1.5, h * 0.5, 0.12), 0.6)
    ry = min(max(h * 0.8, w * 0.8, 0.12), 0.6)
    ellipse = {"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry, "border": 0.2, "mode": "union"}
    return _subtract([ellipse], subject_forms or [])


def _main_box(recipe: dict) -> tuple[float, float, float, float] | None:
    from ai_dt.segment.base import SUBJECT_CATEGORIES, box_of

    best = None
    for item in subject_boxes(recipe):
        parsed = box_of(item)
        if not parsed or parsed[0] not in SUBJECT_CATEGORIES:
            continue
        area = parsed[1][2] * parsed[1][3]
        if best is None or area > best[0]:
            best = (area, parsed[1])
    return best[1] if best else None


def subject_boxes(recipe: dict) -> list[dict]:
    analysis = recipe.get("analysis") if isinstance(recipe.get("analysis"), dict) else {}
    return [item for item in (analysis.get("subjects") or []) if isinstance(item, dict)]


def _open_image(directory, image):
    if image is not None:
        return image
    path = directory / "preview.jpg"
    if not path.is_file():
        return None
    from PIL import Image

    with Image.open(path) as opened:
        return opened.convert("RGB")


def step_forms(recipe: dict, step_id: str, image, backend: str = "") -> tuple[list[dict] | None, str]:
    """The group members for one step and the segmenter that found the region."""

    from ai_dt.dt_shapes import mask_to_forms

    step = STEPS[step_id]
    chosen_backend = backend or str(recipe.get("segmenter") or "") or "auto"
    if step.builder == "bottom_gradient":
        forms = bottom_gradient_forms()
        # An animal standing on the bright ledge: darken the ledge, not its
        # feet and belly (darktable showed the gradient undoing the lift).
        animals = [
            item for item in ((recipe.get("analysis") or {}).get("subjects") or [])
            if isinstance(item, dict) and item.get("category") == "animal"
        ]
        if animals and image is not None:
            from ai_dt.segment.animals import animal_forms, segment_animals

            found = segment_animals(image, subject_boxes(recipe), backend=chosen_backend)
            outline = animal_forms(found, which="all")
            if outline:
                return _subtract(forms, outline), "gradient minus " + ", ".join(sorted({a.source for a in found.animals}))
        return forms, "gradient"
    if step.builder in ("animal_main", "animal_all"):
        from ai_dt.segment.animals import animal_forms, segment_animals

        if image is None:
            return None, ""
        found = segment_animals(image, subject_boxes(recipe), backend=chosen_backend)
        forms = animal_forms(found, which="main" if step.builder == "animal_main" else "all")
        kinds = sorted({animal.source for animal in found.animals})
        union = found.union(image.size) if found.animals else None
        if forms and union is not None and not _plausible_subject(union) and _clear_subject_elsewhere(image, union):
            # A detector box on water or foreground made SAM outline scenery
            # while a clear subject sits somewhere else. A close-up animal that
            # fills the frame is kept: the finder skips edge regions, so it
            # finds nothing apart from it.
            forms = None
        if forms:
            return forms, ", ".join(kinds)
        # The plan says there is an animal (from the picture model, the typed
        # description or the request) but the detector missed it, e.g. a
        # parrot ornament. Use the main subject instead of failing.
        from ai_dt.segment import segment

        subject = segment(image, ["subject"], boxes=subject_boxes(recipe), backend=chosen_backend)
        mask = subject.get("subject")
        source = str(subject.sources.get("subject", ""))
        if mask is not None and not _plausible_subject(mask):
            # A model can return a patch of water or foreground running off the
            # frame. Fall back to the local-contrast finder, which skips those.
            mask, source = None, ""
        if mask is None:
            from ai_dt.segment.heuristic import HeuristicSegmenter

            mask = HeuristicSegmenter().segment(image, ["subject"], boxes=[]).get("subject")
            source = "heuristic"
            if mask is None or not _plausible_subject(mask):
                return None, ""
        fallback = mask_to_forms(mask, border=0.015, max_paths=6, max_points=48, min_area=0.004)
        if not fallback:
            return None, ""
        return fallback, "subject (no animal detected): " + source
    region_forms: list[dict] = []
    source = ""
    box = _main_box(recipe) if step.region == "subject" else None
    if step.builder == "glow" and box is None:
        return None, ""
    if step.region and image is not None and (step.builder in ("segment", "sky_gradient") or box is not None):
        from ai_dt.segment import segment

        found = segment(image, [step.region], boxes=subject_boxes(recipe), backend=chosen_backend)
        mask = found.get(step.region)
        if mask is not None:
            region_forms = mask_to_forms(mask, border=0.015, max_paths=6, max_points=48, min_area=0.004)
            source = found.sources.get(step.region, "")
    if step.builder == "segment":
        return (region_forms or None), source
    if step.builder == "sky_gradient":
        return sky_gradient_forms(region_forms), source or "gradient"
    if step.builder == "edge_burn":
        return edge_burn_forms(region_forms), source or "gradient"
    if step.builder == "glow":
        return glow_forms(box, region_forms), source or "subject box"
    return None, ""


def _plausible_subject(mask) -> bool:
    """False for a mask that looks like scenery: very large, or running along the frame edge."""

    import numpy as np

    binary = np.asarray(mask) >= 0.5
    if not binary.any() or binary.mean() > 0.35:
        return False
    border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
    return float(border.mean()) <= 0.08


def _clear_subject_elsewhere(image, mask) -> bool:
    """True when the local-contrast finder sees a subject that the mask mostly misses."""

    import numpy as np
    from PIL import Image as _Image

    from ai_dt.segment.heuristic import HeuristicSegmenter

    found = HeuristicSegmenter().segment(image, ["subject"], boxes=[]).get("subject")
    if found is None:
        return False
    other = np.asarray(_Image.fromarray((np.asarray(mask) >= 0.5).astype(np.uint8) * 255).resize(found.shape[::-1])) > 127
    subject = found >= 0.5
    if not subject.any():
        return False
    return float((subject & other).sum()) / float(subject.sum()) < 0.2


def write_step_mask(recipe: dict, directory, step_id: str, *, action: str = "", image=None, backend: str = "") -> dict | None:
    """Build the step's group, write ``mask.json``, and return the ``step_mask`` entry.

    None when there is no preview or the region is not in the photo.
    """

    step = STEPS[step_id]
    image = _open_image(directory, image)
    if image is None:
        return None
    forms, source = step_forms(recipe, step_id, image, backend)
    if not forms:
        return None
    spec = spec_from_forms(step_id, forms, action=action, source=source)
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
    if step.builder == "glow":
        return "No subject was found to glow behind, so this step was left off."
    if step.builder in ("animal_main", "animal_all"):
        return "No animal was found in this photo, so this step was left off."
    if step.builder != "segment":
        return "There is no preview to place this step on, so it was left off."
    region = "subject" if step.id == "mask_background" else step.region
    if region == "architecture":
        region = "building"
    return f"No {region} was found in this photo, so this step was left off."
