"""Run the picture guide: masks, local grades, and a global correction.

A Darktable style can carry global module values. It cannot carry a drawn
mask. This render is the picture that has those masks. Which masks exist,
and how hard they push, comes from the guide for the picture type. The
sharp mask is the union of the features the guide keeps: an animal and the
perch it stands on, a person and the chair or the object in their hands.
A facade is not given a bird's background blur, and a night window is not
promoted into a subject. A mirror in the water is sharpened on its own:
a bird's streak, or a wide echo of a house or a mountain. That does not
cut the building or the land out of the picture.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image, ImageFilter

from ai_dt.guides.decide import (
    Blob,
    Picture,
    decide_picture,
    plain_subject_box,
    skin_subject_box,
    warm_subject_box,
)
from ai_dt.guides.library import Guide, GuideMatch, MaskPlan, clamp_grades, match_guide
from ai_dt.guides.support import (
    _components,
    _feather,
    creature_box,
    extend_animal,
    held_mask,
    is_elongated,
    perch_mask,
    reflected_subject,
    seat_mask,
    water_reflection,
)
from ai_dt.reading import mirror_box, mirror_intention
from ai_dt.schema import Atmosphere, Detection
from ai_dt.simulate import contrast_about_luma, grade_array

_REFLECTION_MISS = "The reflection under the bird could not be found, so it was not clarified."


def render_finish(
    pixels: np.ndarray,
    atmosphere: Atmosphere,
    detections: list[Detection],
    vignette: dict | None = None,
    *,
    guide: Guide | None = None,
    picture: Picture | None = None,
    guide_match: GuideMatch | None = None,
    nudges: dict | None = None,
    stages: list[str] | None = None,
    scales: dict[str, float] | None = None,
    intentions: dict | None = None,
    omit: set[str] | None = None,
    asks: list | None = None,
    strong: bool = False,
    caps_open: bool = False,
    intents: list | None = None,
) -> tuple[np.ndarray, dict]:
    """Return float RGB and a description of the type, guide, and masks used.

    ``stages`` is the order to run. When it is omitted the guide's own
    list is used. A measurement is kept after each stage. A stage that
    overshoots is put back, and the stages after it do not run.

    ``caps_open`` raises the exposure ceiling past about ±0.35 EV. It
    defaults closed. ``strong`` is the same switch. ``intents`` is a list
    of ``{"stage", "action"}`` records: darken, lift, hold, or skip.
    Sentences are not read.
    """

    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    if picture is None:
        picture = decide_picture(pixels, detections)
    if guide is None:
        if guide_match is None:
            guide_match = match_guide(picture.type)
        guide = guide_match.guide
    if min(height, width) < 8:
        graded = grade_array(pixels, atmosphere, vignette)
        return graded, _info(
            picture,
            guide,
            applied=False,
            subjects=0,
            eyes=0,
            background="unchanged",
            subject_ev=0.0,
            ran=["grade"],
            skipped=[stage for stage in guide.stages if stage != "grade"],
            reason="The image is too small to isolate a subject.",
            features=[],
        )

    subject, subject_count = _maybe_subject(pixels, detections, picture, guide)
    # Portraits, groups, night windows, and food do not grow a reflection layer.
    if picture.type in {"animal", "landscape", "architecture", "general"}:
        reflection = _reflection_mask(pixels, subject)
    else:
        reflection = None
    sharp, kept_now = _sharp_union(pixels, subject, guide)
    if reflection is not None and sharp is not None:
        # The mirror is part of the subject. A soft edge would let the
        # background blur pull the reflection back down into the water.
        covered = np.clip(reflection / np.float32(0.45), 0.0, 1.0)
        sharp = np.maximum(sharp, covered)
    isolate = subject is not None
    order = list(guide.stages if stages is None else stages)
    echo = mirror_box(pixels)
    reflection_intention = "lift"
    reflection_failure = ""
    if echo is not None:
        reflection_intention = mirror_intention(pixels, echo)
    if intentions and intentions.get("reflection") in {"lift", "hold", "quiet"}:
        reflection_intention = str(intentions["reflection"])
    theme_mask, theme_kind = _theme_field(pixels, subject)
    if intentions and intentions.get("theme") in {"horizon", "window", "island", "reflection"}:
        forced = _forced_theme(pixels, subject, str(intentions["theme"]))
        if forced is not None:
            theme_mask, theme_kind = forced
    # Theme is not one of the shipped guide stages. Paint it unless this
    # run already stopped, or the caller listed stages and left it out
    # on purpose by passing a list that is not the guide.
    paint_theme = stages is None or "theme" in order or "theme" not in set(guide.stages)
    scale_of = scales or {}
    intent_asks, intent_omit = _stage_intents(intents)
    omitted = set(omit or [])
    omitted.update(intent_omit)
    ask_list = [item for item in (asks or []) if isinstance(item, dict)]
    ask_list.extend(intent_asks)
    # Default stays closed. Either the pipeline's strong flag or caps_open lifts it.
    opened = bool(strong) or bool(caps_open)
    allow_dark_surround = opened and _asked(ask_list, "darken", {"foreground", "rock"})
    if _asked(ask_list, "lift", {"reflection"}):
        reflection_intention = "lift"
    elif _asked(ask_list, "hold", {"reflection"}):
        reflection_intention = "hold"
    elif _asked(ask_list, "darken", {"reflection"}):
        reflection_intention = "quiet"

    do_grade = stages is None or "grade" in order
    if do_grade:
        graded = grade_array(pixels, atmosphere, None if isolate else vignette)
    else:
        graded = np.clip(pixels.astype(np.float32, copy=False), 0.0, 1.0)
    current = graded
    ran = ["grade"] if do_grade else []
    skipped: list[str] = []
    eyes = 0
    eye_mask: np.ndarray | None = None
    subject_ev = 0.0
    background = "unchanged"
    saw_subject = False
    measurements: list[dict] = []
    stop_reason = ""
    if do_grade:
        measurements.append(_measurement("grade", current, subject, None, True, ""))

    for index, stage in enumerate(order):
        if stage == "grade":
            continue
        if stage == "theme":
            continue
        if stage in omitted and not (stage == "reflection" and _asked(ask_list, "lift", {"reflection"})):
            # A guide or reasoner skip does not cancel a request to clarify the mirror.
            skipped.append(stage)
            continue
        plan = guide.masks.get(stage)
        if plan is None or plan.build == "never":
            skipped.append(stage)
            continue
        plan = _scaled_plan(clamp_grades(plan, opened=opened), scale_of.get(stage), opened=opened)
        nudge = None if not nudges else nudges.get(stage)
        if nudge is not None:
            plan = _with_nudge(plan, nudge)
        before = current
        proposed = current
        eye_now = eye_mask
        applied_stage = False
        if stage == "subject":
            if subject is None:
                skipped.append(stage)
                continue
            if picture.brightness < 0.32 and picture.type in {"portrait", "group", "street"}:
                plan = replace(plan, exposure=_low_key_face_exposure(pixels, subject, plan.exposure, opened=opened))
            plan = _subject_ask(plan, ask_list, opened)
            proposed = _grade_region(current, subject, plan)
            subject_ev = plan.exposure
            saw_subject = True
            applied_stage = True
        elif stage == "eyes":
            if subject is None:
                skipped.append(stage)
                continue
            eye_now, eyes = _build_eyes(pixels, detections, subject)
            if eye_now is None:
                eyes = 0
                skipped.append(stage)
                continue
            eye_mask = eye_now
            proposed = _grade_region(current, eye_now, plan)
            applied_stage = True
        elif stage == "reflection":
            if reflection is None and _asked(ask_list, "lift", {"reflection"}):
                # The user asked to clarify the mirror. A guide miss does not win.
                echo = mirror_box(pixels)
                if echo is not None:
                    reflection = _soft_rect(height, width, echo)
                    reflection_intention = "lift"
            if reflection is None:
                if _asked(ask_list, "lift", {"reflection"}):
                    reflection_failure = _REFLECTION_MISS
                skipped.append(stage)
                continue
            if _asked(ask_list, "lift", {"reflection"}):
                reflection_intention = "lift"
            proposed = _grade_reflection(current, reflection, plan, reflection_intention)
            applied_stage = True
        elif stage == "skin":
            if subject is None:
                skipped.append(stage)
                continue
            skin = _skin_mask(pixels, subject)
            if skin is None:
                skipped.append(stage)
                continue
            proposed = _grade_region(current, skin, plan)
            applied_stage = True
        elif stage == "background":
            if subject is None or (plan.defocus <= 0.0 and plan.mute <= 0.0 and abs(plan.exposure) < 0.01):
                skipped.append(stage)
                continue
            if picture.brightness < 0.32 and picture.type in {"portrait", "group", "street"}:
                # A night behind a person is part of the picture. Heavy blur
                # and a mute turn the lights into a flat color.
                plan = replace(
                    plan,
                    defocus=min(plan.defocus, 0.28),
                    mute=min(plan.mute, 0.1),
                    saturation=max(plan.saturation, -0.02),
                )
            cover = sharp if sharp is not None else subject
            before_metrics = _measurement(stage, current, subject, eye_mask, True, "")
            if (
                not allow_dark_surround
                and plan.exposure < 0.0
                and before_metrics["background"] is not None
                and before_metrics["subject"] is not None
                and before_metrics["background"] + 0.02 < before_metrics["subject"]
            ):
                # Already darker than the subject. Defocus can stay. Exposure does not.
                plan = replace(plan, exposure=0.0)
            if allow_dark_surround:
                floor = -0.5 if opened else -0.2
                plan = clamp_grades(replace(plan, exposure=min(float(plan.exposure), floor)), opened=opened)
            hold = _behind_mask(subject) if _asked(ask_list, "lift", {"behind"}) else None
            proposed = _grade_background(
                current,
                graded,
                cover,
                plan,
                theme_mask if theme_kind in {"window", "island"} else None,
                hold,
                darken_all=allow_dark_surround,
            )
            if plan.defocus > 0.05:
                background = "defocus"
            applied_stage = True
        elif stage == "sky":
            sky = _sky_mask(pixels)
            if sky is None:
                skipped.append(stage)
                continue
            proposed = _grade_region(current, sky, plan)
            applied_stage = True
        elif stage == "horizon":
            horizon = _horizon_mask(pixels)
            if horizon is None:
                skipped.append(stage)
                continue
            warmth = 0.0 if nudge is None else float(getattr(nudge, "warmth", 0.0))
            proposed = _grade_horizon(current, horizon, plan, warmth)
            applied_stage = True
        elif stage == "foreground":
            if _frame_is_the_request(ask_list):
                # This request is the bird, the dark water, and the mirror.
                # A land or structure grade does not belong on it.
                skipped.append(stage)
                continue
            foreground = _foreground_mask(pixels, plan, picture, detections)
            if foreground is None:
                skipped.append(stage)
                continue
            proposed = _grade_region(current, foreground, plan)
            applied_stage = True
        elif stage == "highlights":
            protected, used = _grade_highlights(current, plan)
            if not used:
                skipped.append(stage)
                continue
            proposed = protected
            applied_stage = True
        else:
            skipped.append(stage)
            continue
        if not applied_stage:
            continue
        reason_stop = _overshot(
            picture, stage, before, proposed, subject, eye_mask, allow_dark_surround=allow_dark_surround
        )
        if reason_stop:
            measurements.append(_measurement(stage, proposed, subject, eye_mask, False, reason_stop))
            current = before
            if stage == "eyes":
                eyes = 0
            if stage == "subject":
                saw_subject = False
                subject_ev = 0.0
            if stage == "background":
                background = "unchanged"
            stop_reason = reason_stop
            for later in order[index + 1 :]:
                if later not in skipped and later != "grade":
                    skipped.append(later)
            break
        current = proposed
        ran.append(stage)
        measurements.append(_measurement(stage, current, subject, eye_mask, True, ""))

    theme_note = ""
    if "theme" in omitted:
        skipped.append("theme")
    elif (
        paint_theme
        and not stop_reason
        and theme_mask is not None
        and not (theme_kind == "horizon" and "horizon" in ran)
    ):
        theme_plan = clamp_grades(
            MaskPlan(
                build="subject",
                skip="",
                exposure=0.0 if theme_kind == "window" else 0.05,
                contrast=0.0,
                saturation=0.0,
                clarity=0.04 if theme_kind != "window" else 0.0,
                defocus=0.0,
                mute=0.0,
            )
        )
        current = _grade_region(current, theme_mask, theme_plan)
        ran.append("theme")
        theme_note = theme_kind
        measurements.append(_measurement("theme", current, subject, eye_mask, True, ""))

    gradient_note: dict = {}
    if not stop_reason:
        current, gradient_note, ran, skipped = _apply_gradient(
            current,
            pixels,
            subject,
            picture,
            ask_list,
            intentions,
            omitted,
            opened,
            ran,
            skipped,
        )
        current, ran = _apply_rock(
            current,
            pixels,
            subject,
            ask_list,
            omitted,
            opened,
            ran,
        )
        current = _apply_behind(current, subject, ask_list, opened)

    if (
        reflection is not None
        and "reflection" in skipped
        and _asked(ask_list, "lift", {"reflection"})
    ):
        # An earlier stage can stop the run. That does not cancel the mirror
        # the user asked to clarify.
        ref_plan = guide.masks.get("reflection")
        if ref_plan is not None and ref_plan.build != "never":
            ref_plan = _scaled_plan(
                clamp_grades(ref_plan, opened=opened), scale_of.get("reflection"), opened=opened
            )
            current = _grade_reflection(current, reflection, ref_plan, "lift")
            skipped = [name for name in skipped if name != "reflection"]
            if "reflection" not in ran:
                ran.append("reflection")
            reflection_failure = ""
            reflection_intention = "lift"

    reason = guide.reason
    if guide_match is not None and guide_match.fallback:
        reason = guide_match.reason
    if stop_reason:
        reason = f"{reason} {stop_reason}".strip()
    features = _kept_features(guide, picture, ran, saw_subject, kept_now)
    return np.clip(current, 0.0, 1.0), _info(
        picture,
        guide,
        applied=saw_subject,
        subjects=subject_count if saw_subject else 0,
        eyes=eyes if "eyes" in ran else 0,
        background=background,
        subject_ev=subject_ev if saw_subject else 0.0,
        ran=ran,
        skipped=skipped,
        reason=reason,
        features=features,
        measurements=measurements,
        stop_reason=stop_reason,
        reflection=reflection_intention if reflection is not None else "",
        reflection_failure=reflection_failure,
        theme=theme_note or theme_kind,
        scales={key: float(value) for key, value in scale_of.items()},
        gradient=gradient_note,
        vignette="vignette" not in omitted,
        strong=opened,
    )


def _info(
    picture: Picture,
    guide: Guide,
    *,
    applied: bool,
    subjects: int,
    eyes: int,
    background: str,
    subject_ev: float,
    ran: list[str],
    skipped: list[str],
    reason: str,
    features: list[str],
    measurements: list[dict] | None = None,
    stop_reason: str = "",
    reflection: str = "",
    reflection_failure: str = "",
    theme: str = "",
    scales: dict | None = None,
    gradient: dict | None = None,
    vignette: bool = True,
    strong: bool = False,
) -> dict:
    return {
        "applied": applied,
        "subject_masks": subjects,
        "eyes": eyes,
        "background": background,
        "subject_ev": subject_ev if applied else 0.0,
        "reason": reason,
        "picture_type": picture.type,
        "guide": guide.id,
        "guide_file": guide.path,
        "stages": list(guide.stages),
        "stages_ran": list(ran),
        "skipped": list(skipped),
        "features": list(features),
        "measurements": list(measurements or []),
        "stop_reason": stop_reason,
        "reflection": reflection,
        "reflection_failure": reflection_failure,
        "theme": theme,
        "scales": dict(scales or {}),
        "gradient": dict(gradient or {}),
        "vignette": bool(vignette),
        "strong": bool(strong),
    }


def _scaled_plan(plan: MaskPlan, scale: float | None, *, opened: bool = False) -> MaskPlan:
    if scale is None or abs(float(scale) - 1.0) < 1e-6:
        return plan
    factor = max(0.0, min(1.5, float(scale)))
    return clamp_grades(
        replace(
            plan,
            exposure=plan.exposure * factor,
            contrast=plan.contrast * factor,
            saturation=plan.saturation * factor,
            clarity=plan.clarity * factor,
        ),
        opened=opened,
    )


def _masked_mean(luma: np.ndarray, mask: np.ndarray | None) -> float | None:
    if mask is None:
        return None
    chosen = mask > np.float32(0.45)
    if int(chosen.sum()) < 8:
        return None
    return float(luma[chosen].mean())


def _measurement(
    stage: str,
    pixels: np.ndarray,
    subject: np.ndarray | None,
    eyes: np.ndarray | None,
    kept: bool,
    reason: str,
) -> dict:
    luma = _luma(pixels)
    background = None
    if subject is not None:
        outside = subject < np.float32(0.35)
        if int(outside.sum()) >= 8:
            background = float(luma[outside].mean())
    return {
        "stage": stage,
        "frame": round(float(luma.mean()), 4),
        "subject": None if _masked_mean(luma, subject) is None else round(_masked_mean(luma, subject), 4),
        "eyes": None if _masked_mean(luma, eyes) is None else round(_masked_mean(luma, eyes), 4),
        "background": None if background is None else round(background, 4),
        "spread": round(float(luma.std()), 4),
        "kept": kept,
        "reason": reason,
    }


def _overshot(
    picture: Picture,
    stage: str,
    before: np.ndarray,
    after: np.ndarray,
    subject: np.ndarray | None,
    eyes: np.ndarray | None,
    *,
    allow_dark_surround: bool = False,
) -> str:
    """A short reason when this stage went too far, else an empty string."""

    after_luma = _luma(after)
    if stage == "eyes" and eyes is not None and subject is not None:
        eye_level = _masked_mean(after_luma, eyes)
        face_level = _masked_mean(after_luma, subject)
        if eye_level is not None and face_level is not None and eye_level > face_level + 0.02:
            return "The eye is hotter than the face."
    if stage == "background" and subject is not None and not allow_dark_surround:
        before_metrics = _measurement(stage, before, subject, eyes, True, "")
        after_metrics = _measurement(stage, after, subject, eyes, True, "")
        if (
            before_metrics["background"] is not None
            and before_metrics["subject"] is not None
            and after_metrics["background"] is not None
            and after_metrics["subject"] is not None
            and before_metrics["background"] + 0.01 >= before_metrics["subject"]
            and after_metrics["background"] + 0.02 < after_metrics["subject"]
        ):
            return "The background is already darker than the subject."
    if picture.type == "night":
        before_spread = float(_luma(before).std())
        after_spread = float(after_luma.std())
        if after_spread < 0.012 and after_spread < before_spread * 0.55:
            return "The night turned into a flat color."
    return ""


def _maybe_subject(
    pixels: np.ndarray,
    detections: list[Detection],
    picture: Picture,
    guide: Guide,
) -> tuple[np.ndarray | None, int]:
    plan = guide.masks.get("subject")
    if plan is None or plan.build == "never" or "subject" not in guide.stages:
        return None, 0
    return _build_subject(pixels, detections, picture, plan.build)


def _build_subject(
    pixels: np.ndarray,
    detections: list[Detection],
    picture: Picture,
    build: str,
) -> tuple[np.ndarray | None, int]:
    boxes = _subject_boxes(pixels, detections, picture, build)
    if not boxes:
        return None, 0
    height, width = pixels.shape[:2]
    subject = np.zeros((height, width), dtype=np.float32)
    for box in boxes:
        subject = np.maximum(subject, _ellipse(height, width, box, expand=1.35, inner=0.62, feather=0.7))
    if build == "animal":
        subject = extend_animal(pixels, subject)
    return subject, len(boxes)


def _sharp_union(
    pixels: np.ndarray,
    subject: np.ndarray | None,
    guide: Guide,
) -> tuple[np.ndarray | None, set[str]]:
    """Union of the subject and the supports the guide says to keep sharp.

    The subject grade still uses the subject alone. This wider mask is what
    the background is not allowed to blur.
    """

    if subject is None:
        return None, set()
    sharp = subject
    kept: set[str] = set()
    if "perch" in guide.keep:
        perch = perch_mask(pixels, subject)
        if perch is not None:
            sharp = np.maximum(sharp, perch)
            kept.add("perch")
    if "seat" in guide.keep:
        seat = seat_mask(pixels, subject)
        if seat is not None:
            sharp = np.maximum(sharp, seat)
            kept.add("seat")
    if "held" in guide.keep:
        held = held_mask(pixels, subject)
        if held is not None:
            sharp = np.maximum(sharp, held)
            kept.add("held")
    return sharp, kept


def _kept_features(
    guide: Guide,
    picture: Picture,
    ran: list[str],
    saw_subject: bool,
    extras: set[str],
) -> list[str]:
    """Feature ids that stayed sharp, in the order the guide listed them."""

    found: list[str] = []
    for name in guide.keep:
        if name in extras:
            found.append(name)
            continue
        if name == "building" and picture.type == "architecture":
            found.append(name)
        elif name == "land" and picture.type == "landscape":
            found.append(name)
        elif name == "sky" and "sky" in ran:
            found.append(name)
        elif saw_subject and name == "animal" and picture.type == "animal":
            found.append(name)
        elif saw_subject and name == "person" and picture.type in {"portrait", "group", "street"}:
            found.append(name)
        elif saw_subject and name == "dish" and picture.type == "food":
            found.append(name)
        elif saw_subject and name == "object" and picture.type in {"still-life", "macro"}:
            found.append(name)
    return found


def _subject_boxes(
    pixels: np.ndarray,
    detections: list[Detection],
    picture: Picture,
    build: str,
) -> list[tuple[float, float, float, float]]:
    if build == "face":
        boxes = _boxes(detections, {"face", "person"})
        if boxes:
            return boxes
        if picture.subject_box is not None and picture.type in {"portrait", "group", "street"}:
            return [picture.subject_box]
        skin = skin_subject_box(pixels)
        if skin is not None and picture.type in {"portrait", "group", "street"}:
            return [skin.box]
        return []
    if build == "animal":
        boxes = _boxes(detections, {"animal"})
        if boxes:
            return boxes
        return _compact_boxes(pixels, picture, allow_warm=True, allow_plain=True)
    if build == "subject":
        boxes = _boxes(detections, {"face", "person", "animal"})
        if boxes:
            return boxes
        return _compact_boxes(pixels, picture, allow_warm=True, allow_plain=True)
    return []


def _compact_boxes(
    pixels: np.ndarray,
    picture: Picture,
    *,
    allow_warm: bool,
    allow_plain: bool,
) -> list[tuple[float, float, float, float]]:
    blob: Blob | None = warm_subject_box(pixels) if allow_warm else None
    if blob is not None and is_elongated(blob.box, pixels.shape):
        blob = None
    if blob is None and allow_plain:
        creature = creature_box(pixels)
        if creature is not None:
            return [creature[0]]
        blob = plain_subject_box(pixels)
        if blob is not None and is_elongated(blob.box, pixels.shape):
            blob = None
    if blob is None and picture.subject_box is not None and picture.type in {"animal", "food", "macro", "still-life"}:
        if not is_elongated(picture.subject_box, pixels.shape):
            return [picture.subject_box]
    if blob is None:
        return []
    return [blob.box]


def _boxes(detections: list[Detection], categories: set[str]) -> list[tuple[float, float, float, float]]:
    return [item.bbox for item in detections if item.category in categories and _usable_box(item.bbox)]


def _usable_box(bbox: tuple[float, float, float, float]) -> bool:
    return bbox[2] > 0.01 and bbox[3] > 0.01


def _build_eyes(
    pixels: np.ndarray,
    detections: list[Detection],
    subject: np.ndarray,
) -> tuple[np.ndarray | None, int]:
    height, width = pixels.shape[:2]
    eye_boxes = [
        box
        for box in _boxes(detections, {"eye"})
        if _center_on_subject(box, subject)
    ]
    eyes = np.zeros((height, width), dtype=np.float32)
    if eye_boxes:
        for box in eye_boxes[:4]:
            eyes = np.maximum(eyes, _ellipse(height, width, box, expand=1.2, inner=0.45, feather=0.85))
        count = min(4, len(eye_boxes))
    else:
        synthetic, count = _synthetic_eyes(pixels, subject)
        if synthetic is None:
            return None, 0
        eyes = synthetic
    if float(eyes.max()) <= 0.0:
        return None, 0
    return eyes, count


def _skin_mask(pixels: np.ndarray, subject: np.ndarray) -> np.ndarray | None:
    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    spread = np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)
    skin = (
        (red > 0.38)
        & (green > 0.18)
        & (blue > 0.28)
        & (red > green + 0.04)
        & (red > blue)
        & (blue > red * 0.45)
        & (spread > 0.08)
        & (subject > np.float32(0.45))
    )
    if int(skin.sum()) < 30:
        return None
    mask = _blur(skin.astype(np.float32), 1.6)
    if float(mask.max()) <= 0.05:
        return None
    return mask


def _sky_mask(pixels: np.ndarray) -> np.ndarray | None:
    height, width = pixels.shape[:2]
    if height < 16:
        return None
    luma = _luma(pixels)
    split = max(1, height // 3)
    upper = pixels[:split]
    upper_blue = float((upper[..., 2] - upper[..., 0]).mean())
    upper_luma = float(luma[:split].mean())
    lower_luma = float(luma[-split:].mean())
    if upper_blue < 0.03 and upper_luma < lower_luma + 0.04:
        return None
    yy = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    blue = np.clip((pixels[..., 2] - pixels[..., 0]) * np.float32(2.2), 0.0, 1.0)
    mask = yy * np.clip(blue + np.float32(0.25), 0.0, 1.0)
    mask = np.where(yy > np.float32(0.42), mask, np.float32(0.0))
    mask = np.broadcast_to(mask, (height, width)).astype(np.float32, copy=True)
    if float(mask.mean()) < 0.02:
        return None
    return mask


def _horizon_mask(pixels: np.ndarray) -> np.ndarray | None:
    """A thin warm line where a sunset meets the land, not the whole sky."""

    height, width = pixels.shape[:2]
    if height < 32 or width < 32:
        return None
    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    warm = (red > green + np.float32(0.03)) & (red > blue + np.float32(0.05)) & (red > np.float32(0.25))
    row = warm.mean(axis=1)
    y0 = int(height * 0.12)
    y1 = int(height * 0.78)
    band = row[y0:y1]
    if band.size < 8 or float(band.max()) < 0.45:
        return None
    peak = int(np.argmax(band)) + y0
    above = row[max(0, peak - max(4, height // 10)) : peak]
    if above.size and float(row[peak]) < float(above.mean()) + 0.2:
        return None
    yy = np.arange(height, dtype=np.float32)
    sigma = max(2.0, height * 0.028)
    stripe = np.exp(np.float32(-0.5) * ((yy - np.float32(peak)) / np.float32(sigma)) ** 2)
    mask = stripe[:, None] * warm.astype(np.float32)
    mask = _blur(mask, 1.4)
    if float(mask.mean()) < 0.008 or float(mask.max()) < 0.35:
        return None
    return mask


def _with_nudge(plan: MaskPlan, nudge: object) -> MaskPlan:
    """Add one joined recipe onto a mask, then clamp it again."""

    clarity = plan.clarity
    extra = getattr(nudge, "clarity", None)
    if extra is not None:
        clarity = plan.clarity + float(extra)
    return clamp_grades(
        replace(
            plan,
            exposure=plan.exposure + float(getattr(nudge, "exposure", 0.0)),
            contrast=plan.contrast + float(getattr(nudge, "contrast", 0.0)),
            saturation=plan.saturation + float(getattr(nudge, "saturation", 0.0)),
            clarity=clarity,
        )
    )


def _grade_horizon(
    pixels: np.ndarray,
    mask: np.ndarray,
    plan: MaskPlan,
    warmth: float = 0.0,
) -> np.ndarray:
    """Warm the sunset line. Green is held back so the band cannot turn lime."""

    adjusted = _tone(pixels, plan)
    adjusted = adjusted.copy()
    boost = np.float32(1.08 + max(0.0, min(0.05, float(warmth))))
    adjusted[..., 0] = np.clip(adjusted[..., 0] * boost + np.float32(0.015), 0.0, 1.0)
    adjusted[..., 1] = np.clip(adjusted[..., 1] * np.float32(0.96), 0.0, 1.0)
    adjusted[..., 2] = np.clip(adjusted[..., 2] * np.float32(0.9), 0.0, 1.0)
    weight = mask[..., None]
    return np.clip(adjusted * weight + pixels * (1.0 - weight), 0.0, 1.0)


def _low_key_face_exposure(pixels: np.ndarray, subject: np.ndarray, base: float, *, opened: bool = False) -> float:
    luma = _luma(pixels)
    chosen = subject > np.float32(0.45)
    if float(chosen.mean()) < 0.005:
        return base
    median = float(np.median(luma[chosen]))
    if median >= 0.4:
        return base
    extra = float(np.log2((0.38 + 1e-4) / (median + 1e-4)))
    extra = max(0.0, min(0.28, extra * 0.45))
    cap = 1.0 if opened else 0.35
    floor = -1.6 if opened else -0.35
    return max(floor, min(cap, base + extra))


def _foreground_mask(
    pixels: np.ndarray,
    plan: MaskPlan,
    picture: Picture,
    detections: list[Detection],
) -> np.ndarray | None:
    height, width = pixels.shape[:2]
    if plan.build == "structure":
        if picture.type == "architecture" or any(item.category == "architecture" for item in detections):
            return np.ones((height, width), dtype=np.float32)
        return None
    if plan.build != "foreground":
        return None
    if picture.type != "landscape" and picture.sky < 0.5:
        return None
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    mask = np.clip((yy - np.float32(0.42)) / np.float32(0.22), 0.0, 1.0)
    return np.broadcast_to(mask, (height, width)).astype(np.float32, copy=True)


def _grade_highlights(pixels: np.ndarray, plan: MaskPlan) -> tuple[np.ndarray, bool]:
    luma = _luma(pixels)
    weight = np.clip((luma - np.float32(0.70)) / np.float32(0.18), 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)
    if float(weight.mean()) < 0.001:
        return pixels, False
    adjusted = _tone(pixels, plan)
    mixed = pixels * (1.0 - weight[..., None]) + adjusted * weight[..., None]
    return np.clip(mixed, 0.0, 1.0), True


def _grade_reflection(
    pixels: np.ndarray,
    mask: np.ndarray,
    plan: MaskPlan,
    intention: str = "lift",
) -> np.ndarray:
    """Sharpen the mirror and lift it a little, without painting a new color.

    Contrast pivots on the echo itself, so a dark streak is not pulled toward
    gray and a bright ridge is not shoved into white. Clarity does most of
    the separating. Exposure stays modest and backs off when the echo is
    already bright.
    """

    exposure = float(plan.exposure)
    clarity = float(plan.clarity)
    contrast = float(plan.contrast)
    if intention == "hold":
        exposure = 0.0
        contrast = min(contrast, 0.04)
    elif intention == "quiet":
        exposure = min(exposure, -0.06)
        clarity = min(clarity, 0.12)
        contrast = min(contrast, 0.04)
    chosen = mask > np.float32(0.5)
    luma0 = _luma(pixels)
    if int(chosen.sum()) > 30:
        origin = float(np.median(luma0[chosen]))
        if origin > 0.62:
            exposure = min(exposure, 0.04)
        elif origin > 0.48:
            exposure = min(exposure, 0.08)
    out = pixels * np.float32(2.0 ** exposure)
    luma = _luma(out)
    pivot = float(np.median(luma[chosen])) if int(chosen.sum()) > 30 else 0.5
    pivot = min(0.8, max(0.05, pivot))
    new_luma = (luma - np.float32(pivot)) * np.float32(1.0 + contrast) + np.float32(pivot)
    head = np.clip((luma - np.float32(0.78)) / np.float32(0.16), 0.0, 1.0)
    new_luma = new_luma * (1.0 - head) + luma * head
    new_luma = np.clip(new_luma, 0.0, 1.0)
    out = out * (new_luma / np.maximum(luma, np.float32(1e-4)))[..., None]
    if abs(clarity) >= 0.01:
        soft = _blur_rgb(np.clip(out, 0.0, 1.0), 1.3)
        detail = (out - soft) * np.float32(clarity * 2.0)
        room = np.clip((np.float32(0.98) - np.max(out, axis=2)) / np.float32(0.18), 0.0, 1.0)
        out = out + detail * room[..., None]
    out = np.clip(out, 0.0, 1.0)
    weight = mask[..., None]
    return np.clip(out * weight + pixels * (1.0 - weight), 0.0, 1.0)


def _grade_region(pixels: np.ndarray, mask: np.ndarray, plan: MaskPlan) -> np.ndarray:
    adjusted = _tone(pixels, plan)
    weight = mask[..., None]
    return np.clip(adjusted * weight + pixels * (1.0 - weight), 0.0, 1.0)


def _grade_background(
    current: np.ndarray,
    source: np.ndarray,
    subject: np.ndarray,
    plan: MaskPlan,
    theme: np.ndarray | None = None,
    hold: np.ndarray | None = None,
    darken_all: bool = False,
) -> np.ndarray:
    """Defocus the surround, and darken only the part that competes with the subject.

    A quiet surround keeps its exposure. A theme window or color island
    is left out of both the darkening and the blur.
    """

    height, width = source.shape[:2]
    if plan.defocus > 0.05:
        radius = max(8.0, plan.defocus * 0.045 * min(height, width))
        blurred = _blur_rgb(source, radius)
    else:
        blurred = source
    luma = _luma(source)
    core = subject > np.float32(0.45)
    subject_level = float(np.median(luma[core])) if int(core.sum()) >= 8 else float(luma.mean())
    outside = subject < np.float32(0.35)
    competing = outside & (luma > np.float32(subject_level + 0.06))
    if theme is not None:
        protected = theme > np.float32(0.4)
        competing = competing & ~protected
        blurred = np.where(protected[..., None], source, blurred)
    if hold is not None:
        protected = hold > np.float32(0.4)
        competing = competing & ~protected
        blurred = np.where(protected[..., None], source, blurred)
    # Both parts of the surround fall soft. Only the competing part is darkened,
    # unless the user asked the end result to go darker. Then the water comes down
    # and the subject mask stays out of it.
    quiet_plan = replace(plan, exposure=0.0 if plan.exposure < 0.0 else plan.exposure)
    quiet = _tone(blurred, quiet_plan, mute=plan.mute)
    bright = _tone(blurred, plan, mute=plan.mute)
    if darken_all and plan.exposure < 0.0:
        base = bright
    else:
        base = np.where(competing[..., None], bright, quiet)
    weight = subject[..., None]
    return np.clip(current * weight + base * (1.0 - weight), 0.0, 1.0)


def _theme_field(
    pixels: np.ndarray,
    subject: np.ndarray | None,
) -> tuple[np.ndarray | None, str]:
    horizon = _horizon_mask(pixels)
    if horizon is not None:
        return horizon, "horizon"
    window = _window_field(pixels, subject)
    if window is not None:
        return window, "window"
    island = _island_field(pixels, subject)
    if island is not None:
        return island, "island"
    return None, ""


def _forced_theme(
    pixels: np.ndarray,
    subject: np.ndarray | None,
    kind: str,
) -> tuple[np.ndarray | None, str] | None:
    if kind == "horizon":
        horizon = _horizon_mask(pixels)
        if horizon is not None:
            return horizon, "horizon"
    if kind == "window":
        window = _window_field(pixels, subject)
        if window is not None:
            return window, "window"
    if kind == "island":
        island = _island_field(pixels, subject)
        if island is not None:
            return island, "island"
    if kind == "reflection":
        echo = mirror_box(pixels)
        if echo is None:
            return None
        height, width = pixels.shape[:2]
        return _soft_rect(height, width, echo), "reflection"
    return None


def _window_field(pixels: np.ndarray, subject: np.ndarray | None) -> np.ndarray | None:
    luma = _luma(pixels)
    chroma = np.max(np.abs(pixels - luma[..., None]), axis=2)
    # A window is a small near-white opening. A warm flower is bright too, and is not one.
    bright = (luma > np.float32(0.78)) & (chroma < np.float32(0.08))
    if subject is not None:
        bright = bright & (subject < np.float32(0.25))
    if float(bright.mean()) > 0.12 or int(bright.sum()) < 12:
        return None
    height, width = luma.shape
    ys, xs = np.nonzero(bright)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    area = ((x1 - x0 + 1) * (y1 - y0 + 1)) / float(height * width)
    if area < 0.004 or area > 0.12:
        return None
    box = (x0 / width, y0 / height, (x1 - x0 + 1) / width, (y1 - y0 + 1) / height)
    mask = _ellipse(height, width, box, expand=1.15, inner=0.55, feather=0.8)
    mask = np.where(bright, np.maximum(mask, np.float32(0.65)), mask * np.float32(0.35))
    if float(mask.max()) < 0.2:
        return None
    return mask.astype(np.float32)


def _island_field(pixels: np.ndarray, subject: np.ndarray | None) -> np.ndarray | None:
    luma = _luma(pixels)
    chroma = np.max(np.abs(pixels - luma[..., None]), axis=2)
    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    skin = (red > 0.38) & (green > 0.18) & (blue > 0.28) & (red > green) & (red > blue)
    leafy = (green > red + np.float32(0.04)) & (green > blue + np.float32(0.02))
    core = (chroma > np.float32(0.18)) & ~skin & ~leafy
    if subject is not None:
        core = core & (subject < np.float32(0.25))
    count = int(core.sum())
    if count < 20 or count > int(0.08 * core.size):
        return None
    height, width = luma.shape
    ys, xs = np.nonzero(core)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    if count / box_area < 0.35:
        return None
    box = (x0 / width, y0 / height, (x1 - x0 + 1) / width, (y1 - y0 + 1) / height)
    return _ellipse(height, width, box, expand=1.2, inner=0.5, feather=0.75)


def _tone(pixels: np.ndarray, plan: MaskPlan, mute: float = 0.0) -> np.ndarray:
    out = pixels * np.float32(2.0 ** plan.exposure)
    out = contrast_about_luma(out, float(plan.contrast))
    luma = _luma(out)
    chroma = np.float32((1.0 - mute) * (1.0 + plan.saturation))
    out = luma[..., None] + (out - luma[..., None]) * chroma
    if abs(plan.clarity) >= 0.01:
        soft = _blur_rgb(np.clip(out, 0.0, 1.0), 1.3)
        out = out + (out - soft) * np.float32(plan.clarity * 2.0)
    return np.clip(out, 0.0, 1.0)


def _luma(pixels: np.ndarray) -> np.ndarray:
    return 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]


def _ellipse(
    height: int,
    width: int,
    bbox: tuple[float, float, float, float],
    *,
    expand: float,
    inner: float,
    feather: float,
) -> np.ndarray:
    x, y, bw, bh = bbox
    cx = (x + bw * 0.5) * width
    cy = (y + bh * 0.5) * height
    rx = max(2.0, bw * width * 0.5 * expand)
    ry = max(2.0, bh * height * 0.5 * expand)
    yy, xx = np.ogrid[0:height, 0:width]
    distance = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    edge = np.clip((distance - inner) / max(feather, 1e-3), 0.0, 1.0)
    edge = edge * edge * (3.0 - 2.0 * edge)
    return (1.0 - edge).astype(np.float32)


def _center_on_subject(
    box: tuple[float, float, float, float],
    subject: np.ndarray,
) -> bool:
    height, width = subject.shape
    cx = min(width - 1, max(0, int((box[0] + box[2] * 0.5) * width)))
    cy = min(height - 1, max(0, int((box[1] + box[3] * 0.5) * height)))
    return float(subject[cy, cx]) > 0.2


def _reflection_mask(pixels: np.ndarray, subject: np.ndarray | None) -> np.ndarray | None:
    """The mirror in the water: a bird's streak, or a wider echo below a shore.

    A compact animal keeps the mirror directly under its body. A house or a
    mountain has no subject mask; only the echo is graded, and the shape
    above is left alone.
    """

    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    compact = reflected_subject(pixels)
    if compact is not None and subject is not None and float(np.max(subject)) > 0.0:
        body, reflection = compact
        if _reflection_under_subject(reflection, subject):
            shaped = _mirror_below_body(pixels, body, reflection)
            if shaped is not None:
                return shaped
            return _ellipse(height, width, reflection, expand=1.2, inner=0.5, feather=0.85)
    found = water_reflection(pixels)
    if found is None:
        return None
    _source, reflection = found
    if subject is not None and float(np.max(subject)) > 0.0:
        if not _reflection_under_subject(reflection, subject):
            return None
    return _soft_rect(height, width, reflection)


def _mirror_below_body(
    pixels: np.ndarray,
    body: tuple[float, float, float, float],
    reflection: tuple[float, float, float, float],
) -> np.ndarray | None:
    """The bright mirror in the water directly under the body.

    The check for a bird is a narrow streak. The mirror itself is the
    connected shape under the body that stands out from the water beside
    it. Pads and the open water stay out because they are separate pieces.
    """

    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    x, _y, bw, bh = body
    x0 = min(width - 1, max(0, int(x * width)))
    x1 = min(width, max(x0 + 2, int((x + bw) * width)))
    y0 = min(height - 1, max(0, int((_y + bh) * height)))
    if x1 - x0 < 4 or height - y0 < 6:
        return None
    y1 = min(height, y0 + max(8, int(0.30 * height)))
    luma = _luma(pixels)
    gap = max(4, int(0.03 * width))
    left = luma[y0:y1, max(0, x0 - gap * 5) : max(0, x0 - gap)]
    right = luma[y0:y1, min(width, x1 + gap) : min(width, x1 + gap * 5)]
    parts = [part for part in (left, right) if part.size >= 8]
    if not parts:
        return None
    level = float(np.median(np.concatenate([part.ravel() for part in parts])))
    window = luma[y0:y1, x0:x1]
    chosen = window > np.float32(level + 0.08)
    if int(chosen.sum()) < 20:
        return None
    pieces = [piece for piece in _components(chosen) if int(piece.sum()) >= 20 and bool(piece[:4].any())]
    if not pieces:
        return None
    main = max(pieces, key=lambda piece: int(piece.sum()))
    ys, xs = np.nonzero(main)
    if int(ys.max()) - int(ys.min()) < int(0.04 * height):
        return None
    if (int(xs.max()) - int(xs.min()) + 1) / float(width) > 0.45:
        return None
    if float(main.mean()) > 0.55:
        return None
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = main
    rx = min(width - 1, max(0, int((reflection[0] + reflection[2] * 0.5) * width)))
    ry = min(height - 1, max(0, int((reflection[1] + reflection[3] * 0.35) * height)))
    if not bool(mask[max(0, ry - 2) : ry + 3, max(0, rx - 2) : rx + 3].any()):
        return None
    return _feather(mask, height, width)


def _reflection_under_subject(
    reflection: tuple[float, float, float, float],
    subject: np.ndarray,
) -> bool:
    """True when the mirror sits below the subject and shares its columns."""

    height, width = subject.shape
    ys, xs = np.nonzero(subject > np.float32(0.45))
    if ys.size == 0:
        return False
    subject_left = float(xs.min()) / width
    subject_right = float(xs.max() + 1) / width
    subject_top = float(ys.min()) / height
    center_x = reflection[0] + reflection[2] * 0.5
    if center_x < subject_left - 0.06 or center_x > subject_right + 0.06:
        return False
    return reflection[1] >= subject_top


def _soft_rect(
    height: int,
    width: int,
    bbox: tuple[float, float, float, float],
) -> np.ndarray | None:
    """A soft box that stays on the mirror and off the water beside it.

    An ellipse around a wide ridge swallows the water on either side. The
    feather here is only a few pixels, so the sides stay the water they are.
    """

    x, y, bw, bh = bbox
    pad_x = 0.008
    pad_y = 0.01
    x0 = max(0, int((x - pad_x) * width))
    y0 = max(0, int((y - pad_y) * height))
    x1 = min(width, int(round((x + bw + pad_x) * width)))
    y1 = min(height, int(round((y + bh + pad_y) * height)))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    radius = max(1.0, 0.012 * min(height, width))
    return _blur(mask, radius)


def _synthetic_eyes(pixels: np.ndarray, subject: np.ndarray) -> tuple[np.ndarray | None, int]:
    """A small dark spot inside the subject, used when no eye detector fired."""

    height, width = subject.shape
    luma = _luma(pixels)
    fine = _blur(luma, 1.2)
    coarse = _blur(luma, max(4.0, 0.03 * min(height, width)))
    blob = coarse - fine
    core = (blob > np.float32(0.05)) & (subject > np.float32(0.45)) & (luma < np.float32(0.62))
    rows = np.nonzero(subject > np.float32(0.45))[0]
    if rows.size:
        top = int(rows.min())
        bottom = int(rows.max())
        # Eyes sit in the upper part of a face. A dark collar is not an eye.
        cut = top + int((bottom - top) * 0.52)
        core[cut:, :] = False
    if int(core.sum()) < 6:
        return None, 0
    strength = np.where(core, blob, np.float32(0.0))
    radius = max(3, int(round(0.012 * min(height, width))))
    mask = np.zeros((height, width), dtype=np.float32)
    found = 0
    for _ in range(2):
        peak = int(np.argmax(strength))
        py, px = divmod(peak, width)
        if float(strength[py, px]) < 0.05:
            break
        y0, y1 = max(0, py - radius), min(height, py + radius + 1)
        x0, x1 = max(0, px - radius), min(width, px + radius + 1)
        box = (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)
        mask = np.maximum(mask, _ellipse(height, width, box, expand=1.05, inner=0.4, feather=0.9))
        strength[y0:y1, x0:x1] = 0
        found += 1
    if found == 0:
        return None, 0
    return mask, found


def _blur_rgb(pixels: np.ndarray, radius: float) -> np.ndarray:
    height, width = pixels.shape[:2]
    scale = 2 if min(height, width) > 500 else 1
    small_size = (max(1, width // scale), max(1, height // scale))
    image = Image.fromarray(np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    image = image.resize(small_size, Image.Resampling.BOX)
    image = image.filter(ImageFilter.GaussianBlur(radius=max(0.6, radius / scale)))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image).astype(np.float32) / np.float32(255.0)


# Closed caps stay inside about a third of a stop. Open caps are the
# strong range, and only when caps_open (or the pipeline's strong flag) is set.
_CLOSED_EV = 0.35
_OPEN_EV = 1.6
_OPEN_GRADIENT_EV = -0.70


def _stage_intents(intents: list | None) -> tuple[list[dict], set[str]]:
    """Turn stage/action records into asks and skips.

    Each item is ``{"stage": ..., "action": ...}``. Actions are darken,
    lift, hold, and skip. Unknown stages are ignored. Sentences are not read.
    """

    targets = {
        "subject": "subject",
        "rock": "rock",
        "foreground": "foreground",
        "behind": "behind",
        "sky": "sky",
        "reflection": "reflection",
        "surround": "foreground",
        "background": "foreground",
    }
    skips = {
        "surround": "background",
        "background": "background",
        "foreground": "background",
        "eye": "eyes",
        "eyes": "eyes",
    }
    asks: list[dict] = []
    omit: set[str] = set()
    for item in intents or []:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or item.get("target") or "").strip().lower()
        action = str(item.get("action") or item.get("verb") or "").strip().lower()
        if not stage or action not in {"darken", "lift", "hold", "skip"}:
            continue
        if stage == "gradient":
            if action == "skip":
                omit.add("gradient")
            elif action in {"darken", "lift"}:
                asks.append({"verb": "gradient", "target": "", "source": "intent"})
            continue
        if action == "skip":
            omit.add(skips.get(stage, stage))
            continue
        target = targets.get(stage)
        if target is None:
            continue
        asks.append({"verb": action, "target": target, "source": "intent"})
    return asks, omit


def _frame_is_the_request(asks: list) -> bool:
    """True when the user asked for the bird, darker water, and a clearer reflection."""

    return (
        (_asked(asks, "lift", {"subject"}) or _asked(asks, "hold", {"subject"}))
        and _asked(asks, "darken", {"foreground", "rock"})
        and _asked(asks, "lift", {"reflection"})
    )


def _asked(asks: list, verb: str, targets: set[str]) -> bool:
    for item in asks:
        if str(item.get("verb") or "") != verb:
            continue
        if not targets or str(item.get("target") or "") in targets:
            return True
    return False


def _subject_ask(plan: MaskPlan, asks: list, opened: bool) -> MaskPlan:
    """Replace the subject exposure when the user named that target."""

    if _asked(asks, "darken", {"subject"}):
        exposure = -0.90 if opened else -0.22
        return clamp_grades(replace(plan, exposure=exposure), opened=opened)
    if _asked(asks, "lift", {"subject"}):
        exposure = max(float(plan.exposure), 0.55 if opened else 0.15)
        return clamp_grades(replace(plan, exposure=exposure), opened=opened)
    if _asked(asks, "hold", {"subject"}):
        return clamp_grades(replace(plan, exposure=0.0), opened=opened)
    return plan


def _behind_mask(subject: np.ndarray) -> np.ndarray | None:
    """The frame above the body. Light behind the subject lives here."""

    height, width = subject.shape
    rows = np.nonzero(subject > np.float32(0.45))[0]
    if rows.size == 0:
        return None
    top = int(rows.min())
    if top < 4:
        return None
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:top, :] = 1.0
    return mask


def _linear_field(shape: tuple[int, int], axis: str, bright: str) -> np.ndarray:
    height, width = shape
    if axis == "horizontal":
        ramp = np.linspace(1.0, 0.0, width, dtype=np.float32)
        if bright == "right":
            ramp = ramp[::-1].copy()
        field = np.broadcast_to(ramp, (height, width)).copy()
    else:
        ramp = np.linspace(1.0, 0.0, height, dtype=np.float32)
        if bright == "bottom":
            ramp = ramp[::-1].copy()
        field = np.broadcast_to(ramp[:, None], (height, width)).copy()
    return _blur(field, max(6.0, 0.06 * min(height, width)))


def _end_mean(luma: np.ndarray, axis: str, bright: str, *, which: str) -> float:
    height, width = luma.shape
    if axis == "horizontal":
        band = max(2, width // 5)
        left = float(luma[:, :band].mean())
        right = float(luma[:, -band:].mean())
        bright_side = left if bright == "left" else right
        dark_side = right if bright == "left" else left
    else:
        band = max(2, height // 5)
        top = float(luma[:band, :].mean())
        bottom = float(luma[-band:, :].mean())
        bright_side = top if bright == "top" else bottom
        dark_side = bottom if bright == "top" else top
    return bright_side if which == "bright" else dark_side


def _apply_gradient(
    current: np.ndarray,
    pixels: np.ndarray,
    subject: np.ndarray | None,
    picture: Picture,
    asks: list,
    intentions: dict | None,
    omitted: set[str],
    opened: bool,
    ran: list[str],
    skipped: list[str],
) -> tuple[np.ndarray, dict, list[str], list[str]]:
    """One soft linear exposure field. A weak one is skipped."""

    if "gradient" in omitted:
        skipped.append("gradient")
        return current, {"skipped": True, "reason": "The gradient was reverted."}, ran, skipped
    regions = list(intentions.get("regions") or []) if isinstance(intentions, dict) else []
    user_gradient = any(
        isinstance(item, dict) and item.get("verb") == "gradient" and item.get("source") == "user"
        for item in asks
    )
    # A request about the bird, the dark water, and the reflection does not
    # grow a sky gradient just because the frame is uneven.
    if _frame_is_the_request(asks) and not user_gradient:
        return current, {}, ran, skipped
    asked = user_gradient or _asked(asks, "gradient", set())
    measured = any(isinstance(item, dict) and item.get("role") == "gradient" for item in regions)
    if not asked and not measured:
        return current, {}, ran, skipped
    close = picture.type in {"portrait", "group"} and (
        picture.compact_area >= 0.08
        or (
            picture.subject_box is not None
            and picture.subject_box[2] * picture.subject_box[3] >= 0.08
        )
    )
    if picture.type == "night":
        skipped.append("gradient")
        return (
            current,
            {"skipped": True, "reason": "A night frame that should stay dark does not get a gradient."},
            ran,
            skipped,
        )
    if close:
        skipped.append("gradient")
        kind = "portrait" if picture.type == "portrait" else "group"
        return (
            current,
            {"skipped": True, "reason": f"A close {kind} does not get a gradient."},
            ran,
            skipped,
        )
    from ai_dt.reading import linear_falloff

    falloff = linear_falloff(pixels)
    if falloff is None:
        skipped.append("gradient")
        return (
            current,
            {"skipped": True, "reason": "There is no directional falloff worth a gradient."},
            ran,
            skipped,
        )
    if abs(float(falloff["strength"])) < 0.08:
        skipped.append("gradient")
        return (
            current,
            {"skipped": True, "reason": "The gradient would move exposure by a negligible amount."},
            ran,
            skipped,
        )
    # A real falloff stays modest until the caps are open and the gradient was asked for.
    strength = float(falloff["strength"])
    if asked and opened:
        strength = min(strength, _OPEN_GRADIENT_EV)
    ceiling = _OPEN_EV if opened else _CLOSED_EV
    strength = max(-ceiling, min(ceiling, strength))
    luma = _luma(current)
    subject_level = _masked_mean(luma, subject) if subject is not None else None
    dark_side = _end_mean(luma, falloff["axis"], falloff["bright"], which="dark")
    if subject_level is not None and dark_side <= subject_level + 0.01:
        skipped.append("gradient")
        return (
            current,
            {
                "axis": falloff["axis"],
                "strength": 0.0,
                "reason": "The dark end is already as dark as the subject.",
            },
            ran,
            skipped,
        )
    field = _linear_field(current.shape[:2], falloff["axis"], falloff["bright"])
    if subject is not None:
        # The subject, including its soft edge, stays out of the darkening.
        field = field * (subject < np.float32(0.2)).astype(np.float32)
        perch = perch_mask(pixels, subject)
        if perch is not None:
            field = np.clip(field * (1.0 - perch), 0.0, 1.0)
        if _asked(asks, "lift", {"behind"}):
            hold = _behind_mask(subject)
            if hold is not None:
                field = np.clip(field * (hold < np.float32(0.4)).astype(np.float32), 0.0, 1.0)
    plan = clamp_grades(
        MaskPlan(
            build="subject",
            skip="",
            exposure=strength,
            contrast=0.0,
            saturation=0.0,
            clarity=0.0,
            defocus=0.0,
            mute=0.0,
        ),
        opened=opened,
    )
    if abs(plan.exposure) < 0.08:
        skipped.append("gradient")
        return (
            current,
            {"skipped": True, "reason": "The gradient would move exposure by a negligible amount."},
            ran,
            skipped,
        )
    proposed = _grade_region(current, field, plan)
    if subject is not None:
        before_subject = _masked_mean(luma, subject)
        after_subject = _masked_mean(_luma(proposed), subject)
        if (
            before_subject is not None
            and after_subject is not None
            and after_subject + 0.03 < before_subject
        ):
            skipped.append("gradient")
            return (
                current,
                {"axis": falloff["axis"], "strength": 0.0, "reason": "The gradient would crush the subject."},
                ran,
                skipped,
            )
    ran.append("gradient")
    return (
        proposed,
        {
            "axis": falloff["axis"],
            "bright": falloff["bright"],
            "strength": round(float(plan.exposure), 4),
            "reason": falloff["reason"],
        },
        ran,
        skipped,
    )


def _apply_rock(
    current: np.ndarray,
    pixels: np.ndarray,
    subject: np.ndarray | None,
    asks: list,
    omitted: set[str],
    opened: bool,
    ran: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Darken the perch when the user asked to darken the foreground or the rock."""

    if subject is None or "background" in omitted:
        return current, ran
    if not _asked(asks, "darken", {"foreground", "rock"}):
        return current, ran
    perch = perch_mask(pixels, subject)
    if perch is None:
        return current, ran
    rock = np.clip(perch * (subject < np.float32(0.35)).astype(np.float32), 0.0, 1.0)
    if float(rock.max()) < 0.2:
        return current, ran
    exposure = -_OPEN_EV if opened else -0.22
    plan = clamp_grades(
        MaskPlan(
            build="subject",
            skip="",
            exposure=exposure,
            contrast=0.0,
            saturation=0.0,
            clarity=0.0,
            defocus=0.0,
            mute=0.0,
        ),
        opened=opened,
    )
    return _grade_region(current, rock, plan), ran


def _apply_behind(
    current: np.ndarray,
    subject: np.ndarray | None,
    asks: list,
    opened: bool,
) -> np.ndarray:
    """Lift the region behind the subject when that intent is present.

    Closed caps stay inside about ±0.35 EV. Open caps may push the light
    behind the body clearly above the subject.
    """

    if subject is None or not _asked(asks, "lift", {"behind"}):
        return current
    hold = _behind_mask(subject)
    if hold is None:
        return current
    exposure = 0.55 if opened else 0.12
    mask = np.clip(hold * (subject < np.float32(0.2)).astype(np.float32), 0.0, 1.0)
    if float(mask.max()) < 0.2:
        return current
    plan = clamp_grades(
        MaskPlan(
            build="subject",
            skip="",
            exposure=exposure,
            contrast=0.0,
            saturation=0.0,
            clarity=0.0,
            defocus=0.0,
            mute=0.0,
        ),
        opened=opened,
    )
    return _grade_region(current, mask, plan)


def _blur(values: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    image = image.filter(ImageFilter.GaussianBlur(radius=max(0.4, radius)))
    return np.asarray(image).astype(np.float32) / np.float32(255.0)
