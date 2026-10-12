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
    creature_box,
    extend_animal,
    held_mask,
    is_elongated,
    perch_mask,
    reflected_subject,
    seat_mask,
    water_reflection,
)
from ai_dt.schema import Atmosphere, Detection
from ai_dt.simulate import contrast_about_luma, grade_array


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
) -> tuple[np.ndarray, dict]:
    """Return float RGB and a description of the type, guide, and masks used."""

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
    graded = grade_array(pixels, atmosphere, None if isolate else vignette)
    current = graded
    ran = ["grade"]
    skipped: list[str] = []
    eyes = 0
    subject_ev = 0.0
    background = "unchanged"
    saw_subject = False

    for stage in guide.stages:
        if stage == "grade":
            continue
        plan = guide.masks.get(stage)
        if plan is None or plan.build == "never":
            skipped.append(stage)
            continue
        plan = clamp_grades(plan)
        nudge = None if not nudges else nudges.get(stage)
        if nudge is not None:
            plan = _with_nudge(plan, nudge)
        if stage == "subject":
            if subject is None:
                skipped.append(stage)
                continue
            if picture.brightness < 0.32 and picture.type in {"portrait", "group", "street"}:
                plan = replace(plan, exposure=_low_key_face_exposure(pixels, subject, plan.exposure))
            current = _grade_region(current, subject, plan)
            subject_ev = plan.exposure
            saw_subject = True
            ran.append(stage)
            continue
        if stage == "eyes":
            if subject is None:
                skipped.append(stage)
                continue
            eye_mask, eyes = _build_eyes(pixels, detections, subject)
            if eye_mask is None:
                eyes = 0
                skipped.append(stage)
                continue
            current = _grade_region(current, eye_mask, plan)
            ran.append(stage)
            continue
        if stage == "reflection":
            if reflection is None:
                skipped.append(stage)
                continue
            current = _grade_reflection(current, reflection, plan)
            ran.append(stage)
            continue
        if stage == "skin":
            if subject is None:
                skipped.append(stage)
                continue
            skin = _skin_mask(pixels, subject)
            if skin is None:
                skipped.append(stage)
                continue
            current = _grade_region(current, skin, plan)
            ran.append(stage)
            continue
        if stage == "background":
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
            current = _grade_background(current, graded, sharp if sharp is not None else subject, plan)
            if plan.defocus > 0.05:
                background = "defocus"
            ran.append(stage)
            continue
        if stage == "sky":
            sky = _sky_mask(pixels)
            if sky is None:
                skipped.append(stage)
                continue
            current = _grade_region(current, sky, plan)
            ran.append(stage)
            continue
        if stage == "horizon":
            horizon = _horizon_mask(pixels)
            if horizon is None:
                skipped.append(stage)
                continue
            warmth = 0.0 if nudge is None else float(getattr(nudge, "warmth", 0.0))
            current = _grade_horizon(current, horizon, plan, warmth)
            ran.append(stage)
            continue
        if stage == "foreground":
            foreground = _foreground_mask(pixels, plan, picture, detections)
            if foreground is None:
                skipped.append(stage)
                continue
            current = _grade_region(current, foreground, plan)
            ran.append(stage)
            continue
        if stage == "highlights":
            protected, used = _grade_highlights(current, plan)
            if not used:
                skipped.append(stage)
                continue
            current = protected
            ran.append(stage)
            continue
        skipped.append(stage)

    reason = guide.reason
    if guide_match is not None and guide_match.fallback:
        reason = guide_match.reason
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
    }


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


def _low_key_face_exposure(pixels: np.ndarray, subject: np.ndarray, base: float) -> float:
    luma = _luma(pixels)
    chosen = subject > np.float32(0.45)
    if float(chosen.mean()) < 0.005:
        return base
    median = float(np.median(luma[chosen]))
    if median >= 0.4:
        return base
    extra = float(np.log2((0.38 + 1e-4) / (median + 1e-4)))
    extra = max(0.0, min(0.28, extra * 0.45))
    return max(-0.35, min(0.35, base + extra))


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


def _grade_reflection(pixels: np.ndarray, mask: np.ndarray, plan: MaskPlan) -> np.ndarray:
    """Sharpen the mirror and lift it a little, without painting a new color.

    Contrast pivots on the echo itself, so a dark streak is not pulled toward
    gray and a bright ridge is not shoved into white. Clarity does most of
    the separating. Exposure stays modest and backs off when the echo is
    already bright.
    """

    exposure = float(plan.exposure)
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
    contrast = float(plan.contrast)
    new_luma = (luma - np.float32(pivot)) * np.float32(1.0 + contrast) + np.float32(pivot)
    head = np.clip((luma - np.float32(0.78)) / np.float32(0.16), 0.0, 1.0)
    new_luma = new_luma * (1.0 - head) + luma * head
    new_luma = np.clip(new_luma, 0.0, 1.0)
    out = out * (new_luma / np.maximum(luma, np.float32(1e-4)))[..., None]
    if abs(plan.clarity) >= 0.01:
        soft = _blur_rgb(np.clip(out, 0.0, 1.0), 1.3)
        detail = (out - soft) * np.float32(plan.clarity * 2.0)
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
) -> np.ndarray:
    height, width = source.shape[:2]
    if plan.defocus > 0.05:
        radius = max(8.0, plan.defocus * 0.045 * min(height, width))
        base = _blur_rgb(source, radius)
    else:
        base = source
    base = _tone(base, plan, mute=plan.mute)
    weight = subject[..., None]
    return np.clip(current * weight + base * (1.0 - weight), 0.0, 1.0)


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

    A compact animal keeps the tight ellipse under its body. A house or a
    mountain has no subject mask; only the echo is graded, and the shape
    above is left alone.
    """

    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    compact = reflected_subject(pixels)
    if compact is not None and subject is not None and float(np.max(subject)) > 0.0:
        _body, reflection = compact
        if _reflection_under_subject(reflection, subject):
            return _ellipse(height, width, reflection, expand=1.2, inner=0.5, feather=0.85)
    found = water_reflection(pixels)
    if found is None:
        return None
    _source, reflection = found
    if subject is not None and float(np.max(subject)) > 0.0:
        if not _reflection_under_subject(reflection, subject):
            return None
    return _soft_rect(height, width, reflection)


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


def _blur(values: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    image = image.filter(ImageFilter.GaussianBlur(radius=max(0.4, radius)))
    return np.asarray(image).astype(np.float32) / np.float32(255.0)
