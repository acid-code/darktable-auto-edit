"""Run the picture guide: masks, local grades, and a global correction.

A Darktable style can carry global module values. It cannot carry a drawn
mask. This render is the picture that has those masks. Which masks exist,
and how hard they push, comes from the guide for the picture type. The
sharp mask is the union of the features the guide keeps: an animal and the
perch it stands on, a person and the chair or the object in their hands.
A facade is not given a bird's background blur, and a night window is not
promoted into a subject.
"""

from __future__ import annotations

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
from ai_dt.guides.support import creature_box, extend_animal, held_mask, is_elongated, perch_mask, seat_mask
from ai_dt.schema import Atmosphere, Detection
from ai_dt.simulate import grade_array


def render_finish(
    pixels: np.ndarray,
    atmosphere: Atmosphere,
    detections: list[Detection],
    vignette: dict | None = None,
    *,
    guide: Guide | None = None,
    picture: Picture | None = None,
    guide_match: GuideMatch | None = None,
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
    sharp, kept_now = _sharp_union(pixels, subject, guide)
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
        if stage == "subject":
            if subject is None:
                skipped.append(stage)
                continue
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
    eye_boxes = _boxes(detections, {"eye"})
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
    out = (out - 0.5) * np.float32(1.0 + plan.contrast) + 0.5
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


def _synthetic_eyes(pixels: np.ndarray, subject: np.ndarray) -> tuple[np.ndarray | None, int]:
    """A small dark spot inside the subject, used when no eye detector fired."""

    height, width = subject.shape
    luma = _luma(pixels)
    fine = _blur(luma, 1.2)
    coarse = _blur(luma, max(4.0, 0.03 * min(height, width)))
    blob = coarse - fine
    core = (blob > np.float32(0.05)) & (subject > np.float32(0.45)) & (luma < np.float32(0.62))
    if int(core.sum()) < 6:
        return None, 0
    strength = np.where(core, blob, np.float32(0.0))
    radius = max(4, int(round(0.018 * min(height, width))))
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
