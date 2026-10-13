"""Say what this frame is doing, from measurements already in hand.

No model and no extra detector. The picture type, the boxes, and the
pixels name the regions, the problems worth ranking, and what to leave
alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ai_dt.guides.decide import Picture, _crop, _luma
from ai_dt.guides.support import reflected_subject, water_reflection
from ai_dt.schema import Detection

# A mirror this close to the water beside it is not worth lifting.
_WEAK_GAP = 0.045


@dataclass
class Reading:
    """Regions, ranked problems, and what this frame should leave alone."""

    sentence: str
    regions: list[dict] = field(default_factory=list)
    problems: list[dict] = field(default_factory=list)
    leave_alone: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "regions": list(self.regions),
            "problems": list(self.problems),
            "leave_alone": list(self.leave_alone),
        }


def _box_list(box: tuple[float, float, float, float] | None) -> list[float] | None:
    if box is None:
        return None
    return [round(float(value), 4) for value in box]


def _mean(pixels: np.ndarray, box: tuple[float, float, float, float] | None) -> float | None:
    if box is None:
        return None
    region = _crop(pixels, box)
    if region.size == 0:
        return None
    return float(_luma(region).mean())


def _core_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """The inner part of a box, so padding and bright ground are not the subject."""

    x, y, width, height = box
    return (x + 0.2 * width, y + 0.2 * height, width * 0.6, height * 0.6)


def mirror_box(pixels: np.ndarray) -> tuple[float, float, float, float] | None:
    """The reflection box, when water holds one."""

    mirrored = reflected_subject(pixels)
    if mirrored is not None:
        return mirrored[1]
    found = water_reflection(pixels)
    if found is None:
        return None
    return found[1]


def mirror_intention(pixels: np.ndarray, box: tuple[float, float, float, float]) -> str:
    """Hold a weak echo, quiet one that is darker than the water, or lift a clear one."""

    height, width = pixels.shape[:2]
    x, y, bw, bh = box
    x0 = min(width - 1, max(0, int(x * width)))
    y0 = min(height - 1, max(0, int(y * height)))
    x1 = min(width, max(x0 + 1, int((x + bw) * width)))
    y1 = min(height, max(y0 + 1, int((y + bh) * height)))
    luma = _luma(pixels)
    echo = luma[y0:y1, x0:x1]
    if echo.size < 8:
        return "hold"
    echo_level = float(np.median(echo))
    pad = max(2, int(0.04 * width))
    left = luma[y0:y1, max(0, x0 - pad) : x0]
    right = luma[y0:y1, x1 : min(width, x1 + pad)]
    beside_parts = [part for part in (left, right) if part.size >= 8]
    if not beside_parts:
        return "lift" if echo_level > 0.12 else "hold"
    beside = float(np.median(np.concatenate([part.ravel() for part in beside_parts])))
    gap = echo_level - beside
    if gap < _WEAK_GAP:
        if echo_level + 0.02 < beside:
            return "quiet"
        return "hold"
    return "lift"


def _theme(pixels: np.ndarray, picture: Picture) -> dict | None:
    """A horizon band, a compact bright window, or a color island.

    Skin and the creature are the subject, so they are not the theme.
    """

    height, width = pixels.shape[:2]
    luma = _luma(pixels)
    subject = picture.subject_box
    horizon = _horizon_box(pixels)
    if horizon is not None:
        return {
            "role": "theme",
            "kind": "horizon",
            "intention": "protect",
            "box": _box_list(horizon),
            "note": "A horizon band carries the picture and should not be swallowed by the surround.",
        }
    window = _window_box(pixels, luma, subject)
    if window is not None:
        return {
            "role": "theme",
            "kind": "window",
            "intention": "protect",
            "box": _box_list(window),
            "note": "A bright window is not the subject, and the surround should not swallow it.",
        }
    island = _island_box(pixels, luma, subject)
    if island is None:
        return None
    return {
        "role": "theme",
        "kind": "island",
        "intention": "protect",
        "box": _box_list(island),
        "note": "A compact color island is not skin and not the creature, so it stays out of the surround.",
    }


def _horizon_box(pixels: np.ndarray) -> tuple[float, float, float, float] | None:
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
    half = max(2, int(height * 0.04))
    top = max(0, peak - half)
    bottom = min(height, peak + half)
    return (0.0, top / height, 1.0, (bottom - top) / height)


def _window_box(
    pixels: np.ndarray,
    luma: np.ndarray,
    subject: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    height, width = luma.shape
    chroma = np.max(np.abs(pixels - luma[..., None]), axis=2)
    bright = (luma > np.float32(0.78)) & (chroma < np.float32(0.08))
    if subject is not None:
        sx, sy, sw, sh = subject
        x0 = int(sx * width)
        y0 = int(sy * height)
        x1 = int((sx + sw) * width)
        y1 = int((sy + sh) * height)
        bright[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = False
    if float(bright.mean()) > 0.12 or int(bright.sum()) < 12:
        return None
    ys, xs = np.nonzero(bright)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    area = ((x1 - x0 + 1) * (y1 - y0 + 1)) / float(height * width)
    if area < 0.004 or area > 0.12:
        return None
    return (x0 / width, y0 / height, (x1 - x0 + 1) / width, (y1 - y0 + 1) / height)


def _island_box(
    pixels: np.ndarray,
    luma: np.ndarray,
    subject: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    chroma = np.max(np.abs(pixels - luma[..., None]), axis=2)
    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    skin = (red > 0.38) & (green > 0.18) & (blue > 0.28) & (red > green) & (red > blue)
    leafy = (green > red + np.float32(0.04)) & (green > blue + np.float32(0.02))
    core = (chroma > np.float32(0.18)) & ~skin & ~leafy
    if subject is not None:
        height, width = luma.shape
        sx, sy, sw, sh = subject
        x0 = int(sx * width)
        y0 = int(sy * height)
        x1 = int((sx + sw) * width)
        y1 = int((sy + sh) * height)
        core[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = False
    count = int(core.sum())
    frame = core.size
    if count < 20 or count > int(0.08 * frame):
        return None
    ys, xs = np.nonzero(core)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    if count / box_area < 0.35:
        return None
    height, width = luma.shape
    return (x0 / width, y0 / height, (x1 - x0 + 1) / width, (y1 - y0 + 1) / height)


def _surround(pixels: np.ndarray, subject_level: float | None) -> list[dict]:
    luma = _luma(pixels)
    height, width = luma.shape
    band_y = max(1, height // 8)
    band_x = max(1, width // 8)
    border = np.concatenate(
        (
            luma[:band_y].ravel(),
            luma[-band_y:].ravel(),
            luma[:, :band_x].ravel(),
            luma[:, -band_x:].ravel(),
        )
    )
    level = float(np.median(border))
    if subject_level is None:
        return [
            {
                "role": "surround",
                "kind": "quiet",
                "intention": "hold",
                "note": "No compact subject, so the surround is the picture.",
            }
        ]
    regions = []
    if level > subject_level + 0.06:
        regions.append(
            {
                "role": "surround",
                "kind": "competing-bright",
                "intention": "darken",
                "note": "Part of the surround is brighter than the subject and can come down a little.",
            }
        )
    regions.append(
        {
            "role": "surround",
            "kind": "quiet",
            "intention": "hold",
            "note": "The quiet surround keeps its exposure.",
        }
    )
    return regions


def linear_falloff(pixels: np.ndarray) -> dict | None:
    """A bright-to-dark axis worth a soft gradient, or None when there is none.

    The bright end is usually the sky. A flat field and a frame with no
    direction return None. Strength here is the modest amount.
    """

    luma = _luma(pixels)
    height, width = luma.shape
    if min(height, width) < 24:
        return None

    def _ends(profile: np.ndarray) -> tuple[float, float]:
        band = max(2, int(profile.size) // 5)
        return float(profile[:band].mean()), float(profile[-band:].mean())

    def _directed(profile: np.ndarray, gap: float) -> bool:
        if abs(gap) < 0.10 or profile.size < 8:
            return False
        index = np.linspace(-1.0, 1.0, profile.size)
        if float(np.std(profile)) < 1e-4:
            return False
        corr = float(np.corrcoef(index, profile.astype(np.float64))[0, 1])
        return abs(corr) >= 0.35

    row = luma.mean(axis=1)
    col = luma.mean(axis=0)
    top, bottom = _ends(row)
    left, right = _ends(col)
    vertical = top - bottom
    horizontal = left - right
    use_vertical = abs(vertical) >= abs(horizontal) and _directed(row, vertical)
    use_horizontal = abs(horizontal) > abs(vertical) and _directed(col, horizontal)
    if not use_vertical and not use_horizontal:
        if abs(vertical) >= 0.12 and abs(vertical) >= abs(horizontal):
            use_vertical = True
        elif abs(horizontal) >= 0.12:
            use_horizontal = True
        else:
            return None
    if use_vertical:
        bright = "top" if vertical > 0 else "bottom"
        gap = abs(vertical)
        axis = "vertical"
        reason = "The sky side is brighter than the land, so a soft gradient darkens the bright end."
    else:
        bright = "left" if horizontal > 0 else "right"
        gap = abs(horizontal)
        axis = "horizontal"
        reason = "One side of the frame is brighter, so a soft gradient darkens that end."
    if gap < 0.10:
        return None
    strength = -max(0.08, min(0.28, (gap - 0.04) * 0.55))
    if abs(strength) < 0.08:
        return None
    return {
        "axis": axis,
        "bright": bright,
        "strength": round(float(strength), 4),
        "gap": round(float(gap), 4),
        "reason": reason,
    }


def read_frame(
    pixels: np.ndarray,
    picture: Picture,
    detections: list[Detection] | None = None,
) -> Reading:
    """Build the reading for one frame. The first sentence is the lead problem."""

    view = np.clip(pixels.astype(np.float32, copy=False), 0.0, 1.0)
    luma = _luma(view)
    found = list(detections or [])
    subject_level = _mean(view, None if picture.subject_box is None else _core_box(picture.subject_box))
    frame_level = float(luma.mean())
    regions: list[dict] = []
    problems: list[dict] = []
    leave: list[str] = []

    if picture.subject_box is not None:
        intention = "hold"
        if subject_level is not None and subject_level + 0.1 < frame_level:
            intention = "lift"
        label = "subject"
        if picture.type in {"portrait", "group", "street"}:
            label = "face"
        elif picture.type == "animal":
            label = "animal"
        regions.append(
            {
                "role": "subject",
                "kind": label,
                "intention": intention,
                "box": _box_list(picture.subject_box),
                "note": f"The {label} is the subject the picture type kept.",
            }
        )

    echo = mirror_box(view)
    echo_intention = None
    if echo is not None and picture.type in {"animal", "landscape", "architecture", "general"}:
        echo_intention = mirror_intention(view, echo)
        regions.append(
            {
                "role": "reflection",
                "kind": "reflection",
                "intention": echo_intention,
                "box": _box_list(echo),
                "note": {
                    "lift": "The reflection is clear enough to lift with the subject.",
                    "hold": "The reflection is weak, so it is held instead of lifted.",
                    "quiet": "The reflection is a distraction, so it is quieted.",
                }[echo_intention],
            }
        )

    theme = _theme(view, picture)
    if theme is not None:
        regions.append(theme)
    falloff = linear_falloff(view)
    close_face = picture.type in {"portrait", "group"} and (
        picture.compact_area >= 0.08
        or (picture.subject_box is not None and picture.subject_box[2] * picture.subject_box[3] >= 0.08)
    )
    if falloff is not None and picture.type != "night" and not close_face:
        regions.append(
            {
                "role": "gradient",
                "kind": "linear",
                "axis": falloff["axis"],
                "intention": "darken",
                "strength": falloff["strength"],
                "note": falloff["reason"],
            }
        )
    regions.extend(_surround(view, subject_level))

    labels = {item.category for item in found}
    face_like = picture.type in {"portrait", "group", "street"} or "face" in labels
    if face_like and subject_level is not None and subject_level + 0.12 < frame_level:
        problems.append(
            {
                "kind": "dark-subject",
                "sentence": "The face is darker than the bright ground.",
            }
        )
    blown = float((luma > np.float32(0.92)).mean())
    if blown > 0.01 or picture.highlight_specks > 0.02:
        problems.append(
            {
                "kind": "blown-highlights",
                "sentence": "Highlights are blown.",
            }
        )
    if echo_intention == "hold":
        problems.append(
            {
                "kind": "weak-reflection",
                "sentence": "The reflection is weak.",
            }
        )
    elif echo_intention == "quiet":
        problems.append(
            {
                "kind": "muddy-reflection",
                "sentence": "The reflection is muddy.",
            }
        )
    if picture.edges > 0.08 and picture.type not in {"architecture", "street"}:
        problems.append(
            {
                "kind": "busy-foliage",
                "sentence": "The foliage around the subject is busy.",
            }
        )

    if picture.type == "night" or picture.brightness < 0.28:
        leave.append("Leave the night dark.")
    if picture.type == "landscape" and picture.saturation >= 0.18:
        leave.append("Leave the land color alone.")
    if picture.type == "architecture":
        leave.append("Leave the facade as structure.")
    if abs(picture.warmth) <= 0.04 and picture.type != "night":
        leave.append("Leave the measured color alone.")

    if problems:
        sentence = problems[0]["sentence"]
    elif leave:
        sentence = leave[0]
    else:
        sentence = picture.reason.strip() or "Nothing in this frame asks for a local change."
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return Reading(sentence=sentence, regions=regions, problems=problems, leave_alone=leave)
