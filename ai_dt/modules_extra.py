"""Extra Darktable modules, packed the way exposure already is.

Layouts are the parameter structs from Darktable 5.4.1. Every field is a
4-byte float, enum, or gboolean, so the blob length is the field count
times 4. A module is left out when its driving number is zero.

Filmic and sigmoid are alternatives. Masked modules are packed only when
``masks`` is true. The sidecar spike keeps that false, so subject lift,
clarity, skin, blurs, and retouch stay in the TIFF until a mask can be
attached without replacing history.
"""

from __future__ import annotations

import struct

from ai_dt.guides.library import Guide
from ai_dt.history_ops import choose_display
from ai_dt.mapping import _plugin, exposure_params
from ai_dt.masks_xmp import masks_can_attach
from ai_dt.schema import Atmosphere, Detection

_EPS = 0.02

# Darktable 5.4.1 module versions (DT_MODULE_INTROSPECTION).
TONE_EQUAL_VERSION = 2
FILMIC_VERSION = 6
SIGMOID_VERSION = 3
COLORZONES_VERSION = 5
CROP_VERSION = 3
CHANNELMIXER_VERSION = 3
GRADUATED_VERSION = 1
DIFFUSE_VERSION = 2
BLURS_VERSION = 1
RETOUCH_VERSION = 3

TONE_EQUAL_BYTES = 72
FILMIC_BYTES = 116
SIGMOID_BYTES = 56
COLORZONES_BYTES = 520
CROP_BYTES = 24
CHANNELMIXER_BYTES = 160
GRADUATED_BYTES = 24
DIFFUSE_BYTES = 60
BLURS_BYTES = 36


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _near(value: float) -> bool:
    return abs(float(value)) < _EPS


def tone_equalizer_params(
    shadows: float,
    highlights: float,
    *,
    hold: bool,
) -> bytes | None:
    """Nine exposure bands. Shadows lift the dark bands. Highlights pull the bright ones."""

    if hold:
        highlights = min(float(highlights), -0.2)
    if _near(shadows) and _near(highlights) and not hold:
        return None
    bands = [
        0.0,  # blacks (field noise)
        _clamp(shadows * 0.35, -1.0, 1.0),
        _clamp(shadows * 0.7, -1.0, 1.0),
        _clamp(shadows, -1.0, 1.0),
        _clamp(shadows * 0.45, -1.0, 1.0),
        0.0,
        _clamp(highlights, -1.0, 1.0),
        _clamp(highlights * 0.7, -1.0, 1.0),
        0.0,
    ]
    if all(_near(band) for band in bands):
        return None
    blob = struct.pack(
        "<15f3i",
        *bands,
        5.0,  # blending
        1.414213562,  # smoothing
        1.0,  # feathering
        0.0,  # quantization
        0.0,  # contrast boost
        0.0,  # exposure boost
        4,  # EIGF
        4,  # RGB euclidean norm
        1,  # iterations
    )
    if len(blob) != TONE_EQUAL_BYTES:
        raise RuntimeError(f"tone equalizer blob is {len(blob)} bytes")
    return blob


def filmic_params(*, blown: bool, contrast: float) -> bytes:
    white = 3.2 if blown else 4.0
    contrast_value = _clamp(1.0 + contrast, 0.6, 2.0)
    floats = [
        18.45,  # grey point source
        -8.0,  # black relative exposure
        white,
        0.0,  # reconstruct threshold
        3.0,  # feather
        100.0,  # bloom vs reconstruct
        100.0,  # grey vs color
        0.0,  # structure vs texture
        0.0,  # security
        18.45,  # target grey
        0.01517634,  # target black
        100.0,  # target white
        4.0,  # hardness
        0.01,  # latitude
        contrast_value,
        0.0,  # saturation
        0.0,  # balance
        0.2,  # noise
    ]
    ints = [
        3,  # power norm
        4,  # color science v7
        1,  # auto hardness
        0,  # custom grey
        1,  # reconstruction iterations
        1,  # gaussian noise
        0,  # hard shadow curve
        0,  # hard highlight curve
        0,  # compensate ICC black
        2,  # spline v3
        1 if blown else 0,
    ]
    blob = struct.pack("<18f11i", *floats, *ints)
    if len(blob) != FILMIC_BYTES:
        raise RuntimeError(f"filmic blob is {len(blob)} bytes")
    return blob


def sigmoid_params(contrast: float) -> bytes:
    middle = _clamp(1.5 + contrast, 0.8, 3.0)
    blob = struct.pack(
        "<4fi8fi",
        middle,
        0.0,  # skew
        100.0,  # target white
        0.0152,  # target black
        0,  # per channel
        100.0,  # preserve hue
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # purity
        0,  # working profile primaries
    )
    if len(blob) != SIGMOID_BYTES:
        raise RuntimeError(f"sigmoid blob is {len(blob)} bytes")
    return blob


def colorzones_params(saturation: float) -> bytes | None:
    """Chroma curve. Strength is the mix. A near-zero saturation adds nothing."""

    if _near(saturation):
        return None
    buf = bytearray(COLORZONES_BYTES)
    struct.pack_into("<i", buf, 0, 2)  # select by hue, the 5.4 default
    y = _clamp(0.5 + saturation * 0.25, 0.05, 0.95)
    for channel in range(3):
        node_y = y if channel == 1 else 0.5
        for node in range(2):
            offset = 4 + ((channel * 20 + node) * 8)
            struct.pack_into("<ff", buf, offset, (node + 0.5) / 2.0, node_y)
        struct.pack_into("<i", buf, 484 + channel * 4, 2)
        struct.pack_into("<i", buf, 496 + channel * 4, 1)  # Catmull-Rom
    struct.pack_into("<f", buf, 508, _clamp(saturation * 100.0, -80.0, 80.0))
    struct.pack_into("<i", buf, 512, 0)  # smooth
    struct.pack_into("<i", buf, 516, 1)  # splines v2
    return bytes(buf)


def crop_params(
    box: tuple[float, float, float, float] | None,
    *,
    force: bool = False,
) -> bytes | None:
    """Trim toward a horizon that sits near the top or the bottom.

    ``box`` is ``x, y, width, height`` in 0–1. A horizon already in the
    middle third is left alone.
    """

    if box is None or len(box) != 4:
        return None
    _x, y, _w, height = (float(item) for item in box)
    center = y + height / 2.0
    if 0.33 <= center <= 0.67 and not force:
        return None
    left, top, right, bottom = 0.0, 0.0, 1.0, 1.0
    if center < 0.33 or (force and center < 0.5):
        bottom = 1.0 - min(0.06, 0.33 - center) if center < 0.33 else 0.96
    elif center > 0.67 or force:
        top = min(0.06, center - 0.67) if center > 0.67 else 0.04
    if top < _EPS and bottom > 1.0 - _EPS:
        return None
    blob = struct.pack("<4f2i", left, top, right, bottom, -1, -1)
    if len(blob) != CROP_BYTES:
        raise RuntimeError(f"crop blob is {len(blob)} bytes")
    return blob


def _daylight_xy(kelvin: float) -> tuple[float, float]:
    """CIE daylight chromaticity for 4000 K to 7000 K, from Darktable's illuminant helper."""

    temperature = _clamp(kelvin, 4000.0, 7000.0)
    x_value = (
        ((-4.6070e9 / temperature + 2.9678e6) / temperature + 0.09911e3) / temperature
        + 0.244063
    )
    y_value = (-3.0 * x_value + 2.87) * x_value - 0.275
    return x_value, y_value


# Darktable registers color calibration as the pipe white balance when the
# illuminant is anything other than "same as pipeline" (0). That replaces
# the camera as-shot balance. On a tungsten night the replacement is a
# full-frame red cast, even when the requested warmth is only a few percent.
_ILLUMINANT_PIPE = 0
_ADAPTATION_BYPASS = 4  # DT_ADAPTATION_RGB: do not run a chromatic adaptation
_GAIN_LIMIT = 0.08


def channelmixer_params(warmth: float, tint: float) -> bytes | None:
    """A small relative channel gain. It does not replace the camera white balance.

    Positive warmth raises red and lowers blue by a few percent. The
    illuminant stays the pipeline white (D50), and adaptation is bypassed,
    so Darktable keeps the as-shot white balance.
    """

    if _near(warmth) and _near(tint):
        return None
    red = _clamp(1.0 + 0.10 * warmth - 0.04 * tint, 1.0 - _GAIN_LIMIT, 1.0 + _GAIN_LIMIT)
    green = _clamp(1.0 - 0.08 * tint, 1.0 - _GAIN_LIMIT, 1.0 + _GAIN_LIMIT)
    blue = _clamp(1.0 - 0.10 * warmth + 0.04 * tint, 1.0 - _GAIN_LIMIT, 1.0 + _GAIN_LIMIT)
    channels = [0.0] * 24
    channels[0] = red
    channels[5] = green
    channels[10] = blue
    # Stored temperature stays D50. The pipe illuminant ignores it and does
    # not install a daylight or custom CAT.
    x_value, y_value = _daylight_xy(5003.0)
    blob = struct.pack(
        "<24f6i4i4fii",
        *channels,
        0, 0, 0, 0, 0, 0,  # normalize flags
        _ILLUMINANT_PIPE,
        2,  # F3, unused unless the illuminant is fluorescent
        4,  # LED B5, unused unless the illuminant is LED
        _ADAPTATION_BYPASS,
        x_value,
        y_value,
        5003.0,
        0.0,  # gamut compression off, same as Darktable's bypass preset
        0,  # clip
        2,  # saturation algorithm v3
    )
    if len(blob) != CHANNELMIXER_BYTES:
        raise RuntimeError(f"color calibration blob is {len(blob)} bytes")
    return blob


def graduated_params(enabled: bool, placement: dict | None = None) -> bytes | None:
    """Pack graduated density. Without a placement this is only the length check.

    The style writer passes a measured line. Density is EV, hardness is
    percent, rotation is degrees, and offset is percent of the frame.
    Offset 50 and rotation 0 is the center horizontal, which is not used
    unless a measurement actually sits there.
    """

    if placement is None and not enabled:
        return None
    if placement is None:
        density, hardness, rotation, offset = 0.35, 0.0, 0.0, 50.0
    else:
        density = _clamp(float(placement.get("density") or 0.0), -8.0, 8.0)
        hardness = _clamp(float(placement.get("hardness") or 0.0), 0.0, 100.0)
        rotation = _clamp(float(placement.get("rotation") or 0.0), -180.0, 180.0)
        offset = _clamp(float(placement.get("offset") or 50.0), 0.0, 100.0)
        if abs(density) < _EPS:
            return None
    blob = struct.pack("<6f", density, hardness, rotation, offset, 0.0, 0.0)
    if len(blob) != GRADUATED_BYTES:
        raise RuntimeError(f"graduated density blob is {len(blob)} bytes")
    return blob


def _regions_of(reading: object) -> list[dict]:
    regions = getattr(reading, "regions", None)
    if regions is None and isinstance(reading, dict):
        regions = reading.get("regions")
    return [item for item in (regions or []) if isinstance(item, dict)]


def _placement_from_falloff(info: dict) -> dict | None:
    """Turn a bright-to-dark axis into the graduated-density line."""

    axis = str(info.get("axis") or "")
    bright = str(info.get("bright") or "")
    if axis not in {"vertical", "horizontal"} or bright not in {"top", "bottom", "left", "right"}:
        return None
    raw = info.get("position")
    position = 0.5 if raw is None else float(raw)
    position = _clamp(position, 0.05, 0.95)
    gap = abs(float(info.get("gap") or 0.0))
    strength = abs(float(info.get("strength") or gap or 0.2))
    density = _clamp(max(0.45, strength * 4.0), 0.4, 1.8)
    hardness = _clamp(28.0 + gap * 80.0, 28.0, 75.0)
    if axis == "vertical" and bright == "top":
        rotation, offset = 0.0, position * 100.0
    elif axis == "vertical" and bright == "bottom":
        rotation, offset = 180.0, (1.0 - position) * 100.0
    elif axis == "horizontal" and bright == "left":
        rotation, offset = 90.0, position * 100.0
    else:
        rotation, offset = -90.0, (1.0 - position) * 100.0
    return {
        "density": round(density, 4),
        "hardness": round(hardness, 4),
        "rotation": round(rotation, 4),
        "offset": round(offset, 4),
        "axis": axis,
        "bright": bright,
    }


def _falloff_from_box(box: object, kind: str) -> dict | None:
    """A sky, a highlight band, or a bright region becomes a line across that edge."""

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    _x, y, _w, height = (float(item) for item in box)
    if height <= 0.02:
        return None
    name = kind.lower()
    # Darken the bright side. An upper region darkens above its lower edge.
    if y + height <= 0.62:
        position = y + height
        bright = "top"
    elif y >= 0.38:
        position = y
        bright = "bottom"
    else:
        position = y + height / 2.0
        bright = "top"
    return {
        "axis": "vertical",
        "bright": bright,
        "position": position,
        "strength": -0.22 if "highlight" in name else -0.18,
        "gap": 0.22,
    }


def gradient_placement(plan: dict, separation: dict, reading: object) -> dict | None:
    """The measured line, or None when the plan has no direction to place."""

    if not plan.get("gradient"):
        return None
    candidates: list[dict] = []
    falloff = plan.get("falloff")
    if isinstance(falloff, dict) and falloff.get("axis"):
        candidates.append(falloff)
    recorded = separation.get("gradient") if isinstance(separation, dict) else None
    if isinstance(recorded, dict) and recorded.get("axis") and not recorded.get("skipped"):
        candidates.append(recorded)
    for region in _regions_of(reading):
        role = str(region.get("role") or "")
        kind = str(region.get("kind") or "")
        if role == "gradient" and region.get("axis"):
            candidates.append(region)
            continue
        if role in {"sky", "highlights"} or kind in {"sky", "highlights", "bright-band", "blown-highlights"}:
            synthesized = _falloff_from_box(region.get("box"), kind or role)
            if synthesized is not None:
                candidates.append(synthesized)
    for info in candidates:
        placed = _placement_from_falloff(info)
        if placed is not None:
            return placed
    return None


def diffuse_params(*, sharpness: float, iterations: int, radius: int) -> bytes | None:
    if _near(sharpness):
        return None
    floats = [0.0] * 11
    blob = struct.pack(
        "<ifi11fi",
        int(iterations),
        _clamp(sharpness, -1.0, 1.0),
        int(radius),
        *floats,
        0,
    )
    if len(blob) != DIFFUSE_BYTES:
        raise RuntimeError(f"diffuse blob is {len(blob)} bytes")
    return blob


def blurs_params(defocus: float) -> bytes | None:
    if defocus < _EPS:
        return None
    radius = int(round(_clamp(4.0 + defocus * 28.0, 4.0, 32.0)))
    blob = struct.pack("<3i6f", 0, radius, 5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    if len(blob) != BLURS_BYTES:
        raise RuntimeError(f"blurs blob is {len(blob)} bytes")
    return blob


def spots(detections: list[Detection]) -> list[Detection]:
    return [item for item in detections if item.category == "spot"]


def _mask_amount(guide: Guide | None, name: str, key: str) -> float:
    if guide is None:
        return 0.0
    plan = guide.masks.get(name)
    if plan is None:
        return 0.0
    return float(getattr(plan, key, 0.0) or 0.0)


def _horizon_box(reading: object) -> tuple[float, float, float, float] | None:
    regions = getattr(reading, "regions", None)
    if regions is None and isinstance(reading, dict):
        regions = reading.get("regions")
    for region in regions or []:
        if not isinstance(region, dict):
            continue
        if region.get("kind") != "horizon" and region.get("role") != "horizon":
            continue
        box = region.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return tuple(float(item) for item in box)  # type: ignore[return-value]
    return None


def _problem_kinds(reading: object) -> list[str]:
    problems = getattr(reading, "problems", None)
    if problems is None and isinstance(reading, dict):
        problems = reading.get("problems")
    kinds = []
    for item in problems or []:
        if isinstance(item, dict):
            kind = str(item.get("kind") or "")
            if kind:
                kinds.append(kind)
        elif isinstance(item, str):
            kinds.append(item)
    return kinds


def _add(
    modules: list[dict],
    blobs: list[dict],
    *,
    operation: str,
    version: int,
    params: dict,
    blob: bytes,
    multi_name: str,
    priority: int = 1,
) -> None:
    modules.append(
        {
            "operation": operation,
            "modversion": version,
            "multi_name": multi_name,
            "multi_priority": priority,
            "enabled": True,
            "params": params,
        }
    )
    blobs.append(_plugin(len(blobs), operation, version, blob, multi_name, priority))


def append_extra_modules(
    modules: list[dict],
    blobs: list[dict],
    *,
    atmosphere: Atmosphere,
    picture_type: str,
    reading: object,
    plan: dict,
    separation: dict,
    guide: Guide | None,
    history_ops: set[str] | None,
    look: str,
    vibe: str,
    masks: bool | None = None,
) -> tuple[list[dict], list[dict]]:
    """Append modules that have a non-zero grade. Return the same lists."""

    attach = masks_can_attach() if masks is None else bool(masks)
    problems = _problem_kinds(reading)
    blown = "blown-highlights" in problems
    hold = bool(plan.get("highlight_hold")) or blown

    tone = tone_equalizer_params(atmosphere.shadows, atmosphere.highlights, hold=hold)
    if tone is not None:
        _add(
            modules,
            blobs,
            operation="toneequal",
            version=TONE_EQUAL_VERSION,
            params={
                "shadows": round(float(atmosphere.shadows), 4),
                "highlights": round(float(atmosphere.highlights), 4),
                "hold": hold,
            },
            blob=tone,
            multi_name="AI tone equalizer",
        )

    display = choose_display(
        history_ops,
        look=look,
        vibe=vibe,
        picture_type=picture_type,
        problems=problems,
    )
    if display == "filmicrgb":
        _add(
            modules,
            blobs,
            operation="filmicrgb",
            version=FILMIC_VERSION,
            params={"blown": blown, "contrast": round(float(atmosphere.contrast), 4)},
            blob=filmic_params(blown=blown, contrast=atmosphere.contrast),
            multi_name="AI filmic",
        )
    elif display == "sigmoid":
        _add(
            modules,
            blobs,
            operation="sigmoid",
            version=SIGMOID_VERSION,
            params={"contrast": round(float(atmosphere.contrast), 4)},
            blob=sigmoid_params(atmosphere.contrast),
            multi_name="AI sigmoid",
        )

    zones = colorzones_params(atmosphere.saturation)
    if zones is not None:
        _add(
            modules,
            blobs,
            operation="colorzones",
            version=COLORZONES_VERSION,
            params={"saturation": round(float(atmosphere.saturation), 4)},
            blob=zones,
            multi_name="AI color zones",
        )

    crop = crop_params(_horizon_box(reading), force=bool(plan.get("crop_horizon")))
    if crop is not None:
        left, top, right, bottom = struct.unpack_from("<4f", crop, 0)
        _add(
            modules,
            blobs,
            operation="crop",
            version=CROP_VERSION,
            params={
                "left": round(left, 4),
                "top": round(top, 4),
                "right": round(right, 4),
                "bottom": round(bottom, 4),
            },
            blob=crop,
            multi_name="AI crop",
        )

    calibration = channelmixer_params(atmosphere.temperature_shift, atmosphere.tint)
    if calibration is not None:
        _add(
            modules,
            blobs,
            operation="channelmixerrgb",
            version=CHANNELMIXER_VERSION,
            params={
                "warmth": round(float(atmosphere.temperature_shift), 4),
                "tint": round(float(atmosphere.tint), 4),
            },
            blob=calibration,
            multi_name="AI color calibration",
        )

    placement = gradient_placement(plan, separation, reading)
    if plan.get("gradient") and placement is None:
        plan["gradient_note"] = (
            "Graduated density was left out. There is no measured direction to place the line."
        )
    gradient = graduated_params(placement is not None, placement)
    if gradient is not None and placement is not None:
        _add(
            modules,
            blobs,
            operation="graduatednd",
            version=GRADUATED_VERSION,
            params={
                "density": placement["density"],
                "hardness": placement["hardness"],
                "rotation": placement["rotation"],
                "offset": placement["offset"],
            },
            blob=gradient,
            multi_name="AI gradient",
        )

    if not attach:
        return modules, blobs

    ran = set(separation.get("stages_ran") or [])
    subject_ev = float(separation.get("subject_ev") or 0.0)
    if "subject" in ran and not _near(subject_ev):
        _add(
            modules,
            blobs,
            operation="exposure",
            version=6,
            params={"exposure": round(subject_ev, 4), "black": 0.0},
            blob=exposure_params(subject_ev, 0.0),
            multi_name="AI subject exposure",
            priority=2,
        )

    clarity = _mask_amount(guide, "eyes", "clarity")
    if "eyes" not in ran:
        clarity = 0.0
    if _near(clarity):
        clarity = _mask_amount(guide, "subject", "clarity") if "subject" in ran else 0.0
    detail = diffuse_params(sharpness=clarity, iterations=8, radius=8)
    if detail is not None and ("eyes" in ran or "subject" in ran):
        _add(
            modules,
            blobs,
            operation="diffuse",
            version=DIFFUSE_VERSION,
            params={"sharpness": round(clarity, 4), "role": "clarity"},
            blob=detail,
            multi_name="AI clarity",
        )

    skin = _mask_amount(guide, "skin", "clarity") if "skin" in ran else 0.0
    smooth = diffuse_params(sharpness=skin, iterations=12, radius=16)
    if smooth is not None and "skin" in ran:
        _add(
            modules,
            blobs,
            operation="diffuse",
            version=DIFFUSE_VERSION,
            params={"sharpness": round(skin, 4), "role": "skin"},
            blob=smooth,
            multi_name="AI skin",
            priority=2,
        )

    defocus = _mask_amount(guide, "background", "defocus") if "background" in ran else 0.0
    blur = blurs_params(defocus)
    if blur is not None:
        _add(
            modules,
            blobs,
            operation="blurs",
            version=BLURS_VERSION,
            params={"defocus": round(defocus, 4)},
            blob=blur,
            multi_name="AI blurs",
        )

    return modules, blobs


def retouch_decision(detections: list[Detection], *, masks: bool) -> dict:
    """Retouch is appended only for a real spot, and only with a mask form.

    There is no spot detector, and the sidecar cannot supply a form id, so
    the style does not gain an empty heal module.
    """

    found = spots(detections)
    if not found:
        return {
            "operation": "retouch",
            "modversion": RETOUCH_VERSION,
            "spots": 0,
            "appended": False,
            "reason": "No spot was detected, so retouch was not added.",
        }
    if not masks:
        return {
            "operation": "retouch",
            "modversion": RETOUCH_VERSION,
            "spots": len(found),
            "appended": False,
            "reason": (
                "A spot was detected, but retouch needs a mask form id and "
                "the sidecar cannot attach one. The module was not added."
            ),
        }
    return {
        "operation": "retouch",
        "modversion": RETOUCH_VERSION,
        "spots": len(found),
        "appended": False,
        "reason": (
            "A spot was detected, but the retouch parameter block stores "
            "mask-table form ids this editor cannot invent. The module was not added."
        ),
    }
