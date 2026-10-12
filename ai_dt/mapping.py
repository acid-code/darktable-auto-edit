"""Turn detections and an atmosphere reading into Darktable module values.

The binary layouts below were checked against a Darktable 5.4.1 camera style:

* exposure module version 6 is 24 bytes: mode, black, exposure,
  deflicker percentile, deflicker target, compensate-exposure-bias.
  Darktable 5.6 upgrades that blob to version 7 on import.
* color balance rgb module version 5 is 132 bytes: 32 floats plus the
  saturation-formula integer. A decoded official style lands
  ``shadows_weight`` at float 12, ``mask_grey_fulcrum`` at float 28,
  ``vibrance`` at float 29, and ``contrast`` at float 31.
* the default blend blob is the 420-byte ``gz08…`` value shipped in those
  same styles, with blendop version 13.

Vignette version 4 is the 44-byte layout from the 5.4.1 module. It is the
spatial edit a style file can carry without a separate mask table. Drawn
ellipse masks are still described on each local edit so a later encoder can
attach them without changing the detector or the advisor.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

import struct

import numpy as np
from PIL import Image

from ai_dt.advisors.histogram import luminance
from ai_dt.schema import Atmosphere, Detection

# Default "no mask, normal blend, full opacity" blob from darktable 5.4.1 styles.
DEFAULT_BLENDOP = (
    "gz08eJxjYGBgYAFiCQYYOOHEgAZY0QWAgBGLGANDgz0Ej1Q+dlAx68oBEMbFxwX+AwGIBgCbGCeh"
)
BLENDOP_VERSION = 13

EXPOSURE_VERSION = 6
COLORBALANCE_VERSION = 5
VIGNETTE_VERSION = 4

_VIGNETTE_PRIORITY = {
    "face": 0,
    "person": 1,
    "animal": 2,
    "architecture": 3,
    "subject": 4,
    "eye": 9,
}

_COLOR_FIELDS = (
    "shadows_Y",
    "shadows_C",
    "shadows_H",
    "midtones_Y",
    "midtones_C",
    "midtones_H",
    "highlights_Y",
    "highlights_C",
    "highlights_H",
    "global_Y",
    "global_C",
    "global_H",
    "shadows_weight",
    "white_fulcrum",
    "highlights_weight",
    "chroma_shadows",
    "chroma_highlights",
    "chroma_global",
    "chroma_midtones",
    "saturation_global",
    "saturation_highlights",
    "saturation_midtones",
    "saturation_shadows",
    "hue_angle",
    "brilliance_global",
    "brilliance_highlights",
    "brilliance_midtones",
    "brilliance_shadows",
    "mask_grey_fulcrum",
    "vibrance",
    "grey_fulcrum",
    "contrast",
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def exposure_params(exposure_ev: float, black: float = 0.0) -> bytes:
    """Pack exposure v6. Percentile and target keep the module defaults."""

    blob = struct.pack(
        "<iffffi",
        0,  # manual
        clamp(black, -1.0, 1.0),
        clamp(exposure_ev, -3.0, 3.0),
        50.0,
        -4.0,
        0,  # do not compensate the camera exposure bias on top of this delta
    )
    if len(blob) != 24:
        raise RuntimeError(f"exposure blob is {len(blob)} bytes, expected 24")
    return blob


def colorbalance_params(values: dict[str, float]) -> bytes:
    """Pack color balance rgb v5. Unspecified fields keep Darktable defaults."""

    params = {name: 0.0 for name in _COLOR_FIELDS}
    params["shadows_weight"] = 1.0
    params["highlights_weight"] = 1.0
    params["mask_grey_fulcrum"] = 0.1845
    params["grey_fulcrum"] = 0.1845
    params.update(values)
    floats = [float(params[name]) for name in _COLOR_FIELDS]
    blob = struct.pack("<32fi", *floats, 1)  # 1 = darktable UCS saturation formula
    if len(blob) != 132:
        raise RuntimeError(f"color balance blob is {len(blob)} bytes, expected 132")
    return blob


def vignette_params(values: dict[str, float]) -> bytes:
    """Pack vignette v4. ``center_x`` and ``center_y`` are -1..1, 0 at the middle."""

    blob = struct.pack(
        "<ffffffiffIi",
        float(values["scale"]),
        float(values["falloff_scale"]),
        float(values["brightness"]),
        float(values["saturation"]),
        float(values["center_x"]),
        float(values["center_y"]),
        1 if values.get("autoratio") else 0,
        float(values["whratio"]),
        float(values["shape"]),
        int(values.get("dithering", 0)),
        1 if values.get("unbound", True) else 0,
    )
    if len(blob) != 44:
        raise RuntimeError(f"vignette blob is {len(blob)} bytes, expected 44")
    return blob


def _warmth_to_color(atmosphere: Atmosphere) -> dict[str, float]:
    shift = clamp(atmosphere.temperature_shift, -1.0, 1.0)
    tint = clamp(atmosphere.tint, -1.0, 1.0)
    if abs(shift) < 0.03 and abs(tint) < 0.03:
        hue = 0.0
        chroma = 0.0
    else:
        hue = 40.0 if shift >= 0 else 210.0
        chroma = abs(shift) * 0.14
        hue = (hue + tint * 40.0) % 360.0
        chroma = clamp(chroma + abs(tint) * 0.08, 0.0, 0.16)
    return {
        "global_H": hue,
        "global_C": chroma,
        "contrast": clamp(atmosphere.contrast, -0.6, 0.6),
        "saturation_global": clamp(atmosphere.saturation, -0.6, 0.6),
        "vibrance": clamp(atmosphere.vibrance, -0.5, 0.5),
        "brilliance_shadows": clamp(atmosphere.shadows, -0.6, 0.6),
        "brilliance_highlights": clamp(atmosphere.highlights, -0.6, 0.6),
    }


def _crop_mean(luma: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    height, width = luma.shape
    x, y, w, h = bbox
    x0 = min(width - 1, max(0, int(x * width)))
    y0 = min(height - 1, max(0, int(y * height)))
    x1 = min(width, max(x0 + 1, int((x + w) * width)))
    y1 = min(height, max(y0 + 1, int((y + h) * height)))
    return float(luma[y0:y1, x0:x1].mean())


def _ellipse(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    x, y, w, h = bbox
    return {
        "cx": round(x + w / 2.0, 5),
        "cy": round(y + h / 2.0, 5),
        "rx": round(w / 2.0, 5),
        "ry": round(h / 2.0, 5),
        "rotation": 0.0,
        "feather": 0.35,
    }


def _local_edit(image: Image.Image, detection: Detection, frame_mean: float) -> dict:
    luma = luminance(image)
    subject_mean = _crop_mean(luma, detection.bbox)
    # Positive when the subject is darker than the frame: the masked exposure
    # we would apply if a drawn mask were attached.
    exposure_ev = clamp((frame_mean - subject_mean) * 1.5, -0.6, 0.75)
    if detection.category == "eye":
        exposure_ev = clamp(exposure_ev, -0.15, 0.35)
    if detection.category == "architecture":
        exposure_ev = clamp(exposure_ev, -0.25, 0.25)
    if subject_mean + 0.06 < frame_mean:
        reason = "subject is darker than the rest of the frame"
    elif subject_mean > frame_mean + 0.06:
        reason = "subject is brighter than the rest of the frame"
    else:
        reason = "subject is close to the frame luminance"
    return {
        "target": detection.label,
        "category": detection.category,
        "confidence": round(detection.confidence, 4),
        "reason": reason,
        "bbox": [round(v, 5) for v in detection.bbox],
        "ellipse": _ellipse(detection.bbox),
        "exposure_ev": round(exposure_ev, 4),
        "subject_luminance": round(subject_mean, 4),
        "frame_luminance": round(frame_mean, 4),
    }


def _shadow_lift(edit: dict) -> float:
    if edit["category"] not in {"face", "person", "animal"}:
        return 0.0
    if edit["subject_luminance"] >= 0.5 or edit["exposure_ev"] < 0.12:
        return 0.0
    return clamp(edit["exposure_ev"] * 0.4, 0.0, 0.22)


def _vignette_for(
    detection: Detection,
    image: Image.Image,
    strength: float = 1.0,
) -> dict[str, float] | None:
    if detection.category == "eye":
        return None
    x, y, w, h = detection.bbox
    area = w * h
    if area < 0.01 or area > 0.72:
        return None
    width, height = image.size
    longest = float(max(width, height))
    radius_px = 0.5 * max(w * width, h * height)
    scale = clamp(100.0 * radius_px / longest, 10.0, 85.0)
    aspect = (w * width) / max(h * height, 1e-6)
    if aspect <= 1.0:
        whratio = clamp(aspect, 0.2, 1.0)
    else:
        whratio = clamp(1.0 + (1.0 / aspect), 1.0, 2.0)
    base = -0.22 if detection.category in {"face", "person", "animal"} else -0.12
    brightness = clamp(base * strength, -0.55, 0.0)
    return {
        "scale": round(scale, 3),
        "falloff_scale": round(clamp(scale * 0.85, 12.0, 90.0), 3),
        "brightness": brightness,
        "saturation": 0.0,
        "center_x": round(clamp((x + w / 2.0) * 2.0 - 1.0, -1.0, 1.0), 4),
        "center_y": round(clamp((y + h / 2.0) * 2.0 - 1.0, -1.0, 1.0), 4),
        "autoratio": False,
        "whratio": round(whratio, 4),
        "shape": 1.0,
        "dithering": 0,
        "unbound": True,
    }


def _choose_vignette_target(detections: list[Detection]) -> Detection | None:
    ranked = sorted(
        detections,
        key=lambda item: (_VIGNETTE_PRIORITY.get(item.category, 8), -item.confidence),
    )
    for detection in ranked:
        if detection.category == "eye":
            continue
        return detection
    return None


def _plugin(
    number: int,
    operation: str,
    version: int,
    params: bytes,
    multi_name: str,
    multi_priority: int = 1,
) -> dict:
    return {
        "num": number,
        "operation": operation,
        "modversion": version,
        "multi_name": multi_name,
        "multi_priority": multi_priority,
        "enabled": True,
        "op_params": params.hex(),
    }


def build_modules(
    image: Image.Image,
    atmosphere: Atmosphere,
    detections: list[Detection],
    vignette_strength: float = 1.0,
) -> tuple[Atmosphere, list[dict], list[dict], list[dict]]:
    """Return the adjusted atmosphere, module records, local edits, and raw blobs.

    Module records are JSON-friendly. The companion list holds the packed
    parameter bytes in the same order, ready for the style writer.
    ``vignette_strength`` scales the subject vignette. 1 is the built-in
    amount. A faithful recipe uses a lighter value. A moody recipe uses more.
    """

    luma = luminance(image)
    frame_mean = float(luma.mean())
    local_edits = [_local_edit(image, detection, frame_mean) for detection in detections[:6]]
    adjusted = Atmosphere(**atmosphere.to_dict())
    extra_shadows = max((_shadow_lift(edit) for edit in local_edits), default=0.0)
    if extra_shadows:
        adjusted.shadows = clamp(adjusted.shadows + extra_shadows, -0.45, 0.45)
        adjusted.rationale = (
            adjusted.rationale
            + " Dark subject tones get an additional shadow lift because a style "
            + "cannot carry a drawn mask."
        ).strip()

    color_values = _warmth_to_color(adjusted)
    exposure_blob = exposure_params(adjusted.exposure_ev, adjusted.black)
    color_blob = colorbalance_params(color_values)
    modules = [
        {
            "operation": "exposure",
            "modversion": EXPOSURE_VERSION,
            "multi_name": "AI exposure",
            "multi_priority": 1,
            "enabled": True,
            "params": {
                "exposure": round(clamp(adjusted.exposure_ev, -3.0, 3.0), 4),
                "black": round(clamp(adjusted.black, -1.0, 1.0), 4),
            },
        },
        {
            "operation": "colorbalancergb",
            "modversion": COLORBALANCE_VERSION,
            "multi_name": "AI color",
            "multi_priority": 1,
            "enabled": True,
            "params": {key: round(float(value), 4) for key, value in color_values.items()},
        },
    ]
    blobs = [
        _plugin(0, "exposure", EXPOSURE_VERSION, exposure_blob, "AI exposure"),
        _plugin(1, "colorbalancergb", COLORBALANCE_VERSION, color_blob, "AI color"),
    ]

    target = _choose_vignette_target(detections)
    vignette = (
        _vignette_for(target, image, vignette_strength) if target is not None else None
    )
    if target is not None and vignette is not None:
        modules.append(
            {
                "operation": "vignette",
                "modversion": VIGNETTE_VERSION,
                "multi_name": f"AI {target.category}",
                "multi_priority": 1,
                "enabled": True,
                "params": vignette,
            }
        )
        blobs.append(
            _plugin(
                2,
                "vignette",
                VIGNETTE_VERSION,
                vignette_params(vignette),
                f"AI {target.category}",
            )
        )
        for edit in local_edits:
            if edit["category"] == target.category and edit["bbox"] == [
                round(v, 5) for v in target.bbox
            ]:
                edit["applied_as"] = (
                    "subject-centered vignette, which darkens around this region"
                )
                break
    for edit in local_edits:
        edit.setdefault(
            "applied_as",
            "measured only; the ellipse is stored for a future drawn-mask encoder",
        )
    return adjusted, modules, local_edits, blobs


def render_style(name: str, description: str, plugins: list[dict]) -> str:
    """Render a .dtstyle document. No module-order list is written.

    A partial order list would replace the photo's pipeline order. Omitting it
    lets Darktable drop each module at that module's own place in the pipe.
    """

    root = ET.Element("darktable_style", {"version": "1.0"})
    info = ET.SubElement(root, "info")
    ET.SubElement(info, "name").text = name
    ET.SubElement(info, "description").text = description
    style = ET.SubElement(root, "style")
    for plugin in plugins:
        node = ET.SubElement(style, "plugin")
        ET.SubElement(node, "num").text = str(plugin["num"])
        ET.SubElement(node, "module").text = str(plugin["modversion"])
        ET.SubElement(node, "operation").text = plugin["operation"]
        ET.SubElement(node, "op_params").text = plugin["op_params"]
        ET.SubElement(node, "enabled").text = "1" if plugin["enabled"] else "0"
        ET.SubElement(node, "blendop_params").text = DEFAULT_BLENDOP
        ET.SubElement(node, "blendop_version").text = str(BLENDOP_VERSION)
        ET.SubElement(node, "multi_priority").text = str(plugin["multi_priority"])
        ET.SubElement(node, "multi_name").text = plugin["multi_name"]
        ET.SubElement(node, "multi_name_hand_edited").text = "1"
    rough = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="UTF-8")
    return pretty.decode("utf-8")
