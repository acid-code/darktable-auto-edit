"""Lighting estimate from the pixels themselves.

This advisor always runs. It is the measurement a vision suggestion is blended
with, and it is the whole suggestion when no vision model is available.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ai_dt.guides.support import is_face_bar
from ai_dt.schema import Atmosphere


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def luminance(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def _white_balance(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """A small correction from near-neutral pixels only.

    Moss, a sunset, and a blue night are the color of the picture. Sampling
    them and pushing the same hue again turns the frame green or blue.
    Positive temperature warms a blue cast. Positive tint pulls green back
    toward magenta.
    """

    if rgb.size == 0 or float(mask.mean()) < 0.02:
        return 0.0, 0.0
    channels = rgb.astype(np.int16)
    span = channels.max(axis=2) - channels.min(axis=2)
    level = channels.mean(axis=2)
    # A real gray is very tight. Muddy brown still has some span, and cooling
    # it drains the red out of earth and leaves.
    gray = mask & (span < 18) & (level > 28) & (level < 235)
    muddy = mask & (span < 48) & (level > 28) & (level < 235)
    neutral = gray if float(gray.mean()) >= 0.025 else muddy
    if float(neutral.mean()) < 0.025:
        return 0.0, 0.0
    mean = rgb[neutral].astype(np.float32).mean(axis=0)
    too_blue = float(mean[2] - mean[0]) / 255.0
    too_green = float(mean[1] - (mean[0] + mean[2]) * 0.5) / 255.0
    temperature = _clamp(too_blue * 1.15, -0.28, 0.28)
    tint = _clamp(too_green * 1.25, -0.2, 0.2)
    frame = rgb.astype(np.float32)
    frame_warm = float((frame[..., 0] - frame[..., 2]).mean())
    frame_green = float((frame[..., 1] - (frame[..., 0] + frame[..., 2]) * 0.5).mean())
    # Earth that is already warm is the picture, not a cast to cool away.
    if frame_warm > 4.0:
        temperature = max(temperature, 0.0)
    if frame_green > 4.0:
        tint = max(tint, 0.0)
    if neutral is muddy:
        temperature = max(temperature, 0.0)
        tint = max(tint, 0.0)
    return temperature, tint


def _region(values: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    height, width = values.shape[:2]
    x, y, w, h = bbox
    x0 = min(width - 1, max(0, int(x * width)))
    y0 = min(height - 1, max(0, int(y * height)))
    x1 = min(width, max(x0 + 1, int((x + w) * width)))
    y1 = min(height, max(y0 + 1, int((y + h) * height)))
    return values[y0:y1, x0:x1]


def refine_for_subjects(image: Image.Image, atmosphere: Atmosphere, detections: list) -> Atmosphere:
    """Prefer a face, person, or animal over the frame median.

    A dark background must not add two stops on top of an already lit face,
    and skin must not be read as a white-balance error.
    """

    subjects = [
        item
        for item in detections
        if item.category in {"face", "person", "animal"}
        and not (item.category in {"face", "person"} and is_face_bar(item.bbox))
    ]
    if not subjects:
        return atmosphere

    rgb = np.asarray(image.convert("RGB"))
    luma = luminance(image)
    subject = max(subjects, key=lambda item: item.bbox[2] * item.bbox[3])
    region = _region(luma, subject.bbox)
    if region.size < 16:
        return atmosphere

    median = float(np.median(region))
    p95 = float(np.percentile(region, 95))
    target = 0.46
    subject_ev = float(np.log2((target + 1e-4) / (median + 1e-4)))
    frame_median = float(np.median(luma))
    if frame_median < 0.22:
        # A night portrait keeps the dark. The face is lifted on its own mask.
        exposure = _clamp(min(atmosphere.exposure_ev, 0.55), -2.0, 0.55)
        follows = (
            f"The frame is dark (median {frame_median:.0%}), so the night stays dark. "
            f"The {subject.category} is lifted on its own mask."
        )
    else:
        exposure = 0.8 * subject_ev + 0.2 * atmosphere.exposure_ev
        follows = (
            f"Exposure follows the {subject.category} (median luminance {median:.0%}) "
            f"rather than the whole frame."
        )
    headroom = float(np.log2(0.92 / max(p95, 0.04)))
    exposure = _clamp(min(exposure, headroom), -2.0, 2.0)

    mid = (luma > 0.15) & (luma < 0.85)
    for item in subjects:
        x, y, w, h = item.bbox
        height, width = luma.shape
        x0 = min(width, max(0, int(x * width)))
        y0 = min(height, max(0, int(y * height)))
        x1 = min(width, max(x0, int((x + w) * width)))
        y1 = min(height, max(y0, int((y + h) * height)))
        mid[y0:y1, x0:x1] = False
    temperature, tint = _white_balance(rgb, mid)

    refined = Atmosphere(**atmosphere.to_dict())
    refined.exposure_ev = exposure
    refined.temperature_shift = temperature
    refined.tint = tint
    refined.mood = _mood(exposure, temperature, atmosphere.contrast)
    refined.rationale = (
        f"{follows} Suggested {exposure:+.2f} EV, "
        f"contrast {atmosphere.contrast:+.2f}, warmth {temperature:+.2f}."
    )
    refined.source = atmosphere.source
    return refined


class HistogramAdvisor:
    name = "histogram"

    def advise(self, image: Image.Image) -> Atmosphere:
        rgb = np.asarray(image.convert("RGB"))
        luma = luminance(image)
        median = float(np.median(luma))
        p5 = float(np.percentile(luma, 5))
        p95 = float(np.percentile(luma, 95))
        target = 0.46
        exposure = _clamp(float(np.log2((target + 1e-4) / (median + 1e-4))), -2.0, 2.0)
        # A bright subject can be a small part of a dark frame. The 95th
        # percentile misses it, and lifting the frame then clips it white.
        bright = float(np.percentile(luma, 99))
        headroom = float(np.log2((0.92 + 1e-4) / (max(bright, 0.04) + 1e-4)))
        exposure = _clamp(min(exposure, headroom), -2.0, 2.0)
        clipped = float(np.mean(luma > 0.98))
        if clipped > 0.015:
            exposure = _clamp(exposure - min(1.0, clipped * 6.0), -2.0, 2.0)

        contrast_range = p95 - p5
        contrast = _clamp((0.62 - contrast_range) * 0.65, -0.35, 0.35)

        mid = (luma > 0.15) & (luma < 0.85)
        temperature, tint = _white_balance(rgb, mid)

        span = (rgb.max(axis=2) - rgb.min(axis=2)).astype(np.float32) / 255.0
        colorfulness = float(span.mean())
        if median < 0.24:
            # Absolute chroma is tiny in a dark frame even when the color is
            # strong. Compare it with the brightness so moss and night color
            # are not lifted into a green or blue grade. A gray frame still lifts.
            relative = span / np.maximum(luma, np.float32(0.05))
            if float(np.median(relative)) > 0.28:
                exposure = min(exposure, 0.4)
                # Mid-gray contrast on a dark color picture crushes it and
                # leaves only the strongest hue.
                contrast = min(contrast, 0.05)
        if colorfulness < 0.04:
            saturation = 0.0
            vibrance = 0.0
        else:
            saturation = _clamp((0.18 - colorfulness) * 0.45, -0.2, 0.16)
            vibrance = _clamp(saturation * 0.6, -0.15, 0.12)

        shadows = _clamp((0.05 - p5) * 2.2, 0.0, 0.3) if p5 < 0.05 else 0.0
        highlights = -_clamp((p95 - 0.90) * 1.4, 0.0, 0.35) if p95 > 0.90 else 0.0

        mood = _mood(exposure, temperature, contrast)
        rationale = (
            f"Median luminance is {median:.0%} of the preview. "
            f"Suggested {exposure:+.2f} EV, contrast {contrast:+.2f}, "
            f"warmth {temperature:+.2f}."
        )
        return Atmosphere(
            exposure_ev=exposure,
            contrast=contrast,
            temperature_shift=temperature,
            tint=tint,
            saturation=saturation,
            shadows=shadows,
            highlights=highlights,
            vibrance=vibrance,
            mood=mood,
            rationale=rationale,
            source=self.name,
        )


def _mood(exposure: float, temperature: float, contrast: float) -> str:
    if exposure > 0.4:
        tone = "dark"
    elif exposure < -0.4:
        tone = "bright"
    else:
        tone = "balanced"
    if temperature > 0.2:
        color = "cool"
    elif temperature < -0.2:
        color = "warm"
    else:
        color = "neutral"
    if contrast > 0.15:
        shape = "flat"
    elif contrast < -0.12:
        shape = "contrasty"
    else:
        shape = "moderate"
    return f"{tone}, {color}, {shape} light"
