"""Approximate a recipe with Pillow.

Darktable is not required, and this is not Darktable's render. Exposure,
contrast, warmth, and the vignette are applied in order so a preview can
show which way the numbers moved.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ai_dt.schema import Atmosphere


def sample_image(name: str) -> Image.Image:
    """Synthetic previews the local page can analyze without a photo library."""

    if name == "portrait":
        image = Image.new("RGB", (880, 640), (32, 30, 34))
        draw = ImageDraw.Draw(image)
        draw.ellipse((250, 70, 640, 560), fill=(206, 154, 122))
        draw.ellipse((340, 210, 400, 270), fill=(28, 20, 20))
        draw.ellipse((500, 210, 560, 270), fill=(28, 20, 20))
        return image
    if name == "architecture":
        image = Image.new("RGB", (880, 640), (168, 186, 196))
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 360, 840, 640), fill=(150, 146, 138))
        draw.rectangle((180, 80, 700, 560), fill=(214, 208, 196))
        for x in range(210, 680, 36):
            draw.rectangle((x, 110, x + 18, 520), fill=(36, 40, 46))
        for y in range(140, 500, 48):
            draw.rectangle((190, y, 690, y + 8), fill=(36, 40, 46))
        return image
    if name == "night":
        # Dark enough to read as night, light enough that the measured
        # exposure does not swamp the night grade in the preview.
        image = Image.new("RGB", (880, 640), (78, 96, 148))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 390, 880, 640), fill=(92, 84, 74))
        for origin in ((90, 160), (240, 200), (470, 130), (660, 210)):
            x, y = origin
            draw.rectangle((x, y, x + 140, y + 230), fill=(58, 66, 92))
            draw.rectangle((x + 18, y + 24, x + 48, y + 58), fill=(240, 196, 110))
            draw.rectangle((x + 70, y + 24, x + 100, y + 58), fill=(230, 170, 90))
            draw.rectangle((x + 18, y + 90, x + 48, y + 124), fill=(220, 160, 80))
            draw.rectangle((x + 90, y + 140, x + 120, y + 180), fill=(236, 188, 100))
        return image
    if name == "blue":
        image = Image.new("RGB", (880, 640), (92, 128, 168))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 400, 880, 640), fill=(70, 108, 140))
        draw.ellipse((280, 120, 620, 520), fill=(150, 176, 198))
        return image
    raise ValueError(f"Unknown sample {name!r}")


def _as_atmosphere(data: dict) -> Atmosphere:
    fields = {key: data[key] for key in Atmosphere.__dataclass_fields__ if key in data}
    return Atmosphere(**fields)


def grade_array(
    pixels: np.ndarray,
    atmosphere: Atmosphere,
    vignette: dict | None = None,
) -> np.ndarray:
    """Apply the global grade to a float RGB image in the range 0 to 1."""

    arr = np.array(pixels, dtype=np.float32, copy=True)
    arr *= np.float32(2.0 ** float(atmosphere.exposure_ev))
    arr = (arr - 0.5) * np.float32(1.0 + float(atmosphere.contrast)) + 0.5
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    shadow = np.clip((0.45 - luma) / 0.45, 0.0, 1.0)
    highlight = np.clip((luma - 0.55) / 0.45, 0.0, 1.0)
    arr = arr + np.float32(atmosphere.shadows) * 0.45 * shadow[..., None]
    arr = arr + np.float32(atmosphere.highlights) * 0.28 * highlight[..., None]
    warmth = np.float32(atmosphere.temperature_shift)
    arr[..., 0] *= np.float32(1.0) + 0.22 * warmth
    arr[..., 2] *= np.float32(1.0) - 0.22 * warmth
    tint = np.float32(atmosphere.tint)
    arr[..., 0] *= np.float32(1.0) + 0.08 * tint
    arr[..., 1] *= np.float32(1.0) - 0.10 * tint
    arr[..., 2] *= np.float32(1.0) + 0.06 * tint
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    saturation = np.float32(1.0 + float(atmosphere.saturation) + 0.5 * float(atmosphere.vibrance))
    arr = luma[..., None] + (arr - luma[..., None]) * saturation
    if vignette is not None:
        arr = _vignette(arr, vignette)
    return np.clip(arr, 0.0, 1.0)


def approximate(image: Image.Image, atmosphere: Atmosphere, vignette: dict | None = None) -> Image.Image:
    """Return a new RGB image with the recipe's global moves roughly applied."""

    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    graded = grade_array(arr, atmosphere, vignette)
    return Image.fromarray((graded * 255.0).astype(np.uint8), mode="RGB")


def _vignette(arr: np.ndarray, vignette: dict) -> np.ndarray:
    height, width = arr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    center_x = (float(vignette.get("center_x", 0.0)) + 1.0) * 0.5 * width
    center_y = (float(vignette.get("center_y", 0.0)) + 1.0) * 0.5 * height
    longest = float(max(width, height))
    radius = max(8.0, float(vignette.get("scale", 40.0)) / 100.0 * longest)
    whratio = max(0.2, float(vignette.get("whratio", 1.0)))
    dx = (xx - center_x) / radius
    dy = (yy - center_y) / (radius * whratio)
    distance = np.sqrt(dx * dx + dy * dy)
    falloff = max(0.2, float(vignette.get("falloff_scale", 40.0)) / max(float(vignette.get("scale", 40.0)), 1.0))
    edge = np.clip((distance - 0.65) / falloff, 0.0, 1.0)
    edge = edge * edge * (3.0 - 2.0 * edge)
    gain = 1.0 + float(vignette.get("brightness", -0.2)) * edge
    return arr * gain.astype(np.float32)[..., None]


def vignette_from_modules(modules: list[dict]) -> dict | None:
    for module in modules:
        if module.get("operation") == "vignette":
            params = module.get("params")
            if isinstance(params, dict):
                return params
    return None


def render_preview(image: Image.Image, recipe: dict) -> Image.Image:
    atmosphere = _as_atmosphere(recipe["analysis"]["atmosphere"])
    vignette = vignette_from_modules(recipe.get("modules") or [])
    return approximate(image, atmosphere, vignette)
