"""What Darktable should open after a run.

The masked TIFF holds the local grades and the accepted features.
The style file is the global grade only. The camera file is left alone.
"""

from __future__ import annotations

from pathlib import Path


def build_handoff(recipe: dict) -> dict:
    """A small record the panel and the Lua script can both follow."""

    render = recipe.get("render") if isinstance(recipe.get("render"), dict) else {}
    style = recipe.get("style") if isinstance(recipe.get("style"), dict) else {}
    info = recipe.get("input") if isinstance(recipe.get("input"), dict) else {}
    resolution = "full" if str(info.get("resolution") or "") == "full" else "half"
    included: list[str] = []
    for key in ("features_committed", "features"):
        for item in recipe.get(key) or []:
            name = str(item).strip()
            if name and name not in included:
                included.append(name)
    masked = str(render.get("file") or "")
    picture_name = Path(masked).name or "edited.tif"
    size = "full resolution" if resolution == "full" else "half resolution"
    if included:
        body = f"It includes {', '.join(included)}. "
    else:
        body = "This frame kept the measured grade. "
    sentence = (
        f"The masked picture is {picture_name} (camera white balance, {size}). "
        f"{body}"
        "The original file stays unchanged. "
        "The style file is the global grade only and does not contain these masks."
    )
    return {
        "version": 1,
        "open": "masked-picture",
        "masked_file": masked,
        "style_file": str(style.get("file") or ""),
        "style_role": "global-grade",
        "apply_style_to_source": False,
        "source_unchanged": True,
        "white_balance": "camera",
        "resolution": resolution,
        "features": included,
        "sentence": sentence,
    }
