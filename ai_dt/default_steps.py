"""Standard editor steps every plan starts with, measured from the photo.

1. ``default_clipping``: a tone equalizer (v2, same in darktable 5.4 and 5.6)
   whose bands are found by iterating a preview estimate until the clipped
   highlights and crushed shadows almost vanish, the way an editor drags a
   slider in the clipping view. A contrast floor keeps it from flattening.
2. ``default_dehaze``: haze removal (v3), only when the dark-channel haze
   score says so; strength scales with the score.
3. ``mask_sharpen_subject``: sharpen (v1) inside the subject mask with a
   threshold. Reduced on noisy photos, skipped on very noisy ones.

They are ordinary queue steps (Accept, Skip, Back) and go first.
"""

from __future__ import annotations

import struct

import numpy as np

CLIP_HI = 250 / 255.0
CLIP_LO = 5 / 255.0
CLIP_HI_TRIGGER = 0.005  # 0.5 % of pixels at or above 250
CLIP_LO_TRIGGER = 0.01  # 1 % at or below 5
CONTRAST_FLOOR = 0.85  # keep at least 85 % of the p5..p95 range
HAZE_TRIGGER = 0.2
NOISE_REDUCE = 2.5  # sigma in 0..255 units
NOISE_SKIP = 5.0

CLIPPING_ID = "default_clipping"
DEHAZE_ID = "default_dehaze"
SHARPEN_ID = "mask_sharpen_subject"
DEFAULT_IDS = (CLIPPING_ID, DEHAZE_ID, SHARPEN_ID)
GLOBAL_IDS = frozenset({CLIPPING_ID, DEHAZE_ID})
NAMES = {
    CLIPPING_ID: "Recover clipping",
    DEHAZE_ID: "Remove haze",
    SHARPEN_ID: "Sharpen the subject",
}
TITLES = {
    CLIPPING_ID: "Pull clipped highlights and lift crushed shadows",
    DEHAZE_ID: "Remove the haze",
    SHARPEN_ID: "Sharpen the main subject",
}
TONE_EQUAL_VERSION = 2
HAZE_VERSION = 3
SHARPEN_VERSION = 1

# Band centres in EV (-8 .. 0) of the tone equalizer, for the estimate.
_BAND_EV = np.arange(-8.0, 1.0, 1.0)


def _small(image, width: int = 256) -> np.ndarray:
    rgb = image.convert("RGB")
    if rgb.width > width:
        rgb = rgb.resize((width, max(1, round(rgb.height * width / rgb.width))))
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def clipped_fractions(luma: np.ndarray) -> tuple[float, float]:
    return float(np.mean(luma >= CLIP_HI)), float(np.mean(luma <= CLIP_LO))


def _bands(shadows: float, highlights: float) -> list[float]:
    """Nine band offsets in EV: lift -8..-4 EV (blacks most), pull -2..0 EV."""

    return [
        shadows, shadows, shadows * 0.9, shadows * 0.6, shadows * 0.3,
        0.0, highlights * 0.4, highlights * 0.8, highlights,
    ]


def simulate(luma: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
    """Cheap preview estimate of the tone equalizer on display luma."""

    lin = np.clip(luma, 1e-4, 1.0) ** 2.2
    # A clipped pixel is at least as bright as white; treat it as +0.5 EV over.
    lin = np.where(luma >= 254.5 / 255.0, lin * 1.41, lin)
    ev = np.log2(lin)
    gain = np.interp(ev, _BAND_EV, _bands(shadows, highlights))
    return np.clip((lin * 2.0 ** gain) ** (1 / 2.2), 0.0, 1.0)


def _spread(luma: np.ndarray) -> float:
    low, high = np.percentile(luma, [5, 95])
    return float(high - low)


def solve_clipping(luma: np.ndarray, peak: np.ndarray | None = None) -> dict | None:
    """Search shadow lift and highlight pull that shrink clipping most.

    None when the photo is not clipped (below both triggers).
    """

    # Highlights clip per channel (the brightest one), shadows on luma.
    peak = luma if peak is None else peak

    def fractions(lum, pk):
        return clipped_fractions(pk)[0], clipped_fractions(lum)[1]

    hi, lo = fractions(luma, peak)
    if hi < CLIP_HI_TRIGGER and lo < CLIP_LO_TRIGGER:
        return None
    base_spread = max(_spread(luma), 1e-3)
    best = (hi + lo, 0.0, 0.0, hi, lo)
    shadow_grid = np.arange(0.0, 1.51, 0.1) if lo >= CLIP_LO_TRIGGER else [0.0]
    high_grid = -np.arange(0.0, 1.21, 0.1) if hi >= CLIP_HI_TRIGGER else [0.0]
    for shadows in shadow_grid:
        for highlights in high_grid:
            out = simulate(luma, float(shadows), float(highlights))
            if _spread(out) < CONTRAST_FLOOR * base_spread:
                continue
            new_hi, new_lo = fractions(out, simulate(peak, float(shadows), float(highlights)))
            # Smallest move wins among near-equal results.
            score = new_hi + new_lo + 0.002 * (abs(shadows) + abs(highlights))
            if score < best[0] - 1e-6:
                best = (score, float(shadows), float(highlights), new_hi, new_lo)
    _score, shadows, highlights, new_hi, new_lo = best
    if abs(shadows) < 0.05 and abs(highlights) < 0.05:
        return None
    return {
        "shadows": round(shadows, 2),
        "highlights": round(highlights, 2),
        "clipped_before": round(hi, 4),
        "crushed_before": round(lo, 4),
        "clipped_after": round(new_hi, 4),
        "crushed_after": round(new_lo, 4),
    }


def haze_score(rgb: np.ndarray) -> float:
    """0..1: a bright dark channel, low saturation and low contrast mean haze."""

    from numpy.lib.stride_tricks import sliding_window_view

    channel = np.pad(rgb.min(axis=2), 3, mode="edge")
    dark = sliding_window_view(channel, (7, 7)).min(axis=(2, 3))
    dark_level = float(np.median(dark))
    top = rgb[: max(1, rgb.shape[0] // 2)]
    mx, mn = top.max(axis=2), top.min(axis=2)
    saturation = float(np.mean((mx - mn) / np.maximum(mx, 1e-3)))
    spread = _spread(_luma(rgb))
    score = (
        np.clip((dark_level - 0.12) / 0.25, 0.0, 1.0) * 0.6
        + np.clip((0.3 - saturation) / 0.2, 0.0, 1.0) * 0.2
        + np.clip((0.55 - spread) / 0.3, 0.0, 1.0) * 0.2
    )
    # Haze lifts the dark channel; without that it is just a soft photo.
    if dark_level < 0.12:
        score = min(score, 0.1)
    return round(float(score), 3)


def noise_sigma(luma: np.ndarray) -> float:
    """Immerkaer's fast noise estimate, in 0..255 units."""

    height, width = luma.shape
    if height < 3 or width < 3:
        return 0.0
    y = luma.astype(np.float64) * 255.0
    kernel = ((1, -2, 1), (-2, 4, -2), (1, -2, 1))
    response = sum(
        kernel[i][j] * y[i : height - 2 + i, j : width - 2 + j] for i in range(3) for j in range(3)
    )
    response = np.abs(response)
    sigma = np.sum(response) * np.sqrt(0.5 * np.pi) / (6.0 * (width - 2) * (height - 2))
    return round(float(sigma), 2)


def measure(image, *, iso: float | None = None) -> dict:
    """All defaults for one photo: ``{id: values}``, only the steps it needs."""

    rgb = _small(image)
    luma = _luma(rgb)
    found: dict = {}
    clip = solve_clipping(luma, rgb.max(axis=2))
    if clip:
        found[CLIPPING_ID] = clip
    haze = haze_score(rgb)
    if haze >= HAZE_TRIGGER:
        found[DEHAZE_ID] = {
            "score": haze,
            "strength": round(0.1 + 0.3 * min(1.0, haze), 2),
            "distance": round(0.15 + 0.15 * min(1.0, haze), 2),
        }
    # Noise on the full-size preview, before downsizing hides it.
    full = _luma(_small(image, width=1024))
    sigma = noise_sigma(full)
    noisy = sigma >= NOISE_REDUCE or (iso is not None and iso >= 3200)
    if sigma < NOISE_SKIP and not (iso is not None and iso >= 12800):
        found[SHARPEN_ID] = {"noise": sigma, "action": "hold" if noisy else "lift"}
    return found


def tone_blob(shadows: float, highlights: float) -> bytes:
    bands = [max(-1.5, min(1.5, band)) for band in _bands(shadows, highlights)]
    return struct.pack(
        "<15f3i", *bands, 5.0, 1.414213562, 1.0, 0.0, 0.0, 0.0, 4, 4, 1
    )


def haze_blob(strength: float, distance: float) -> bytes:
    # dt_iop_hazeremoval_params_t v3: strength, distance, compatibility_mode.
    return struct.pack("<ffi", max(-1.0, min(1.0, strength)), max(0.0, min(1.0, distance)), 0)


def sharpen_blob(radius: float, amount: float, threshold: float) -> bytes:
    # dt_iop_sharpen_params_t v1: radius, amount, threshold.
    return struct.pack("<fff", radius, amount, threshold)


def style_for(step_id: str, values: dict | None) -> str | None:
    """The .dtstyle for a global default step, or None."""

    from ai_dt.mapping import _plugin, render_style

    values = values if isinstance(values, dict) else {}
    if step_id == CLIPPING_ID:
        if not values:
            return None
        blob = tone_blob(float(values.get("shadows", 0)), float(values.get("highlights", 0)))
        plugin = _plugin(0, "toneequal", TONE_EQUAL_VERSION, blob, NAMES[step_id], multi_priority=8)
    elif step_id == DEHAZE_ID:
        if not values:
            return None
        blob = haze_blob(float(values.get("strength", 0.2)), float(values.get("distance", 0.2)))
        plugin = _plugin(0, "hazeremoval", HAZE_VERSION, blob, NAMES[step_id], multi_priority=8)
    else:
        return None
    return render_style("AI accepted step", f"Accepted step: {NAMES[step_id]}.", [plugin])


def queue_items(found: dict) -> list[dict]:
    items = []
    for ident in DEFAULT_IDS:
        if ident not in found:
            continue
        action = str(found[ident].get("action") or "lift")
        items.append({"id": ident, "action": action, "title": TITLES[ident], "default": True})
    return items


def add_defaults(plan: dict, found: dict, skip: list[str] | tuple = ()) -> dict:
    """Put the default steps first on the queue; never twice, never over an applied one."""

    blocked = set(skip) | {str(name) for name in (plan.get("features_done") or [])}
    queue = [dict(item) for item in (plan.get("step_queue") or []) if isinstance(item, dict)]
    present = {str(item.get("id") or "") for item in queue}
    if not queue and plan.get("features_next"):
        queue = [
            {
                "id": str(plan["features_next"]),
                "action": str(plan.get("features_next_action") or ""),
                "title": str(plan.get("features_next_title") or ""),
            }
        ]
        present = {queue[0]["id"]}
    head = [item for item in queue_items(found) if item["id"] not in blocked and item["id"] not in present]
    if not head:
        return plan
    queue = head + queue
    plan["default_steps"] = {key: value for key, value in found.items()}
    plan["step_queue"] = queue
    plan["features_path"] = [str(item.get("id") or "") for item in queue]
    plan["features_next"] = queue[0]["id"]
    plan["features_next_title"] = queue[0]["title"]
    plan["features_next_action"] = queue[0]["action"]
    return plan
