"""Find the features a guide wants kept sharp, without a detection model.

The compact subject is not the whole picture. A bird and the branch it
stands on are one subject: the creature stays sharp, and so does the thick
woody support under its feet. Fine green behind them does not. The same
idea, more quietly, keeps a chair under a person or an object in their
hands. Nothing here downloads a model.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# A body wider than this, relative to its height, is a branch or a horizon
# rather than the creature. The padded warm box of a full-width branch
# still trips the bar test below.
ELONGATED_ASPECT = 2.8
_MAX_SIDE = 360


def is_elongated(
    box: tuple[float, float, float, float],
    shape: tuple[int, ...] | None = None,
) -> bool:
    """True when a box is a bar across the frame, not a compact body.

    Aspect is measured in pixels when the frame shape is known. A tall
    subject on a wide frame is not a branch just because the normalized
    box looks skinny.
    """

    bw = float(box[2])
    bh = float(box[3])
    if bw <= 1e-6 or bh <= 1e-6:
        return True
    if shape is not None and len(shape) >= 2 and shape[0] > 0 and shape[1] > 0:
        pixel_h = bh * float(shape[0])
        pixel_w = bw * float(shape[1])
        aspect = max(pixel_w / max(pixel_h, 1e-6), pixel_h / max(pixel_w, 1e-6))
    else:
        aspect = max(bw / bh, bh / bw)
    if aspect >= ELONGATED_ASPECT:
        return True
    if bw >= 0.9 and bh <= 0.5:
        return True
    if bh >= 0.9 and bw <= 0.5:
        return True
    return False


def creature_box(
    pixels: np.ndarray,
) -> tuple[tuple[float, float, float, float], float, float] | None:
    """A compact body that is not the green background and not the perch.

    Brown, woody, elongated structure is left out on purpose so a branch
    the animal is standing on is not mistaken for the animal.
    """

    small = _small(pixels)
    candidate = _creature_pixels(small)
    thick = _thick(candidate, 0.016, 0.5)
    best: tuple[tuple[float, float, float, float], float, float] | None = None
    best_count = 0
    for component in _components(thick):
        measured = _compact_region(component, small)
        if measured is None:
            continue
        count = int(component.sum())
        if count > best_count:
            best = measured
            best_count = count
    return best


def extend_animal(pixels: np.ndarray, subject: np.ndarray) -> np.ndarray:
    """Grow the animal mask along its own color, so a long tail stays sharp.

    Green and woody pixels are not eligible. A growth that swallows the
    frame is discarded and the original mask is kept.
    """

    if subject is None or float(subject.max()) <= 0.0:
        return subject
    small = _small(pixels)
    seeds = _resize_mask(subject, small.shape) > np.float32(0.45)
    if int(seeds.sum()) < 20:
        return subject
    median = np.median(small[seeds], axis=0).astype(np.float32)
    distance = np.sqrt(np.sum((small - median) ** 2, axis=2))
    candidate = (distance < np.float32(0.2)) & _creature_pixels(small)
    thick = _thick(candidate, 0.01, 0.42)
    extra = np.zeros(thick.shape, dtype=bool)
    for component in _components(thick):
        if np.any(component & seeds):
            extra |= component
    frame = extra.size
    if int(extra.sum()) < 20 or int(extra.sum()) > int(0.4 * frame):
        return subject
    soft = _feather(extra, small.shape[0], small.shape[1])
    grown = _resize_mask(soft, subject.shape)
    return np.maximum(subject, grown)


def perch_mask(pixels: np.ndarray, subject: np.ndarray) -> np.ndarray | None:
    """The branch, rock, hand, or post the subject is actually touching.

    From the subject, keep nearby pixels that are brown and woody, thick
    enough to be a support rather than a twig, and connected to the body.
    The region stops where the picture becomes fine green. A warm patch
    that is only the animal's own color, and does not extend outside it,
    is not a perch.
    """

    return _attached_mask(pixels, subject, kind="perch")


def seat_mask(pixels: np.ndarray, subject: np.ndarray) -> np.ndarray | None:
    """A chair or other support directly under a person, when it is clearly separate."""

    return _attached_mask(pixels, subject, kind="seat")


def held_mask(pixels: np.ndarray, subject: np.ndarray) -> np.ndarray | None:
    """A compact object overlapping a person and sticking out of the body."""

    return _attached_mask(pixels, subject, kind="held")


def _attached_mask(
    pixels: np.ndarray,
    subject: np.ndarray,
    *,
    kind: str,
) -> np.ndarray | None:
    if subject is None or float(np.max(subject)) <= 0.0:
        return None
    small_pixels = _small(pixels)
    small_subject = _resize_mask(subject, small_pixels.shape)
    if float(small_subject.max()) < 0.35:
        return None
    if kind == "perch":
        candidate = _woody(small_pixels)
        min_outside = 0.012
        min_aspect_unless_outside = 2.0
        outside_ratio_bar = 0.55
        area_limit = (0.0, 0.42)
        contact = small_subject > np.float32(0.22)
    elif kind == "seat":
        candidate = _seat_pixels(small_pixels, small_subject)
        min_outside = 0.02
        min_aspect_unless_outside = 0.0
        outside_ratio_bar = 0.45
        area_limit = (0.02, 0.22)
        contact = small_subject > np.float32(0.3)
    else:
        candidate = _held_pixels(small_pixels, small_subject)
        min_outside = 0.004
        min_aspect_unless_outside = 0.0
        outside_ratio_bar = 0.35
        area_limit = (0.004, 0.06)
        contact = small_subject > np.float32(0.35)
    thick = _thick(candidate, 0.012 if kind == "perch" else 0.014, 0.46)
    if not thick.any():
        return None
    radius = max(2, int(round(0.02 * min(small_pixels.shape[:2]))))
    touching = _dilate(contact, radius) & thick
    if not touching.any():
        return None
    chosen = np.zeros(thick.shape, dtype=bool)
    frame = thick.size
    for component in _components(thick):
        if not np.any(component & touching):
            continue
        total = int(component.sum())
        outside = int((component & (small_subject < np.float32(0.25))).sum())
        if outside < min_outside * frame or total > area_limit[1] * frame:
            continue
        if total < area_limit[0] * frame:
            continue
        aspect = _aspect(component)
        outside_ratio = outside / float(total)
        if aspect < min_aspect_unless_outside and outside_ratio < outside_ratio_bar:
            continue
        if kind == "held" and _reaches_bottom(component):
            continue
        if kind == "seat" and _touches_top(component):
            continue
        chosen |= component
    if int(chosen.sum()) < 20:
        return None
    soft = _feather(chosen, small_pixels.shape[0], small_pixels.shape[1])
    full = _resize_mask(soft, subject.shape)
    full = _blur(full, max(0.8, 0.006 * min(subject.shape[:2])))
    if float(full.max()) < 0.25:
        return None
    return full


def _creature_pixels(pixels: np.ndarray) -> np.ndarray:
    background = _corner_median(pixels)
    distance = np.sqrt(np.sum((pixels - background) ** 2, axis=2))
    return (distance > np.float32(0.16)) & ~_green(pixels) & ~_woody(pixels)


def _seat_pixels(pixels: np.ndarray, subject: np.ndarray) -> np.ndarray:
    background = _corner_median(pixels)
    distance = np.sqrt(np.sum((pixels - background) ** 2, axis=2))
    rows = np.arange(pixels.shape[0], dtype=np.float32)[:, None]
    body = subject > np.float32(0.45)
    if not body.any():
        return np.zeros(subject.shape, dtype=bool)
    y_mid = float(np.median(np.nonzero(body)[0]))
    below = rows > y_mid
    candidate = (distance > np.float32(0.16)) & ~_green(pixels) & (subject < np.float32(0.22)) & below
    top = max(1, pixels.shape[0] // 10)
    candidate[:top, :] = False
    return candidate


def _held_pixels(pixels: np.ndarray, subject: np.ndarray) -> np.ndarray:
    background = _corner_median(pixels)
    distance = np.sqrt(np.sum((pixels - background) ** 2, axis=2))
    # Skin and the woody chair stay out. A held object is some other color,
    # overlapping the person and continuing outside the body.
    return (distance > np.float32(0.18)) & ~_green(pixels) & ~_skin(pixels) & ~_woody(pixels) & (
        subject < np.float32(0.85)
    )


def _green(pixels: np.ndarray) -> np.ndarray:
    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    return (green > red + np.float32(0.04)) & (green > blue + np.float32(0.02))


def _woody(pixels: np.ndarray) -> np.ndarray:
    """Brown bark: red ahead of blue, not the green behind it."""

    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    return (
        (red > np.float32(0.18))
        & (red > blue + np.float32(0.06))
        & (red + np.float32(0.02) > green)
        & (green + np.float32(0.02) > blue * np.float32(0.5))
        & ~_green(pixels)
    )


def _skin(pixels: np.ndarray) -> np.ndarray:
    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    spread = np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)
    return (
        (red > np.float32(0.38))
        & (green > np.float32(0.18))
        & (blue > np.float32(0.28))
        & (red > green + np.float32(0.04))
        & (red > blue)
        & (blue > red * np.float32(0.45))
        & (spread > np.float32(0.08))
    )


def _corner_median(pixels: np.ndarray) -> np.ndarray:
    height, width = pixels.shape[:2]
    band_y = max(1, height // 12)
    band_x = max(1, width // 12)
    corners = np.concatenate(
        (
            pixels[:band_y, :band_x].reshape(-1, 3),
            pixels[:band_y, -band_x:].reshape(-1, 3),
            pixels[-band_y:, :band_x].reshape(-1, 3),
            pixels[-band_y:, -band_x:].reshape(-1, 3),
        ),
        axis=0,
    )
    return np.median(corners, axis=0).astype(np.float32)


def _compact_region(
    component: np.ndarray,
    pixels: np.ndarray,
) -> tuple[tuple[float, float, float, float], float, float] | None:
    count = int(component.sum())
    height, width = component.shape
    frame = height * width
    fraction = count / float(frame)
    if fraction < 0.012 or fraction > 0.38:
        return None
    # A facade is a grid of edges. A creature is a solid body with an outline.
    if _edge_mean(pixels, component) > 0.16:
        return None
    ys, xs = np.nonzero(component)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if x0 <= 1 and y0 <= 1 and x1 >= width - 2 and y1 >= height - 2:
        return None
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    fill = count / float(max(1, box_w * box_h))
    if fill < 0.28:
        return None
    norm = (x0 / width, y0 / height, box_w / width, box_h / height)
    if is_elongated(norm, component.shape):
        return None
    pad_x = int(box_w * 0.12)
    pad_y = int(box_h * 0.14)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width - 1, x1 + pad_x)
    y1 = min(height - 1, y1 + pad_y)
    box = (x0 / width, y0 / height, (x1 - x0 + 1) / width, (y1 - y0 + 1) / height)
    return box, float(fill), float(fraction)


def _edge_mean(pixels: np.ndarray, mask: np.ndarray) -> float:
    luma = 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]
    gy = np.zeros_like(luma)
    gx = np.zeros_like(luma)
    gy[1:, :] = np.abs(luma[1:, :] - luma[:-1, :])
    gx[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
    edge = gx + gy
    if not mask.any():
        return 0.0
    return float(edge[mask].mean())


def _aspect(component: np.ndarray) -> float:
    ys, xs = np.nonzero(component)
    if ys.size == 0:
        return 1.0
    box_w = int(xs.max()) - int(xs.min()) + 1
    box_h = int(ys.max()) - int(ys.min()) + 1
    if box_w <= 0 or box_h <= 0:
        return 1.0
    return max(box_w / float(box_h), box_h / float(box_w))


def _reaches_bottom(component: np.ndarray) -> bool:
    ys = np.nonzero(component)[0]
    if ys.size == 0:
        return False
    return int(ys.max()) >= int(component.shape[0] * 0.88)


def _touches_top(component: np.ndarray) -> bool:
    ys = np.nonzero(component)[0]
    if ys.size == 0:
        return False
    return int(ys.min()) <= int(component.shape[0] * 0.08)


def _thick(candidate: np.ndarray, radius_scale: float, level: float) -> np.ndarray:
    """Keep a solid support and drop needles, twigs, and one-pixel lines.

    A blur narrower than the branch stays high inside the wood and falls
    apart on the fine background. That is the stand-in for local contrast.
    """

    if candidate.dtype != np.float32:
        values = candidate.astype(np.float32)
    else:
        values = candidate
    if float(values.max()) <= 0.0:
        return np.zeros(values.shape, dtype=bool)
    height, width = values.shape
    radius = max(1.05, radius_scale * min(height, width))
    return _blur(values, radius) > np.float32(level)


def _feather(chosen: np.ndarray, height: int, width: int) -> np.ndarray:
    """A soft edge so the kept feature is not a paper cutout."""

    solid = _dilate(chosen, 1).astype(np.float32)
    radius = max(1.3, 0.012 * min(height, width))
    soft = _blur(solid, radius)
    core = chosen.astype(np.float32)
    return np.maximum(soft, core)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected pieces of a boolean mask. Empty when nothing is set."""

    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return []
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    ys, xs = np.nonzero(mask)
    pieces: list[np.ndarray] = []
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        coords_y: list[int] = []
        coords_x: list[int] = []
        while stack:
            y, x = stack.pop()
            coords_y.append(y)
            coords_x.append(x)
            y_lo = y - 1 if y else 0
            y_hi = y + 2 if y + 1 < height else height
            x_lo = x - 1 if x else 0
            x_hi = x + 2 if x + 1 < width else width
            for ny in range(y_lo, y_hi):
                row_seen = seen[ny]
                row_mask = mask[ny]
                for nx in range(x_lo, x_hi):
                    if row_seen[nx] or not row_mask[nx]:
                        continue
                    row_seen[nx] = True
                    stack.append((ny, nx))
        piece = np.zeros((height, width), dtype=bool)
        piece[coords_y, coords_x] = True
        pieces.append(piece)
    return pieces


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return np.asarray(mask).astype(bool)
    size = radius * 2 + 1
    image = Image.fromarray(np.asarray(mask).astype(np.uint8) * 255, mode="L")
    image = image.filter(ImageFilter.MaxFilter(size=size))
    return np.asarray(image) > 127


def _small(pixels: np.ndarray) -> np.ndarray:
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    longest = max(height, width)
    clipped = np.clip(pixels.astype(np.float32, copy=False), 0.0, 1.0)
    if longest <= _MAX_SIDE:
        return clipped
    scale = _MAX_SIDE / float(longest)
    image = Image.fromarray(np.rint(clipped * 255.0).astype(np.uint8), mode="RGB")
    image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BOX)
    return np.asarray(image).astype(np.float32) / np.float32(255.0)


def _resize_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    image = Image.fromarray(np.rint(np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image).astype(np.float32) / np.float32(255.0)


def _blur(values: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    image = image.filter(ImageFilter.GaussianBlur(radius=max(0.4, float(radius))))
    return np.asarray(image).astype(np.float32) / np.float32(255.0)
