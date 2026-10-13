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


def is_face_bar(box: tuple[float, float, float, float]) -> bool:
    """True when a skin box is a strip across the frame, not a head.

    Warm ground along a horizon can pass a loose skin test. A head is not
    several times wider than it is tall.
    """

    bw = float(box[2])
    bh = float(box[3])
    if bw <= 1e-6 or bh <= 1e-6:
        return True
    return bw / bh >= 3.2


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


def reflected_subject(
    pixels: np.ndarray,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None:
    """A compact body standing in dark water, and the narrow mirror under it.

    Lily pads and pale flowers are brighter than the bird and can look like
    a face. The reflection is a thin bright streak in the dark water, and
    the body is the neutral mass directly above that streak. Open water that
    is green, or a bright tail against foliage, is not a mirror. A wide echo
    of a house or a mountain is a water reflection, not this body.
    """

    small = _small(pixels)
    height, width = small.shape[:2]
    if min(height, width) < 48:
        return None
    luma = 0.2126 * small[..., 0] + 0.7152 * small[..., 1] + 0.0722 * small[..., 2]
    row = luma.mean(axis=1)
    band_top = int(0.78 * height)
    band_bottom = int(0.94 * height)
    if band_bottom <= band_top + 4:
        return None
    bottom = small[band_top:band_bottom]
    bottom_luma = float(row[band_top:band_bottom].mean())
    middle_luma = float(row[int(0.42 * height) : int(0.68 * height)].mean())
    if bottom_luma > 0.30 or bottom_luma > middle_luma - 0.05:
        return None
    if float(_green(bottom).mean()) > 0.28:
        return None
    chroma = np.max(np.abs(bottom - luma[band_top:band_bottom, :, None]), axis=2)
    if float(chroma.mean()) > 0.08:
        return None

    streak_top = int(0.72 * height)
    streak_bottom = int(0.93 * height)
    column = luma[streak_top:streak_bottom].mean(axis=0)
    smooth = np.convolve(column, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    offset = max(6, width // 16)
    score = smooth - 0.5 * (np.roll(smooth, offset) + np.roll(smooth, -offset))
    score[:offset] = 0.0
    score[-offset:] = 0.0
    center = int(np.argmax(score))
    peak = float(score[center])
    if peak < 0.028:
        return None
    half_score = peak * 0.55
    left_edge = center
    while left_edge > 0 and float(score[left_edge]) > half_score:
        left_edge -= 1
    right_edge = center
    while right_edge < width - 1 and float(score[right_edge]) > half_score:
        right_edge += 1
    if (right_edge - left_edge) / float(width) > 0.16:
        return None

    water_level = float(np.median(luma[streak_top:streak_bottom]))
    # The edge of a wide mirror also makes a narrow peak. A bird's streak is
    # the bright part of the water, not a sliver of a house or a ridge.
    if float(np.mean(column > water_level + 0.03)) > 0.2:
        return None
    half_w = max(3, int(round(0.04 * width)))
    x0 = max(0, center - half_w)
    x1 = min(width, center + half_w)
    mirror = luma[streak_top:streak_bottom, x0:x1] > np.float32(water_level + 0.03)
    if int(mirror.sum()) < 24 or float(mirror.mean()) < 0.12:
        return None
    mirror_rows = np.nonzero(mirror.any(axis=1))[0]
    if mirror_rows.size < int(0.06 * height):
        return None
    reflection = (
        x0 / width,
        (streak_top + int(mirror_rows.min())) / height,
        (x1 - x0) / width,
        (int(mirror_rows.max()) - int(mirror_rows.min()) + 1) / height,
    )

    body_bottom = streak_top
    body_top = max(0, body_bottom - int(0.30 * height))
    body_x0 = max(0, center - int(0.16 * width))
    body_x1 = min(width, center + int(0.11 * width))
    red = small[..., 0]
    blue = small[..., 2]
    leafy = _green(small)
    warm = red > blue + np.float32(0.05)
    local_luma = luma[body_top:body_bottom, body_x0:body_x1]
    body_pixels = (
        (local_luma > np.float32(0.20))
        & (local_luma < np.float32(0.78))
        & ~leafy[body_top:body_bottom, body_x0:body_x1]
        & ~warm[body_top:body_bottom, body_x0:body_x1]
    )
    if float(body_pixels.mean()) < 0.22 or int(body_pixels.sum()) < 40:
        return None
    if float(local_luma[body_pixels].mean()) < bottom_luma + 0.08:
        return None
    ys, xs = np.nonzero(body_pixels)
    box_w = (int(xs.max()) - int(xs.min()) + 1) / float(width)
    box_h = (int(ys.max()) - int(ys.min()) + 1) / float(height)
    if box_w < 0.07 or box_h < 0.06 or box_w > 0.42:
        return None
    body = (
        (body_x0 + int(xs.min())) / width,
        (body_top + int(ys.min())) / height,
        box_w,
        box_h,
    )
    return body, reflection


def water_reflection(
    pixels: np.ndarray,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None:
    """A shape above a waterline, and the mirror of it in the water below.

    The mirror may be a thin streak or a wide echo of a ridge, a tree, or a
    house. Water is the darker, smoother, lower-chroma band. The mirror is
    the part of that band that is brighter or more structured than the water
    beside it and lines up with a shape above. A warm horizon, green land,
    window lights, and a plain dark foreground are not mirrors.
    """

    small = _small(pixels)
    height, width = small.shape[:2]
    if min(height, width) < 48:
        return None
    luma = 0.2126 * small[..., 0] + 0.7152 * small[..., 1] + 0.0722 * small[..., 2]
    chroma = np.max(np.abs(small - luma[..., None]), axis=2)
    green = _green(small)
    warmth = small[..., 0] - small[..., 2]
    best: tuple[float, tuple[float, float, float, float], tuple[float, float, float, float]] | None = None
    for y in range(int(0.38 * height), int(0.78 * height)):
        measured = _mirror_at(luma, chroma, green, warmth, y)
        if measured is None:
            continue
        score, source, reflection = measured
        if best is None or score > best[0]:
            best = (score, source, reflection)
    if best is None:
        return None
    return best[1], best[2]


def _mirror_at(
    luma: np.ndarray,
    chroma: np.ndarray,
    green: np.ndarray,
    warmth: np.ndarray,
    y: int,
) -> tuple[float, tuple[float, float, float, float], tuple[float, float, float, float]] | None:
    """One candidate waterline. None when this row is not a shore."""

    height, width = luma.shape
    if y < int(0.08 * height) or height - y < int(0.14 * height):
        return None
    span = max(3, height // 48)
    if y < span or y + span >= height:
        return None
    if float(luma[y - span : y].mean()) < float(luma[y : y + span].mean()) + 0.035:
        return None

    lower = luma[y:]
    dark = lower <= np.float32(np.percentile(lower, 45))
    if int(dark.sum()) < 20:
        return None
    if float(chroma[y:][dark].mean()) > 0.09:
        return None
    if float(green[y:][dark].mean()) > 0.2:
        return None
    if float(warmth[y:][dark].mean()) > 0.045:
        return None
    if float(np.percentile(lower, 25)) > 0.2 or float(lower.mean()) > 0.34:
        return None
    # The shore itself has to be water. A drop from sky onto a ridge is not
    # a waterline just because the lake is somewhere further down.
    shore_h = max(4, int(0.05 * height))
    shore = slice(y, min(height, y + shore_h))
    if float(np.median(luma[shore])) > 0.30:
        return None
    if float(chroma[shore].mean()) > 0.09 or float(green[shore].mean()) > 0.22:
        return None
    if float(warmth[shore].mean()) > 0.05:
        return None

    above_top = max(0, y - int(0.36 * height))
    if y - above_top < int(0.08 * height):
        return None
    if float(lower.mean()) > float(luma[above_top:y].mean()) - 0.04:
        return None

    band_h = max(6, int(0.24 * height))
    below = luma[y : min(height, y + band_h)]
    above = luma[max(0, y - band_h) : y]
    if above.shape[0] < 6 or below.shape[0] < 6:
        return None
    if float(below.mean()) > 0.3 or float(np.percentile(below, 30)) > 0.22:
        return None

    above_col = above.mean(axis=0)
    below_col = below.mean(axis=0)
    floor = float(np.percentile(below_col, 20))
    brighter = below_col > np.float32(floor + 0.028)
    # Vertical ridges show up across columns. A flat echo shows up as brightness.
    vert = np.abs(np.diff(below, axis=0, prepend=below[:1])).mean(axis=0)
    horiz = np.zeros(width, dtype=np.float32)
    if width > 1:
        gx = np.abs(below[:, 1:] - below[:, :-1]).mean(axis=0)
        horiz[:-1] += gx
        horiz[1:] += gx
        horiz *= np.float32(0.5)
    detail = np.maximum(vert, horiz)
    busier = detail > np.float32(float(np.percentile(detail, 45)) + 0.01)
    mirror_cols = brighter | busier
    mirror_cols[:2] = False
    mirror_cols[-2:] = False
    if int(mirror_cols.sum()) < max(4, int(0.045 * width)):
        return None

    chosen = _best_mirror_run(mirror_cols, above_col, below_col, detail, width)
    if chosen is None:
        return None
    x0, x1, lift, structure = chosen
    outside = np.ones(width, dtype=bool)
    outside[x0:x1] = False
    outside[:2] = False
    outside[-2:] = False
    if int(outside.sum()) < max(4, int(0.06 * width)):
        return None
    inside_above = float(above_col[x0:x1].mean())
    outside_above = float(above_col[outside].mean())
    inside_std = float(above[:, x0:x1].std())
    outside_std = float(above[:, outside].std())
    if abs(inside_above - outside_above) < 0.03 and inside_std < outside_std + 0.008:
        return None
    # A facade can match the sky in brightness and still echo through its
    # windows. A ridge is darker than the sky and brighter than the lake.
    # Either sign is a horizontal lineup.
    pad = max(2, int(0.06 * width))
    left = max(0, x0 - pad)
    right = min(width, x1 + pad)
    agreement = _agreement(above_col[left:right], below_col[left:right])
    same_place = abs(inside_above - outside_above) >= 0.04 or inside_std >= outside_std + 0.01
    if abs(agreement) < 0.2 and not same_place:
        return None

    side_level = float(np.median(luma[y:, outside]))
    row_mean = luma[y:, x0:x1].mean(axis=1)
    row_edge = np.abs(np.diff(luma[y:, x0:x1], axis=1, prepend=luma[y:, x0 : x0 + 1])).mean(axis=1)
    side_edge = float(np.abs(np.diff(luma[y:, outside], axis=1, prepend=luma[y:, outside][:, :1])).mean())
    # The subject itself can cross the shore. The mirror is the dimmer echo.
    above_level = float(np.percentile(above[:, x0:x1], 55))
    echo = ((row_mean > side_level + 0.018) | (row_edge > side_edge + 0.008)) & (
        row_mean < above_level - 0.04
    )
    span_rows = _top_run(echo, int(0.12 * height), gap=max(3, int(0.05 * height)))
    if span_rows is None:
        return None
    y0_rel, y1_rel = span_rows
    if y1_rel - y0_rel < int(0.045 * height):
        return None

    block = luma[y + y0_rel : y + y1_rel, x0:x1]
    if block.size == 0:
        return None
    # Scattered window lights are warm pinpoints. A mirror fills its box.
    if float(block.mean()) > 0.72 and (x1 - x0) / float(width) < 0.2:
        return None
    if float((warmth[y + y0_rel : y + y1_rel, x0:x1]).mean()) > 0.07 and (x1 - x0) / float(width) < 0.16:
        return None
    solid = float(np.mean(block > side_level + 0.015))
    if solid < 0.28 and structure < 0.012:
        return None

    source_top = _source_top(luma, y, x0, x1, outside)
    if y - source_top < int(0.04 * height):
        return None
    source = (
        x0 / width,
        source_top / height,
        (x1 - x0) / width,
        (y - source_top) / height,
    )
    reflection = (
        x0 / width,
        (y + y0_rel) / height,
        (x1 - x0) / width,
        (y1_rel - y0_rel) / height,
    )
    score = lift + structure * 2.0 + (x1 - x0) / float(width)
    return score, source, reflection


def _best_mirror_run(
    mirror_cols: np.ndarray,
    above_col: np.ndarray,
    below_col: np.ndarray,
    vert: np.ndarray,
    width: int,
) -> tuple[int, int, float, float] | None:
    """The widest coherent echo, when it stands out from the water beside it."""

    runs = _merge_runs(_column_runs(mirror_cols), max(2, width // 48))
    best: tuple[int, int, float, float] | None = None
    best_score = 0.0
    for x0, x1 in runs:
        frac = (x1 - x0) / float(width)
        if frac < 0.05 or frac > 0.9:
            continue
        if float(mirror_cols[x0:x1].mean()) < 0.62:
            continue
        outside = np.ones(width, dtype=bool)
        outside[x0:x1] = False
        if int(outside.sum()) < max(4, int(0.06 * width)):
            continue
        lift = float(below_col[x0:x1].mean() - below_col[outside].mean())
        structure = float(vert[x0:x1].mean() - vert[outside].mean())
        if lift < 0.025 and structure < 0.008:
            continue
        # The columns have to agree with a shape above, not only a bright pool.
        if float(np.std(above_col[x0:x1])) + abs(float(above_col[x0:x1].mean() - above_col[outside].mean())) < 0.02:
            continue
        score = max(lift, 0.0) + max(structure, 0.0) * 2.0
        if score > best_score:
            best_score = score
            best = (x0, x1, lift, structure)
    return best


def _source_top(
    luma: np.ndarray,
    y: int,
    x0: int,
    x1: int,
    outside: np.ndarray,
) -> int:
    """Walk up from the shore while those columns still hold the shape.

    A wall can match the sky in brightness for a few rows. Windows and
    ridges still differ, so a short run of similar rows does not end it.
    """

    height = luma.shape[0]
    limit = max(0, y - int(0.46 * height))
    max_gap = max(3, int(0.04 * height))
    gap = 0
    top = y
    while top > limit:
        row_in = luma[top - 1, x0:x1]
        row_side = luma[top - 1, outside]
        color_gap = abs(float(row_in.mean()) - float(row_side.mean()))
        detail_gap = abs(float(row_in.std()) - float(row_side.std()))
        if color_gap < 0.02 and detail_gap < 0.012:
            gap += 1
            if gap > max_gap:
                return min(y, top + gap - 1)
        else:
            gap = 0
        top -= 1
    return min(y, top + gap)


def _column_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(mask.size)))
    return runs


def _merge_runs(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged: list[list[int]] = [[runs[0][0], runs[0][1]]]
    for x0, x1 in runs[1:]:
        if x0 - merged[-1][1] <= gap:
            merged[-1][1] = x1
        else:
            merged.append([x0, x1])
    return [(item[0], item[1]) for item in merged]


def _agreement(left: np.ndarray, right: np.ndarray) -> float:
    """How similarly two column profiles rise and fall. Zero when either is flat."""

    if left.size < 4 or right.size < 4 or left.shape != right.shape:
        return 0.0
    if float(np.std(left)) < 1e-4 or float(np.std(right)) < 1e-4:
        return 0.0
    return float(np.corrcoef(left.astype(np.float64), right.astype(np.float64))[0, 1])


def _top_run(flags: np.ndarray, limit: int, gap: int = 0) -> tuple[int, int] | None:
    """The run that begins near the shore, skipping a few dark rows inside it.

    A window band in the echo is darker than the water for a moment. That
    gap is still the mirror. A glint that only starts far below the shore
    is not.
    """

    start: int | None = None
    end: int | None = None
    missed = 0
    for index, flag in enumerate(flags.tolist()):
        if flag:
            if start is None:
                if index > limit:
                    return None
                start = index
            missed = 0
            end = index + 1
            continue
        if start is None:
            continue
        missed += 1
        if missed > gap:
            return start, end
    if start is None or end is None:
        return None
    return start, end


def boxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """True when two normalized boxes share any area."""

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    width = min(ax + aw, bx + bw) - max(ax, bx)
    height = min(ay + ah, by + bh) - max(ay, by)
    return width > 0.0 and height > 0.0


def _strong_foliage(pixels: np.ndarray) -> np.ndarray:
    """Green that is clearly leaves. A mildly olive body is not this."""

    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    return (green > red + np.float32(0.08)) & (green > blue + np.float32(0.06))


def _reflection_block(pixels: np.ndarray) -> np.ndarray:
    """Pixels that belong to a water mirror, not to the body above it."""

    height, width = pixels.shape[:2]
    blocked = np.zeros((height, width), dtype=bool)
    boxes: list[tuple[float, float, float, float]] = []
    mirrored = reflected_subject(pixels)
    if mirrored is not None:
        boxes.append(mirrored[1])
    found = water_reflection(pixels)
    if found is not None:
        boxes.append(found[1])
    for box in boxes:
        x, y, bw, bh = box
        x0 = min(width - 1, max(0, int((x - 0.01) * width)))
        y0 = min(height - 1, max(0, int(y * height)))
        x1 = min(width, max(x0 + 1, int((x + bw + 0.01) * width)))
        y1 = min(height, max(y0 + 1, int((y + bh) * height)))
        # The echo can touch the body through a thin gap. Cut from the
        # shore downward so the mirror is not part of the creature.
        blocked[y0:, x0:x1] = True
    return blocked


def creature_box(
    pixels: np.ndarray,
) -> tuple[tuple[float, float, float, float], float, float] | None:
    """A compact body that is not the green background and not the perch.

    An elongated brown branch is left out. A compact body of its own color
    is kept, including a brown or olive animal that is not a branch. A
    water reflection is not part of the body. A textured cluster fails the
    fill and edge gates, the same way any other busy patch does.
    """

    small = _small(pixels)
    blocked = _reflection_block(small)
    background = _corner_median(small)
    distance = np.sqrt(np.sum((small - background) ** 2, axis=2))
    apart = distance > np.float32(0.16)
    foliage = _strong_foliage(small)
    woody = _woody(small)
    skin = _skin(small)
    # Mild olive stays. Strong leaves and the mirror do not.
    mild = apart & ~foliage & ~woody & ~blocked
    # A brown animal is woody and compact. A branch is the same color and
    # is rejected later because it is a bar. Skin is a face, not this body.
    brown = apart & woody & ~skin & ~foliage & ~blocked
    best: tuple[tuple[float, float, float, float], float, float] | None = None
    best_count = 0
    for candidate in (mild, brown):
        thick = _thick(candidate, 0.016, 0.5)
        for component in _components(thick):
            measured = _compact_region(component, small)
            if measured is None:
                continue
            count = int(component.sum())
            if count > best_count:
                best = measured
                best_count = count
    # A compact body over a narrow mirror is the creature. A wide echo of a
    # house or a ridge never reaches here, so a reflection alone is not an animal.
    # A wide shallow band, such as pads on the water, is not that body.
    mirrored = reflected_subject(pixels)
    if mirrored is not None:
        body, _echo = mirrored
        area = float(body[2] * body[3])
        compact_body = (
            not is_elongated(body, small.shape)
            and body[2] < 0.72
            and body[3] < 0.72
            and area >= 0.02
        )
        if compact_body:
            picked = (body, 0.55, area)
            if best is None:
                return picked
            bw, bh = float(best[0][2]), float(best[0][3])
            band = bw > bh * 1.45 and bh < 0.45
            if band and not boxes_overlap(best[0], body):
                return picked
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
    # A compact body sits inside the frame. A slab that runs from one edge
    # to the opposite edge is a wall, a field, or a shore.
    if (y0 <= 1 and y1 >= height - 2) or (x0 <= 1 and x1 >= width - 2):
        return None
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    fill = count / float(max(1, box_w * box_h))
    if fill < 0.28:
        return None
    norm = (x0 / width, y0 / height, box_w / width, box_h / height)
    # A band of pads or shore can be one connected mass. A creature fills
    # its box. A wide, half-empty band does not.
    if norm[2] >= 0.8 and fill < 0.5:
        return None
    if is_elongated(norm, component.shape):
        return None
    # A compact body does not run most of the way across the frame.
    # A wall or a field does, and it is not a creature.
    if norm[2] >= 0.72 or norm[3] >= 0.72:
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
