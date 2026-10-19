"""Subject finder that needs only Pillow and NumPy.

Replace or precede this with :class:`ai_dt.detectors.yolo.YoloDetector` when a
detection model is installed. The heuristic is the fallback so a click still
produces a recipe on a machine with no weights downloaded.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ai_dt.guides.decide import Blob, _face_is_skin, _interior_edge
from ai_dt.guides.support import boxes_overlap, creature_box, is_face_bar
from ai_dt.schema import Detection


def _downsample(image: Image.Image, max_side: int = 240) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BILINEAR,
    )


def _luma(rgb: np.ndarray) -> np.ndarray:
    color = rgb.astype(np.float32) / 255.0
    return 0.2126 * color[:, :, 0] + 0.7152 * color[:, :, 1] + 0.0722 * color[:, :, 2]


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    spread = np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)
    return (
        (red > 95)
        & (green > 40)
        & (blue > 20)
        &         (red > green)
        & (red > blue)
        # A near-neutral wall (spread of about 16) is not a cheek.
        & (spread > 20)
    )


def _largest_bbox(mask: np.ndarray, min_fraction: float) -> tuple[int, int, int, int] | None:
    """Bounding box of the largest connected component, as x, y, w, h in pixels."""

    height, width = mask.shape
    if height == 0 or width == 0 or not mask.any():
        return None
    visited = np.zeros(mask.shape, dtype=bool)
    seeds = np.argwhere(mask)
    best_area = 0
    best: tuple[int, int, int, int] | None = None
    for seed_y, seed_x in seeds:
        if visited[seed_y, seed_x]:
            continue
        stack = [(int(seed_y), int(seed_x))]
        visited[seed_y, seed_x] = True
        area = 0
        min_x = max_x = int(seed_x)
        min_y = max_y = int(seed_y)
        while stack:
            y, x = stack.pop()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
        if area > best_area:
            best_area = area
            best = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    if best is None or best_area < min_fraction * height * width:
        return None
    return best


def _normalize(box: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (
        max(0.0, x / width),
        max(0.0, y / height),
        min(1.0, w / width),
        min(1.0, h / height),
    )


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """Mean of a square neighborhood. The radius stays small so fur is not an eye."""

    radius = max(1, int(radius))
    padded = np.pad(values, radius, mode="edge")
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    height, width = values.shape
    y = np.arange(height)
    x = np.arange(width)
    reach = 2 * radius + 1
    total = (
        integral[np.ix_(y + reach, x + reach)]
        - integral[np.ix_(y, x + reach)]
        - integral[np.ix_(y + reach, x)]
        + integral[np.ix_(y, x)]
    )
    return total / float(reach * reach)


def _shrink_luma(crop: np.ndarray, max_side: int = 220) -> tuple[np.ndarray, float, float]:
    height, width = crop.shape
    longest = max(height, width)
    if longest <= max_side:
        return crop, 1.0, 1.0
    scale = max_side / float(longest)
    resized = (
        max(1, int(round(height * scale))),
        max(1, int(round(width * scale))),
    )
    image = Image.fromarray(np.clip(crop * 255.0, 0, 255).astype(np.uint8), mode="L")
    small = np.asarray(image.resize((resized[1], resized[0]), Image.Resampling.BILINEAR))
    return small.astype(np.float32) / np.float32(255.0), height / small.shape[0], width / small.shape[1]


def _saturation_peaks(
    saturation: np.ndarray,
    colorful: np.ndarray,
    minimum: float,
) -> list[tuple[float, float, int, int, int, float]]:
    """Bright colored points, with each iris kept once."""

    work = np.where(colorful, saturation, np.float32(0.0)).astype(np.float32, copy=True)
    reach = max(2, int(round(0.045 * min(work.shape))))
    peaks: list[tuple[float, float, int, int, int, float]] = []
    spot = max(4, reach)
    for _ in range(12):
        index = int(np.argmax(work))
        score = float(work.ravel()[index])
        if score < minimum:
            break
        cy, cx = divmod(index, work.shape[1])
        y0 = max(0, cy - reach)
        y1 = min(work.shape[0], cy + reach + 1)
        x0 = max(0, cx - reach)
        x1 = min(work.shape[1], cx + reach + 1)
        work[y0:y1, x0:x1] = 0
        near = colorful[max(0, cy - 3) : cy + 4, max(0, cx - 3) : cx + 4]
        if int(near.sum()) < 8:
            continue
        peaks.append((float(cx), float(cy), spot, spot, spot, score))
    return _drop_bands(peaks)


def _drop_bands(peaks: list[tuple]) -> list[tuple]:
    """Drop colored peaks that line up along an edge.

    The peak search clears a square around each maximum, so a saturated
    edge (red plumage meeting yellow, a coloured stripe) comes back as a
    row of evenly spaced peaks on one line. Any two of them make a level
    "pair". Irises are isolated spots: four or more peaks on one row or
    column are an edge, not eyes.
    """

    banded: set[int] = set()
    for axis in (0, 1):
        for peak in peaks:
            row = sorted(
                (index for index, item in enumerate(peaks) if abs(item[1 - axis] - peak[1 - axis]) <= 2.0),
                key=lambda index: peaks[index][axis],
            )
            # Only an unbroken chain is an edge: two irises side by side can
            # each give two peaks on one row, with a wide gap between them.
            chain = row[:1]
            for index in row[1:]:
                step = abs(peaks[index][axis] - peaks[chain[-1]][axis])
                if step <= 2.5 * max(1, int(peaks[index][2])):
                    chain.append(index)
                else:
                    if len(chain) >= 4:
                        banded.update(chain)
                    chain = [index]
            if len(chain) >= 4:
                banded.update(chain)
    return [peak for index, peak in enumerate(peaks) if index not in banded]


def _single_eye(
    small: np.ndarray,
    dark: np.ndarray,
    local: np.ndarray,
) -> tuple[float, float, int, int, int, float] | None:
    """One eye seen in profile: a round, solid dark spot holding a catchlight.

    A bird or animal side-on shows one eye. A nose, a shadow or the dark
    neck beside bright water is dark too, so a lone spot only counts when
    it is compact and round, clearly darker than its surround, and holds a
    bright catchlight *enclosed* by the dark (a hole in the pupil, not light
    at its rim), and it is the only such spot in the search box.
    """

    region_area = int(small.size)
    filled = _filled(dark)
    found: list[tuple[float, float, int, int, int, float]] = []
    for cx, cy, area, spot_w, spot_h in _component_spots(filled):
        if area < 12 or area > int(0.02 * region_area):
            continue
        aspect = spot_w / float(max(1, spot_h))
        if aspect < 0.7 or aspect > 1.45:
            continue
        if area < 0.6 * spot_w * spot_h:
            continue
        x0 = max(0, int(round(cx - spot_w / 2.0)))
        y0 = max(0, int(round(cy - spot_h / 2.0)))
        x1 = min(small.shape[1], x0 + spot_w + 1)
        y1 = min(small.shape[0], y0 + spot_h + 1)
        inside = small[y0:y1, x0:x1]
        pupil = dark[y0:y1, x0:x1]
        hole = filled[y0:y1, x0:x1] & ~pupil
        if inside.size == 0 or not pupil.any():
            continue
        holes = int(hole.sum())
        if holes < 2 or holes > 0.4 * area:
            continue
        depth = float(np.median(local[y0:y1, x0:x1]) - float(np.min(inside)))
        catchlight = float(np.max(inside[hole])) - float(np.median(inside[pupil]))
        if depth < 0.15 or catchlight < 0.35:
            continue
        found.append((cx, cy, area, spot_w, spot_h, depth))
    if len(found) != 1:
        return None
    return found[0]


def _with_pupil(
    peaks: list[tuple[float, float, int, int, int, float]],
    luma: np.ndarray,
    radius: int,
) -> list[tuple[float, float, int, int, int, float]]:
    """Keep a colored spot only when a darker pupil sits inside it."""

    kept: list[tuple[float, float, int, int, int, float]] = []
    reach = max(2, int(radius))
    for peak in peaks:
        cx = min(luma.shape[1] - 1, max(0, int(peak[0])))
        cy = min(luma.shape[0] - 1, max(0, int(peak[1])))
        patch = luma[
            max(0, cy - reach) : min(luma.shape[0], cy + reach + 1),
            max(0, cx - reach) : min(luma.shape[1], cx + reach + 1),
        ]
        if patch.size == 0:
            continue
        here = float(luma[cy, cx])
        pupil = patch < min(0.42, here - 0.12)
        if int(pupil.sum()) < 2:
            continue
        ys, xs = np.nonzero(pupil)
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        longer = max(width, height)
        shorter = max(1, min(width, height))
        # A grid line is long and thin. A pupil is a small dark spot.
        if longer / float(shorter) > 3.0 and longer > 0.7 * max(patch.shape):
            continue
        if float(pupil.mean()) > 0.4:
            # A sunlit iris sits in a dark socket. The wide window is mostly
            # shadow, while a tighter window still holds the pupil.
            tight = max(2, reach // 2)
            core = luma[
                max(0, cy - tight) : min(luma.shape[0], cy + tight + 1),
                max(0, cx - tight) : min(luma.shape[1], cx + tight + 1),
            ]
            if core.size == 0:
                continue
            slit = core < min(0.42, here - 0.12)
            if int(slit.sum()) < 3 or float(slit.mean()) > 0.35:
                continue
        kept.append(peak)
    return kept


def _colorful_iris(
    colorful: np.ndarray,
    luma: np.ndarray,
    cx: int,
    cy: int,
    reach: int,
) -> tuple[float, float] | None:
    """Center of the lower colored pixels under a brow."""

    y0 = min(colorful.shape[0] - 1, cy + 1)
    y1 = min(colorful.shape[0], cy + reach + 1)
    half = max(3, reach // 2)
    x0 = max(0, cx - half)
    x1 = min(colorful.shape[1], cx + half + 1)
    mask = colorful[y0:y1, x0:x1]
    if int(mask.sum()) < 4:
        return None
    ys, xs = np.nonzero(mask)
    if int(ys.max() - ys.min()) < 0.08 * min(colorful.shape):
        return None
    lower = ys >= np.percentile(ys, 65)
    if int(lower.sum()) >= 2:
        ys, xs = ys[lower], xs[lower]
    values = luma[y0:y1, x0:x1][ys, xs]
    bright = values >= float(np.median(values))
    if int(bright.sum()) >= 2:
        ys, xs = ys[bright], xs[bright]
    return (x0 + float(xs.mean()), y0 + float(ys.mean()))


def _shaded_pupil(
    luma: np.ndarray,
    cx: int,
    cy: int,
) -> tuple[float, float] | None:
    """A dark pupil under a brow, with iris around it and shadow rejected.

    The pupil can sit beside the bright fur, not only straight below it.
    A nose or an ear shadow is dark without a lighter ring, so it is skipped.
    """

    span = min(luma.shape)
    side = max(4, int(round(0.14 * span)))
    y0 = cy + max(6, int(round(0.08 * span)))
    y1 = min(luma.shape[0] - 6, cy + int(round(0.30 * span)))
    x0 = max(6, cx - side)
    x1 = min(luma.shape[1] - 6, cx + side)
    if y1 <= y0 or x1 <= x0:
        return None
    best: tuple[float, int, int] | None = None
    for y in range(y0, y1):
        for x in range(x0, x1):
            here = float(luma[y, x])
            if here > 0.20:
                continue
            ring = float(luma[y - 3 : y + 4, x - 3 : x + 4].mean())
            if ring < 0.14 or ring > 0.45 or ring < here + 0.08:
                continue
            # A shadow is dark on one side. A pupil has iris on every side.
            sides = (
                float(luma[y - 5 : y - 2, x - 1 : x + 2].mean()),
                float(luma[y + 3 : y + 6, x - 1 : x + 2].mean()),
                float(luma[y - 1 : y + 2, x - 5 : x - 2].mean()),
                float(luma[y - 1 : y + 2, x + 3 : x + 6].mean()),
            )
            if min(sides) < 0.12:
                continue
            if best is None or here < best[0]:
                best = (here, x, y)
    if best is None:
        return None
    level, x, y = best
    core = luma[y - 3 : y + 4, x - 3 : x + 4] <= level + np.float32(0.06)
    ys, xs = np.nonzero(core)
    if ys.size < 2:
        return (float(x), float(y))
    return (x - 3 + float(xs.mean()), y - 3 + float(ys.mean()))


def _drop_onto_iris(
    pair: tuple[tuple, tuple],
    colorful: np.ndarray,
    luma: np.ndarray,
) -> tuple[tuple, tuple]:
    """Move a bright brow down onto the iris beneath it.

    A sunlit iris is the colored pixels under the brow. A shaded iris is
    the dark pupil below that brow, with lighter iris on every side.
    """

    reach = max(8, int(round(0.16 * min(luma.shape))))
    dropped = []
    for peak in pair:
        cx = min(luma.shape[1] - 1, max(0, int(peak[0])))
        cy = min(luma.shape[0] - 1, max(0, int(peak[1])))
        landed = _colorful_iris(colorful, luma, cx, cy, reach)
        if landed is None:
            landed = _shaded_pupil(luma, cx, cy)
        if landed is None:
            dropped.append(peak)
            continue
        dropped.append((landed[0], landed[1], peak[2], peak[3], peak[4], peak[5]))
    return (dropped[0], dropped[1])


def _kept_spots(
    mask: np.ndarray,
    score_map: np.ndarray,
) -> list[tuple[float, float, int, int, int, float]]:
    region_area = int(mask.size)
    kept: list[tuple[float, float, int, int, int, float]] = []
    for cx, cy, area, spot_w, spot_h in _component_spots(mask):
        if area < 4 or area > int(0.02 * region_area):
            continue
        aspect = spot_w / float(max(1, spot_h))
        if aspect < 0.35 or aspect > 2.8:
            continue
        if spot_w > 0.22 * mask.shape[1] or spot_h > 0.22 * mask.shape[0]:
            continue
        iy = min(mask.shape[0] - 1, max(0, int(cy)))
        ix = min(mask.shape[1] - 1, max(0, int(cx)))
        kept.append((cx, cy, area, spot_w, spot_h, float(score_map[iy, ix])))
    return kept


def _best_eye_pair(
    kept: list[tuple[float, float, int, int, int, float]],
    span: float,
    minimum: float,
    ratio_limit: float = 2.5,
    max_sep: float = 0.32,
    min_sep: float = 0.04,
    close_weight: float = 0.0,
) -> tuple[tuple, tuple] | None:
    best: tuple[tuple, tuple] | None = None
    best_score = 0.0
    for index, left in enumerate(kept):
        for right in kept[index + 1 :]:
            dx = abs(left[0] - right[0])
            dy = abs(left[1] - right[1])
            sep = (dx * dx + dy * dy) ** 0.5
            if sep < min_sep * span or sep > max_sep * span:
                continue
            if dx < 1.0 or dy > dx * 0.9:
                continue
            ratio = max(left[2], right[2]) / float(max(1, min(left[2], right[2])))
            if ratio > ratio_limit:
                continue
            score = left[5] + right[5] - close_weight * (sep / span)
            if score > best_score:
                best_score = score
                best = (left, right)
    if best is None or best_score < minimum:
        return None
    return best


def _pair_to_boxes(
    pair: tuple[tuple, tuple],
    small_shape: tuple[int, int],
    origin: tuple[int, int],
    scale_y: float,
    scale_x: float,
    width: int,
    height: int,
    expand: float = 3.0,
) -> list[tuple[float, float, float, float]]:
    x0, y0 = origin
    boxes: list[tuple[float, float, float, float]] = []
    for cx, cy, _area, spot_w, spot_h, _score in pair:
        eye_w = max(int(spot_w * expand), int(0.035 * small_shape[1]), 2)
        eye_h = max(int(spot_h * expand), int(0.045 * small_shape[0]), 2)
        full_cx = x0 + (cx + 0.5) * scale_x
        full_cy = y0 + (cy + 0.5) * scale_y
        full_w = max(2.0, eye_w * scale_x)
        full_h = max(2.0, eye_h * scale_y)
        ex = int(round(full_cx - full_w / 2.0))
        ey = int(round(full_cy - full_h / 2.0))
        ew = max(2, int(round(full_w)))
        eh = max(2, int(round(full_h)))
        ex = min(max(0, ex), max(0, width - ew))
        ey = min(max(0, ey), max(0, height - eh))
        boxes.append(_normalize((ex, ey, ew, eh), width, height))
    return boxes


def animal_eye_boxes(
    luma: np.ndarray,
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    color: np.ndarray | None = None,
) -> list[tuple[float, float, float, float]]:
    """Two eyes inside an animal, or nothing when no pair is there.

    The search covers the whole animal. A lying cat does not keep its eyes
    in the upper half of the box the way a human face does. A single dark
    spot, such as a nose, is not a pair. Colored irises are preferred over
    dark ears when the picture has color.
    """

    x, y, w, h = box
    if w <= 0.0 or h <= 0.0 or w * h < 0.01 or w * h > 0.9:
        return []
    x0 = max(0, min(width - 1, int(x * width)))
    y0 = max(0, min(height - 1, int(y * height)))
    x1 = max(x0 + 1, min(width, int((x + w) * width)))
    y1 = max(y0 + 1, min(height, int((y + h) * height)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return []
    crop = luma[y0:y1, x0:x1].astype(np.float32, copy=False)
    if crop.size and float(np.max(crop)) > 1.5:
        crop = crop / np.float32(255.0)
    small, scale_y, scale_x = _shrink_luma(crop)
    # Wider than a pupil, so the pupil is darker than the fur around it.
    # A tight blur sits inside the pupil and the center no longer looks dark.
    radius = max(4, int(round(0.08 * min(small.shape))))
    if color is not None and getattr(color, "ndim", 0) == 3:
        patch = np.asarray(color[y0:y1, x0:x1], dtype=np.float32)
        if patch.size and float(patch.max()) > 1.5:
            patch = patch / np.float32(255.0)
        sat = np.max(patch, axis=2) - np.min(patch, axis=2)
        small_sat, sat_scale_y, sat_scale_x = _shrink_luma(sat)
        local_sat = _box_mean(small_sat, radius)
        luma_gate = small
        if luma_gate.shape != small_sat.shape:
            luma_gate = luma_gate[: small_sat.shape[0], : small_sat.shape[1]]
        colorful = (
            (small_sat > local_sat + np.float32(0.10))
            & (small_sat > np.float32(0.22))
            & (luma_gate > np.float32(0.28))
            & (luma_gate < np.float32(0.82))
        )
        peaks = _with_pupil(
            _saturation_peaks(small_sat, colorful, 0.34),
            luma_gate,
            max(3, int(round(0.055 * min(small_sat.shape)))),
        )
        color_pair = _best_eye_pair(
            peaks,
            float(max(small_sat.shape)),
            0.7,
            ratio_limit=8.0,
            max_sep=0.30,
            min_sep=0.07,
            close_weight=0.8,
        )
        ear_line = 0.28 * small_sat.shape[0]
        if color_pair is not None and (color_pair[0][1] + color_pair[1][1]) * 0.5 < ear_line:
            # The brightest pair is the ear tips. A lower pair can still be
            # the eyes. Ear shadows are not a second try.
            lower = [peak for peak in peaks if peak[1] >= ear_line]
            color_pair = _best_eye_pair(
                lower,
                float(max(small_sat.shape)),
                0.6,
                ratio_limit=8.0,
                max_sep=0.55,
                min_sep=0.07,
                close_weight=0.4,
            )
            if color_pair is None:
                return []
        if color_pair is not None:
            color_pair = _drop_onto_iris(color_pair, colorful, luma_gate)
            placed = _separate_eyes(
                [
                    box
                    for box in _pair_to_boxes(
                        color_pair,
                        small_sat.shape,
                        (x0, y0),
                        sat_scale_y,
                        sat_scale_x,
                        width,
                        height,
                        expand=2.6,
                    )
                    if box[2] > 0.015 and box[3] > 0.015
                ]
            )
            if placed:
                return placed
    local = _box_mean(small, radius)
    # Pupils are dark. Orange or white fur around them is not.
    dark = (small < local - np.float32(0.14)) & (small < np.float32(0.38))
    pair = _best_eye_pair(_kept_spots(dark, local - small), float(max(small.shape)), 0.22)
    if pair is not None:
        boxes = _pair_to_boxes(pair, small.shape, (x0, y0), scale_y, scale_x, width, height)
        placed = _separate_eyes([box for box in boxes if box[2] > 0.015 and box[3] > 0.015])
        if placed:
            return placed
    # Side-on, only one eye shows (and a "pair" of specks too small to be
    # eyes is no pair). The catchlight leaves a hole in the dark mask, so
    # fill the spot before reading its shape.
    single = _single_eye(small, dark, local)
    if single is None:
        return []
    boxes = _pair_to_boxes((single,), small.shape, (x0, y0), scale_y, scale_x, width, height, expand=1.4)
    return [box for box in boxes if box[2] > 0.015 and box[3] > 0.015]


def attach_animal_eyes(pixels: np.ndarray, detections: list[Detection]) -> list[Detection]:
    """Add eye boxes for an animal that does not already have eyes inside it."""

    # The smaller box is the body. A loose blob around it finds the sign
    # or the ears, and those spots then block the real eyes.
    animals = sorted(
        (item for item in detections if item.category == "animal"),
        key=lambda item: float(item.bbox[2]) * float(item.bbox[3]),
    )
    if not animals:
        return list(detections)
    view = np.asarray(pixels, dtype=np.float32)
    if view.ndim == 3:
        if view.size and float(view.max()) > 1.5:
            view = view / np.float32(255.0)
        luma = 0.2126 * view[..., 0] + 0.7152 * view[..., 1] + 0.0722 * view[..., 2]
    else:
        luma = view
        if luma.size and float(luma.max()) > 1.5:
            luma = luma / np.float32(255.0)
    frame_h, frame_w = int(luma.shape[0]), int(luma.shape[1])
    # A downsampled pass can mark ears. Search this picture again and keep
    # a face detector's eyes when they already sit on the animal.
    kept = [
        item
        for item in detections
        if not (
            item.category == "eye"
            and item.source != "mediapipe"
            and any(_center_inside(item.bbox, animal.bbox) for animal in animals)
        )
    ]
    eyes = [item for item in kept if item.category == "eye"]
    added: list[Detection] = []
    color = view if view.ndim == 3 else None
    searched: list[Detection] = []
    for animal in animals:
        if any(_center_inside(eye.bbox, animal.bbox) for eye in eyes + added):
            continue
        # A loose blob around a body we already searched would invent ears.
        if any(
            _center_inside(earlier.bbox, animal.bbox)
            and earlier.bbox[2] * earlier.bbox[3] < animal.bbox[2] * animal.bbox[3] * 0.65
            for earlier in searched
        ):
            continue
        searched.append(animal)
        for box in animal_eye_boxes(luma, _eye_search_box(animal), frame_w, frame_h, color=color):
            added.append(Detection("eye", "eye", 0.66, box, "animal"))
    if not added:
        return kept
    return kept + added


def _eye_search_box(animal: Detection) -> tuple[float, float, float, float]:
    """Where an animal's eyes can be. A standing bird keeps them in the head.

    A tall detector "bird" box runs down to the feet, and two orange feet on
    a ledge read as a coloured eye pair (the duck photo). The head is in the
    upper part of that box. Other animals keep the whole box: a lying cat
    does not hold its eyes high.
    """

    x, y, w, h = (float(value) for value in animal.bbox)
    if str(animal.label or "").lower() == "bird" and animal.source != "heuristic" and h > 1.3 * w:
        return (x, y, w, h * 0.45)
    return (x, y, w, h)


def _filled(mask: np.ndarray) -> np.ndarray:
    """``mask`` with enclosed holes (a catchlight inside a pupil) filled."""

    outside = np.zeros(mask.shape, dtype=bool)
    free = ~mask
    stack = [(y, x) for y in range(mask.shape[0]) for x in (0, mask.shape[1] - 1) if free[y, x]]
    stack += [(y, x) for x in range(mask.shape[1]) for y in (0, mask.shape[0] - 1) if free[y, x]]
    while stack:
        y, x = stack.pop()
        if outside[y, x] or not free[y, x]:
            continue
        outside[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y + 1 < mask.shape[0]:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x + 1 < mask.shape[1]:
            stack.append((y, x + 1))
    return mask | (free & ~outside)


def _box_blur(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    total = np.zeros_like(values)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
    return total / 9.0


def _component_spots(mask: np.ndarray) -> list[tuple[float, float, int, int, int]]:
    """Connected dark cores as center x, center y, area, width, height."""

    height, width = mask.shape
    if height == 0 or width == 0 or not mask.any():
        return []
    visited = np.zeros(mask.shape, dtype=bool)
    spots: list[tuple[float, float, int, int, int]] = []
    for seed_y, seed_x in np.argwhere(mask):
        if visited[seed_y, seed_x]:
            continue
        stack = [(int(seed_y), int(seed_x))]
        visited[seed_y, seed_x] = True
        count = 0
        sum_x = 0
        sum_y = 0
        min_x = max_x = int(seed_x)
        min_y = max_y = int(seed_y)
        while stack:
            y, x = stack.pop()
            count += 1
            sum_x += x
            sum_y += y
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
        spots.append((sum_x / count, sum_y / count, count, max_x - min_x + 1, max_y - min_y + 1))
    return spots


def _separated_eyes(
    luma: np.ndarray,
    face: tuple[float, float, float, float],
    width: int,
    height: int,
) -> list[tuple[float, float, float, float]]:
    """Two dark spots in the upper half, far enough apart to be eyes.

    One dark pixel, or the darkest pixel of a flat half, is not an eye.
    More than a handful of spots is texture, not a pair of eyes.
    """

    fx, fy, fw, fh = face
    x0 = int(fx * width)
    y0 = int(fy * height)
    x1 = int((fx + fw) * width)
    upper = y0 + max(1, int(fh * height * 0.55))
    if x1 - x0 < 8 or upper - y0 < 6:
        return []
    region = _box_blur(luma[y0:upper, x0:x1])
    if region.size < 16:
        return []
    level = float(np.median(region))
    dark = region < level - 0.10
    spots = _component_spots(dark)
    span = max(1, region.shape[1])
    limit = max(4, int(0.004 * region.size))
    kept: list[tuple[float, float]] = []
    for cx, cy, area, spot_w, _spot_h in spots:
        if area < limit or area > int(0.18 * region.size):
            continue
        if spot_w > 0.42 * span:
            continue
        kept.append((cx, cy))
    if len(kept) < 2 or len(kept) > 6:
        return []
    pair: tuple[tuple[float, float], tuple[float, float]] | None = None
    best = -1.0
    for index, left in enumerate(kept):
        for right in kept[index + 1 :]:
            sep = abs(left[0] - right[0]) / float(span)
            dy = abs(left[1] - right[1]) / float(max(1, region.shape[0]))
            if sep < 0.18 or sep > 0.75 or dy > 0.45:
                continue
            score = sep
            if score > best:
                best = score
                pair = (left, right)
    if pair is None:
        return []
    boxes: list[tuple[float, float, float, float]] = []
    eye_w = max(2, int(fw * width * 0.16))
    eye_h = max(2, int(fh * height * 0.10))
    for cx, cy in pair:
        ex = int(x0 + cx) - eye_w // 2
        ey = int(y0 + cy) - eye_h // 2
        ex = min(max(0, ex), max(0, width - eye_w))
        ey = min(max(0, ey), max(0, height - eye_h))
        boxes.append(_normalize((ex, ey, eye_w, eye_h), width, height))
    return boxes


def heuristic_face_ok(pixels: np.ndarray, bbox: tuple[float, float, float, float]) -> bool:
    """A heuristic face has to be skin, quiet inside, and have two eyes.

    A busy interior is full of edges. A cheek is not.
    MediaPipe faces are not passed through here.
    """

    view = pixels.astype(np.float32, copy=False)
    if view.size and float(view.max()) > 1.5:
        view = view / np.float32(255.0)
    if not _face_is_skin(view, bbox):
        return False
    if _interior_edge(view, Blob(box=bbox, fill=1.0, area=float(bbox[2] * bbox[3]))) > 0.1:
        return False
    luma = 0.2126 * view[..., 0] + 0.7152 * view[..., 1] + 0.0722 * view[..., 2]
    height, width = int(luma.shape[0]), int(luma.shape[1])
    return len(_separated_eyes(luma, bbox, width, height)) >= 2


def _separate_eyes(boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Two boxes on the same spot are one mark, not a pair of eyes."""

    if len(boxes) < 2:
        return []
    ax, ay, aw, ah = boxes[0]
    bx, by, bw, bh = boxes[1]
    if abs((ax + aw / 2.0) - (bx + bw / 2.0)) < max(aw, bw) * 0.6 and abs(
        (ay + ah / 2.0) - (by + bh / 2.0)
    ) < max(ah, bh) * 0.6:
        return []
    return boxes


def _center_inside(box: tuple[float, float, float, float], face: tuple[float, float, float, float]) -> bool:
    cx = box[0] + box[2] * 0.5
    cy = box[1] + box[3] * 0.5
    return face[0] <= cx <= face[0] + face[2] and face[1] <= cy <= face[1] + face[3]


def gate_faces(pixels: np.ndarray, detections: list[Detection]) -> list[Detection]:
    """Drop heuristic faces that fail the skin, quiet-interior, and eye checks.

    A person box stays a person. A skin-colored blob beside it is not
    promoted into a face, and a MediaPipe face is left alone.
    """

    rejected: list[tuple[float, float, float, float]] = []
    kept: list[Detection] = []
    for item in detections:
        if item.category == "person":
            kept.append(item)
            continue
        if item.category == "face" and item.source == "heuristic":
            if not heuristic_face_ok(pixels, item.bbox):
                rejected.append(item.bbox)
                continue
        kept.append(item)
    if not rejected:
        return kept
    return [
        item
        for item in kept
        if not (
            item.category == "eye"
            and item.source == "heuristic"
            and any(_center_inside(item.bbox, face) for face in rejected)
        )
    ]


def _iou_norm(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


class HeuristicDetector:
    """Find a skin-toned face, dark eye spots inside it, and a line-heavy region."""

    name = "heuristic"

    def detect(self, image: Image.Image) -> list[Detection]:
        small = _downsample(image.convert("RGB"))
        rgb = np.asarray(small)
        height, width = rgb.shape[:2]
        luma = _luma(rgb)
        skin = _skin_mask(rgb)

        faces = self._faces(skin, luma, width, height)
        small_pixels = rgb.astype(np.float32) / np.float32(255.0)
        faces = [face for face in faces if heuristic_face_ok(small_pixels, face.bbox)]
        full = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        creature = creature_box(full)
        # A face that misses the compact body is dropped. A textured cluster
        # beside it is not that body, for example a flower cluster.
        if creature is not None:
            body = creature[0]
            faces = [face for face in faces if boxes_overlap(face.bbox, body)]
        eyes: list[Detection] = []
        for face in faces:
            for box in _separated_eyes(luma, face.bbox, width, height):
                eyes.append(
                    Detection(
                        label="eye",
                        category="eye",
                        confidence=0.62,
                        bbox=box,
                        source=self.name,
                    )
                )
        if creature is not None and not any(
            _center_inside(eye.bbox, creature[0]) for eye in eyes
        ):
            for box in animal_eye_boxes(luma, creature[0], width, height, color=small_pixels):
                eyes.append(
                    Detection(
                        label="eye",
                        category="eye",
                        confidence=0.66,
                        bbox=box,
                        source=self.name,
                    )
                )
        architecture = self._architecture(rgb, luma, skin, faces, width, height)

        found = faces + eyes + architecture
        if creature is not None:
            found.append(
                Detection(
                    label="animal",
                    category="animal",
                    confidence=0.74,
                    bbox=creature[0],
                    source=self.name,
                )
            )
        if found:
            return found
        salient = self._salient(luma, width, height)
        return [salient] if salient is not None else []

    def _faces(self, skin: np.ndarray, _luma: np.ndarray, width: int, height: int) -> list[Detection]:
        box = _largest_bbox(skin, min_fraction=0.01)
        if box is None:
            return []
        x, y, w, h = box
        # Grow the box a little so hair and the chin stay inside the region.
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.08)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(width - x, w + pad_x * 2)
        h = min(height - y, h + pad_y * 2)
        area_fraction = (w * h) / float(width * height)
        skin_fraction = float(skin[y : y + h, x : x + w].mean()) if w and h else 0.0
        if skin_fraction < 0.35 or area_fraction > 0.9:
            return []
        bbox = _normalize((x, y, w, h), width, height)
        # A warm horizon is a short band across the frame. It is not a head.
        if is_face_bar(bbox):
            return []
        confidence = 0.45 + 0.4 * min(1.0, skin_fraction)
        # A face-like region is taller than it is wide, but a close crop may not be.
        if h >= w * 0.75:
            confidence = min(0.9, confidence + 0.05)
        return [
            Detection(
                label="face",
                category="face",
                confidence=confidence,
                bbox=bbox,
                source=self.name,
            )
        ]

    def _eyes(
        self,
        luma: np.ndarray,
        face: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> list[Detection]:
        fx, fy, fw, fh = face
        x0 = int(fx * width)
        y0 = int(fy * height)
        x1 = int((fx + fw) * width)
        upper = y0 + max(1, int(fh * height * 0.48))
        if x1 - x0 < 8 or upper - y0 < 6:
            return []
        region = luma[y0:upper, x0:x1]
        blurred = _box_blur(region)
        face_mean = float(blurred.mean())
        mid = blurred.shape[1] // 2
        eyes: list[Detection] = []
        halves = (
            (blurred[:, :mid], x0),
            (blurred[:, mid:], x0 + mid),
        )
        for half, origin_x in halves:
            if half.size == 0 or half.shape[0] < 2 or half.shape[1] < 2:
                continue
            flat = int(np.argmin(half))
            iy, ix = divmod(flat, half.shape[1])
            darkness = face_mean - float(half[iy, ix])
            if darkness < 0.08:
                continue
            eye_w = max(2, int(fw * width * 0.16))
            eye_h = max(2, int(fh * height * 0.1))
            cx = origin_x + ix
            cy = y0 + iy
            ex = max(0, cx - eye_w // 2)
            ey = max(0, cy - eye_h // 2)
            ex = min(ex, width - eye_w)
            ey = min(ey, height - eye_h)
            confidence = min(0.8, 0.35 + darkness)
            eyes.append(
                Detection(
                    label="eye",
                    category="eye",
                    confidence=confidence,
                    bbox=_normalize((ex, ey, eye_w, eye_h), width, height),
                    source=self.name,
                )
            )
        return eyes

    def _architecture(
        self,
        rgb: np.ndarray,
        luma: np.ndarray,
        skin: np.ndarray,
        faces: list[Detection],
        width: int,
        height: int,
    ) -> list[Detection]:
        gy = np.zeros_like(luma)
        gx = np.zeros_like(luma)
        gy[1:, :] = np.abs(luma[1:, :] - luma[:-1, :])
        gx[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
        edges = gx + gy
        rows, cols = 6, 6
        cell_h = max(1, height // rows)
        cell_w = max(1, width // cols)
        scores: list[tuple[float, int, int]] = []
        for row in range(rows):
            for col in range(cols):
                y0, x0 = row * cell_h, col * cell_w
                if y0 >= height or x0 >= width:
                    continue
                y1 = min(height, height if row == rows - 1 else (row + 1) * cell_h)
                x1 = min(width, width if col == cols - 1 else (col + 1) * cell_w)
                if y1 <= y0 or x1 <= x0:
                    continue
                if float(skin[y0:y1, x0:x1].mean()) > 0.3:
                    continue
                scores.append((float(edges[y0:y1, x0:x1].mean()), row, col))
        if not scores:
            return []
        strongest = max(score for score, _, _ in scores)
        if strongest < 0.08:
            return []
        chosen = [(row, col) for score, row, col in scores if score >= strongest * 0.72]
        if not chosen:
            return []
        min_row = min(row for row, _ in chosen)
        max_row = max(row for row, _ in chosen)
        min_col = min(col for _, col in chosen)
        max_col = max(col for _, col in chosen)
        x = min_col * cell_w
        y = min_row * cell_h
        x2 = width if max_col == cols - 1 else (max_col + 1) * cell_w
        y2 = height if max_row == rows - 1 else (max_row + 1) * cell_h
        box = _normalize((x, y, max(1, x2 - x), max(1, y2 - y)), width, height)
        area = box[2] * box[3]
        if area < 0.04 or area > 0.85:
            return []
        region = rgb[y:y2, x:x2]
        if region.size:
            red = region[:, :, 0].astype(np.int16)
            green = region[:, :, 1].astype(np.int16)
            blue = region[:, :, 2].astype(np.int16)
            leafy = (green > red + 8) & (green > blue + 8)
            # Furrows in a field are edges, and they are green. A facade is not.
            if float(leafy.mean()) > 0.45:
                return []
        if any(_iou_norm(box, face.bbox) > 0.25 for face in faces):
            return []
        return [
            Detection(
                label="structure",
                category="architecture",
                confidence=min(0.8, 0.4 + strongest),
                bbox=box,
                source=self.name,
            )
        ]

    def _salient(
        self,
        luma: np.ndarray,
        width: int,
        height: int,
    ) -> Detection | None:
        rows, cols = 6, 6
        cell_h = max(1, height // rows)
        cell_w = max(1, width // cols)
        best = 0.0
        best_at = (0, 0)
        found = False
        for row in range(rows):
            for col in range(cols):
                y0, x0 = row * cell_h, col * cell_w
                if y0 >= height or x0 >= width:
                    continue
                y1 = min(height, height if row == rows - 1 else (row + 1) * cell_h)
                x1 = min(width, width if col == cols - 1 else (col + 1) * cell_w)
                if y1 <= y0 or x1 <= x0:
                    continue
                score = float(luma[y0:y1, x0:x1].var())
                if score > best:
                    best = score
                    best_at = (row, col)
                    found = True
        if not found or best < 0.002:
            return None
        row, col = best_at
        row0, col0 = max(0, row - 1), max(0, col - 1)
        row1, col1 = min(rows, row + 2), min(cols, col + 2)
        x = col0 * cell_w
        y = row0 * cell_h
        x2 = width if col1 == cols else col1 * cell_w
        y2 = height if row1 == rows else row1 * cell_h
        return Detection(
            label="subject",
            category="subject",
            confidence=0.34,
            bbox=_normalize((x, y, max(1, x2 - x), max(1, y2 - y)), width, height),
            source=self.name,
        )


def drop_animal_heads(detections: list[Detection]) -> list[Detection]:
    """Drop a person or face box that is really the head of a detected animal.

    YOLO keeps people down to 0.15 confidence for recall. On a duck it put a
    0.17 (0.49 at full size) "person" on the head inside a 0.9 "bird" box,
    and that box then made the photo a portrait (skin, person, face steps).
    A person or face whose centre sits inside a confident detector animal,
    that is clearly smaller than the animal and less certain than it, is
    the animal's head. A rider or a person holding a pet is larger or
    surer than that and stays.
    """

    animals = [
        item
        for item in detections
        if item.category == "animal" and item.source != "heuristic" and float(item.confidence) >= 0.5
    ]
    if not animals:
        return list(detections)
    kept: list[Detection] = []
    for item in detections:
        if item.category in {"person", "face"} and item.source != "mediapipe":
            area = float(item.bbox[2]) * float(item.bbox[3])
            head = any(
                _center_inside(item.bbox, animal.bbox)
                and area < 0.5 * float(animal.bbox[2]) * float(animal.bbox[3])
                and float(item.confidence) < max(0.5, float(animal.confidence) - 0.2)
                for animal in animals
            )
            if head:
                continue
        kept.append(item)
    return kept
