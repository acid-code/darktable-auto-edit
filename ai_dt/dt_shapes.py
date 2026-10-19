"""Darktable drawn-mask shapes: binary layouts, outlines, and a reference renderer.

Every layout here follows ``src/develop/masks.h`` in darktable 5.6.0
(``DEVELOP_MASKS_VERSION`` 6). The records are what ``masks_history`` in an
XMP sidecar stores in ``darktable:mask_points``, little endian, packed:

- ellipse ``dt_masks_point_ellipse_t``: ``ffffffI`` (center, radius, rotation, border, flags), 28 bytes
- path ``dt_masks_point_path_t``: ``ffffffffI`` per point (corner, ctrl1, ctrl2, border, state), 36 bytes
- gradient ``dt_masks_point_gradient_t``: ``ffffffI`` (anchor, rotation, compression, steepness, curvature, state), 28 bytes
- group ``dt_masks_point_group_t``: ``IIIf`` per member (form id, parent id, state, opacity), 16 bytes

Coordinates are 0..1 of the input image, origin top left. The path
helpers turn a binary mask into closed paths the way darktable 5.6's own
AI object tool does after potrace: one group, one path per outline, holes
as ``difference`` members. Here the outline is traced with plain NumPy, so
no OpenCV is needed, then simplified with Douglas-Peucker and given smooth
Catmull-Rom handles (the same formula darktable uses for a point in its
``NORMAL`` state).

``render_forms`` is a small reference renderer for tests. It mirrors the
group combine rules in ``src/develop/masks/group.c`` and the gradient
formula in ``src/develop/masks/gradient.c``. Paths are filled as polygons
of their corners, without the feather, which is close enough to check an
outline round trip.
"""

from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# dt_masks_type_t
CIRCLE = 1 << 0
PATH = 1 << 1
GROUP = 1 << 2
CLONE = 1 << 3
GRADIENT = 1 << 4
ELLIPSE = 1 << 5
BRUSH = 1 << 6

MASKS_VERSION = 6

# dt_masks_state_t
STATE_USE = 1 << 0
STATE_SHOW = 1 << 1
STATE_INVERSE = 1 << 2
STATE_UNION = 1 << 3
STATE_INTERSECTION = 1 << 4
STATE_DIFFERENCE = 1 << 5
STATE_EXCLUSION = 1 << 6
STATE_SUM = 1 << 7

COMBINE_STATES = {
    "union": STATE_UNION,
    "intersect": STATE_INTERSECTION,
    "difference": STATE_DIFFERENCE,
    "exclusion": STATE_EXCLUSION,
    "sum": STATE_SUM,
}

# dt_masks_points_states_t
POINT_STATE_NORMAL = 1
POINT_STATE_USER = 2

# dt_masks_gradient_states_t
GRADIENT_LINEAR = 1
GRADIENT_SIGMOIDAL = 2

ELLIPSE_PROPORTIONAL = 1

_ELLIPSE = struct.Struct("<ffffffI")
_PATH_POINT = struct.Struct("<ffffffffI")
_GRADIENT = struct.Struct("<ffffffI")
_GROUP_MEMBER = struct.Struct("<IIIf")

POINT_SIZES = {
    ELLIPSE: _ELLIPSE.size,
    PATH: _PATH_POINT.size,
    GRADIENT: _GRADIENT.size,
    GROUP: _GROUP_MEMBER.size,
}

MAX_PATH_POINTS = 64


class ShapeError(ValueError):
    """A shape that darktable would not read back the same way."""


# --------------------------------------------------------------------------
# Binary records


@dataclass
class PathPoint:
    x: float
    y: float
    ctrl1: tuple[float, float]
    ctrl2: tuple[float, float]
    border: tuple[float, float] = (0.01, 0.01)
    state: int = POINT_STATE_USER


def encode_ellipse(cx: float, cy: float, rx: float, ry: float, *, rotation: float = 0.0,
                   border: float = 0.05, flags: int = ELLIPSE_PROPORTIONAL) -> bytes:
    return _ELLIPSE.pack(float(cx), float(cy), float(rx), float(ry), float(rotation), float(border), int(flags))


def decode_ellipse(blob: bytes) -> dict:
    if len(blob) != _ELLIPSE.size:
        raise ShapeError(f"an ellipse record is {_ELLIPSE.size} bytes, got {len(blob)}")
    cx, cy, rx, ry, rotation, border, flags = _ELLIPSE.unpack(blob)
    return {"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry,
            "rotation": rotation, "border": border, "flags": flags}


def encode_path(points: list[PathPoint]) -> bytes:
    if len(points) < 3:
        raise ShapeError("a closed path needs at least three corners")
    return b"".join(
        _PATH_POINT.pack(
            float(p.x), float(p.y),
            float(p.ctrl1[0]), float(p.ctrl1[1]),
            float(p.ctrl2[0]), float(p.ctrl2[1]),
            float(p.border[0]), float(p.border[1]),
            int(p.state),
        )
        for p in points
    )


def decode_path(blob: bytes) -> list[PathPoint]:
    if not blob or len(blob) % _PATH_POINT.size:
        raise ShapeError(f"a path record is a multiple of {_PATH_POINT.size} bytes, got {len(blob)}")
    points = []
    for offset in range(0, len(blob), _PATH_POINT.size):
        x, y, c1x, c1y, c2x, c2y, b0, b1, state = _PATH_POINT.unpack_from(blob, offset)
        points.append(PathPoint(x, y, (c1x, c1y), (c2x, c2y), (b0, b1), state))
    return points


def encode_gradient(anchor_x: float, anchor_y: float, rotation: float, *, compression: float = 0.25,
                    steepness: float = 0.0, curvature: float = 0.0,
                    state: int = GRADIENT_SIGMOIDAL) -> bytes:
    return _GRADIENT.pack(float(anchor_x), float(anchor_y), float(rotation), float(compression),
                          float(steepness), float(curvature), int(state))


def decode_gradient(blob: bytes) -> dict:
    if len(blob) != _GRADIENT.size:
        raise ShapeError(f"a gradient record is {_GRADIENT.size} bytes, got {len(blob)}")
    ax, ay, rotation, compression, steepness, curvature, state = _GRADIENT.unpack(blob)
    return {"type": "gradient", "anchor": (ax, ay), "rotation": rotation, "compression": compression,
            "steepness": steepness, "curvature": curvature, "state": state}


def member_state(mode: str = "union", *, first: bool = False, invert: bool = False) -> int:
    """The group state darktable writes for one member.

    The first member is copied, so it carries no combine flag. Every later
    member names how it joins what came before. The group is a left fold,
    so ``[a, b union, c intersect]`` is ``(a | b) & c``.
    """

    state = STATE_USE | STATE_SHOW
    if not first:
        try:
            state |= COMBINE_STATES[mode]
        except KeyError as exc:
            raise ShapeError(f"unknown combine mode {mode!r}") from exc
    if invert:
        state |= STATE_INVERSE
    return state


def encode_group(group_id: int, members: list[tuple[int, int, float]]) -> bytes:
    """``members`` is ``(form id, state, opacity)`` in combine order."""

    if not members:
        raise ShapeError("a group needs at least one member")
    return b"".join(
        _GROUP_MEMBER.pack(int(form_id) & 0xFFFFFFFF, int(group_id) & 0xFFFFFFFF, int(state), float(opacity))
        for form_id, state, opacity in members
    )


def decode_group(blob: bytes) -> list[dict]:
    if not blob or len(blob) % _GROUP_MEMBER.size:
        raise ShapeError(f"a group record is a multiple of {_GROUP_MEMBER.size} bytes, got {len(blob)}")
    members = []
    for offset in range(0, len(blob), _GROUP_MEMBER.size):
        form_id, parent, state, opacity = _GROUP_MEMBER.unpack_from(blob, offset)
        members.append({"form_id": form_id, "parent_id": parent, "state": state, "opacity": opacity,
                        "mode": state_mode(state), "invert": bool(state & STATE_INVERSE)})
    return members


def state_mode(state: int) -> str:
    for name, flag in COMBINE_STATES.items():
        if state & flag:
            return name
    return "copy"


# --------------------------------------------------------------------------
# Gradients


_EDGE_ROTATION = {"top": 0.0, "left": 90.0, "bottom": 180.0, "right": 270.0}


def edge_gradient(edge: str, extent: float, *, compression: float = 0.25) -> dict:
    """A linear gradient that is fully on at ``edge`` and fades out past ``extent``.

    ``extent`` is the 0..1 distance from that edge to the gradient line.
    In darktable's formula a rotation of 0 keeps the part above the anchor,
    90 the left part, 180 the bottom and 270 the right.
    """

    if edge not in _EDGE_ROTATION:
        raise ShapeError(f"edge must be one of {', '.join(_EDGE_ROTATION)}")
    extent = min(max(float(extent), 0.02), 0.98)
    if edge == "top":
        anchor = (0.5, extent)
    elif edge == "bottom":
        anchor = (0.5, 1.0 - extent)
    elif edge == "left":
        anchor = (extent, 0.5)
    else:
        anchor = (1.0 - extent, 0.5)
    return {
        "type": "gradient",
        "anchor": anchor,
        "rotation": _EDGE_ROTATION[edge],
        "compression": min(max(float(compression), 0.01), 1.0),
        "steepness": 0.0,
        "curvature": 0.0,
        "state": GRADIENT_SIGMOIDAL,
    }


def gradient_values(form: dict, width: int, height: int) -> np.ndarray:
    """darktable's gradient mask on a ``height x width`` grid of pixel centers."""

    wd = float(width)
    ht = float(height)
    hwscale = 1.0 / math.hypot(wd, ht)
    v = math.radians(-float(form["rotation"]))
    sinv = math.sin(v)
    cosv = math.cos(v)
    ax, ay = form["anchor"]
    xoffset = cosv * ax * wd + sinv * ay * ht
    yoffset = sinv * ax * wd - cosv * ay * ht
    compression = max(float(form.get("compression", 0.25)), 0.001)
    curvature = float(form.get("curvature", 0.0))
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    xs += 0.5
    ys += 0.5
    x0 = (cosv * xs + sinv * ys - xoffset) * hwscale
    y0 = (sinv * xs - cosv * ys - yoffset) * hwscale
    distance = y0 - curvature * x0 * x0
    if int(form.get("state", GRADIENT_SIGMOIDAL)) == GRADIENT_LINEAR:
        value = 0.5 + 0.5 * distance / compression
    else:
        erf = np.vectorize(math.erf, otypes=[np.float64])
        value = 0.5 + 0.5 * erf(distance / compression)
    return np.clip(value, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Binary mask -> outlines

_DIRECTIONS = ((0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1))


def _downsample(mask: np.ndarray, longest: int) -> np.ndarray:
    height, width = mask.shape
    scale = max(height, width) / float(longest)
    if scale <= 1.0:
        return mask
    new_h = max(2, int(round(height / scale)))
    new_w = max(2, int(round(width / scale)))
    ys = (np.arange(new_h) * height / new_h).astype(int)
    xs = (np.arange(new_w) * width / new_w).astype(int)
    # Area-ish downsample: average over the block each output pixel covers.
    y_edges = np.linspace(0, height, new_h + 1).astype(int)
    x_edges = np.linspace(0, width, new_w + 1).astype(int)
    integral = np.pad(mask.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y0, y1 = y_edges[:-1], np.maximum(y_edges[1:], y_edges[:-1] + 1)
    x0, x1 = x_edges[:-1], np.maximum(x_edges[1:], x_edges[:-1] + 1)
    total = (integral[y1][:, x1] - integral[y0][:, x1] - integral[y1][:, x0] + integral[y0][:, x0])
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    del ys, xs
    return (total / area).astype(np.float32)


def label_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected labels, 1..n. Pure Python breadth first; fine for <= 512 px masks."""

    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    count = 0
    flat = binary.ravel()
    lab = labels.ravel()
    for start in np.flatnonzero(flat):
        if lab[start]:
            continue
        count += 1
        lab[start] = count
        queue = deque([int(start)])
        while queue:
            index = queue.popleft()
            y, x = divmod(index, width)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor = ny * width + nx
                    if flat[neighbor] and not lab[neighbor]:
                        lab[neighbor] = count
                        queue.append(neighbor)
    return labels, count


def trace_boundary(component: np.ndarray) -> list[tuple[int, int]]:
    """Moore-neighbor trace of one 8-connected blob, as ``(x, y)`` pixel corners, clockwise."""

    padded = np.pad(component.astype(bool), 1)
    ys, xs = np.nonzero(padded)
    if ys.size == 0:
        return []
    order = np.lexsort((xs, ys))
    start = (int(ys[order[0]]), int(xs[order[0]]))
    if ys.size == 1:
        return [(start[1] - 1, start[0] - 1)]
    boundary = [start]
    current = start
    backtrack = 0  # west of the first raster pixel is always background
    first_move = None
    limit = 4 * int(ys.size) + 16
    for _ in range(limit):
        found = None
        for step in range(1, 9):
            direction = (backtrack + step) % 8
            dy, dx = _DIRECTIONS[direction]
            candidate = (current[0] + dy, current[1] + dx)
            if padded[candidate]:
                found = (direction, candidate)
                break
        if found is None:
            break
        direction, nxt = found
        previous = (direction - 1) % 8
        py, px = current[0] + _DIRECTIONS[previous][0], current[1] + _DIRECTIONS[previous][1]
        rel = (py - nxt[0], px - nxt[1])
        backtrack = _DIRECTIONS.index(rel)
        if current == start and first_move is not None and nxt == first_move:
            break
        if first_move is None:
            first_move = nxt
        current = nxt
        boundary.append(current)
    if len(boundary) > 1 and boundary[-1] == start:
        boundary.pop()
    return [(x - 1, y - 1) for y, x in boundary]


def _perpendicular(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    segment = b - a
    length = float(np.hypot(*segment))
    if length < 1e-12:
        return np.hypot(*(points - a).T)
    return np.abs(segment[0] * (points[:, 1] - a[1]) - segment[1] * (points[:, 0] - a[0])) / length


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        distances = _perpendicular(points[start + 1 : end], points[start], points[end])
        index = int(np.argmax(distances))
        if distances[index] > epsilon:
            split = start + 1 + index
            keep[split] = True
            stack.append((start, split))
            stack.append((split, end))
    return points[keep]


def simplify_closed(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Douglas-Peucker on a closed outline, split at the two farthest corners."""

    array = np.asarray(points, dtype=np.float64)
    if len(array) <= 3:
        return [tuple(p) for p in array]
    far = int(np.argmax(np.hypot(*(array - array[0]).T)))
    first = _rdp(array[: far + 1], epsilon)
    second = _rdp(np.vstack([array[far:], array[:1]]), epsilon)
    merged = np.vstack([first[:-1], second[:-1]])
    return [(float(x), float(y)) for x, y in merged]


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Signed shoelace area. Positive is clockwise on screen (y down)."""

    if len(points) < 3:
        return 0.0
    array = np.asarray(points, dtype=np.float64)
    x, y = array[:, 0], array[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@dataclass
class Outline:
    """One closed outline in 0..1 coordinates. ``hole`` outlines are subtracted."""

    points: list[tuple[float, float]]
    hole: bool = False
    area: float = 0.0
    extra: dict = field(default_factory=dict)


def mask_to_outlines(
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
    max_paths: int = 6,
    max_points: int = 48,
    min_area: float = 0.004,
    holes: bool = True,
    working_size: int = 256,
    tolerance: float = 1.2,
) -> list[Outline]:
    """Closed outlines around the parts of ``mask`` above ``threshold``.

    The biggest ``max_paths`` blobs above ``min_area`` (fraction of the
    frame) are kept, each simplified until it has at most ``max_points``
    corners. With ``holes``, a background pocket fully inside a kept blob
    becomes a hole outline. Corners within about a pixel of the frame
    edge are snapped onto the edge so a sky or a ground region reaches it.
    """

    values = np.asarray(mask, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 4:
        raise ShapeError("the mask must be a 2-D array at least 4 pixels on a side")
    # The writer keeps at most MAX_PATH_POINTS corners per path.
    max_points = max(3, min(int(max_points), MAX_PATH_POINTS))
    small = _downsample(values, working_size)
    height, width = small.shape
    binary = small >= float(threshold)
    total = float(height * width)
    outlines: list[Outline] = []

    def corners(component: np.ndarray, hole: bool) -> Outline | None:
        traced = trace_boundary(component)
        if len(traced) < 3:
            return None
        # Pixel centers, normalized. Snap the frame edge.
        pts = []
        for x, y in traced:
            fx = (x + 0.5) / width
            fy = (y + 0.5) / height
            if x <= 0:
                fx = 0.0
            elif x >= width - 1:
                fx = 1.0
            if y <= 0:
                fy = 0.0
            elif y >= height - 1:
                fy = 1.0
            pts.append((fx, fy))
        epsilon = tolerance / float(max(width, height))
        simplified = simplify_closed(pts, epsilon)
        while len(simplified) > max_points:
            epsilon *= 1.35
            simplified = simplify_closed(pts, epsilon)
        if len(simplified) < 3:
            return None
        area = abs(polygon_area(simplified))
        return Outline(simplified, hole=hole, area=area)

    labels, count = label_components(binary)
    if count == 0:
        return []
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    ranked = [index for index in np.argsort(sizes[1:])[::-1] + 1 if sizes[index] / total >= min_area]
    for index in ranked[:max_paths]:
        component = labels == index
        outline = corners(component, False)
        if outline is not None:
            outlines.append(outline)
        if not holes:
            continue
        pockets, pocket_count = label_components(~component)
        if pocket_count < 2:
            continue
        border_labels = set(np.unique(np.concatenate([pockets[0], pockets[-1], pockets[:, 0], pockets[:, -1]])))
        pocket_sizes = np.bincount(pockets.ravel(), minlength=pocket_count + 1)
        for pocket in range(1, pocket_count + 1):
            if pocket in border_labels or pocket_sizes[pocket] / total < min_area:
                continue
            hole = corners(pockets == pocket, True)
            if hole is not None:
                outlines.append(hole)
    return outlines


SHARP_TURN_DEG = 60.0


def _sharp_turn(px: float, py: float, x: float, y: float, nx: float, ny: float) -> bool:
    ax, ay, bx, by = x - px, y - py, nx - x, ny - y
    norm = math.hypot(ax, ay) * math.hypot(bx, by)
    if norm <= 0.0:
        return False
    return (ax * bx + ay * by) / norm < math.cos(math.radians(SHARP_TURN_DEG))


def smooth_path(points: list[tuple[float, float]], *, border: float = 0.01,
                smooth: bool = True, sharp_turns: bool = False) -> list[PathPoint]:
    """Path points with Catmull-Rom handles, the same rule darktable uses for a NORMAL point.

    A corner on the frame edge keeps sharp handles so the path does not
    bulge out of the frame and back. With ``sharp_turns`` (traced
    outlines) so does a turn of more than SHARP_TURN_DEG: Catmull-Rom handles there point across the corner and the
    curve swings out past the long side next to it, a few pixels outside the
    outline the mask was traced from (a gap ring round a subject).
    """

    count = len(points)
    if count < 3:
        raise ShapeError("a closed path needs at least three corners")
    out: list[PathPoint] = []
    border = float(min(max(border, 0.0005), 1.0))
    for index, (x, y) in enumerate(points):
        px, py = points[index - 1]
        nx, ny = points[(index + 1) % count]
        on_edge = x in (0.0, 1.0) or y in (0.0, 1.0)
        if smooth and not on_edge and not (sharp_turns and _sharp_turn(px, py, x, y, nx, ny)):
            dx = (nx - px) / 6.0
            dy = (ny - py) / 6.0
            ctrl1 = (x - dx, y - dy)
            ctrl2 = (x + dx, y + dy)
        else:
            ctrl1 = (x, y)
            ctrl2 = (x, y)
        out.append(PathPoint(x, y, ctrl1, ctrl2, (border, border), POINT_STATE_USER))
    return out


def mask_to_forms(mask: np.ndarray, *, border: float = 0.01, mode: str = "union", **options) -> list[dict]:
    """Path forms for a mask, ready for ``drawn_mask.splice_shapes``.

    The first outline is copied, later ones join with ``mode`` and holes
    are a ``difference``. Holes follow the blob they cut.
    """

    outlines = mask_to_outlines(mask, **options)
    forms: list[dict] = []
    for outline in outlines:
        forms.append(
            {
                "type": "path",
                "points": [[x, y] for x, y in outline.points],
                "border": border,
                "mode": "difference" if outline.hole else mode,
            }
        )
    if forms:
        forms[0]["mode"] = mode
    return forms


# --------------------------------------------------------------------------
# Reference renderer (tests and previews)


def _fill_polygon(points: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    xs = array[:, 0] * width
    ys = array[:, 1] * height
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float64)
    grid_x += 0.5
    grid_y += 0.5
    inside = np.zeros((height, width), dtype=bool)
    count = len(array)
    for index in range(count):
        x1, y1 = xs[index], ys[index]
        x2, y2 = xs[(index + 1) % count], ys[(index + 1) % count]
        if y1 == y2:
            continue
        crosses = (grid_y >= min(y1, y2)) & (grid_y < max(y1, y2))
        at_x = x1 + (grid_y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (grid_x < at_x)
    return inside.astype(np.float32)


def render_form(form: dict, width: int, height: int) -> np.ndarray:
    kind = form["type"]
    if kind == "path":
        points = [tuple(p) for p in form["points"]]
        return _fill_polygon(points, width, height)
    if kind == "gradient":
        return gradient_values(form, width, height)
    if kind == "ellipse":
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
        dx = ((xs + 0.5) / width - float(form["cx"])) / max(float(form["rx"]), 1e-6)
        dy = ((ys + 0.5) / height - float(form["cy"])) / max(float(form["ry"]), 1e-6)
        return ((dx * dx + dy * dy) <= 1.0).astype(np.float32)
    raise ShapeError(f"cannot render a {kind} form")


def render_forms(forms: list[dict], width: int, height: int, *, invert: bool = False) -> np.ndarray:
    """Combine forms the way a darktable group does: a left fold over members."""

    buffer = np.zeros((height, width), dtype=np.float32)
    for index, form in enumerate(forms):
        values = render_form(form, width, height)
        if form.get("invert"):
            values = 1.0 - values
        opacity = float(form.get("opacity", 1.0))
        mode = "copy" if index == 0 else str(form.get("mode") or "union")
        if mode == "copy":
            buffer = values * opacity
        elif mode in {"union", "sum"}:
            buffer = np.maximum(buffer, values * opacity)
        elif mode == "intersect":
            buffer = np.where((buffer > 0) & (values > 0), np.minimum(buffer, values * opacity), 0.0)
        elif mode == "difference":
            other = values * opacity
            buffer = np.where((buffer > 0) & (other > 0), buffer * (1.0 - other), buffer)
        elif mode == "exclusion":
            other = values * opacity
            buffer = np.maximum(buffer, other) - np.minimum(buffer, other)
        else:
            raise ShapeError(f"unknown combine mode {mode!r}")
    if invert:
        buffer = 1.0 - buffer
    return buffer.astype(np.float32)


def iou(first: np.ndarray, second: np.ndarray, threshold: float = 0.5) -> float:
    a = np.asarray(first) >= threshold
    b = np.asarray(second) >= threshold
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)
