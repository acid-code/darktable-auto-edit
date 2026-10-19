"""Add one drawn ellipse to a Darktable sidecar without replacing its history.

Darktable 5.4 stores drawn shapes in ``masks_history`` and points a module
at a shape through ``blendop_params`` (``mask_mode`` and ``mask_id``).
A style file cannot carry that shape. This writer keeps every history item
already in the sidecar and appends one exposure whose blend blob points at
a new ellipse. The layout follows ``dt_masks_point_ellipse_t`` and the
420-byte ``dt_develop_blend_params_t``. Darktable 5.6.2 still uses that blob and mask version 6. It upgrades an exposure version 6 module to version 7 on import.

The caller must apply the file with ``image:apply_sidecar()`` only after
the darkroom has closed the photo. Applying it while the photo is on
screen deadlocks the pixelpipe.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import sys
import zlib
from html import escape
from pathlib import Path

# Darktable runs this file by path. The script's own folder is then the only
# entry on the path, and `import ai_dt` cannot see the package above it.
_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from ai_dt.mapping import DEFAULT_BLENDOP, BLENDOP_VERSION, EXPOSURE_VERSION, exposure_params
from ai_dt.revert import _HISTORY_ITEM, _attr, is_ai_editor_history_name

_BLEND_BYTES = 420
_OFFSET_MASK_MODE = 0
_OFFSET_OPACITY = 16
_OFFSET_MASK_ID = 24
_MASK_MODE_DRAWN = 3  # ENABLED | MASK
_ELLIPSE_TYPE = 1 << 5
_PATH_TYPE = 1 << 1
_GROUP_TYPE = 4
_ELLIPSE_VERSION = 6  # DEVELOP_MASKS_VERSION in darktable 5.4.1
_GROUP_FIRST_STATE = 3
_GROUP_NEXT_STATE = 131
_ELLIPSE_PROPORTIONAL = 1
_MASK_ID_BASE = 0x10000000


class MaskSpliceError(ValueError):
    """The sidecar was left unchanged because the splice was not safe."""


def _decode_gz(encoded: str) -> bytes:
    if not encoded.startswith("gz"):
        raise MaskSpliceError("the default blend blob is not a darktable gz value")
    return zlib.decompress(base64.b64decode(encoded[4:]))


def _encode_gz(raw: bytes) -> str:
    compressed = zlib.compress(raw, 9)
    factor = min(len(raw) // len(compressed) + 1, 99)
    return f"gz{factor:02d}{base64.b64encode(compressed).decode('ascii')}"


def _default_blend() -> bytes:
    raw = _decode_gz(DEFAULT_BLENDOP)
    if len(raw) != _BLEND_BYTES:
        raise MaskSpliceError(
            f"the default blend blob is {len(raw)} bytes, expected {_BLEND_BYTES}"
        )
    return raw


def _check_unit(name: str, value: float) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise MaskSpliceError(f"{name} must sit inside the frame, got {value}")
    return number


def eye_ellipse(local_edits: list) -> dict | None:
    """One ellipse covering every detected eye, or None when no eye was found."""

    boxes: list[tuple[float, float, float, float]] = []
    lifts: list[float] = []
    for edit in local_edits:
        if not isinstance(edit, dict):
            continue
        category = str(edit.get("category") or "")
        target = str(edit.get("target") or "")
        if category != "eye" and "eye" not in target.lower():
            continue
        box = edit.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            boxes.append(tuple(float(part) for part in box))
            lifts.append(float(edit.get("exposure_ev") or 0.0))
        except (TypeError, ValueError):
            continue
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    width = x1 - x0
    height = y1 - y0
    if width <= 0.0 or height <= 0.0:
        return None
    return {
        "cx": round(x0 + width / 2.0, 5),
        "cy": round(y0 + height / 2.0, 5),
        "rx": round(width / 2.0, 5),
        "ry": round(height / 2.0, 5),
        "exposure_ev": round(max(lifts), 4),
        "name": "AI eyes",
    }


def _stack_for_append(document: str) -> tuple[str, list[str], int]:
    """Every history row, and the index where one new module is inserted.

    ``history_end`` counts the active rows. The new module takes that index.
    Later rows stay in the stack and move up one number. A count past the
    last row inserts at the end.
    """

    history = [
        match.group(0)
        for match in _HISTORY_ITEM.finditer(document)
        if _attr(match.group(0), "darktable:operation")
    ]
    end_match = re.search(r'darktable:history_end="(\d+)"', document)
    if not history or end_match is None:
        raise MaskSpliceError("this sidecar has no history stack, so nothing was written")
    history_end = int(end_match.group(1))
    if history_end > len(history):
        history_end = len(history)
    if history_end < 0:
        history_end = 0
    return document, history, history_end


def _bump_attr(tag: str, attr: str, start: int) -> str:
    found = re.search(rf'{attr}="(\d+)"', tag)
    if found is None:
        return tag
    value = int(found.group(1))
    if value < start:
        return tag
    return tag[: found.start(1)] + str(value + 1) + tag[found.end(1) :]


def _shift_numbers_from(document: str, start: int) -> str:
    """Free ``start`` by moving every later history number, and its masks, up one."""

    def bump(match: re.Match[str]) -> str:
        tag = match.group(0)
        if _attr(tag, "darktable:operation"):
            return _bump_attr(tag, "darktable:num", start)
        return _bump_attr(_bump_attr(tag, "darktable:mask_num", start), "darktable:num", start)

    return _HISTORY_ITEM.sub(bump, document)


def _drop_inactive_rows(document: str, history: list[str], history_end: int) -> str:
    """Remove the rows past ``history_end`` and their shapes, as darktable does.

    A new edit in darktable's darkroom discards every row above the
    selected one. Keeping them left the new module in the middle of the
    stack with older, inactive rows above it, so the top of the history
    was not the step just accepted.
    """

    dropped = history[history_end:]
    nums = {_attr(tag, "darktable:num") for tag in dropped if _attr(tag, "darktable:num")}
    dropped_set = set(dropped)

    def drop(match: re.Match[str]) -> str:
        tag = match.group(0)
        if tag in dropped_set:
            return ""
        if not _attr(tag, "darktable:operation") and _attr(tag, "darktable:mask_num") in nums:
            return ""
        return tag

    return _HISTORY_ITEM.sub(drop, document)


def _prepare_insert(document: str) -> tuple[str, list[str], int, int]:
    """Return the document, its history, the active count, and the new item number.

    The new module always becomes the top of the stack: rows past
    ``history_end`` (undone in darktable's history panel) are discarded
    first, the way darktable discards them on any new edit. A count past
    the last row appends with the next free number, so a gap in the
    existing numbers does not turn an active row off. The caller writes
    ``history_end = active + 1``, which is then the row count.
    """

    document, history, history_end = _stack_for_append(document)
    if history_end < len(history):
        document = _drop_inactive_rows(document, history, history_end)
        document, history, history_end = _stack_for_append(document)
    numbers = [int(_attr(tag, "darktable:num") or "0") for tag in history]
    return document, history, history_end, (max(numbers) + 1) if numbers else 0


def _forms_snapshot(document: str, before_num: int) -> list[str]:
    """The shape records darktable treats as current just before history row ``before_num``.

    darktable keeps a full copy of every shape on each history row that
    changed masks, and uses the copy on the last such row
    (``dt_masks_read_masks_history``). A new row that carries only its own
    shapes makes every earlier masked module lose its mask.
    """

    records = [
        match.group(0)
        for match in _HISTORY_ITEM.finditer(document)
        if not _attr(match.group(0), "darktable:operation") and _attr(match.group(0), "darktable:mask_num")
    ]
    nums = sorted(
        {int(_attr(tag, "darktable:mask_num")) for tag in records if int(_attr(tag, "darktable:mask_num")) < before_num}
    )
    if not nums:
        return []
    latest = str(nums[-1])
    return [tag for tag in records if _attr(tag, "darktable:mask_num") == latest]


def _carried_forms(document: str, num: int, new_ids: set[int]) -> str:
    """The current shapes, re-filed under the new row ``num`` (darktable's snapshot rule)."""

    carried = []
    for tag in _forms_snapshot(document, num):
        if int(_attr(tag, "darktable:mask_id") or "0") in new_ids:
            continue
        carried.append(re.sub(r'darktable:mask_num="\d+"', f'darktable:mask_num="{num}"', tag, count=1))
    return "".join(carried)


def verify_top(document: str, name: str) -> None:
    """The step just written is the active top row and its shapes are filed under it."""

    _document, history, history_end = _stack_for_append(document)
    end_match = re.search(r'darktable:history_end="(\d+)"', document)
    written_end = int(end_match.group(1)) if end_match else -1
    if written_end != len(history):
        raise MaskSpliceError(
            f"history_end is {written_end} but the stack has {len(history)} rows, "
            "so the new step would not be the active top"
        )
    top = history[-1]
    if _attr(top, "darktable:multi_name") != escape(name, quote=True):
        raise MaskSpliceError(f"the top history row is {_attr(top, 'darktable:multi_name')!r}, not {name!r}")
    num = _attr(top, "darktable:num")
    blend_id = None
    try:
        blend = _decode_gz(_attr(top, "darktable:blendop_params"))
        blend_id = struct.unpack_from("<I", blend, _OFFSET_MASK_ID)[0]
    except (MaskSpliceError, ValueError, zlib.error, struct.error):
        blend_id = None
    if blend_id:
        filed = {
            int(_attr(tag, "darktable:mask_id") or "0")
            for tag in (m.group(0) for m in _HISTORY_ITEM.finditer(document))
            if not _attr(tag, "darktable:operation") and _attr(tag, "darktable:mask_num") == num
        }
        if blend_id not in filed:
            raise MaskSpliceError("the new module's mask is not filed under its history row, so darktable would drop it")


def _place_history_item(document: str, item: str, num: int) -> str:
    """Insert one history tag so its ``num`` sits in order."""

    for match in _HISTORY_ITEM.finditer(document):
        tag = match.group(0)
        if not _attr(tag, "darktable:operation"):
            continue
        existing = int(_attr(tag, "darktable:num") or "0")
        if existing > num:
            return document[: match.start()] + item + document[match.start() :]
    last_end = None
    for match in _HISTORY_ITEM.finditer(document):
        if _attr(match.group(0), "darktable:operation"):
            last_end = match.end()
    if last_end is None:
        raise MaskSpliceError("the last history item could not be found again")
    return document[:last_end] + item + document[last_end:]


def splice_ellipse_exposure(
    document: str,
    *,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    exposure_ev: float,
    name: str = "AI eyes",
) -> str:
    """Return a sidecar with one masked exposure inserted at the active end.

    Rows after ``history_end`` stay in the stack and move up one number.
    A count past the last row appends. Refuses a document with no history
    and an ellipse that leaves the frame. The input string is not modified.
    """

    center_x = _check_unit("center_x", center_x)
    center_y = _check_unit("center_y", center_y)
    radius_x = _check_unit("radius_x", radius_x)
    radius_y = _check_unit("radius_y", radius_y)
    if radius_x <= 0.0 or radius_y <= 0.0:
        raise MaskSpliceError("the ellipse radius must be greater than zero")
    if not name.startswith("AI "):
        raise MaskSpliceError("the added module name must start with 'AI '")

    document, history, history_end, next_num = _prepare_insert(document)
    priorities = [
        int(_attr(tag, "darktable:multi_priority") or "0")
        for tag in history
        if _attr(tag, "darktable:operation") == "exposure"
    ]
    # The default pipe already has exposure,0. A first eye at priority 1
    # has no slot, and Darktable drops it. history_end stays one past the
    # rows. A stack that already has exposure keeps the next priority.
    priority = (max(priorities) + 1) if priorities else 0
    versions = [
        int(value)
        for value in (_attr(tag, "darktable:blendop_version") for tag in history)
        if value
    ]
    blend_version = max(versions) if versions else BLENDOP_VERSION
    mask_id = _MASK_ID_BASE
    while f'darktable:mask_id="{mask_id}"' in document:
        mask_id += 1

    points = struct.pack(
        "<ffffffI",
        center_x,
        center_y,
        radius_x,
        radius_y,
        0.0,
        0.05,
        _ELLIPSE_PROPORTIONAL,
    )
    blend = bytearray(_default_blend())
    struct.pack_into("<I", blend, _OFFSET_MASK_MODE, _MASK_MODE_DRAWN)
    struct.pack_into("<f", blend, _OFFSET_OPACITY, 100.0)
    struct.pack_into("<I", blend, _OFFSET_MASK_ID, mask_id & 0xFFFFFFFF)
    params = exposure_params(float(exposure_ev))
    safe_name = escape(name, quote=True)
    item = (
        f'<rdf:li darktable:num="{next_num}" darktable:operation="exposure" '
        f'darktable:enabled="1" darktable:modversion="{EXPOSURE_VERSION}" '
        f'darktable:params="{params.hex()}" '
        f'darktable:blendop_params="{_encode_gz(bytes(blend))}" '
        f'darktable:blendop_version="{blend_version}" '
        f'darktable:multi_name="{safe_name}" darktable:multi_priority="{priority}" '
        f'darktable:multi_name_hand_edited="1"/>'
    )
    spliced = _place_history_item(document, item, next_num)
    spliced = re.sub(
        r'darktable:history_end="\d+"',
        f'darktable:history_end="{history_end + 1}"',
        spliced,
        count=1,
    )
    spliced = _pipeline_slot(spliced, "exposure", priority)

    mask_li = _carried_forms(spliced, next_num, {mask_id}) + (
        f'<rdf:li darktable:mask_num="{next_num}" '
        f'darktable:mask_id="{mask_id}" '
        f'darktable:mask_type="{_ELLIPSE_TYPE}" '
        f'darktable:mask_name="{safe_name}" '
        f'darktable:mask_version="{_ELLIPSE_VERSION}" '
        f'darktable:mask_points="{points.hex()}" '
        f'darktable:mask_nb="1" '
        f'darktable:mask_src="{"00" * 8}"/>'
    )
    return _insert_masks(spliced, mask_li)


def splice_path_exposure(
    document: str,
    *,
    points: list[dict],
    exposure_ev: float,
    name: str,
) -> str:
    """Append one exposure limited to a closed path.

    Each point is ``dt_masks_point_path_t``: the corner, both Bézier
    handles on that corner, a short feather, and the normal point state.
    Darktable closes the path. The same insert rules as an eye ellipse
    apply, including the open exposure slot.
    """

    if not is_ai_editor_history_name(name):
        raise MaskSpliceError("the added module name is not an editor step")
    packed = []
    for point in points[:48]:
        try:
            x = _check_unit("x", point["x"])
            y = _check_unit("y", point["y"])
        except (KeyError, TypeError) as exc:
            raise MaskSpliceError("a path point is missing its place") from exc
        packed.append(struct.pack("<ffffffffi", x, y, x, y, x, y, 0.02, 0.02, 1))
    if len(packed) < 5:
        raise MaskSpliceError("the object outline needs at least five corners")

    document, history, history_end, next_num = _prepare_insert(document)
    priorities = [
        int(_attr(tag, "darktable:multi_priority") or "0")
        for tag in history
        if _attr(tag, "darktable:operation") == "exposure"
    ]
    priority = (max(priorities) + 1) if priorities else 0
    versions = [
        int(value)
        for value in (_attr(tag, "darktable:blendop_version") for tag in history)
        if value
    ]
    blend_version = max(versions) if versions else BLENDOP_VERSION
    mask_id = _MASK_ID_BASE
    while f'darktable:mask_id="{mask_id}"' in document:
        mask_id += 1
    blend = bytearray(_default_blend())
    struct.pack_into("<I", blend, _OFFSET_MASK_MODE, _MASK_MODE_DRAWN)
    struct.pack_into("<f", blend, _OFFSET_OPACITY, 100.0)
    struct.pack_into("<I", blend, _OFFSET_MASK_ID, mask_id & 0xFFFFFFFF)
    params = exposure_params(float(exposure_ev))
    safe_name = escape(name, quote=True)
    item = (
        f'<rdf:li darktable:num="{next_num}" darktable:operation="exposure" '
        f'darktable:enabled="1" darktable:modversion="{EXPOSURE_VERSION}" '
        f'darktable:params="{params.hex()}" '
        f'darktable:blendop_params="{_encode_gz(bytes(blend))}" '
        f'darktable:blendop_version="{blend_version}" '
        f'darktable:multi_name="{safe_name}" darktable:multi_priority="{priority}" '
        f'darktable:multi_name_hand_edited="1"/>'
    )
    spliced = _place_history_item(document, item, next_num)
    spliced = re.sub(
        r'darktable:history_end="\d+"',
        f'darktable:history_end="{history_end + 1}"',
        spliced,
        count=1,
    )
    spliced = _pipeline_slot(spliced, "exposure", priority)
    blob = b"".join(packed)
    mask_li = _carried_forms(spliced, next_num, {mask_id}) + (
        f'<rdf:li darktable:mask_num="{next_num}" '
        f'darktable:mask_id="{mask_id}" '
        f'darktable:mask_type="{_PATH_TYPE}" '
        f'darktable:mask_name="{safe_name}" '
        f'darktable:mask_version="{_ELLIPSE_VERSION}" '
        f'darktable:mask_points="{blob.hex()}" '
        f'darktable:mask_nb="{len(packed)}" '
        f'darktable:mask_src="{"00" * 8}"/>'
    )
    return _insert_masks(spliced, mask_li)


# Operations a shape step may carry, and the parameter blob length each one has.
SHAPE_OPERATIONS = {
    "exposure": 24,
    "colorbalancergb": 132,
    "colorzones": 520,
}
_OFFSET_MASK_COMBINE = 20
_COMBINE_MASKS_POS = 0x04  # DEVELOP_COMBINE_MASKS_POS: drawn mask polarity, inverted


def _fresh_ids(document: str, count: int) -> list[int]:
    taken = document
    ids = []
    for _ in range(count):
        mask_id = _MASK_ID_BASE
        while f'darktable:mask_id="{mask_id}"' in taken:
            mask_id += 1
        ids.append(mask_id)
        taken += f'darktable:mask_id="{mask_id}"'
    return ids


def _form_record(form: dict) -> tuple[int, bytes, int]:
    """Mask type, packed points, and point count for one shape spec."""

    from ai_dt import dt_shapes

    kind = str(form.get("type") or "")
    if kind == "path":
        raw = form.get("points") or []
        corners = []
        for point in raw[: dt_shapes.MAX_PATH_POINTS]:
            try:
                if isinstance(point, dict):
                    x, y = float(point["x"]), float(point["y"])
                else:
                    x, y = float(point[0]), float(point[1])
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise MaskSpliceError("a path point is missing its place") from exc
            corners.append((_check_unit("x", x), _check_unit("y", y)))
        if len(corners) < 3:
            raise MaskSpliceError("a path needs at least three corners")
        points = dt_shapes.smooth_path(
            corners,
            border=float(form.get("border", 0.01)),
            smooth=bool(form.get("smooth", True)),
        )
        return dt_shapes.PATH, dt_shapes.encode_path(points), len(points)
    if kind == "gradient":
        anchor = form.get("anchor") or (0.5, 0.5)
        try:
            ax, ay = float(anchor[0]), float(anchor[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise MaskSpliceError("a gradient needs an anchor") from exc
        _check_unit("anchor_x", ax)
        _check_unit("anchor_y", ay)
        blob = dt_shapes.encode_gradient(
            ax,
            ay,
            float(form.get("rotation", 0.0)) % 360.0,
            compression=min(max(float(form.get("compression", 0.25)), 0.001), 1.0),
            steepness=float(form.get("steepness", 0.0)),
            curvature=min(max(float(form.get("curvature", 0.0)), -2.0), 2.0),
            state=int(form.get("state", dt_shapes.GRADIENT_SIGMOIDAL)),
        )
        return dt_shapes.GRADIENT, blob, 1
    if kind == "ellipse":
        cx = _check_unit("center_x", form["cx"])
        cy = _check_unit("center_y", form["cy"])
        rx = _check_unit("radius_x", form["rx"])
        ry = _check_unit("radius_y", form["ry"])
        if rx <= 0.0 or ry <= 0.0:
            raise MaskSpliceError("the ellipse radius must be greater than zero")
        blob = dt_shapes.encode_ellipse(
            cx, cy, rx, ry,
            rotation=float(form.get("rotation", 0.0)),
            border=float(form.get("border", 0.05)),
        )
        return dt_shapes.ELLIPSE, blob, 1
    raise MaskSpliceError(f"unknown shape type {kind!r}")


def splice_shapes(
    document: str,
    *,
    forms: list[dict],
    operation: str,
    params: bytes,
    modversion: int,
    name: str,
    invert: bool = False,
    opacity: float = 100.0,
) -> str:
    """Insert one module whose drawn mask is a group of paths, gradients, and ellipses.

    ``forms`` are combined in order, the way darktable's group does it:
    the first form is copied, then each later form joins with its
    ``mode`` (``union``, ``intersect``, ``difference``, ``exclusion``).
    A form with ``invert`` is inverted before it joins. ``invert`` on the
    whole call flips the drawn-mask polarity in the blend blob, which is
    how a background is the inverse of a subject. The blend points at the
    group, as darktable 5.6's AI object tool does. The usual insert rules
    apply: insert at ``history_end``, priority 0 for a first instance.
    """

    if operation not in SHAPE_OPERATIONS:
        raise MaskSpliceError(f"a shape step cannot carry {operation}")
    if len(params) != SHAPE_OPERATIONS[operation]:
        raise MaskSpliceError(f"the {operation} blob is the wrong length")
    if not (name.startswith("AI ") or is_ai_editor_history_name(name)):
        raise MaskSpliceError("the added module name is not an editor step")
    cleaned = [form for form in forms if isinstance(form, dict)][:16]
    if not cleaned:
        raise MaskSpliceError("there is no shape to write")
    records = [_form_record(form) for form in cleaned]

    document, history, history_end, next_num = _prepare_insert(document)
    priorities = [
        int(_attr(tag, "darktable:multi_priority") or "0")
        for tag in history
        if _attr(tag, "darktable:operation") == operation
    ]
    priority = (max(priorities) + 1) if priorities else 0
    versions = [
        int(value)
        for value in (_attr(tag, "darktable:blendop_version") for tag in history)
        if value
    ]
    blend_version = max(versions) if versions else BLENDOP_VERSION
    ids = _fresh_ids(document, len(records) + 1)
    form_ids, group_id = ids[:-1], ids[-1]
    blend = bytearray(_default_blend())
    struct.pack_into("<I", blend, _OFFSET_MASK_MODE, _MASK_MODE_DRAWN)
    struct.pack_into("<f", blend, _OFFSET_OPACITY, min(max(float(opacity), 0.0), 100.0))
    struct.pack_into("<I", blend, _OFFSET_MASK_ID, group_id & 0xFFFFFFFF)
    if invert:
        combine = struct.unpack_from("<I", blend, _OFFSET_MASK_COMBINE)[0]
        struct.pack_into("<I", blend, _OFFSET_MASK_COMBINE, combine | _COMBINE_MASKS_POS)
    safe_name = escape(name, quote=True)
    item = (
        f'<rdf:li darktable:num="{next_num}" darktable:operation="{operation}" '
        f'darktable:enabled="1" darktable:modversion="{int(modversion)}" '
        f'darktable:params="{params.hex()}" '
        f'darktable:blendop_params="{_encode_gz(bytes(blend))}" '
        f'darktable:blendop_version="{blend_version}" '
        f'darktable:multi_name="{safe_name}" darktable:multi_priority="{priority}" '
        f'darktable:multi_name_hand_edited="1"/>'
    )
    spliced = _place_history_item(document, item, next_num)
    spliced = re.sub(
        r'darktable:history_end="\d+"',
        f'darktable:history_end="{history_end + 1}"',
        spliced,
        count=1,
    )
    spliced = _pipeline_slot(spliced, operation, priority)

    from ai_dt import dt_shapes

    entries = []
    members = []
    for index, (form, form_id, (mask_type, points, count)) in enumerate(zip(cleaned, form_ids, records)):
        entries.append(_mask_record(next_num, form_id, mask_type, safe_name, points, count))
        state = dt_shapes.member_state(
            str(form.get("mode") or "union"),
            first=index == 0,
            invert=bool(form.get("invert")),
        )
        members.append((form_id, state, min(max(float(form.get("opacity", 1.0)), 0.0), 1.0)))
    group_points = dt_shapes.encode_group(group_id, members)
    entries.append(_mask_record(next_num, group_id, _GROUP_TYPE, safe_name, group_points, len(members)))
    carried = _carried_forms(spliced, next_num, set(ids))
    return _insert_masks(spliced, carried + "".join(entries))


def read_mask_records(document: str) -> list[dict]:
    """Every shape in ``masks_history``, with its points decoded where the type is known."""

    from ai_dt import dt_shapes

    found = []
    for match in _HISTORY_ITEM.finditer(document):
        tag = match.group(0)
        if _attr(tag, "darktable:operation") or not _attr(tag, "darktable:mask_id"):
            continue
        raw = _attr(tag, "darktable:mask_points") or ""
        try:
            blob = _decode_gz(raw) if raw.startswith("gz") else bytes.fromhex(raw)
        except (ValueError, MaskSpliceError, zlib.error):
            blob = b""
        mask_type = int(_attr(tag, "darktable:mask_type") or "0")
        record = {
            "num": int(_attr(tag, "darktable:mask_num") or "0"),
            "id": int(_attr(tag, "darktable:mask_id") or "0"),
            "type": mask_type,
            "name": _attr(tag, "darktable:mask_name") or "",
            "version": int(_attr(tag, "darktable:mask_version") or "0"),
            "nb": int(_attr(tag, "darktable:mask_nb") or "0"),
            "blob": blob,
        }
        try:
            if mask_type == dt_shapes.PATH:
                record["points"] = dt_shapes.decode_path(blob)
            elif mask_type == dt_shapes.GRADIENT:
                record["points"] = dt_shapes.decode_gradient(blob)
            elif mask_type == dt_shapes.ELLIPSE:
                record["points"] = dt_shapes.decode_ellipse(blob)
            elif mask_type & dt_shapes.GROUP:
                record["points"] = dt_shapes.decode_group(blob)
        except dt_shapes.ShapeError:
            record["points"] = None
        found.append(record)
    return found


def decode_blend(encoded: str) -> dict:
    """The fields of a blend blob this editor writes."""

    raw = _decode_gz(encoded) if encoded.startswith("gz") else bytes.fromhex(encoded)
    mode, _cst, _blend_mode, _parameter, opacity, combine, mask_id = struct.unpack_from("<IiIffII", raw, 0)
    return {
        "length": len(raw),
        "mask_mode": mode,
        "opacity": opacity,
        "mask_combine": combine,
        "mask_id": mask_id,
        "inverted": bool(combine & _COMBINE_MASKS_POS),
    }


def _pipeline_slot(document: str, operation: str, priority: int) -> str:
    """Keep the new instance in the pipeline list the darkroom reloads."""

    match = re.search(r'darktable:iop_order_list="([^"]*)"', document)
    if match is None:
        return _write_pipeline_list(document, operation, priority)
    parts = match.group(1).split(",")
    if len(parts) % 2:
        return document
    slots = list(zip(parts[0::2], parts[1::2]))
    pair = (operation, str(priority))
    if pair in slots:
        return document
    insert_at = len(slots)
    for index, (name, _instance) in enumerate(slots):
        if name == operation:
            insert_at = index + 1
    slots.insert(insert_at, pair)
    rebuilt = ",".join(f"{name},{instance}" for name, instance in slots)
    return document[: match.start(1)] + rebuilt + document[match.end(1) :]


def _history_instances(document: str) -> list[tuple[str, int]]:
    """``(operation, multi_priority)`` of every history row, in order."""

    found: list[tuple[str, int]] = []
    history = re.search(r"<darktable:history>(.*?)</darktable:history>", document, re.DOTALL)
    if history is None:
        return found
    for row in re.finditer(r"<rdf:li\b([^>]*)>", history.group(1)):
        attrs = row.group(1)
        op = re.search(r'darktable:operation="([^"]+)"', attrs)
        if op is None:
            continue
        prio = re.search(r'darktable:multi_priority="(-?\d+)"', attrs)
        found.append((op.group(1), int(prio.group(1)) if prio else 0))
    return found


def _write_pipeline_list(document: str, operation: str, priority: int) -> str:
    """Write the full ``iop_order_list`` darktable needs for a second instance.

    A stock sidecar has only ``iop_order_version`` (5 for a JPEG in 5.x) and
    no list: darktable rebuilds instance 0 of every module from its built-in
    table. A second instance is not in that table, so darktable 5.6.2 logs
    "cannot get iop-order for colorbalancergb instance 1" and "can't find
    history entry N while adding mask" and drops the row and its masks.
    darktable itself writes the list as soon as a module has two instances
    (``dt_ioppr_write_iop_order``), with the version unchanged, so this does
    the same: the built-in table, each extra instance right after the
    instances of its module, in priority order.
    """

    if priority <= 0:
        return document
    from ai_dt.iop_order import ORDERS

    version = re.search(r'darktable:iop_order_version="(\d+)"', document)
    if version is None or int(version.group(1)) not in ORDERS:
        return document
    slots = [(name, 0) for name in ORDERS[int(version.group(1))]]
    extra = sorted(
        {(name, prio) for name, prio in _history_instances(document) if prio > 0} | {(operation, int(priority))},
        key=lambda pair: pair[1],
    )
    for name, prio in extra:
        if (name, prio) in slots:
            continue
        at = None
        for index, (slot_name, _instance) in enumerate(slots):
            if slot_name == name:
                at = index + 1
        if at is None:
            # Not a built-in module: darktable could not place it either.
            return document
        slots.insert(at, (name, prio))
    listed = ",".join(f"{name},{prio}" for name, prio in slots)
    return document[: version.end()] + f'\n   darktable:iop_order_list="{listed}"' + document[version.end() :]


def splice_region_module(
    document: str,
    *,
    shapes: list[dict],
    params: bytes,
    operation: str,
    modversion: int,
    name: str,
) -> str:
    """Append one color-zones module limited to the given ellipses.

    One ellipse is referenced directly. Several ellipses are wrapped in a
    group, which is the mask the module blends through. The module is
    inserted at ``history_end``. Later rows stay and move up one number.
    """

    if operation != "colorzones":
        raise MaskSpliceError("only a color zones module can carry these shapes")
    if len(params) != 520:
        raise MaskSpliceError("the color zones blob is the wrong length")
    if not (name.startswith("AI ") or is_ai_editor_history_name(name)):
        raise MaskSpliceError("the added module name is not an editor step")
    cleaned = []
    for shape in shapes[:4]:
        cx = _check_unit("center_x", shape["cx"])
        cy = _check_unit("center_y", shape["cy"])
        rx = min(_check_unit("radius_x", shape["rx"]), cx, 1.0 - cx)
        ry = min(_check_unit("radius_y", shape["ry"]), cy, 1.0 - cy)
        if rx <= 0.0 or ry <= 0.0:
            continue
        cleaned.append((cx, cy, rx, ry))
    if not cleaned:
        raise MaskSpliceError("there is no ellipse inside the frame")

    document, history, history_end, next_num = _prepare_insert(document)
    priorities = [
        int(_attr(tag, "darktable:multi_priority") or "0")
        for tag in history
        if _attr(tag, "darktable:operation") == operation
    ]
    # The pipeline lists colorzones,0 even when history has no color zones.
    # A new instance at priority 1 has no slot, and Darktable drops it on reload.
    priority = (max(priorities) + 1) if priorities else 0
    versions = [
        int(value)
        for value in (_attr(tag, "darktable:blendop_version") for tag in history)
        if value
    ]
    blend_version = max(versions) if versions else BLENDOP_VERSION

    def fresh_id(taken: str) -> int:
        mask_id = _MASK_ID_BASE
        while f'darktable:mask_id="{mask_id}"' in taken:
            mask_id += 1
        return mask_id

    taken = document
    ellipse_ids = []
    for _shape in cleaned:
        mask_id = fresh_id(taken)
        ellipse_ids.append(mask_id)
        taken += f'darktable:mask_id="{mask_id}"'
    group_id = fresh_id(taken) if len(ellipse_ids) > 1 else 0
    blend_id = group_id or ellipse_ids[0]
    blend = bytearray(_default_blend())
    struct.pack_into("<I", blend, _OFFSET_MASK_MODE, _MASK_MODE_DRAWN)
    struct.pack_into("<f", blend, _OFFSET_OPACITY, 100.0)
    struct.pack_into("<I", blend, _OFFSET_MASK_ID, blend_id & 0xFFFFFFFF)
    safe_name = escape(name, quote=True)
    item = (
        f'<rdf:li darktable:num="{next_num}" darktable:operation="{operation}" '
        f'darktable:enabled="1" darktable:modversion="{int(modversion)}" '
        f'darktable:params="{params.hex()}" '
        f'darktable:blendop_params="{_encode_gz(bytes(blend))}" '
        f'darktable:blendop_version="{blend_version}" '
        f'darktable:multi_name="{safe_name}" darktable:multi_priority="{priority}" '
        f'darktable:multi_name_hand_edited="1"/>'
    )
    spliced = _place_history_item(document, item, next_num)
    spliced = re.sub(
        r'darktable:history_end="\d+"',
        f'darktable:history_end="{history_end + 1}"',
        spliced,
        count=1,
    )
    spliced = _pipeline_slot(spliced, operation, priority)
    records = []
    for index, (mask_id, (cx, cy, rx, ry)) in enumerate(zip(ellipse_ids, cleaned)):
        points = struct.pack("<ffffffI", cx, cy, rx, ry, 0.0, 0.04, _ELLIPSE_PROPORTIONAL)
        records.append(_mask_record(next_num, mask_id, _ELLIPSE_TYPE, safe_name, points, 1))
    if group_id:
        group_points = b"".join(
            struct.pack(
                "<IIIf",
                form_id,
                group_id,
                _GROUP_FIRST_STATE if index == 0 else _GROUP_NEXT_STATE,
                1.0,
            )
            for index, form_id in enumerate(ellipse_ids)
        )
        records.append(
            _mask_record(next_num, group_id, _GROUP_TYPE, safe_name, group_points, len(ellipse_ids))
        )
    return _insert_masks(spliced, _carried_forms(spliced, next_num, set(ellipse_ids) | {group_id}) + "".join(records))


def _mask_record(
    mask_num: int,
    mask_id: int,
    mask_type: int,
    name: str,
    points: bytes,
    count: int,
) -> str:
    return (
        f'<rdf:li darktable:mask_num="{mask_num}" '
        f'darktable:mask_id="{mask_id}" '
        f'darktable:mask_type="{mask_type}" '
        f'darktable:mask_name="{name}" '
        f'darktable:mask_version="{_ELLIPSE_VERSION}" '
        f'darktable:mask_points="{points.hex()}" '
        f'darktable:mask_nb="{count}" '
        f'darktable:mask_src="{"00" * 8}"/>'
    )


def _insert_masks(spliced: str, records: str) -> str:
    marker = "<darktable:masks_history"
    start = spliced.find(marker)
    if start >= 0:
        close = spliced.find("</darktable:masks_history>", start)
        if close < 0:
            raise MaskSpliceError("masks_history does not close, so nothing was written")
        block = spliced[start:close]
        seq = block.rfind("</rdf:Seq>")
        if seq >= 0:
            at = start + seq
            return spliced[:at] + records + spliced[at:]
        # Revert leaves an empty list as <rdf:Seq/>, which has no closing tag.
        empty = re.search(r"<rdf:Seq\s*/>", block)
        if empty is None:
            raise MaskSpliceError("masks_history has no list, so nothing was written")
        at = start + empty.start()
        end = start + empty.end()
        return spliced[:at] + "<rdf:Seq>" + records + "</rdf:Seq>" + spliced[end:]
    rdf_close = spliced.rfind("</rdf:RDF>")
    if rdf_close < 0:
        raise MaskSpliceError("this is not an XMP document, so nothing was written")
    block = (
        "<darktable:masks_history>"
        "<rdf:Seq>"
        f"{records}"
        "</rdf:Seq>"
        "</darktable:masks_history>"
    )
    return spliced[:rdf_close] + block + spliced[rdf_close:]


def _next_mask_num(document: str) -> int:
    numbers = [int(match) for match in re.findall(r'darktable:mask_num="(\d+)"', document)]
    return (max(numbers) + 1) if numbers else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append one masked module to a darktable sidecar."
    )
    parser.add_argument("--xmp", required=True)
    parser.add_argument("--xmp-out", required=True)
    parser.add_argument("--mask-json")
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--rx", type=float)
    parser.add_argument("--ry", type=float)
    parser.add_argument("--ev", type=float)
    parser.add_argument("--name", default="AI eyes")
    args = parser.parse_args(argv)
    source = Path(args.xmp)
    destination = Path(args.xmp_out)
    if not source.is_file():
        print(f"sidecar not found: {source}", file=sys.stderr)
        return 1
    document = source.read_text(encoding="utf-8")
    try:
        if args.mask_json:
            spec = json.loads(Path(args.mask_json).read_text(encoding="utf-8"))
            if spec.get("kind") == "shapes":
                spliced = splice_shapes(
                    document,
                    forms=list(spec["forms"]),
                    operation=str(spec["operation"]),
                    params=bytes.fromhex(str(spec["params"])),
                    modversion=int(spec["modversion"]),
                    name=str(spec["name"]),
                    invert=bool(spec.get("invert")),
                    opacity=float(spec.get("opacity", 100.0)),
                )
            elif spec.get("kind") == "path":
                spliced = splice_path_exposure(
                    document,
                    points=list(spec["points"]),
                    exposure_ev=float(spec["exposure_ev"]),
                    name=str(spec["name"]),
                )
            else:
                spliced = splice_region_module(
                    document,
                    shapes=list(spec["shapes"]),
                    params=bytes.fromhex(str(spec["params"])),
                    operation=str(spec["operation"]),
                    modversion=int(spec["modversion"]),
                    name=str(spec["name"]),
                )
        else:
            if None in (args.cx, args.cy, args.rx, args.ry, args.ev):
                parser.error("an eye ellipse needs --cx --cy --rx --ry --ev")
            spliced = splice_ellipse_exposure(
                document,
                center_x=args.cx,
                center_y=args.cy,
                radius_x=args.rx,
                radius_y=args.ry,
                exposure_ev=args.ev,
                name=args.name,
            )
        written_name = str(spec["name"]) if args.mask_json else args.name
        verify_top(spliced, written_name)
    except (MaskSpliceError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(spliced, encoding="utf-8", newline="\n")
    print("added 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
