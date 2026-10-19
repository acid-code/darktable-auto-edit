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


def _prepare_insert(document: str) -> tuple[str, list[str], int, int]:
    """Return the document, its history, the active count, and the new item number.

    A shorter active count inserts at that index and shifts later numbers.
    A count at or past the end appends with the next free number, so a gap
    in the existing numbers does not turn an active row off.
    """

    document, history, history_end = _stack_for_append(document)
    numbers = [int(_attr(tag, "darktable:num") or "0") for tag in history]
    if history_end < len(history):
        document = _shift_numbers_from(document, history_end)
        return document, history, history_end, history_end
    return document, history, history_end, (max(numbers) + 1) if numbers else 0


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

    mask_li = (
        f'<rdf:li darktable:mask_num="{_next_mask_num(spliced)}" '
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
    mask_li = (
        f'<rdf:li darktable:mask_num="{_next_mask_num(spliced)}" '
        f'darktable:mask_id="{mask_id}" '
        f'darktable:mask_type="{_PATH_TYPE}" '
        f'darktable:mask_name="{safe_name}" '
        f'darktable:mask_version="{_ELLIPSE_VERSION}" '
        f'darktable:mask_points="{blob.hex()}" '
        f'darktable:mask_nb="{len(packed)}" '
        f'darktable:mask_src="{"00" * 8}"/>'
    )
    return _insert_masks(spliced, mask_li)


def _pipeline_slot(document: str, operation: str, priority: int) -> str:
    """Keep the new instance in the pipeline list the darkroom reloads."""

    match = re.search(r'darktable:iop_order_list="([^"]*)"', document)
    if match is None:
        return document
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
    return _insert_masks(spliced, "".join(records))


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
            if spec.get("kind") == "path":
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
    except MaskSpliceError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(spliced, encoding="utf-8", newline="\n")
    print("added 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
