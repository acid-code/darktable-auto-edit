"""Add one drawn ellipse to a Darktable sidecar without replacing its history.

Darktable 5.4 stores drawn shapes in ``masks_history`` and points a module
at a shape through ``blendop_params`` (``mask_mode`` and ``mask_id``).
A style file cannot carry that shape. This writer keeps every history item
already in the sidecar and appends one exposure whose blend blob points at
a new ellipse. The layout follows ``dt_masks_point_ellipse_t`` and the
420-byte ``dt_develop_blend_params_t`` used by Darktable 5.4.1.

The caller must apply the file with ``image:apply_sidecar()`` only after
the darkroom has closed the photo. Applying it while the photo is on
screen deadlocks the pixelpipe.
"""

from __future__ import annotations

import argparse
import base64
import re
import struct
import sys
import zlib
from html import escape
from pathlib import Path

from ai_dt.mapping import DEFAULT_BLENDOP, BLENDOP_VERSION, EXPOSURE_VERSION, exposure_params
from ai_dt.revert import _HISTORY_ITEM, _attr

_BLEND_BYTES = 420
_OFFSET_MASK_MODE = 0
_OFFSET_OPACITY = 16
_OFFSET_MASK_ID = 24
_MASK_MODE_DRAWN = 3  # ENABLED | MASK
_ELLIPSE_TYPE = 1 << 5
_ELLIPSE_VERSION = 6  # DEVELOP_MASKS_VERSION in darktable 5.4.1
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
    """Return a sidecar with the same history plus one masked exposure.

    Refuses a document whose ``history_end`` does not match the history
    items, a document with no history, and an ellipse that leaves the frame.
    The input string is not modified.
    """

    center_x = _check_unit("center_x", center_x)
    center_y = _check_unit("center_y", center_y)
    radius_x = _check_unit("radius_x", radius_x)
    radius_y = _check_unit("radius_y", radius_y)
    if radius_x <= 0.0 or radius_y <= 0.0:
        raise MaskSpliceError("the ellipse radius must be greater than zero")
    if not name.startswith("AI "):
        raise MaskSpliceError("the added module name must start with 'AI '")

    history = [
        match.group(0)
        for match in _HISTORY_ITEM.finditer(document)
        if _attr(match.group(0), "darktable:operation")
    ]
    end_match = re.search(r'darktable:history_end="(\d+)"', document)
    if not history or end_match is None:
        raise MaskSpliceError("this sidecar has no history stack, so nothing was written")
    history_end = int(end_match.group(1))
    if history_end != len(history):
        raise MaskSpliceError(
            "history_end does not match the history stack, so nothing was written"
        )

    numbers = [int(_attr(tag, "darktable:num") or "0") for tag in history]
    next_num = max(numbers) + 1
    priorities = [
        int(_attr(tag, "darktable:multi_priority") or "0")
        for tag in history
        if _attr(tag, "darktable:operation") == "exposure"
    ]
    priority = (max(priorities) + 1) if priorities else 0
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
        f'darktable:blendop_version="{BLENDOP_VERSION}" '
        f'darktable:multi_name="{safe_name}" darktable:multi_priority="{priority}" '
        f'darktable:multi_name_hand_edited="1"/>'
    )
    last = history[-1]
    insert_at = document.rfind(last)
    if insert_at < 0:
        raise MaskSpliceError("the last history item could not be found again")
    insert_at += len(last)
    spliced = document[:insert_at] + item + document[insert_at:]
    spliced = re.sub(
        r'darktable:history_end="\d+"',
        f'darktable:history_end="{history_end + 1}"',
        spliced,
        count=1,
    )

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
    marker = "<darktable:masks_history"
    start = spliced.find(marker)
    if start >= 0:
        close = spliced.find("</darktable:masks_history>", start)
        if close < 0:
            raise MaskSpliceError("masks_history does not close, so nothing was written")
        seq = spliced.rfind("</rdf:Seq>", start, close)
        if seq < 0:
            raise MaskSpliceError("masks_history has no list, so nothing was written")
        return spliced[:seq] + mask_li + spliced[seq:]
    rdf_close = spliced.rfind("</rdf:RDF>")
    if rdf_close < 0:
        raise MaskSpliceError("this is not an XMP document, so nothing was written")
    block = (
        "<darktable:masks_history>"
        "<rdf:Seq>"
        f"{mask_li}"
        "</rdf:Seq>"
        "</darktable:masks_history>"
    )
    return spliced[:rdf_close] + block + spliced[rdf_close:]


def _next_mask_num(document: str) -> int:
    numbers = [int(match) for match in re.findall(r'darktable:mask_num="(\d+)"', document)]
    return (max(numbers) + 1) if numbers else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append one masked exposure to a darktable sidecar."
    )
    parser.add_argument("--xmp", required=True)
    parser.add_argument("--xmp-out", required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--rx", type=float, required=True)
    parser.add_argument("--ry", type=float, required=True)
    parser.add_argument("--ev", type=float, required=True)
    parser.add_argument("--name", default="AI eyes")
    args = parser.parse_args(argv)
    source = Path(args.xmp)
    destination = Path(args.xmp_out)
    if not source.is_file():
        print(f"sidecar not found: {source}", file=sys.stderr)
        return 1
    try:
        spliced = splice_ellipse_exposure(
            source.read_text(encoding="utf-8"),
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
