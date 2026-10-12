"""A tiny RGGB DNG that LibRaw can open.

This is a test fixture, not a camera file. The color matrix is only there so
rawpy will postprocess it.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_dng(path: Path, raw: np.ndarray | None = None) -> Path:
    if raw is None:
        raw = np.zeros((32, 32), dtype=np.uint16)
        raw[0::2, 0::2] = 12000
        raw[0::2, 1::2] = 24000
        raw[1::2, 0::2] = 24000
        raw[1::2, 1::2] = 8000
    height, width = raw.shape
    pixel_bytes = np.ascontiguousarray(raw.astype("<u2")).tobytes()
    model = b"TestCamera\x00"
    version = bytes((1, 4, 0, 0))
    backward = bytes((1, 1, 0, 0))
    matrix = [
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041,
    ]

    def srational(values: list[float]) -> bytes:
        blob = b""
        for value in values:
            blob += struct.pack("<ii", int(round(value * 1_000_000)), 1_000_000)
        return blob

    def rational(values: list[int]) -> bytes:
        return b"".join(struct.pack("<II", int(value), 1) for value in values)

    def short(value: int) -> bytes:
        return struct.pack("<H", value).ljust(4, b"\x00")

    def long(value: int) -> bytes:
        return struct.pack("<I", value)

    tags: list[tuple[int, int, int, bytes | None, bytes | None]] = [
        (256, 4, 1, long(width), None),
        (257, 4, 1, long(height), None),
        (258, 3, 1, short(16), None),
        (259, 3, 1, short(1), None),
        (262, 3, 1, short(32803), None),
        (273, 4, 1, long(8), None),
        (277, 3, 1, short(1), None),
        (278, 4, 1, long(height), None),
        (279, 4, 1, long(len(pixel_bytes)), None),
        (284, 3, 1, short(1), None),
        (33421, 3, 2, struct.pack("<HH", 2, 2), None),
        (33422, 1, 4, bytes((0, 1, 1, 2)), None),
        (50706, 1, 4, version, None),
        (50707, 1, 4, backward, None),
        (50708, 2, len(model), None, model),
        (50714, 5, 1, None, rational([0])),
        (50717, 4, 1, long(65535), None),
        (50721, 10, 9, None, srational(matrix)),
        (50728, 5, 3, None, rational([1, 1, 1])),
        (50778, 3, 1, short(21), None),
    ]
    tags.sort(key=lambda item: item[0])

    blob = b""
    blob_start = 8 + len(pixel_bytes)
    if blob_start % 2:
        blob_start += 1
    entries: list[tuple[int, int, int, bytes]] = []
    for tag, kind, count, inline, extra in tags:
        if extra is None:
            if inline is None or len(inline) != 4:
                raise RuntimeError(f"DNG tag {tag} is missing an inline value")
            entries.append((tag, kind, count, inline))
            continue
        if len(blob) % 2:
            blob += b"\x00"
        offset = blob_start + len(blob)
        entries.append((tag, kind, count, long(offset)))
        blob += extra

    ifd_offset = blob_start + len(blob)
    if ifd_offset % 2:
        blob += b"\x00"
        ifd_offset += 1
    ifd = struct.pack("<H", len(entries))
    for tag, kind, count, value in entries:
        ifd += struct.pack("<HHI", tag, kind, count) + value
    ifd += struct.pack("<I", 0)

    out = bytearray(ifd_offset + len(ifd))
    out[0:8] = struct.pack("<2sHI", b"II", 42, ifd_offset)
    out[8 : 8 + len(pixel_bytes)] = pixel_bytes
    out[blob_start : blob_start + len(blob)] = blob
    out[ifd_offset:] = ifd
    destination = Path(path)
    destination.write_bytes(out)
    return destination
