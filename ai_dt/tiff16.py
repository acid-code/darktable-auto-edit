"""Write a 16-bit RGB TIFF without an extra image library.

The file is uncompressed, little-endian, and interleaved. Pillow and Darktable
both read that form. Half-resolution RAW renders land around a few tens of
megabytes; ``--full-res`` is larger.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_rgb16(path: Path, pixels: np.ndarray) -> None:
    """Write ``pixels`` (float RGB, 0 to 1) as a 16-bit TIFF."""

    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("The rendered picture must be RGB")
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    if height < 1 or width < 1:
        raise ValueError("The rendered picture is empty")
    scaled = np.clip(pixels, 0.0, 1.0)
    samples = np.rint(scaled * np.float32(65535.0)).astype("<u2")
    payload = np.ascontiguousarray(samples).tobytes()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_tiff(width, height, payload))


def _tiff(width: int, height: int, payload: bytes) -> bytes:
    # BitsPerSample is three shorts, so it lives outside the IFD entry.
    bits = struct.pack("<HHH", 16, 16, 16)
    x_res = struct.pack("<II", 72, 1)
    y_res = struct.pack("<II", 72, 1)
    header = 8
    strip_offset = header
    cursor = header + len(payload)
    if cursor % 2:
        cursor += 1
    bits_offset = cursor
    cursor += len(bits)
    if cursor % 2:
        cursor += 1
    x_offset = cursor
    cursor += len(x_res)
    y_offset = cursor
    cursor += len(y_res)
    if cursor % 2:
        cursor += 1
    ifd_offset = cursor

    def short(value: int) -> bytes:
        return struct.pack("<H", value).ljust(4, b"\x00")

    def long(value: int) -> bytes:
        return struct.pack("<I", value)

    # tag, type, count, value. Types: SHORT=3, LONG=4, RATIONAL=5.
    entries = [
        (256, 4, 1, long(width)),
        (257, 4, 1, long(height)),
        (258, 3, 3, long(bits_offset)),
        (259, 3, 1, short(1)),
        (262, 3, 1, short(2)),
        (273, 4, 1, long(strip_offset)),
        (277, 3, 1, short(3)),
        (278, 4, 1, long(height)),
        (279, 4, 1, long(len(payload))),
        (282, 5, 1, long(x_offset)),
        (283, 5, 1, long(y_offset)),
        (284, 3, 1, short(1)),
        (296, 3, 1, short(2)),
    ]
    ifd = struct.pack("<H", len(entries))
    for tag, kind, count, value in entries:
        ifd += struct.pack("<HHI", tag, kind, count) + value
    ifd += struct.pack("<I", 0)

    blob = bytearray(ifd_offset + len(ifd))
    blob[0:8] = struct.pack("<2sHI", b"II", 42, ifd_offset)
    blob[strip_offset : strip_offset + len(payload)] = payload
    blob[bits_offset : bits_offset + len(bits)] = bits
    blob[x_offset : x_offset + len(x_res)] = x_res
    blob[y_offset : y_offset + len(y_res)] = y_res
    blob[ifd_offset:] = ifd
    return bytes(blob)
