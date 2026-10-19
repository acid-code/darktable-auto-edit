"""Print a Darktable sidecar's history stack and vignette numbers."""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path


def _attr(tag: str, name: str) -> str | None:
    found = re.search(rf'{name}="([^"]*)"', tag)
    return found.group(1) if found else None


def _vignette(params: str) -> str:
    raw = bytes.fromhex(params)
    if len(raw) < 44:
        return f"vignette params {len(raw)} bytes"
    scale, falloff, brightness, _saturation, cx, cy = struct.unpack("<ffffff", raw[:24])
    whratio = struct.unpack_from("<f", raw, 28)[0]
    return (
        f"scale={scale:.2f} falloff={falloff:.2f} brightness={brightness:.3f} "
        f"center=({cx:.3f},{cy:.3f}) whratio={whratio:.3f}"
    )


def main(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    end = re.search(r'history_end="(\d+)"', text)
    items = [
        match.group(0)
        for match in re.finditer(r"<rdf:li\b[^>]*darktable:operation=\"[^\"]+\"[^>]*/>", text)
    ]
    print(f"history_end {end.group(1) if end else '?'} items {len(items)}")
    for tag in items:
        operation = _attr(tag, "darktable:operation")
        extra = ""
        if operation == "vignette":
            params = _attr(tag, "darktable:params") or ""
            if params and not params.startswith("gz"):
                extra = " " + _vignette(params)
        print(
            f"{_attr(tag, 'darktable:num'):>3} {operation} "
            f"prio={_attr(tag, 'darktable:multi_priority') or '0'} "
            f"blend={_attr(tag, 'darktable:blendop_version') or '?'} "
            f"name={_attr(tag, 'darktable:multi_name') or ''}{extra}"
        )
    masks = re.findall(r'darktable:mask_type="(\d+)"', text)
    if masks:
        print(f"masks {len(masks)} types {','.join(masks)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: read_sidecar.py SIDECAR.xmp", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
