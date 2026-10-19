"""Print Accept and Back snapshots for one photo's work folder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

READER = (
    Path(__file__).resolve().parents[2]
    / "darktable-history"
    / "scripts"
    / "read_sidecar.py"
)
SNAPSHOTS = (
    "history.xmp",
    "history-before-mask.xmp",
    "masked.xmp",
    "history-step.xmp",
)


def main(folder: Path) -> int:
    if not folder.is_dir():
        print(f"work folder not found: {folder}", file=sys.stderr)
        return 1
    python = sys.executable
    for name in SNAPSHOTS:
        path = folder / name
        print(f"==== {name} ====", flush=True)
        if not path.is_file():
            print("missing")
            continue
        subprocess.run([python, str(READER), str(path)], check=False)
    redo = folder / "history-redo.json"
    print("==== history-redo.json ====", flush=True)
    if not redo.is_file():
        print("missing")
        return 0
    try:
        payload = json.loads(redo.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"unreadable: {exc}")
        return 1
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        print("items 0")
        return 0
    print(f"items {len(items)}")
    for item in items:
        if isinstance(item, dict):
            print(item.get("name") or "")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: read_accept.py WORKDIR", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
