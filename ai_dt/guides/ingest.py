"""Copy one hand-written guide into the library.

This does not invent a guide from a photo. You write the note, this checks
the fields, and it files the note under the first picture type. The folder
is for you. Matching uses the words inside the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_dt.guides.library import DEFAULT_LIBRARY, _read_guide_file


def ingest_note(source: Path, library: Path, force: bool = False) -> Path:
    """Validate ``source`` and write it into ``library``. Returns the new path."""

    cleaned = _read_guide_file(source)
    picture_type = cleaned["types"][0].split()[0].lower()
    destination_dir = library / picture_type
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{cleaned['id']}.json"
    if destination.exists() and not force:
        raise ValueError(f"{destination} already exists. Pass --force to replace it.")
    destination.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a guide note and file it in the local guide library.",
    )
    parser.add_argument(
        "note",
        type=Path,
        help="JSON or text file you wrote. This tool does not download anything.",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=DEFAULT_LIBRARY,
        help="Guide folder. Defaults to the guides directory beside the analyzer.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing guide with the same id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    note = args.note.expanduser()
    if not note.is_file():
        print(f"Not a file: {note}", file=sys.stderr)
        return 2
    try:
        destination = ingest_note(note, args.library.expanduser(), force=args.force)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"filed {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
