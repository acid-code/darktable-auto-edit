"""Copy one hand-written recipe note into the library.

This does not fetch presets from the web. You write the JSON, this checks
the fields, and it files the note under the first scene name. The folder
is for you. Matching uses the words inside the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_dt.recipes.library import DEFAULT_LIBRARY, validate_recipe


def ingest_note(source: Path, library: Path, force: bool = False) -> Path:
    """Validate ``source`` and write it into ``library``. Returns the new path."""

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source.name} is not valid JSON: {exc}") from exc
    cleaned = validate_recipe(payload, source=source.name)
    scene = cleaned["scenes"][0]
    destination_dir = library / scene
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{cleaned['id']}.json"
    if destination.exists() and not force:
        raise ValueError(f"{destination} already exists. Pass --force to replace it.")
    destination.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a recipe note and file it in the local recipe library.",
    )
    parser.add_argument("note", type=Path, help="JSON file you wrote. This tool does not download anything.")
    parser.add_argument(
        "--library",
        type=Path,
        default=DEFAULT_LIBRARY,
        help="Recipe folder. Defaults to the recipes directory beside the analyzer.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing recipe with the same id")
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
