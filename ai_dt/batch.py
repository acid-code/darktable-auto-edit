"""Develop a folder of RAW and picture files into finished TIFFs.

Each file is written to its own directory under the output folder:

    edited.tif
    output.json
    edit.dtstyle

One file that cannot be read does not stop the rest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_dt import __version__
from ai_dt.errors import AdvisorUnavailable, DetectorUnavailable, RawReadError, RawSupportUnavailable
from ai_dt.io_image import is_picture
from ai_dt.guides.library import format_picture_line
from ai_dt.pipeline import ADVISORS, DETECTORS, analyze, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a folder of RAW and picture files.")
    parser.add_argument("folder", type=Path, help="Folder of RAW, JPEG, PNG, TIFF, or WebP files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where the finished files go")
    parser.add_argument("--detector", choices=DETECTORS, default="auto")
    parser.add_argument("--advisor", choices=ADVISORS, default="histogram")
    parser.add_argument("--vibe", default="", help="Open text for the look. Empty stays a faithful correction.")
    parser.add_argument("--full-res", action="store_true", help="Decode each RAW at full resolution")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many files. 0 means the whole folder.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def iter_pictures(folder: Path, output_dir: Path) -> list[Path]:
    output = output_dir.resolve()
    found: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if not is_picture(path):
            continue
        resolved = path.resolve()
        if resolved == output or output in resolved.parents:
            continue
        found.append(path)
    return found


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    folder = args.folder.expanduser()
    if not folder.is_dir():
        print(f"Not a folder: {folder}", file=sys.stderr)
        return 2
    output = args.output_dir.expanduser()
    pictures = iter_pictures(folder, output)
    if args.limit > 0:
        pictures = pictures[: args.limit]
    if not pictures:
        print(f"No RAW or picture files in {folder}", file=sys.stderr)
        return 2

    shared = _shared_stems(folder, pictures)
    failures = 0
    for index, path in enumerate(pictures, start=1):
        destination = _destination(output, folder, path, shared)
        print(f"[{index}/{len(pictures)}] {path}")
        try:
            recipe = analyze(
                path,
                detector=args.detector,
                advisor=args.advisor,
                vibe=args.vibe,
                output_dir=destination,
                full_res=args.full_res,
            )
            _, _, edited = write_outputs(recipe, destination)
        except (DetectorUnavailable, AdvisorUnavailable, RawSupportUnavailable, RawReadError, OSError, ValueError) as exc:
            failures += 1
            print(f"failed {path}: {exc}", file=sys.stderr)
            continue
        kind = (recipe.get("input") or {}).get("kind") or "raster"
        separation = recipe.get("separation") or {}
        picture = recipe.get("picture") or {}
        guide = recipe.get("guide") or {}
        picture_type = picture.get("type") or "general"
        guide_title = guide.get("title") or "Faithful general"
        named = format_picture_line(picture_type, guide_title, recipe.get("features"))
        masks = "subject isolated" if separation.get("applied") else "global grade"
        print(f"wrote {edited} ({kind}, {named}, {masks})")
    print(f"{len(pictures) - failures} written, {failures} failed")
    return 1 if failures else 0


def _shared_stems(folder: Path, pictures: list[Path]) -> set[str]:
    counts: dict[str, int] = {}
    for path in pictures:
        counts[_stem_key(folder, path)] = counts.get(_stem_key(folder, path), 0) + 1
    return {key for key, count in counts.items() if count > 1}


def _stem_key(folder: Path, path: Path) -> str:
    try:
        relative = path.relative_to(folder)
    except ValueError:
        relative = Path(path.name)
    return str(relative.with_suffix(""))


def _destination(output: Path, folder: Path, path: Path, shared: set[str]) -> Path:
    key = _stem_key(folder, path)
    destination = output / key
    if key in shared:
        suffix = path.suffix.lower().lstrip(".") or "file"
        destination = output / f"{key}-{suffix}"
    return destination


if __name__ == "__main__":
    raise SystemExit(main())
