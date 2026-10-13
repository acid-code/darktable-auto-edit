"""Command line entry for the Darktable AI editor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_dt import __version__
from ai_dt.errors import AdvisorUnavailable, DetectorUnavailable, RawReadError, RawSupportUnavailable
from ai_dt.guides.library import format_picture_line
from ai_dt.pipeline import ADVISORS, DETECTORS, analyze, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read a RAW or picture file and write a masked 16-bit TIFF plus a Darktable style.",
    )
    parser.add_argument(
        "image",
        type=Path,
        help="RAW file (NEF, CR2, ARW, DNG, …) or a JPEG, PNG, TIFF, or WebP",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for edited.tif, output.json, and edit.dtstyle",
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=None,
        help="Original file when the positional image is only a developed preview",
    )
    parser.add_argument(
        "--full-res",
        action="store_true",
        help="Decode a RAW at full resolution. The default is half resolution.",
    )
    parser.add_argument("--detector", choices=DETECTORS, default="auto")
    parser.add_argument("--advisor", choices=ADVISORS, default="auto")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--ollama-model", default="llava")
    parser.add_argument(
        "--prompt",
        type=Path,
        default=None,
        help="Replacement for ai_dt/prompts/llava_lighting.txt",
    )
    parser.add_argument("--style-name", default=None, help="Override the generated style name")
    parser.add_argument(
        "--vibe",
        default="",
        help="Look to apply. Empty keeps a faithful correction. A Fuji sim name, or fuji / fujicam, selects one film simulation.",
    )
    parser.add_argument(
        "--steer",
        default="",
        help="Explicit clauses: strong or epic, darken/lift/hold a target, gradient, revert or skip a stage.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = args.image.expanduser()
    if not image.is_file():
        print(f"Not an image file: {image}", file=sys.stderr)
        return 2
    try:
        recipe = analyze(
            image,
            detector=args.detector,
            advisor=args.advisor,
            yolo_model=args.yolo_model,
            ollama_model=args.ollama_model,
            prompt_path=args.prompt,
            style_name=args.style_name,
            output_dir=args.output_dir,
            vibe=args.vibe,
            steer=args.steer,
            source_path=args.original,
            full_res=args.full_res,
        )
        json_path, style_path, edited_path = write_outputs(recipe, args.output_dir)
    except (DetectorUnavailable, AdvisorUnavailable, RawSupportUnavailable, RawReadError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    atmosphere = recipe["analysis"]["atmosphere"]
    subjects = recipe["analysis"]["subjects"]
    match = recipe.get("recipe_match") or {}
    look = match.get("title") or "none"
    kind = (recipe.get("input") or {}).get("kind") or "raster"
    separation = recipe.get("separation") or {}
    picture = recipe.get("picture") or {}
    guide = recipe.get("guide") or {}
    picture_type = picture.get("type") or "general"
    guide_title = guide.get("title") or "Faithful general"
    print(format_picture_line(picture_type, guide_title, recipe.get("features")))
    sentence = str((recipe.get("reading") or {}).get("sentence") or "").strip()
    if sentence:
        print(sentence)
    print(
        f"{atmosphere['mood']}: exposure {atmosphere['exposure_ev']:+.2f} EV, "
        f"contrast {atmosphere['contrast']:+.2f}, "
        f"warmth {atmosphere['temperature_shift']:+.2f}, "
        f"{len(subjects)} subject(s). "
        f"look {look}. "
        f"input {kind}. "
        f"style {recipe['style']['name']}"
    )
    if separation.get("applied"):
        background = "background defocus" if separation.get("background") == "defocus" else "background kept"
        print(
            f"rendered subject {separation['subject_ev']:+.2f} EV, "
            f"{separation['eyes']} eye mask(s), {background}"
        )
    else:
        print("rendered the global grade")
    print(f"wrote {json_path}")
    print(f"wrote {style_path}")
    print(f"wrote {edited_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
