"""Command line entry for the Darktable AI editor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_dt import __version__
from ai_dt.env import load_project_env
from ai_dt.errors import AdvisorUnavailable, DetectorUnavailable, RawReadError, RawSupportUnavailable
from ai_dt.guides.library import format_picture_line
from ai_dt.pipeline import ADVISORS, DETECTORS, analyze, write_outputs
from ai_dt.reasoner import propose_gemini, propose_ollama

load_project_env()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read a RAW or picture file and write a masked 16-bit TIFF plus a Darktable style.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=None,
        help="RAW file (NEF, CR2, ARW, DNG, …) or a JPEG, PNG, TIFF, or WebP",
    )
    parser.add_argument(
        "--align-history",
        action="store_true",
        help="Move the plan with one history step. Use --direction back or next.",
    )
    parser.add_argument("--history-name", default="", help="History instance name that crossed the cursor.")
    parser.add_argument("--direction", default="", help="back or next, with --align-history.")
    parser.add_argument(
        "--commit-step",
        action="store_true",
        help="Write the waiting step only. Do not rebuild the look.",
    )
    parser.add_argument(
        "--skip-step",
        action="store_true",
        help="Leave the waiting step out and show the one after it.",
    )
    parser.add_argument(
        "--restore-step",
        action="store_true",
        help="Put the last accepted step back. The history write did not land.",
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
    parser.add_argument(
        "--scene",
        default="",
        help="The photo description the user confirmed. Empty keeps the stored description.",
    )
    parser.add_argument(
        "--rewrite-scene",
        action="store_true",
        help="Ask the picture model to write the photo description again.",
    )
    parser.add_argument(
        "--look",
        default="",
        help="A choice id to apply: faithful, a recommendation, or a recipe id. Empty proposes the measured grade.",
    )
    parser.add_argument("--image-id", type=int, default=None, help="Darktable image id for the version record.")
    parser.add_argument(
        "--duplicate-index",
        type=int,
        default=None,
        help="Darktable duplicate index for the version record.",
    )
    parser.add_argument("--parent-id", type=int, default=None, help="Image id this duplicate was branched from.")
    parser.add_argument(
        "--parent-index",
        type=int,
        default=None,
        help="Duplicate index this version was branched from.",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help="Darktable library.db, used to avoid stacking a second display transform.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--stage",
        choices=("all", "picture", "describe", "gemini", "ollama", "finish", "chat"),
        default="all",
        help="Run one visible step. all is the single-process path.",
    )
    return parser


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} was not an object")
    return data


def _analyze_kwargs(args) -> dict:
    return {
        "detector": args.detector,
        "advisor": args.advisor,
        "yolo_model": args.yolo_model,
        "ollama_model": args.ollama_model,
        "prompt_path": args.prompt,
        "style_name": args.style_name,
        "output_dir": args.output_dir,
        "vibe": args.vibe,
        "steer": args.steer,
        "scene_text": args.scene,
        "rewrite_scene": args.rewrite_scene,
        "describe_only": args.stage == "describe",
        "look": args.look,
        "image_id": args.image_id,
        "duplicate_index": args.duplicate_index,
        "parent_id": args.parent_id,
        "parent_index": args.parent_index,
        "library_path": args.library,
        "source_path": args.original,
        "full_res": args.full_res,
    }


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    args = build_parser().parse_args(argv)
    if args.align_history:
        from ai_dt.features import align_plan_to_history

        try:
            align_plan_to_history(Path(args.output_dir), args.history_name, args.direction)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Analysis failed: {exc}", file=sys.stderr)
            return 3
        print("Plan aligned.")
        return 0
    if args.commit_step or args.skip_step or args.restore_step or args.stage == "chat":
        from ai_dt.features import (
            commit_waiting_step,
            plan_chat,
            restore_waiting_step,
            skip_waiting_step,
        )

        try:
            if args.restore_step:
                restore_waiting_step(Path(args.output_dir))
                print("Step restored.")
            elif args.skip_step:
                skip_waiting_step(Path(args.output_dir))
                print("Step skipped.")
            elif args.stage == "chat":
                plan_chat(Path(args.output_dir), args.steer, args.scene)
                print("Plan ready.")
            else:
                commit_waiting_step(Path(args.output_dir))
                print("Step written.")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Analysis failed: {exc}", file=sys.stderr)
            return 3
        return 0
    if args.image is None:
        print("An image file is required.", file=sys.stderr)
        return 2
    image = args.image.expanduser()
    if not image.is_file():
        print(f"Not an image file: {image}", file=sys.stderr)
        return 2
    try:
        kwargs = _analyze_kwargs(args)
        if args.stage == "describe":
            recipe = analyze(image, **kwargs)
            summary = str((recipe.get("scene") or {}).get("summary") or "").strip()
            print(summary or "Description ready.")
            return 0
        if args.stage == "picture":
            record = analyze(image, picture_only=True, **kwargs)
            _write_json(Path(args.output_dir) / "stage.json", record)
            print("Picture stage finished.")
            return 0
        if args.stage in {"gemini", "ollama"}:
            saved = _read_json(Path(args.output_dir) / "stage.json")
            sentence = str(saved.get("sentence") or "")
            preset = str(saved.get("preset") or "")
            vibe = str(saved.get("vibe") or args.vibe)
            steer = str(saved.get("steer") or args.steer)
            if args.stage == "gemini":
                payload = propose_gemini(sentence, vibe, preset, steer, scene=str(saved.get("scene") or args.scene))
            else:
                previous: dict = {}
                proposal_path = Path(args.output_dir) / "proposal.json"
                if proposal_path.is_file():
                    previous = _read_json(proposal_path)
                payload = propose_ollama(
                    sentence,
                    vibe,
                    preset,
                    steer,
                    failure=str(previous.get("failure") or ""),
                    gemini_tried=bool(previous.get("gemini_tried")),
                    scene=str(saved.get("scene") or args.scene),
                )
            _write_json(Path(args.output_dir) / "proposal.json", payload)
            print(f"{args.stage} stage finished.")
            return 0
        saved_advice = None
        proposal = None
        if args.stage == "finish":
            saved_advice = _read_json(Path(args.output_dir) / "stage.json")
            wrapped = _read_json(Path(args.output_dir) / "proposal.json")
            proposal = wrapped.get("proposal") if isinstance(wrapped.get("proposal"), dict) else None
            if proposal is None:
                raise ValueError("The text plan stage did not write a proposal.")
        recipe = analyze(image, saved_advice=saved_advice, proposal=proposal, **kwargs)
        json_path, style_path, edited_path = write_outputs(recipe, args.output_dir)
    except (DetectorUnavailable, AdvisorUnavailable, RawSupportUnavailable, RawReadError, OSError, ValueError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"Analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

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
