"""Remove the history rows this editor appended.

Darktable 5.4 Lua can reset a whole history stack (`image:reset()`) and can
replace one image's history from a sidecar (`image:apply_sidecar()`). It
cannot delete one history row. The editor strips the AI items out of a
sidecar and applies that file.

Call ``image:apply_sidecar()`` only after the darkroom has closed this
image. Applying it while the photo is on screen reloads develop on the
Lua thread with the pixelpipe still running, and that deadlocks the
window. Opening the photo again loads the replaced stack.

Deleting the same rows from ``library.db`` while Darktable is open does not
reload the darkroom history panel. The panel keeps develop's copy and writes
it back, so the next attempt finds the same AI rows. The ``--library`` path
remains for a closed library and for tests.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path


# Darktable launches this file by path, so the ai_dt package is not importable.
# These are the history names step_history_name writes.
_ACCEPTED_STEP_NAMES = frozenset(
    {
        "Subject lift",
        "Shine the subject",
        "Darken the foreground",
        "Darken the bright gap",
        "Darken the shafts",
        "Warm the highlights",
        "Highlight the color",
        "Highlight the red",
        "Highlight the orange",
        "Highlight the yellow",
        "Highlight the green",
        "Highlight the blue",
        "Recolor",
        "Eyes",
        "Vignette the subject",
        # Drawn-mask steps (ai_dt/mask_steps.py). Keep in sync with
        # editor_history_names in ai_editor.lua.
        "Grade the sky",
        "Grade the buildings",
        "Grade the vegetation",
        "Grade the water",
        "Grade the ground",
        "Lift the people",
        "Quiet the background",
        # Lesson steps (photoserge examples 4-11) and animal masks.
        "Darken the sky gradient",
        "Darken the bottom",
        "Burn the edges",
        "Glow behind the subject",
        "Lift the animal",
        "Lift the animals",
        "Quiet around the animals",
        "Split tone",
        "Warm-only look",
        "Keep only the red",
        "Keep only the orange",
        "Keep only the yellow",
        "Keep only the green",
        "Keep only the blue",
    }
)


def accepted_step_names() -> frozenset[str]:
    """History instance names Accept writes. These do not start with ``AI ``."""

    return _ACCEPTED_STEP_NAMES


def is_ai_editor_history_name(
    multi_name: str | None,
    extra_names: frozenset[str] | set[str] | None = None,
) -> bool:
    """Return whether a history instance was named by this plugin.

    Older rows still start with ``AI `` (``AI exposure``, ``AI color``,
    ``AI eyes``, ``AI animal``). An accepted step uses its own name, such as
    ``Subject lift`` or ``Highlight the red``. A name recorded on this photo's
    plan counts too, which covers an eye step named from its sentence.
    A module renamed by hand no longer matches. A module the photographer
    named with one of these names does.
    """

    if not multi_name:
        return False
    if multi_name in ("AI exposure", "AI color") or multi_name.startswith("AI "):
        return True
    if multi_name in accepted_step_names():
        return True
    return bool(extra_names) and multi_name in extra_names


def revert_image_history(
    connection: sqlite3.Connection,
    image_id: int,
    stage: str | None = None,
) -> int:
    """Delete this plugin's history rows for one image. Return how many.

    ``history_end`` is an index into the history list ordered by ``num``, not
    a history number. It moves back only for removed rows that were still
    inside the active prefix, so an undone step stays undone.
    """

    image = connection.execute(
        "SELECT history_end FROM main.images WHERE id = ?",
        (image_id,),
    ).fetchone()
    if image is None:
        raise LookupError(f"image {image_id} is not in the library")

    rows = connection.execute(
        "SELECT num, multi_name, operation FROM main.history WHERE imgid = ? ORDER BY num",
        (image_id,),
    ).fetchall()
    history_end = image[0]
    if history_end is None:
        history_end = len(rows)

    delete_nums: list[int] = []
    removed_before_end = 0
    named = (stage or "").strip().lower()
    for index, (num, multi_name, operation) in enumerate(rows):
        if not is_ai_editor_history_name(multi_name):
            continue
        # The vignette is the history module. Other stages are in the TIFF.
        if named and named != "vignette":
            continue
        if named == "vignette" and operation != "vignette":
            continue
        delete_nums.append(int(num))
        if index < history_end:
            removed_before_end += 1
    if not delete_nums:
        return 0

    marks = ",".join("?" * len(delete_nums))
    connection.execute(
        f"DELETE FROM main.history WHERE imgid = ? AND num IN ({marks})",
        (image_id, *delete_nums),
    )
    connection.execute(
        f"DELETE FROM main.masks_history WHERE imgid = ? AND num IN ({marks})",
        (image_id, *delete_nums),
    )
    connection.execute(
        "UPDATE main.images SET history_end = ?, thumb_timestamp = -1, thumb_maxmip = 0 WHERE id = ?",
        (max(0, history_end - removed_before_end), image_id),
    )
    connection.execute("DELETE FROM main.history_hash WHERE imgid = ?", (image_id,))
    return len(delete_nums)


# One history or mask entry. Attribute values are quoted, so a ">" inside
# base64 does not end the tag. Darktable writes these as self-closing items.
_HISTORY_ITEM = re.compile(
    r"<rdf:li\b(?:[^>\"]|\"[^\"]*\")*(?:/>|>\s*</rdf:li>)",
    re.IGNORECASE,
)


def _attr(tag: str, name: str) -> str:
    match = re.search(rf'\b{name}="([^"]*)"', tag)
    return match.group(1) if match else ""


def _extra_names(extra_names: list[str] | None) -> set[str]:
    return {str(name).strip() for name in (extra_names or []) if str(name).strip()}


def strip_ai_history(document: str, extra_names: list[str] | None = None) -> tuple[str, int]:
    """Remove AI history items from a Darktable XMP sidecar.

    The photographer's other history items stay. Mask entries that belonged
    to a removed item are removed with it. ``history_end`` moves back by the
    number of history items removed.
    """

    known = _extra_names(extra_names)
    history_tags = [
        match.group(0)
        for match in _HISTORY_ITEM.finditer(document)
        if _attr(match.group(0), "darktable:operation")
    ]
    end_match = re.search(r'darktable:history_end="(\d+)"', document)
    history_end = int(end_match.group(1)) if end_match else len(history_tags)
    removed_nums: set[str] = set()
    removed = 0
    # history_end counts active items from the start of the stack. An AI
    # step the photographer already undid sits past that prefix and must
    # not pull the prefix back over their own modules.
    removed_before_end = 0
    for index, tag in enumerate(history_tags):
        if not is_ai_editor_history_name(_attr(tag, "darktable:multi_name"), known):
            continue
        removed += 1
        number = _attr(tag, "darktable:num")
        if number:
            removed_nums.add(number)
        if index < history_end:
            removed_before_end += 1

    if removed == 0:
        return document, 0

    def drop_ai(match: re.Match[str]) -> str:
        tag = match.group(0)
        if not is_ai_editor_history_name(_attr(tag, "darktable:multi_name"), known):
            return tag
        return ""

    cleaned = _HISTORY_ITEM.sub(drop_ai, document)
    if removed_nums:

        def drop_masks(match: re.Match[str]) -> str:
            tag = match.group(0)
            if _attr(tag, "darktable:num") in removed_nums:
                return ""
            return tag

        cleaned = _HISTORY_ITEM.sub(drop_masks, cleaned)

    def drop_named_masks(match: re.Match[str]) -> str:
        tag = match.group(0)
        if is_ai_editor_history_name(_attr(tag, "darktable:mask_name"), known):
            return ""
        return tag

    cleaned = _HISTORY_ITEM.sub(drop_named_masks, cleaned)
    if removed_before_end:

        def move_end(match: re.Match[str]) -> str:
            value = max(0, int(match.group(1)) - removed_before_end)
            return f'darktable:history_end="{value}"'

        cleaned = re.sub(r'darktable:history_end="(\d+)"', move_end, cleaned, count=1)
    return cleaned, removed


def _history_entries(document: str) -> list[tuple[int, int, str]]:
    entries: list[tuple[int, int, str]] = []
    for match in _HISTORY_ITEM.finditer(document):
        tag = match.group(0)
        if _attr(tag, "darktable:operation"):
            entries.append((match.start(), match.end(), tag))
    return entries


def pop_history_level(document: str) -> tuple[str, dict, int]:
    """Remove the last active history row and the masks that belong to it.

    ``history_end`` is the count of active rows. Back used to lower that
    count and leave the row in the stack, which only moved the highlight.
    The returned entry is what Next splices back in.
    """

    items = _history_entries(document)
    end_match = re.search(r'darktable:history_end="(\d+)"', document)
    if not items or end_match is None:
        return document, {}, 0
    active = min(int(end_match.group(1)), len(items))
    if active <= 0:
        return document, {}, 0
    start, end, tag = items[active - 1]
    number = _attr(tag, "darktable:num")
    spans = [(start, end)]
    masks: list[str] = []
    if number:
        for match in _HISTORY_ITEM.finditer(document):
            mask = match.group(0)
            if _attr(mask, "darktable:operation"):
                continue
            if _attr(mask, "darktable:mask_num") == number:
                masks.append(mask)
                spans.append((match.start(), match.end()))
    updated = document
    for span_start, span_end in sorted(spans, reverse=True):
        updated = updated[:span_start] + updated[span_end:]
    updated = re.sub(
        r'darktable:history_end="\d+"',
        f'darktable:history_end="{active - 1}"',
        updated,
        count=1,
    )
    entry = {
        "name": _attr(tag, "darktable:multi_name"),
        "history": tag,
        "masks": masks,
    }
    return updated, entry, 1


def _insert_history_tag(document: str, tag: str) -> str:
    number = int(_attr(tag, "darktable:num") or "0")
    for start, _end, existing in _history_entries(document):
        if int(_attr(existing, "darktable:num") or "0") > number:
            return document[:start] + tag + document[start:]
    items = _history_entries(document)
    if items:
        _start, end, _existing = items[-1]
        return document[:end] + tag + document[end:]
    marker = "<darktable:history"
    start = document.find(marker)
    if start < 0:
        return document + tag
    close = document.find("</darktable:history>", start)
    if close < 0:
        return document + tag
    block = document[start:close]
    seq = block.rfind("</rdf:Seq>")
    if seq >= 0:
        at = start + seq
        return document[:at] + tag + document[at:]
    empty = re.search(r"<rdf:Seq\s*/>", block)
    if empty is None:
        return document[:close] + "<rdf:Seq>" + tag + "</rdf:Seq>" + document[close:]
    at = start + empty.start()
    end = start + empty.end()
    return document[:at] + "<rdf:Seq>" + tag + "</rdf:Seq>" + document[end:]


def _insert_mask_tags(document: str, records: str) -> str:
    marker = "<darktable:masks_history"
    start = document.find(marker)
    if start >= 0:
        close = document.find("</darktable:masks_history>", start)
        if close < 0:
            return document
        block = document[start:close]
        seq = block.rfind("</rdf:Seq>")
        if seq >= 0:
            at = start + seq
            return document[:at] + records + document[at:]
        empty = re.search(r"<rdf:Seq\s*/>", block)
        if empty is None:
            return document
        at = start + empty.start()
        end = start + empty.end()
        return document[:at] + "<rdf:Seq>" + records + "</rdf:Seq>" + document[end:]
    rdf_close = document.rfind("</rdf:RDF>")
    block = (
        "<darktable:masks_history><rdf:Seq>"
        + records
        + "</rdf:Seq></darktable:masks_history>"
    )
    if rdf_close < 0:
        return document + block
    return document[:rdf_close] + block + document[rdf_close:]


def push_history_level(document: str, entry: dict) -> tuple[str, str, int]:
    """Splice one removed history row, and its masks, back onto the stack."""

    tag = str((entry or {}).get("history") or "")
    if "darktable:operation" not in tag:
        return document, "", 0
    updated = _insert_history_tag(document, tag)
    masks = [str(item) for item in ((entry or {}).get("masks") or []) if str(item)]
    if masks:
        updated = _insert_mask_tags(updated, "".join(masks))
    end_match = re.search(r'darktable:history_end="(\d+)"', updated)
    history_end = int(end_match.group(1)) if end_match else 0
    updated = re.sub(
        r'darktable:history_end="\d+"',
        f'darktable:history_end="{history_end + 1}"',
        updated,
        count=1,
    )
    return updated, _attr(tag, "darktable:multi_name"), 1


def _load_redo(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and item.get("history")]


def _save_redo(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}) + "\n", encoding="utf-8")


def move_history_cursor(document: str, direction: int) -> tuple[str, str, int]:
    """Move ``history_end`` by one step and keep every history item.

    ``direction`` -1 steps back. ``direction`` 1 steps forward into the
    redo stack. The returned name is the module instance that crossed the
    cursor. Masks stay where they are; the module that owns one simply
    becomes active or inactive with the cursor. A cursor already at the
    end of the stack does not write a new document.
    """

    history_tags = [
        match.group(0)
        for match in _HISTORY_ITEM.finditer(document)
        if _attr(match.group(0), "darktable:operation")
    ]
    end_match = re.search(r'darktable:history_end="(\d+)"', document)
    if not history_tags or end_match is None:
        return document, "", 0
    history_end = int(end_match.group(1))
    if direction < 0:
        if history_end <= 0:
            return document, "", 0
        crossed = history_tags[history_end - 1]
        new_end = history_end - 1
    elif direction > 0:
        if history_end >= len(history_tags):
            return document, "", 0
        crossed = history_tags[history_end]
        new_end = history_end + 1
    else:
        return document, "", 0
    updated = re.sub(
        r'darktable:history_end="\d+"',
        f'darktable:history_end="{new_end}"',
        document,
        count=1,
    )
    return updated, _attr(crossed, "darktable:multi_name"), 1


def keep_history_prefix(document: str, end: int) -> tuple[str, int]:
    """Keep the highlighted history prefix and drop every later row.

    ``end`` is the active count, the same number as ``history_end``. Masks
    that belong to a dropped row go with it. Copy uses this. Accept does not.
    """

    history = [
        match.group(0)
        for match in _HISTORY_ITEM.finditer(document)
        if _attr(match.group(0), "darktable:operation")
    ]
    try:
        limit = int(end)
    except (TypeError, ValueError):
        limit = len(history)
    if limit < 0:
        limit = 0
    if limit >= len(history):
        return document, 0
    dropped = set(history[limit:])
    drop_nums = {
        _attr(tag, "darktable:num") for tag in history[limit:] if _attr(tag, "darktable:num")
    }

    def drop(match: re.Match[str]) -> str:
        tag = match.group(0)
        if tag in dropped:
            return ""
        if _attr(tag, "darktable:operation"):
            return tag
        number = _attr(tag, "darktable:mask_num") or _attr(tag, "darktable:num")
        if number and number in drop_nums:
            return ""
        return tag

    cleaned = _HISTORY_ITEM.sub(drop, document)
    cleaned = re.sub(
        r'darktable:history_end="\d+"',
        f'darktable:history_end="{limit}"',
        cleaned,
        count=1,
    )
    return cleaned, len(history) - limit


def collapse_history_to_cursor(document: str) -> tuple[str, int]:
    """Keep every history row.

    Accept used to delete the rows after ``history_end``. Those rows are
    later steps the photographer still has. A new module is inserted at
    that marker instead, so this leaves the sidecar unchanged.
    """

    return document, 0


def revert_library_image(library_path: str, image_id: int, stage: str | None = None) -> int:
    """Open the library, remove this plugin's rows, and commit."""

    # A short wait. Darktable holds the library for the whole session, and
    # the caller rewrites the sidecar when this wait runs out.
    connection = sqlite3.connect(library_path, timeout=1.5)
    try:
        connection.execute("BEGIN IMMEDIATE")
        removed = revert_image_history(connection, image_id, stage)
        connection.commit()
        return removed
    except Exception:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        raise
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remove AI editor history rows from a darktable library.")
    parser.add_argument("--library", default="", help="Path to library.db")
    parser.add_argument("--image", type=int, default=0, help="Darktable image id")
    parser.add_argument("--stage", default="", help="One stage to drop, such as vignette. Empty removes every AI row.")
    parser.add_argument("--xmp", default="", help="Sidecar to strip when the library is locked.")
    parser.add_argument("--xmp-out", default="", help="Where to write the stripped sidecar.")
    parser.add_argument(
        "--collapse-xmp",
        action="store_true",
        help="Leave history after history_end in place.",
    )
    parser.add_argument(
        "--keep-prefix",
        action="store_true",
        help="Keep only the highlighted history prefix. Used by Copy.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="How many history rows the highlighted prefix contains.",
    )
    parser.add_argument(
        "--step-xmp",
        action="store_true",
        help="Remove or restore one history row. Back stores it for Next.",
    )
    parser.add_argument(
        "--direction",
        default="",
        help="back or next. Used with --step-xmp.",
    )
    parser.add_argument(
        "--redo",
        default="",
        help="Where Back stores a removed history row so Next can splice it back.",
    )
    parser.add_argument(
        "--also",
        action="append",
        default=[],
        help="Another history name this photo's plan recorded, such as an eye sentence.",
    )
    args = parser.parse_args(argv)
    if args.xmp and args.step_xmp:
        source = Path(args.xmp)
        if not source.is_file():
            print(f"sidecar not found: {source}", file=sys.stderr)
            return 1
        if not args.xmp_out:
            print("a stepped sidecar needs --xmp-out", file=sys.stderr)
            return 1
        direction = -1 if args.direction == "back" else 1 if args.direction == "next" else 0
        if direction == 0:
            print("direction must be back or next", file=sys.stderr)
            return 1
        redo_path = Path(args.redo) if args.redo else Path(args.xmp_out).parent / "history-redo.json"
        redo = _load_redo(redo_path)
        document = source.read_text(encoding="utf-8")
        if direction < 0:
            updated, entry, moved = pop_history_level(document)
            name = str(entry.get("name") or "") if moved else ""
            if moved and redo_path is not None:
                redo.append(entry)
                _save_redo(redo_path, redo)
        else:
            if not redo:
                print("moved 0")
                return 0
            entry = redo.pop()
            updated, name, moved = push_history_level(document, entry)
            if moved and redo_path is not None:
                _save_redo(redo_path, redo)
        if moved == 0:
            print("moved 0")
            return 0
        destination = Path(args.xmp_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(updated, encoding="utf-8", newline="\n")
        print(f"moved 1\nname {name}")
        return 0
    if args.xmp and args.keep_prefix:
        source = Path(args.xmp)
        if not source.is_file():
            print(f"sidecar not found: {source}", file=sys.stderr)
            return 1
        if not args.xmp_out:
            print("a copied sidecar needs --xmp-out", file=sys.stderr)
            return 1
        document = source.read_text(encoding="utf-8")
        end = args.end
        if end < 0:
            found = re.search(r'darktable:history_end="(\d+)"', document)
            end = int(found.group(1)) if found else 0
        cleaned, removed = keep_history_prefix(document, end)
        destination = Path(args.xmp_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(cleaned, encoding="utf-8", newline="\n")
        print(f"removed {removed}")
        return 0
    if args.xmp and args.collapse_xmp:
        source = Path(args.xmp)
        if not source.is_file():
            print(f"sidecar not found: {source}", file=sys.stderr)
            return 1
        if not args.xmp_out:
            print("a collapsed sidecar needs --xmp-out", file=sys.stderr)
            return 1
        cleaned, removed = collapse_history_to_cursor(source.read_text(encoding="utf-8"))
        destination = Path(args.xmp_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(cleaned, encoding="utf-8", newline="\n")
        print(f"removed {removed}")
        return 0
    if args.xmp:
        source = Path(args.xmp)
        if not source.is_file():
            print(f"sidecar not found: {source}", file=sys.stderr)
            return 1
        if not args.xmp_out:
            print("a stripped sidecar needs --xmp-out", file=sys.stderr)
            return 1
        cleaned, removed = strip_ai_history(source.read_text(encoding="utf-8"), args.also)
        destination = Path(args.xmp_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(cleaned, encoding="utf-8", newline="\n")
        print(f"removed {removed}")
        return 0
    if not args.library or not args.image:
        print("a library path and an image id are required", file=sys.stderr)
        return 1
    try:
        removed = revert_library_image(args.library, args.image, args.stage or None)
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except sqlite3.OperationalError as exc:
        print(f"the library database is busy: {exc}", file=sys.stderr)
        return 1
    print(f"removed {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
