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
import re
import sqlite3
import sys
from pathlib import Path


def is_ai_editor_history_name(multi_name: str | None) -> bool:
    """Return whether a history instance was named by this plugin.

    Matches ``AI exposure``, ``AI color``, and any instance name that starts
    with ``AI `` (``AI face``, ``AI subject``, ``AI animal``, ``AI architecture``,
    and later categories). A module renamed by hand no longer matches. A module
    the photographer named with that same prefix does.
    """

    if not multi_name:
        return False
    if multi_name in ("AI exposure", "AI color"):
        return True
    return multi_name.startswith("AI ")


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


def strip_ai_history(document: str) -> tuple[str, int]:
    """Remove AI history items from a Darktable XMP sidecar.

    The photographer's other history items stay. Mask entries that belonged
    to a removed item are removed with it. ``history_end`` moves back by the
    number of history items removed.
    """

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
        if not is_ai_editor_history_name(_attr(tag, "darktable:multi_name")):
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
        if not is_ai_editor_history_name(_attr(tag, "darktable:multi_name")):
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
        if is_ai_editor_history_name(_attr(tag, "darktable:mask_name")):
            return ""
        return tag

    cleaned = _HISTORY_ITEM.sub(drop_named_masks, cleaned)
    if removed_before_end:

        def move_end(match: re.Match[str]) -> str:
            value = max(0, int(match.group(1)) - removed_before_end)
            return f'darktable:history_end="{value}"'

        cleaned = re.sub(r'darktable:history_end="(\d+)"', move_end, cleaned, count=1)
    return cleaned, removed


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


def collapse_history_to_cursor(document: str) -> tuple[str, int]:
    """Drop history items after the one the photographer is viewing.

    ``history_end`` counts active items from the start of the stack. Later
    items are still stored. A new edit starts from the viewed item, so those
    later items are removed and ``history_end`` stays where it is.
    """

    history_tags = [
        match.group(0)
        for match in _HISTORY_ITEM.finditer(document)
        if _attr(match.group(0), "darktable:operation")
    ]
    end_match = re.search(r'darktable:history_end="(\d+)"', document)
    history_end = int(end_match.group(1)) if end_match else len(history_tags)
    if history_end >= len(history_tags):
        return document, 0
    drop_nums: set[str] = set()
    for index, tag in enumerate(history_tags):
        if index < history_end:
            continue
        number = _attr(tag, "darktable:num")
        if number:
            drop_nums.add(number)
    if not drop_nums:
        return document, 0

    def drop(match: re.Match[str]) -> str:
        tag = match.group(0)
        if _attr(tag, "darktable:num") in drop_nums:
            return ""
        return tag

    cleaned = _HISTORY_ITEM.sub(drop, document)
    return cleaned, len(drop_nums)


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
        help="Drop history after history_end instead of removing AI rows.",
    )
    parser.add_argument(
        "--step-xmp",
        action="store_true",
        help="Move history_end by one and keep every history item.",
    )
    parser.add_argument(
        "--direction",
        default="",
        help="back or next. Used with --step-xmp.",
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
        updated, name, moved = move_history_cursor(source.read_text(encoding="utf-8"), direction)
        if moved == 0:
            print("moved 0")
            return 0
        destination = Path(args.xmp_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(updated, encoding="utf-8", newline="\n")
        print(f"moved 1\nname {name}")
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
        cleaned, removed = strip_ai_history(source.read_text(encoding="utf-8"))
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
