"""Remove the history rows this editor appended.

Darktable 5.4 Lua can reset a whole history stack (`image:reset()`) and can
apply or replace history from a sidecar or a style. It cannot delete one
history row. Replacing the stack would also rewrite the photographer's own
modules. This module deletes only the rows whose instance name this plugin
wrote, by the history number stored on each row. Other numbers stay put, so
the rest of the stack is unchanged.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys


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


def revert_library_image(library_path: str, image_id: int, stage: str | None = None) -> int:
    """Open the library, remove this plugin's rows, and commit."""

    connection = sqlite3.connect(library_path, timeout=8)
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
    parser.add_argument("--library", required=True, help="Path to library.db")
    parser.add_argument("--image", required=True, type=int, help="Darktable image id")
    parser.add_argument("--stage", default="", help="One stage to drop, such as vignette. Empty removes every AI row.")
    args = parser.parse_args(argv)
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
