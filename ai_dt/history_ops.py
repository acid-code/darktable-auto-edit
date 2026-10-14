"""Read which module operations are already on one image.

Filmic and sigmoid are display transforms. A second one doubles the
roll-off, so the style appends one only when the history has neither.
When the library cannot be read, the set is unknown and neither is added.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DISPLAY_TRANSFORMS = frozenset({"filmicrgb", "sigmoid"})


def history_operations(library: Path, image_id: int) -> set[str] | None:
    """Operation names on this image, or None when the library cannot be read."""

    path = Path(library)
    if not path.is_file():
        return None
    try:
        # Read-only so a library Darktable already has open can still be queried.
        uri = path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=2)
    except (sqlite3.Error, ValueError):
        return None
    try:
        rows = connection.execute(
            "SELECT operation FROM main.history WHERE imgid = ?",
            (int(image_id),),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    return {str(row[0]) for row in rows if row and row[0]}


def choose_display(
    history_ops: set[str] | None,
    *,
    look: str,
    vibe: str,
    picture_type: str,
    problems: list[str],
) -> str | None:
    """``filmicrgb``, ``sigmoid``, or None when one must not be appended.

    Unknown history adds nothing. A history that already has either
    transform adds nothing. Sigmoid wins when the look asks for it.
    Filmic is for a night frame or blown highlights.
    """

    if history_ops is None:
        return None
    if DISPLAY_TRANSFORMS & set(history_ops):
        return None
    text = f"{look} {vibe}".lower()
    wants_sigmoid = "sigmoid" in text
    wants_filmic = picture_type == "night" or "blown-highlights" in problems
    if wants_sigmoid:
        return "sigmoid"
    if wants_filmic:
        return "filmicrgb"
    return None
