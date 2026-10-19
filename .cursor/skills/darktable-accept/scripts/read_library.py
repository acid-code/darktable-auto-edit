"""Print read-only library history for photos whose filename contains a fragment."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

LIBRARY = Path(r"C:/Users/asaf2/AppData/Local/darktable/library.db")


def main(fragment: str) -> int:
    if not LIBRARY.is_file():
        print(f"library not found: {LIBRARY}", file=sys.stderr)
        return 1
    connection = sqlite3.connect(f"file:{LIBRARY.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, filename, version, history_end
            FROM images
            WHERE filename LIKE ?
            ORDER BY id, version
            """,
            (f"%{fragment}%",),
        ).fetchall()
        if not rows:
            print("no image")
            return 0
        for row in rows:
            print(
                f"id {row['id']} version {row['version']} "
                f"history_end {row['history_end']} {row['filename']}"
            )
            items = connection.execute(
                """
                SELECT num, operation, multi_name
                FROM history
                WHERE imgid = ?
                ORDER BY num
                """,
                (row["id"],),
            ).fetchall()
            print(f"  rows {len(items)}")
            for item in items:
                print(
                    f"  {item['num']:>3} {item['operation']} {item['multi_name'] or ''}"
                )
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: read_library.py FILENAME_FRAGMENT", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
