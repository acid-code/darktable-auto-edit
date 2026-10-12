"""Name matching and history-row removal. These tests do not open Darktable."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dt.revert import is_ai_editor_history_name, main, revert_image_history


def _library() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE images (
          id INTEGER PRIMARY KEY,
          history_end INTEGER,
          thumb_timestamp INTEGER,
          thumb_maxmip INTEGER
        );
        CREATE TABLE history (
          imgid INTEGER,
          num INTEGER,
          operation TEXT,
          multi_name TEXT
        );
        CREATE TABLE masks_history (
          imgid INTEGER,
          num INTEGER,
          formid INTEGER
        );
        CREATE TABLE history_hash (
          imgid INTEGER PRIMARY KEY,
          current_hash BLOB
        );
        """
    )
    return connection


class NameMatchTests(unittest.TestCase):
    def test_plugin_instance_names(self):
        for name in (
            "AI exposure",
            "AI color",
            "AI face",
            "AI subject",
            "AI animal",
            "AI architecture",
            "AI future",
        ):
            self.assertTrue(is_ai_editor_history_name(name), name)

    def test_photographer_modules_stay(self):
        for name in ("", " ", "exposure", "color", "vignette", "portrait", "AI", "My AI look"):
            self.assertFalse(is_ai_editor_history_name(name), repr(name))
        self.assertFalse(is_ai_editor_history_name(None))


class RevertHistoryTests(unittest.TestCase):
    def test_removes_only_ai_rows_and_their_masks(self):
        connection = _library()
        connection.execute("INSERT INTO images VALUES (7, 6, 10, 2)")
        connection.execute("INSERT INTO history_hash VALUES (7, X'abcd')")
        rows = [
            (7, 0, "exposure", ""),
            (7, 1, "colorbalancergb", "portrait"),
            (7, 2, "vignette", ""),
            (7, 3, "exposure", "AI exposure"),
            (7, 4, "colorbalancergb", "AI color"),
            (7, 5, "vignette", "AI face"),
        ]
        connection.executemany("INSERT INTO history VALUES (?, ?, ?, ?)", rows)
        connection.executemany(
            "INSERT INTO masks_history VALUES (?, ?, ?)",
            [(7, 2, 1), (7, 5, 2)],
        )

        removed = revert_image_history(connection, 7)

        self.assertEqual(removed, 3)
        kept = connection.execute(
            "SELECT num, operation, multi_name FROM history ORDER BY num"
        ).fetchall()
        self.assertEqual(
            kept,
            [(0, "exposure", ""), (1, "colorbalancergb", "portrait"), (2, "vignette", "")],
        )
        self.assertEqual(
            connection.execute("SELECT num FROM masks_history ORDER BY num").fetchall(),
            [(2,)],
        )
        image = connection.execute(
            "SELECT history_end, thumb_timestamp, thumb_maxmip FROM images WHERE id = 7"
        ).fetchone()
        self.assertEqual(image, (3, -1, 0))
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM history_hash").fetchone()[0],
            0,
        )

    def test_several_runs_are_all_removed(self):
        connection = _library()
        connection.execute("INSERT INTO images VALUES (1, 4, 1, 1)")
        connection.executemany(
            "INSERT INTO history VALUES (?, ?, ?, ?)",
            [
                (1, 0, "exposure", ""),
                (1, 1, "exposure", "AI exposure"),
                (1, 2, "vignette", "AI animal"),
                (1, 3, "exposure", "AI exposure"),
            ],
        )

        self.assertEqual(revert_image_history(connection, 1), 3)
        self.assertEqual(
            connection.execute("SELECT multi_name FROM history ORDER BY num").fetchall(),
            [("",)],
        )
        self.assertEqual(connection.execute("SELECT history_end FROM images").fetchone()[0], 1)

    def test_undone_ai_steps_do_not_change_the_active_prefix(self):
        connection = _library()
        connection.execute("INSERT INTO images VALUES (4, 2, 5, 1)")
        connection.executemany(
            "INSERT INTO history VALUES (?, ?, ?, ?)",
            [
                (4, 0, "exposure", ""),
                (4, 1, "colorbalancergb", ""),
                (4, 2, "colorbalancergb", "AI color"),
                (4, 3, "exposure", "AI exposure"),
                (4, 4, "vignette", "AI architecture"),
            ],
        )

        self.assertEqual(revert_image_history(connection, 4), 3)
        self.assertEqual(
            connection.execute("SELECT num, multi_name FROM history ORDER BY num").fetchall(),
            [(0, ""), (1, "")],
        )
        self.assertEqual(connection.execute("SELECT history_end FROM images").fetchone()[0], 2)

    def test_no_ai_rows_leaves_the_image_alone(self):
        connection = _library()
        connection.execute("INSERT INTO images VALUES (9, 1, 4, 3)")
        connection.execute("INSERT INTO history VALUES (9, 0, 'exposure', '')")
        connection.execute("INSERT INTO history_hash VALUES (9, X'aa')")

        self.assertEqual(revert_image_history(connection, 9), 0)
        self.assertEqual(
            connection.execute("SELECT history_end, thumb_timestamp FROM images").fetchone(),
            (1, 4),
        )
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM history_hash").fetchone()[0], 1)

    def test_missing_image_is_an_error(self):
        connection = _library()
        with self.assertRaises(LookupError):
            revert_image_history(connection, 99)

    def test_command_line_prints_how_many_rows_were_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "library.db"
            connection = sqlite3.connect(library)
            connection.executescript(
                """
                CREATE TABLE images (
                  id INTEGER PRIMARY KEY,
                  history_end INTEGER,
                  thumb_timestamp INTEGER,
                  thumb_maxmip INTEGER
                );
                CREATE TABLE history (
                  imgid INTEGER,
                  num INTEGER,
                  operation TEXT,
                  multi_name TEXT
                );
                CREATE TABLE masks_history (
                  imgid INTEGER,
                  num INTEGER,
                  formid INTEGER
                );
                CREATE TABLE history_hash (
                  imgid INTEGER PRIMARY KEY,
                  current_hash BLOB
                );
                """
            )
            connection.execute("INSERT INTO images VALUES (3, 2, 1, 1)")
            connection.executemany(
                "INSERT INTO history VALUES (?, ?, ?, ?)",
                [(3, 0, "exposure", ""), (3, 1, "exposure", "AI exposure")],
            )
            connection.commit()
            connection.close()

            buffer = StringIO()
            with redirect_stdout(buffer):
                code = main(["--library", str(library), "--image", "3"])

            self.assertEqual(code, 0)
            self.assertIn("removed 1", buffer.getvalue())
            check = sqlite3.connect(library)
            self.assertEqual(
                check.execute("SELECT multi_name FROM history").fetchall(),
                [("",)],
            )
            check.close()


if __name__ == "__main__":
    unittest.main()
