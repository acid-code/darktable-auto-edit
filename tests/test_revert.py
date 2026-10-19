"""Name matching and history-row removal. These tests do not open Darktable."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dt.revert import (
    accepted_step_names,
    collapse_history_to_cursor,
    keep_history_prefix,
    is_ai_editor_history_name,
    main,
    move_history_cursor,
    pop_history_level,
    push_history_level,
    revert_image_history,
    strip_ai_history,
)


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

    def test_an_accepted_step_name_is_an_ai_edit(self):
        from ai_dt.mapping import step_history_name

        names = []
        for feature in (
            "subject",
            "foreground",
            "opening",
            "shafts",
            "highlights",
            "colorzones",
            "eyes",
        ):
            names.append(step_history_name(feature))
            names.append(step_history_name(feature, again=True))
        for hue in ("red", "orange", "yellow", "green", "blue"):
            names.append(step_history_name("colorzones", title=f"highlight the {hue}"))
        names.append(step_history_name("recolor", title="too much red"))
        for name in names:
            self.assertTrue(name, name)
            self.assertTrue(is_ai_editor_history_name(name), name)
        self.assertFalse(is_ai_editor_history_name("Highlight"))
        self.assertFalse(is_ai_editor_history_name("lift the catchlight, then hold the fur"))
        lua = (ROOT / "ai_editor.lua").read_text(encoding="utf-8")
        for name in accepted_step_names():
            self.assertIn(f'"{name}"', lua, name)


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

    def test_a_named_stage_removes_only_that_history_module(self):
        connection = _library()
        connection.execute("INSERT INTO images VALUES (3, 3, 1, 0)")
        connection.executemany(
            "INSERT INTO history VALUES (?, ?, ?, ?)",
            [
                (3, 0, "exposure", "AI exposure"),
                (3, 1, "colorbalancergb", "AI color"),
                (3, 2, "vignette", "AI animal"),
            ],
        )
        self.assertEqual(revert_image_history(connection, 3, "vignette"), 1)
        self.assertEqual(
            connection.execute("SELECT operation, multi_name FROM history ORDER BY num").fetchall(),
            [("exposure", "AI exposure"), ("colorbalancergb", "AI color")],
        )

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


class SidecarRevertTests(unittest.TestCase):
    def test_ai_history_is_removed_and_the_rest_stays(self) -> None:
        document = """
        <rdf:Description darktable:history_end="3">
          <rdf:li darktable:num="0" darktable:operation="exposure" darktable:multi_name=""/>
          <rdf:li darktable:num="1" darktable:operation="colorbalancergb" darktable:multi_name="portrait"/>
          <rdf:li darktable:num="2" darktable:operation="exposure" darktable:multi_name="AI exposure" darktable:params="abcd"/>
          <rdf:li darktable:num="2" darktable:mask="1"/>
        </rdf:Description>
        """
        cleaned, removed = strip_ai_history(document)
        self.assertEqual(removed, 1)
        self.assertNotIn("AI exposure", cleaned)
        self.assertIn('darktable:multi_name="portrait"', cleaned)
        self.assertIn('darktable:multi_name=""', cleaned)
        self.assertNotIn('darktable:num="2"', cleaned)
        self.assertIn('darktable:history_end="2"', cleaned)

    def test_subject_lift_and_the_red_highlight_are_removed(self) -> None:
        document = """
        <rdf:Description darktable:history_end="4">
          <rdf:li darktable:num="0" darktable:operation="exposure" darktable:multi_name=""/>
          <rdf:li darktable:num="1" darktable:operation="colorbalancergb" darktable:multi_name="portrait"/>
          <rdf:li darktable:num="2" darktable:operation="exposure" darktable:multi_name="Subject lift"/>
          <rdf:li darktable:num="3" darktable:operation="colorzones" darktable:multi_name="Highlight the red"/>
        </rdf:Description>
        """
        cleaned, removed = strip_ai_history(document)
        self.assertEqual(removed, 2)
        self.assertNotIn("Subject lift", cleaned)
        self.assertNotIn("Highlight the red", cleaned)
        self.assertIn('darktable:multi_name="portrait"', cleaned)
        self.assertIn('darktable:multi_name=""', cleaned)
        self.assertIn('darktable:history_end="2"', cleaned)

    def test_a_logged_eye_sentence_is_removed_when_the_plan_names_it(self) -> None:
        document = (
            '<rdf:Description darktable:history_end="2">'
            '<rdf:li darktable:num="0" darktable:operation="colorbalancergb" darktable:multi_name="portrait"/>'
            '<rdf:li darktable:num="1" darktable:operation="colorbalancergb"'
            ' darktable:multi_name="lift the catchlight, then hold the fur"/>'
            "</rdf:Description>"
        )
        kept, left = strip_ai_history(document)
        self.assertEqual(left, 0)
        self.assertIn("catchlight", kept)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "photo.xmp"
            destination = Path(directory) / "revert.xmp"
            source.write_text(document, encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--xmp",
                        str(source),
                        "--xmp-out",
                        str(destination),
                        "--also",
                        "lift the catchlight, then hold the fur",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("removed 1", output.getvalue())
            cleaned = destination.read_text(encoding="utf-8")
            self.assertNotIn("catchlight", cleaned)
            self.assertIn('darktable:multi_name="portrait"', cleaned)
            self.assertIn('darktable:history_end="1"', cleaned)

    def test_history_after_the_viewed_step_stays_in_the_stack(self) -> None:
        document = """
        <rdf:Description darktable:history_end="2">
          <rdf:li darktable:num="0" darktable:operation="exposure" darktable:multi_name=""/>
          <rdf:li darktable:num="1" darktable:operation="colorbalancergb" darktable:multi_name="AI color"/>
          <rdf:li darktable:num="2" darktable:operation="exposure" darktable:multi_name="Subject lift"/>
          <rdf:li darktable:num="3" darktable:operation="vignette" darktable:multi_name="AI animal"/>
        </rdf:Description>
        """
        cleaned, removed = collapse_history_to_cursor(document)
        self.assertEqual(removed, 0)
        self.assertIn('darktable:history_end="2"', cleaned)
        self.assertIn("Subject lift", cleaned)
        self.assertIn("AI animal", cleaned)

    def test_a_copy_keeps_only_the_highlighted_prefix(self) -> None:
        document = """
        <rdf:Description darktable:history_end="2">
          <rdf:li darktable:num="0" darktable:operation="exposure" darktable:multi_name=""/>
          <rdf:li darktable:num="1" darktable:operation="colorbalancergb" darktable:multi_name="AI color"/>
          <rdf:li darktable:num="2" darktable:operation="exposure" darktable:multi_name="Subject lift"/>
          <rdf:li darktable:num="3" darktable:operation="vignette" darktable:multi_name="AI animal"/>
          <rdf:li darktable:mask_num="3" darktable:mask_name="tail"/>
          <rdf:li darktable:mask_num="1" darktable:mask_name="kept"/>
        </rdf:Description>
        """
        cleaned, removed = keep_history_prefix(document, 2)
        self.assertEqual(removed, 2)
        self.assertIn('darktable:history_end="2"', cleaned)
        self.assertIn("AI color", cleaned)
        self.assertNotIn("Subject lift", cleaned)
        self.assertNotIn("AI animal", cleaned)
        self.assertNotIn('darktable:mask_name="tail"', cleaned)
        self.assertIn('darktable:mask_name="kept"', cleaned)

    def test_undone_ai_steps_do_not_pull_history_end_back(self) -> None:
        document = """
        <rdf:Description darktable:history_end="2">
          <rdf:li darktable:num="0" darktable:operation="exposure" darktable:multi_name=""/>
          <rdf:li darktable:num="1" darktable:operation="sigmoid" darktable:multi_name=""/>
          <rdf:li darktable:num="2" darktable:operation="exposure" darktable:multi_name="AI exposure"/>
          <rdf:li darktable:num="3" darktable:operation="colorbalancergb" darktable:multi_name="AI color"/>
        </rdf:Description>
        """
        cleaned, removed = strip_ai_history(document)
        self.assertEqual(removed, 2)
        self.assertNotIn("AI exposure", cleaned)
        self.assertNotIn("AI color", cleaned)
        self.assertIn('darktable:operation="sigmoid"', cleaned)
        self.assertIn('darktable:history_end="2"', cleaned)

    def test_a_second_strip_of_the_same_sidecar_removes_nothing(self) -> None:
        # Same shape as IMG_8489 version 0: 11 active modules, then AI color
        # and AI exposure past history_end. The tag text is not a history name.
        items = [
            f'<rdf:li darktable:num="{num}" darktable:operation="exposure" darktable:multi_name=""/>'
            for num in range(10)
        ]
        items.append(
            '<rdf:li\n'
            '  darktable:num="10"\n'
            '  darktable:operation="sigmoid"\n'
            '  darktable:multi_name="_builtin_scene-referred default"/>'
        )
        items.append(
            '<rdf:li\n'
            '  darktable:num="11"\n'
            '  darktable:operation="colorbalancergb"\n'
            '  darktable:multi_name="AI color"/>'
        )
        items.append(
            '<rdf:li\n'
            '  darktable:num="12"\n'
            '  darktable:operation="exposure"\n'
            '  darktable:multi_name="AI exposure"/>'
        )
        document = (
            '<rdf:Description darktable:history_end="11">'
            "<darktable:history><rdf:Seq>"
            + "".join(items)
            + "</rdf:Seq></darktable:history>"
            "<dc:subject><rdf:Bag><rdf:li>AI Edit 11364</rdf:li></rdf:Bag></dc:subject>"
            "</rdf:Description>"
        )
        cleaned, removed = strip_ai_history(document)
        self.assertEqual(removed, 2)
        self.assertNotIn('darktable:multi_name="AI color"', cleaned)
        self.assertNotIn('darktable:multi_name="AI exposure"', cleaned)
        self.assertIn('darktable:operation="sigmoid"', cleaned)
        self.assertIn('darktable:history_end="11"', cleaned)
        self.assertIn("<rdf:li>AI Edit 11364</rdf:li>", cleaned)
        again, left = strip_ai_history(cleaned)
        self.assertEqual(left, 0)
        self.assertEqual(again, cleaned)

    def test_only_ai_steps_inside_the_active_prefix_move_history_end(self) -> None:
        # Same shape as IMG_8489 version 2: one AI module inside history_end,
        # and more AI modules after it.
        items = [
            f'<rdf:li darktable:num="{num}" darktable:operation="exposure" darktable:multi_name=""/>'
            for num in range(11)
        ]
        items.append(
            '<rdf:li darktable:num="11" darktable:operation="channelmixerrgb"'
            ' darktable:multi_name="AI color calibration"/>'
        )
        items.extend(
            f'<rdf:li darktable:num="{num}" darktable:operation="exposure" darktable:multi_name="AI exposure"/>'
            for num in range(12, 20)
        )
        document = '<rdf:Description darktable:history_end="12">' + "".join(items) + "</rdf:Description>"
        cleaned, removed = strip_ai_history(document)
        self.assertEqual(removed, 9)
        self.assertEqual(cleaned.count('darktable:operation="exposure"'), 11)
        self.assertNotIn("AI ", cleaned)
        self.assertIn('darktable:history_end="11"', cleaned)
        self.assertEqual(strip_ai_history(cleaned)[1], 0)

    def test_command_line_second_pass_on_the_written_sidecar_removes_nothing(self) -> None:
        document = (
            '<rdf:Description darktable:history_end="2">'
            '<rdf:li darktable:num="0" darktable:operation="exposure" darktable:multi_name=""/>'
            '<rdf:li darktable:num="1" darktable:operation="exposure" darktable:multi_name="AI exposure"/>'
            "</rdf:Description>"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "photo.xmp"
            destination = Path(directory) / "revert.xmp"
            source.write_text(document, encoding="utf-8")
            first = StringIO()
            with redirect_stdout(first):
                code = main(["--xmp", str(source), "--xmp-out", str(destination)])
            self.assertEqual(code, 0)
            self.assertIn("removed 1", first.getvalue())
            self.assertNotIn("AI exposure", destination.read_text(encoding="utf-8"))
            second = StringIO()
            with redirect_stdout(second):
                code = main(["--xmp", str(destination), "--xmp-out", str(destination)])
            self.assertEqual(code, 0)
            self.assertIn("removed 0", second.getvalue())

    def test_the_script_strips_a_subject_lift_without_the_package_on_the_path(self) -> None:
        document = (
            '<rdf:Description darktable:history_end="2">'
            '<rdf:li darktable:num="0" darktable:operation="exposure" darktable:multi_name="portrait"/>'
            '<rdf:li darktable:num="1" darktable:operation="exposure" darktable:multi_name="Subject lift"/>'
            '<rdf:li darktable:num="2" darktable:operation="colorzones" darktable:multi_name="Highlight the red"/>'
            "</rdf:Description>"
        )
        script = ROOT / "ai_dt" / "revert.py"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "photo.xmp"
            destination = Path(directory) / "revert.xmp"
            source.write_text(document, encoding="utf-8")
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(script),
                    "--xmp",
                    str(source),
                    "--xmp-out",
                    str(destination),
                ],
                capture_output=True,
                text=True,
                env=env,
                cwd=directory,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("removed 2", completed.stdout)
            cleaned = destination.read_text(encoding="utf-8")
            self.assertNotIn("Subject lift", cleaned)
            self.assertNotIn("Highlight the red", cleaned)
            self.assertIn('darktable:multi_name="portrait"', cleaned)


class HistoryCursorTests(unittest.TestCase):
    def _stack(self) -> str:
        return (
            '<rdf:Description darktable:history_end="3">'
            '<rdf:li darktable:num="0" darktable:operation="exposure" darktable:multi_name=""/>'
            '<rdf:li darktable:num="1" darktable:operation="colorbalancergb" darktable:multi_name="portrait"/>'
            '<rdf:li darktable:num="2" darktable:operation="exposure" darktable:multi_name="AI eyes"/>'
            '<rdf:li darktable:mask_num="1" darktable:mask_name="window" darktable:mask_id="7"/>'
            "</rdf:Description>"
        )

    def test_back_and_next_keep_every_item_and_the_mask(self) -> None:
        original = self._stack()
        backed, name, moved = move_history_cursor(original, -1)
        self.assertEqual(moved, 1)
        self.assertEqual(name, "AI eyes")
        self.assertIn('darktable:history_end="2"', backed)
        self.assertIn('darktable:multi_name="portrait"', backed)
        self.assertIn('darktable:multi_name="AI eyes"', backed)
        self.assertIn('darktable:mask_name="window"', backed)
        restored, forward_name, forward = move_history_cursor(backed, 1)
        self.assertEqual(forward, 1)
        self.assertEqual(forward_name, "AI eyes")
        self.assertIn('darktable:history_end="3"', restored)
        self.assertEqual(restored.count("rdf:li"), original.count("rdf:li"))

    def test_back_at_the_start_and_next_at_the_end_write_nothing(self) -> None:
        start = self._stack().replace('history_end="3"', 'history_end="0"', 1)
        same, name, moved = move_history_cursor(start, -1)
        self.assertEqual(moved, 0)
        self.assertEqual(name, "")
        self.assertEqual(same, start)
        end = self._stack()
        same, name, moved = move_history_cursor(end, 1)
        self.assertEqual(moved, 0)
        self.assertEqual(same, end)

    def test_a_step_that_cannot_move_does_not_write_the_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "photo.xmp"
            destination = Path(directory) / "out.xmp"
            source.write_text(self._stack(), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--step-xmp",
                        "--direction",
                        "next",
                        "--xmp",
                        str(source),
                        "--xmp-out",
                        str(destination),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("moved 0", output.getvalue())
            self.assertFalse(destination.exists())
            self.assertIn('history_end="3"', source.read_text(encoding="utf-8"))


class HistoryLevelTests(unittest.TestCase):
    def _stack(self) -> str:
        return (
            '<rdf:Description darktable:history_end="2">'
            "<darktable:masks_history><rdf:Seq>"
            '<rdf:li darktable:mask_num="0" darktable:mask_name="keep"/>'
            '<rdf:li darktable:mask_num="1" darktable:mask_name="Highlight the red"/>'
            "</rdf:Seq></darktable:masks_history>"
            "<darktable:history><rdf:Seq>"
            '<rdf:li darktable:num="0" darktable:operation="exposure" darktable:multi_name="keep"/>'
            "<rdf:li"
            ' darktable:num="1"'
            ' darktable:operation="colorzones"'
            ' darktable:multi_name="Highlight the red"/>'
            "</rdf:Seq></darktable:history>"
            "</rdf:Description>"
        )

    def test_back_removes_the_top_level_and_next_puts_that_level_back(self) -> None:
        original = self._stack()
        backed, entry, moved = pop_history_level(original)
        self.assertEqual(moved, 1)
        self.assertEqual(entry["name"], "Highlight the red")
        self.assertIn('darktable:history_end="1"', backed)
        self.assertNotIn("colorzones", backed)
        self.assertNotIn("Highlight the red", backed)
        self.assertIn('darktable:multi_name="keep"', backed)
        self.assertIn('darktable:mask_name="keep"', backed)
        restored, name, forward = push_history_level(backed, entry)
        self.assertEqual(forward, 1)
        self.assertEqual(name, "Highlight the red")
        self.assertIn('darktable:history_end="2"', restored)
        self.assertIn('darktable:operation="colorzones"', restored)
        self.assertIn('darktable:mask_num="1"', restored)
        self.assertIn('darktable:mask_name="keep"', restored)

    def test_back_at_the_first_level_removes_nothing(self) -> None:
        start = self._stack().replace('history_end="2"', 'history_end="0"', 1)
        same, entry, moved = pop_history_level(start)
        self.assertEqual(moved, 0)
        self.assertEqual(entry, {})
        self.assertEqual(same, start)

    def test_back_leaves_a_later_undone_row_and_removes_the_active_one(self) -> None:
        document = self._stack().replace('history_end="2"', 'history_end="1"', 1)
        backed, entry, moved = pop_history_level(document)
        self.assertEqual(moved, 1)
        self.assertEqual(entry["name"], "keep")
        self.assertNotIn('darktable:multi_name="keep"', backed)
        self.assertIn("colorzones", backed)
        self.assertIn('darktable:history_end="0"', backed)

    def test_the_button_round_trip_stores_the_removed_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "photo.xmp"
            destination = Path(directory) / "out.xmp"
            redo = Path(directory) / "history-redo.json"
            source.write_text(self._stack(), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--step-xmp",
                        "--direction",
                        "back",
                        "--xmp",
                        str(source),
                        "--xmp-out",
                        str(destination),
                        "--redo",
                        str(redo),
                    ]
                )
            self.assertEqual(code, 0, output.getvalue())
            self.assertIn("moved 1", output.getvalue())
            self.assertNotIn("colorzones", destination.read_text(encoding="utf-8"))
            source.write_text(destination.read_text(encoding="utf-8"), encoding="utf-8")
            forward = StringIO()
            with redirect_stdout(forward):
                code = main(
                    [
                        "--step-xmp",
                        "--direction",
                        "next",
                        "--xmp",
                        str(source),
                        "--xmp-out",
                        str(destination),
                        "--redo",
                        str(redo),
                    ]
                )
            self.assertEqual(code, 0, forward.getvalue())
            self.assertIn("moved 1", forward.getvalue())
            restored = destination.read_text(encoding="utf-8")
            self.assertIn("colorzones", restored)
            self.assertIn('darktable:mask_num="1"', restored)
            self.assertIn('darktable:history_end="2"', restored)


if __name__ == "__main__":
    unittest.main()
