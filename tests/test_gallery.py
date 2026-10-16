"""The contact sheet and the Darktable panel offer the same looks."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from PIL import Image

from ai_dt.gallery import list_cards, read_pick, render_option, wait_for_pick, work_is_gallery, write_pick
from ai_dt.pipeline import analyze, write_outputs


class CardTests(unittest.TestCase):
    def test_every_choice_and_the_feature_pair_are_on_the_sheet(self) -> None:
        cards = list_cards(
            {
                "choices": [
                    {"id": "faithful", "title": "Faithful", "why": "Measured."},
                    {"id": "warmer", "title": "Warmer", "why": "A little warmth."},
                ],
                "features_next": "foreground",
                "features_next_title": "Darken the foreground",
            }
        )
        self.assertEqual(
            [item["id"] for item in cards],
            ["faithful", "warmer", "accept-feature", "skip-feature"],
        )
        self.assertEqual(cards[2]["title"], "Accept Darken the foreground")

    def test_a_preview_pick_and_a_panel_pick_are_the_same_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_pick(work, "warmer", source="preview", title="Warmer")
            preview = read_pick(work)
            write_pick(work, "faithful", source="editor", title="Faithful", applied=True)
            panel = read_pick(work)
        self.assertEqual(preview["id"], "warmer")
        self.assertEqual(preview["source"], "preview")
        self.assertFalse(preview["applied"])
        self.assertEqual(panel["id"], "faithful")
        self.assertTrue(panel["applied"])
        self.assertGreater(panel["at"], preview["at"])

    def test_only_an_editor_work_folder_can_feed_the_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "notes"
            outside.mkdir()
            (outside / "output.json").write_text("{}", encoding="utf-8")
            inside = root / "ai-editor-work" / "4" / "v0"
            inside.mkdir(parents=True)
            (inside / "output.json").write_text("{}", encoding="utf-8")
            self.assertFalse(work_is_gallery(outside))
            self.assertTrue(work_is_gallery(inside))


class WaitTests(unittest.TestCase):
    def test_the_wait_ends_when_a_look_is_chosen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)

            def choose() -> None:
                write_pick(work, "warmer", source="preview", title="Warmer")

            threading.Timer(0.2, choose).start()
            code = wait_for_pick(work, since=0, timeout=2)
            self.assertEqual(code, 0)
            self.assertEqual(read_pick(work)["id"], "warmer")


class RenderTests(unittest.TestCase):
    def test_two_looks_each_get_a_picture(self) -> None:
        saved = os.environ.get("AI_EDITOR_REASONER")
        os.environ["AI_EDITOR_REASONER"] = "off"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "ai-editor-work" / "sheet"
                root.mkdir(parents=True)
                source = root / "frame.png"
                Image.new("RGB", (48, 32), (24, 36, 48)).save(source)
                recipe = analyze(
                    source,
                    detector="heuristic",
                    advisor="histogram",
                    output_dir=root,
                )
                write_outputs(recipe, root)
                recipe = json.loads((root / "output.json").read_text(encoding="utf-8"))
                faithful = render_option(root, source, "faithful")
                warmer = render_option(root, source, "warmer")
                self.assertTrue(faithful.is_file())
                self.assertTrue(warmer.is_file())
                self.assertNotEqual(faithful, warmer)
                self.assertGreater(faithful.stat().st_size, 32)
                self.assertGreater(warmer.stat().st_size, 32)
        finally:
            if saved is None:
                os.environ.pop("AI_EDITOR_REASONER", None)
            else:
                os.environ["AI_EDITOR_REASONER"] = saved


if __name__ == "__main__":
    unittest.main()
