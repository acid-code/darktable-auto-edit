"""The recipe tells Darktable which file to open and which file to leave alone."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ai_dt.handoff import build_handoff
from ai_dt.pipeline import analyze
from ai_dt.status_text import panel_status


class HandoffTests(unittest.TestCase):
    def test_the_masked_tiff_is_what_opens_and_the_source_stays(self) -> None:
        handoff = build_handoff(
            {
                "render": {"file": "/work/edited.tif", "format": "tiff16"},
                "style": {"file": "/work/edit.dtstyle", "name": "AI Edit"},
                "input": {"resolution": "half"},
                "features": ["animal", "perch"],
                "features_committed": ["separate"],
            }
        )
        self.assertEqual(handoff["open"], "masked-picture")
        self.assertEqual(handoff["masked_file"], "/work/edited.tif")
        self.assertEqual(handoff["style_file"], "/work/edit.dtstyle")
        self.assertEqual(handoff["style_role"], "global-grade")
        self.assertFalse(handoff["apply_style_to_source"])
        self.assertTrue(handoff["source_unchanged"])
        self.assertEqual(handoff["white_balance"], "camera")
        self.assertEqual(handoff["resolution"], "half")
        self.assertIn("separate", handoff["features"])
        self.assertIn("perch", handoff["features"])
        self.assertIn("original file stays unchanged", handoff["sentence"])
        self.assertIn("half resolution", handoff["sentence"])
        self.assertIn("does not contain these masks", handoff["sentence"])

    def test_full_resolution_is_named(self) -> None:
        handoff = build_handoff(
            {
                "render": {"file": "/work/edited.tif"},
                "style": {"file": "/work/edit.dtstyle"},
                "input": {"resolution": "full"},
            }
        )
        self.assertEqual(handoff["resolution"], "full")
        self.assertIn("full resolution", handoff["sentence"])

    def test_the_panel_says_where_the_masks_are(self) -> None:
        text = panel_status(
            {
                "picture_backend": "Picture: measured locally",
                "plan_backend": "Plan: not used",
                "handoff": {
                    "sentence": (
                        "The masked picture is edited.tif (camera white balance, half resolution). "
                        "The original file stays unchanged. "
                        "The style file is the global grade only and does not contain these masks."
                    )
                },
            }
        )
        self.assertIn("masked picture", text)
        self.assertIn("original file stays unchanged", text)
        self.assertNotIn("History changes only when you pick a look", text)


class AnalyzeHandoffTests(unittest.TestCase):
    def test_a_run_records_the_picture_to_open(self) -> None:
        saved = os.environ.get("AI_EDITOR_REASONER")
        os.environ["AI_EDITOR_REASONER"] = "off"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                path = directory / "frame.png"
                Image.new("RGB", (32, 24), (18, 18, 22)).save(path)
                recipe = analyze(
                    path,
                    detector="heuristic",
                    advisor="histogram",
                    output_dir=directory,
                )
        finally:
            if saved is None:
                os.environ.pop("AI_EDITOR_REASONER", None)
            else:
                os.environ["AI_EDITOR_REASONER"] = saved
        self.assertEqual(recipe["handoff"]["open"], "masked-picture")
        self.assertTrue(recipe["handoff"]["source_unchanged"])
        self.assertFalse(recipe["handoff"]["apply_style_to_source"])
        self.assertTrue(str(recipe["handoff"]["masked_file"]).endswith("edited.tif"))
        self.assertIn("original file stays unchanged", recipe["status_text"])
        self.assertIn("half resolution", recipe["status_text"])


if __name__ == "__main__":
    unittest.main()
