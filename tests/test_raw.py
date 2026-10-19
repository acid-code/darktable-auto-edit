"""RAW files decode, and a folder of them can be rendered."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ai_dt.batch import main as batch_main
from ai_dt.cli import main
from ai_dt.pipeline import analyze, write_outputs
from ai_dt.tiff16 import write_rgb16
from tests.minimal_dng import write_dng

try:  # rawpy is optional (requirements.txt); without it a DNG cannot be decoded.
    import rawpy  # noqa: F401

    HAVE_RAWPY = True
except ImportError:
    HAVE_RAWPY = False

needs_rawpy = unittest.skipUnless(HAVE_RAWPY, "rawpy is not installed (optional RAW decoder; pip install -r requirements.txt)")


class TiffTests(unittest.TestCase):
    def test_rgb16_round_trip(self) -> None:
        pixels = np.zeros((12, 18, 3), dtype=np.float32)
        pixels[..., 0] = 1.0
        pixels[3, 4] = (0.0, 0.5, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "picture.tif"
            write_rgb16(path, pixels)
            with Image.open(path) as handle:
                self.assertEqual(handle.size, (18, 12))
                self.assertEqual(handle.mode, "RGB")
                red, green, blue = handle.getpixel((4, 3))
        self.assertLess(red, 20)
        self.assertGreater(green, 100)
        self.assertGreater(blue, 240)


class RawPipelineTests(unittest.TestCase):
    @needs_rawpy
    def test_a_dng_is_decoded_and_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = write_dng(directory / "frame.dng")
            recipe = analyze(
                source,
                detector="heuristic",
                advisor="histogram",
                style_name="AI Edit raw",
                output_dir=directory / "out",
            )
            json_path, style_path, edited = write_outputs(recipe, directory / "out")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            with Image.open(edited) as handle:
                size = handle.size
            self.assertEqual(payload["input"]["kind"], "raw")
            self.assertTrue(str(payload["input"]["path"]).endswith("frame.dng"))
            self.assertEqual(payload["width"], 16)
            self.assertEqual(payload["height"], 16)
            self.assertEqual(size, (16, 16))
            self.assertTrue(style_path.is_file())
            self.assertIn("half resolution", " ".join(payload["input"]["notes"]))

    @needs_rawpy
    def test_full_res_keeps_the_sensor_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = write_dng(directory / "frame.dng")
            recipe = analyze(
                source,
                detector="heuristic",
                advisor="histogram",
                output_dir=directory,
                full_res=True,
            )
        self.assertEqual(recipe["width"], 32)
        self.assertEqual(recipe["height"], 32)
        self.assertEqual(recipe["input"]["notes"], [])

    def test_a_broken_raw_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / "broken.NEF"
            source.write_bytes(b"not a camera raw")
            code = main([str(source), "--output-dir", str(directory)])
        self.assertEqual(code, 3)

    def test_a_bad_original_falls_back_to_the_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            preview = directory / "preview.png"
            Image.new("RGB", (24, 24), (20, 20, 20)).save(preview)
            original = directory / "missing-decode.dng"
            original.write_bytes(b"not really a dng")
            recipe = analyze(
                preview,
                detector="heuristic",
                advisor="histogram",
                source_path=original,
            )
        self.assertEqual(recipe["input"]["kind"], "raster")
        self.assertTrue(any("developed picture" in note for note in recipe["input"]["notes"]))

    def test_a_developed_preview_is_used_instead_of_the_camera_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            preview = directory / "preview.jpg"
            Image.new("RGB", (20, 10), (180, 40, 40)).save(preview, quality=90)
            original = write_dng(directory / "frame.dng")
            recipe = analyze(
                preview,
                detector="heuristic",
                advisor="histogram",
                source_path=original,
            )
        self.assertEqual(recipe["input"]["kind"], "raster")
        self.assertEqual(recipe["width"], 20)
        self.assertEqual(recipe["height"], 10)
        self.assertTrue(any("was not decoded" in note for note in recipe["input"]["notes"]))

    @needs_rawpy
    def test_batch_writes_one_tiff_per_picture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "inbox"
            folder.mkdir()
            Image.new("RGB", (20, 20), (30, 30, 30)).save(folder / "one.png")
            write_dng(folder / "two.dng")
            output = Path(tmp) / "done"
            code = batch_main(
                [
                    str(folder),
                    "--output-dir",
                    str(output),
                    "--detector",
                    "heuristic",
                    "--advisor",
                    "histogram",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((output / "one" / "edited.tif").is_file())
            self.assertTrue((output / "two" / "edited.tif").is_file())
            self.assertTrue((output / "two" / "edit.dtstyle").is_file())

    @needs_rawpy
    def test_a_raw_and_a_jpeg_with_the_same_name_both_survive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "inbox"
            folder.mkdir()
            Image.new("RGB", (20, 20), (40, 40, 40)).save(folder / "bird.jpg")
            write_dng(folder / "bird.dng")
            output = Path(tmp) / "done"
            code = batch_main(
                [str(folder), "--output-dir", str(output), "--detector", "heuristic", "--advisor", "histogram"]
            )
            written = sorted(path.parent.name for path in output.glob("*/edited.tif"))
        self.assertEqual(code, 0)
        self.assertEqual(written, ["bird-dng", "bird-jpg"])


if __name__ == "__main__":
    unittest.main()
