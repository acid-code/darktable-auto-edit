"""Tests for the analyzer. They do not need Darktable, YOLO, or Ollama."""

from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from ai_dt.advisors.combine import combine_atmospheres
from ai_dt.cli import main
from ai_dt.detectors.yolo import category_for
from ai_dt.mapping import (
    DEFAULT_BLENDOP,
    colorbalance_params,
    exposure_params,
    vignette_params,
)
from ai_dt.pipeline import analyze, write_outputs
from ai_dt.schema import Atmosphere


def _save(image: Image.Image, directory: Path, name: str) -> Path:
    path = directory / name
    image.save(path)
    return path


def _portrait() -> Image.Image:
    image = Image.new("RGB", (800, 600), (28, 28, 32))
    draw = ImageDraw.Draw(image)
    draw.ellipse((250, 80, 560, 480), fill=(214, 162, 127))
    draw.ellipse((330, 200, 370, 240), fill=(25, 18, 18))
    draw.ellipse((450, 200, 490, 240), fill=(25, 18, 18))
    return image


def _architecture() -> Image.Image:
    image = Image.new("RGB", (640, 480), (186, 184, 176))
    draw = ImageDraw.Draw(image)
    for x in range(16, 300, 14):
        draw.line((x, 24, x, 450), fill=(18, 18, 18), width=4)
    for y in range(24, 450, 14):
        draw.line((16, y, 300, y), fill=(18, 18, 18), width=4)
    return image


class BlobTests(unittest.TestCase):
    def test_parameter_blobs_match_darktable_sizes(self) -> None:
        exposure = exposure_params(0.7, 0.0)
        self.assertEqual(len(exposure), 24)
        mode, black, value, percentile, target, bias = struct.unpack("<iffffi", exposure)
        self.assertEqual(mode, 0)
        self.assertAlmostEqual(value, 0.7)
        self.assertEqual(percentile, 50.0)
        self.assertEqual(target, -4.0)
        self.assertEqual(bias, 0)
        self.assertAlmostEqual(black, 0.0)

        color = colorbalance_params({"contrast": 0.2, "global_C": 0.1, "global_H": 40})
        self.assertEqual(len(color), 132)
        floats = struct.unpack("<32f", color[:128])
        formula = struct.unpack("<i", color[128:])[0]
        self.assertAlmostEqual(floats[10], 0.1)  # global chroma
        self.assertAlmostEqual(floats[11], 40.0)  # global hue
        self.assertAlmostEqual(floats[31], 0.2)  # contrast
        self.assertEqual(formula, 1)
        self.assertAlmostEqual(floats[12], 1.0)  # shadows weight default
        self.assertAlmostEqual(floats[28], 0.1845, places=4)

        vignette = vignette_params(
            {
                "scale": 30,
                "falloff_scale": 24,
                "brightness": -0.2,
                "saturation": 0,
                "center_x": -0.25,
                "center_y": 0.1,
                "autoratio": False,
                "whratio": 0.8,
                "shape": 1,
                "dithering": 0,
                "unbound": True,
            }
        )
        self.assertEqual(len(vignette), 44)
        unpacked = struct.unpack("<ffffffiffIi", vignette)
        self.assertAlmostEqual(unpacked[4], -0.25)
        self.assertAlmostEqual(unpacked[5], 0.1)

    def test_default_blend_blob_is_the_420_byte_darktable_value(self) -> None:
        raw = DEFAULT_BLENDOP[4:]
        padding = "=" * ((4 - len(raw) % 4) % 4)
        compressed = base64.b64decode(raw + padding)
        decoded = None
        for wbits in (15, 31, -15):
            try:
                decoded = zlib.decompress(compressed, wbits)
                break
            except zlib.error:
                continue
        self.assertIsNotNone(decoded)
        self.assertEqual(len(decoded), 420)


class AnalysisTests(unittest.TestCase):
    def test_dark_preview_gets_positive_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(Image.new("RGB", (240, 160), (30, 30, 30)), directory, "dark.png")
            recipe = self._run(path, directory)
        exposure = recipe["modules"][0]["params"]["exposure"]
        self.assertGreater(exposure, 0.8)
        self.assertEqual(recipe["modules"][0]["operation"], "exposure")
        self.assertEqual(recipe["modules"][1]["operation"], "colorbalancergb")
        self.assertLess(abs(recipe["analysis"]["atmosphere"]["temperature_shift"]), 0.08)
        self.assertGreater(recipe["analysis"]["atmosphere"]["contrast"], 0.1)

    def test_bright_preview_gets_negative_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(Image.new("RGB", (240, 160), (245, 245, 245)), directory, "bright.png")
            recipe = self._run(path, directory)
        self.assertLess(recipe["modules"][0]["params"]["exposure"], -0.4)
        self.assertLess(recipe["analysis"]["atmosphere"]["highlights"], 0)

    def test_a_flat_blue_field_is_not_repainted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(Image.new("RGB", (240, 160), (70, 110, 190)), directory, "blue.png")
            recipe = self._run(path, directory)
        # A saturated blue field is the picture, not a white-balance error.
        self.assertLess(abs(recipe["analysis"]["atmosphere"]["temperature_shift"]), 0.08)
        self.assertLess(abs(recipe["analysis"]["atmosphere"]["tint"]), 0.08)

    def test_a_mild_blue_cast_on_gray_is_warmed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(Image.new("RGB", (240, 160), (150, 156, 188)), directory, "cast.png")
            recipe = self._run(path, directory)
        self.assertGreater(recipe["analysis"]["atmosphere"]["temperature_shift"], 0.08)
        self.assertLess(recipe["analysis"]["atmosphere"]["temperature_shift"], 0.28)
        self.assertGreater(recipe["modules"][1]["params"]["global_C"], 0)

    def test_brown_red_leaves_keep_their_red(self) -> None:
        image = Image.new("RGB", (240, 160), (72, 132, 48))
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 50, 200, 120), fill=(168, 96, 52))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(image, directory, "earth.png")
            recipe = analyze(path, detector="heuristic", advisor="histogram", output_dir=directory)
        rendered = np.asarray(recipe["_rendered"])
        before = np.asarray(image.crop((40, 50, 200, 120))).astype(np.float32)
        after = rendered[50:120, 40:200] * 255.0
        before_red = float((before[..., 0] - before[..., 1]).mean())
        after_red = float((after[..., 0] - after[..., 1]).mean())
        self.assertGreater(after_red, before_red * 0.85)
        self.assertGreaterEqual(recipe["analysis"]["atmosphere"]["temperature_shift"], -0.02)

    def test_a_dark_green_picture_keeps_its_color(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(Image.new("RGB", (240, 160), (18, 42, 16)), directory, "moss.png")
            recipe = analyze(path, detector="heuristic", advisor="histogram", output_dir=directory)
        atmosphere = recipe["analysis"]["atmosphere"]
        self.assertGreaterEqual(atmosphere["tint"], -0.02)
        self.assertLess(abs(atmosphere["temperature_shift"]), 0.08)
        self.assertLessEqual(atmosphere["exposure_ev"], 0.5)

    def test_portrait_finds_a_face_and_writes_a_vignette(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(_portrait(), directory, "portrait.png")
            recipe = self._run(path, directory)
            style = (directory / "edit.dtstyle").read_text(encoding="utf-8")
        categories = {item["category"] for item in recipe["analysis"]["subjects"]}
        self.assertIn("face", categories)
        self.assertIn("eye", categories)
        self.assertLess(recipe["modules"][0]["params"]["exposure"], 1.0)
        self.assertGreater(recipe["analysis"]["atmosphere"]["temperature_shift"], -0.35)
        operations = [item["operation"] for item in recipe["modules"]]
        self.assertIn("vignette", operations)
        self.assertIn("<operation>exposure</operation>", style)
        self.assertIn("AI exposure", style)
        root = ET.fromstring(style)
        self.assertEqual(root.findtext("./info/name"), "AI Edit test")
        exposure = next(
            node for node in root.findall("./style/plugin") if node.findtext("operation") == "exposure"
        )
        raw = bytes.fromhex(exposure.findtext("op_params") or "")
        decoded = struct.unpack("<iffffi", raw)[2]
        self.assertAlmostEqual(decoded, recipe["modules"][0]["params"]["exposure"], places=3)
        for edit in recipe["local_edits"]:
            self.assertIn("ellipse", edit)
            self.assertEqual(len(edit["bbox"]), 4)

    def test_line_grid_is_read_as_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(_architecture(), directory, "building.png")
            recipe = self._run(path, directory)
        categories = {item["category"] for item in recipe["analysis"]["subjects"]}
        self.assertIn("architecture", categories)
        self.assertNotIn("face", categories)

    def test_tiny_image_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(Image.new("RGB", (1, 1), (20, 20, 20)), directory, "tiny.png")
            recipe = self._run(path, directory)
        self.assertEqual(recipe["version"], 1)
        self.assertEqual(recipe["width"], 1)

    def test_auto_mode_falls_back_without_optional_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(Image.new("RGB", (80, 60), (40, 40, 40)), directory, "plain.png")
            recipe = analyze(
                path,
                detector="auto",
                advisor="auto",
                output_dir=directory,
                style_name="AI Edit auto",
            )
            write_outputs(recipe, directory)
        self.assertIn("histogram", recipe["backends"]["advisor"])
        if importlib.util.find_spec("ultralytics") is None:
            self.assertTrue(
                any("ultralytics" in note for note in recipe["backends"]["detector_notes"])
            )

    def test_command_line_accepts_a_path_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "work dir"
            directory.mkdir()
            path = _save(Image.new("RGB", (32, 32), (25, 25, 25)), directory, "my photo.png")
            code = main(
                [
                    str(path),
                    "--output-dir",
                    str(directory),
                    "--detector",
                    "heuristic",
                    "--advisor",
                    "histogram",
                    "--style-name",
                    "AI Edit cli",
                ]
            )
            self.assertEqual(code, 0)
            payload = json.loads((directory / "output.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["style"]["name"], "AI Edit cli")
        self.assertGreater(payload["analysis"]["atmosphere"]["exposure_ev"], 0.5)

    def test_missing_image_returns_a_usage_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main([str(Path(tmp) / "missing.png"), "--output-dir", tmp])
        self.assertEqual(code, 2)

    def test_explicit_yolo_without_the_package_explains_why(self) -> None:
        if importlib.util.find_spec("ultralytics") is not None:
            self.skipTest("ultralytics is installed")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(Image.new("RGB", (16, 16), (10, 10, 10)), directory, "x.png")
            code = main(
                [
                    str(path),
                    "--output-dir",
                    str(directory),
                    "--detector",
                    "yolo",
                    "--advisor",
                    "histogram",
                ]
            )
        self.assertEqual(code, 3)

    def _run(self, path: Path, directory: Path) -> dict:
        recipe = analyze(
            path,
            detector="heuristic",
            advisor="histogram",
            style_name="AI Edit test",
            output_dir=directory,
        )
        write_outputs(recipe, directory)
        return recipe


class CombineTests(unittest.TestCase):
    def test_a_wild_model_suggestion_cannot_override_the_measurement(self) -> None:
        measured = Atmosphere(exposure_ev=0.2, contrast=0.0, mood="measured", rationale="pixels", source="histogram")
        suggestion = Atmosphere(exposure_ev=4.0, contrast=1.0, mood="dramatic", rationale="model", source="ollama:llava")
        # The advisor clamps the model before this function sees it. Simulate that clamp.
        suggestion.exposure_ev = 1.5
        suggestion.contrast = 0.4
        combined = combine_atmospheres(measured, suggestion, suggestion_weight=0.35)
        self.assertAlmostEqual(combined.exposure_ev, 0.65 * 0.2 + 0.35 * 1.5)
        self.assertLess(combined.contrast, 0.2)
        self.assertEqual(combined.mood, "dramatic")
        self.assertIn("pixels", combined.rationale)
        self.assertIn("model", combined.rationale)

    def test_category_map_ignores_unrelated_coco_labels(self) -> None:
        self.assertEqual(category_for("person"), "person")
        self.assertEqual(category_for("dog"), "animal")
        self.assertEqual(category_for("Left Eye"), "eye")
        self.assertIsNone(category_for("chair"))


class LuaJsonTests(unittest.TestCase):
    def test_lua_parser_reads_a_recipe(self) -> None:
        lua = shutil.which("lua5.4") or shutil.which("lua") or shutil.which("luajit")
        if lua is None:
            self.skipTest("lua is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _save(_portrait(), directory, "portrait.png")
            recipe = analyze(
                path,
                detector="heuristic",
                advisor="histogram",
                style_name="AI Edit lua",
                output_dir=directory,
            )
            json_path, _, _ = write_outputs(recipe, directory)
            result = subprocess.run(
                [lua, str(ROOT / "tests" / "test_json.lua"), str(ROOT / "lib" / "json.lua"), str(json_path)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
