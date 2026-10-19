"""tools/verify_dt.py: the cases build without darktable; the render check
runs only where darktable-cli is installed (``$DTCLI`` or on PATH)."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "dt562_jpeg.xmp"
_spec = importlib.util.spec_from_file_location("verify_dt", ROOT / "tools" / "verify_dt.py")
verify_dt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_dt)
CLI = verify_dt.find_cli()


def _duck() -> Image.Image:
    pixels = np.zeros((300, 400, 3), dtype=np.uint8)
    pixels[:] = (60, 110, 70)
    pixels[250:] = (215, 210, 200)
    pixels[100:260, 210:290] = (90, 70, 50)
    return Image.fromarray(pixels)


class BuildTests(unittest.TestCase):
    def test_every_case_builds_and_the_stack_lists_its_second_instances(self) -> None:
        cases = verify_dt.build_cases(FIXTURE.read_text(encoding="utf-8"), _duck(), (0.52, 0.33, 0.21, 0.55), "duck", set())
        names = [case["name"] for case in cases]
        for name in ("mask_animal", "mask_animal_background", "mask_bottom_gradient", "mask_edge_burn", "mask_glow",
                     "group_intersect", "colorzones_region", "style_split_tone", "style_colorzones", "animal_stack"):
            self.assertIn(name, names)
        errors = [case for case in cases if case["xmp"] is None and "no region" not in case["detail"]]
        self.assertEqual(errors, [])
        stack = next(case for case in cases if case["name"] == "animal_stack")["xmp"]
        self.assertIn("darktable:iop_order_list=", stack)
        self.assertIn("colorbalancergb,0,colorbalancergb,1", stack)
        self.assertIn("exposure,0,exposure,1", stack)

    def test_log_filter(self) -> None:
        lines = ["     0.98 cannot get iop-order for colorbalancergb instance 1", "  0.1 dbind-WARNING: x", "fine"]
        self.assertEqual(verify_dt.warnings_in(lines), ["cannot get iop-order for colorbalancergb instance 1"])


@unittest.skipUnless(CLI, "darktable-cli is not installed (set $DTCLI to run the real render check)")
class DarktableTests(unittest.TestCase):
    def test_the_animal_stack_loads_without_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "duck.jpg"
            _duck().save(image, quality=92)
            code = verify_dt.main([
                str(image), "--base", str(FIXTURE), "--box", "0.52,0.33,0.21,0.55", "--out", str(Path(tmp) / "out"),
                "--only", "animal_stack,mask_animal_background", "--width", "400", "--darktable-cli", CLI,
            ])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
