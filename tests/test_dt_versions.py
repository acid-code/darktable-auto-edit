"""darktable 5.4 and 5.6: the same params, but 5.6 adds colorharmonizer to the order."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from unittest import mock

from ai_dt.drawn_mask import _write_pipeline_list
from ai_dt.iop_order import ORDERS, darktable_version, order_for

FIXTURE = Path(__file__).parent / "fixtures" / "dt562_jpeg.xmp"
LUA = (Path(__file__).resolve().parents[1] / "ai_editor.lua").read_text(encoding="utf-8")


def _names(document: str) -> list[str]:
    listed = re.search(r'darktable:iop_order_list="([^"]*)"', document).group(1).split(",")
    return listed[0::2]


class VersionParseTests(unittest.TestCase):
    def test_parse(self) -> None:
        self.assertEqual(darktable_version("5.4.1"), (5, 4))
        self.assertEqual(darktable_version("darktable 5.6.2+12~gabc"), (5, 6))
        self.assertEqual(darktable_version(""), (5, 6))

    def test_environment_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AI_DT_DARKTABLE_VERSION", None)
            self.assertEqual(darktable_version(), (5, 6))
        with mock.patch.dict(os.environ, {"AI_DT_DARKTABLE_VERSION": "5.4.1"}):
            self.assertEqual(darktable_version(), (5, 4))


class OrderTests(unittest.TestCase):
    def test_5_4_has_no_colorharmonizer_and_is_otherwise_the_same(self) -> None:
        for version, order in ORDERS.items():
            old = order_for(version, (5, 4))
            self.assertNotIn("colorharmonizer", old)
            self.assertEqual(len(old), len(order) - 1)
            self.assertEqual(list(old), [name for name in order if name != "colorharmonizer"])
            self.assertEqual(order_for(version, (5, 6)), order)

    def test_written_list_follows_the_running_version(self) -> None:
        base = FIXTURE.read_text(encoding="utf-8")
        with mock.patch.dict(os.environ, {"AI_DT_DARKTABLE_VERSION": "5.4.1"}):
            old = _names(_write_pipeline_list(base, "colorbalancergb", 1))
        with mock.patch.dict(os.environ, {"AI_DT_DARKTABLE_VERSION": "5.6.2"}):
            new = _names(_write_pipeline_list(base, "colorbalancergb", 1))
        self.assertNotIn("colorharmonizer", old)
        self.assertIn("colorharmonizer", new)
        self.assertEqual(old.count("colorbalancergb"), 2)
        self.assertEqual(len(new), len(old) + 1)

    def test_cli_sets_the_version(self) -> None:
        from ai_dt.cli import build_parser

        args = build_parser().parse_args(["x.jpg", "--output-dir", "out", "--dt-version", "5.4.1"])
        self.assertEqual(args.dt_version, "5.4.1")

    def test_lua_passes_the_version_to_every_processor_call(self) -> None:
        self.assertIn("dt.configuration.version", LUA)
        self.assertEqual(LUA.count('join_path(script_directory(), "ai_processor.py")'), 3)
        self.assertEqual(LUA.count("dt_version_arg(),"), 3)


if __name__ == "__main__":
    unittest.main()
