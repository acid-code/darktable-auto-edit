"""Accepted steps chain the way darktable reads them. No darktable needed.

Reproduces the duck report: an accepted step must be the active (highlighted)
top of the history, ``history_end`` must equal the number of rows after every
step, and each step's mask must still be found once later masked steps are
stacked on top. darktable keeps the full shape list on every row that changed
masks and uses the last such row below ``history_end``
(``dt_masks_read_masks_history`` in darktable 5.6), which ``_current_forms``
below mirrors.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dt import dt_shapes as shapes
from ai_dt.drawn_mask import (
    MaskSpliceError,
    decode_blend,
    main,
    read_mask_records,
    splice_ellipse_exposure,
    splice_path_exposure,
    splice_region_module,
    splice_shapes,
    verify_top,
)
from ai_dt.mapping import exposure_params
from ai_dt.revert import _HISTORY_ITEM, _attr, pop_history_level, strip_ai_history
from ai_dt.modules_extra import colorzones_hue_params
from tests.test_drawn_mask import _document

_ZONES = colorzones_hue_params("red", center=0.075)

OUTLINE = [
    {"x": 0.35, "y": 0.30},
    {"x": 0.62, "y": 0.28},
    {"x": 0.70, "y": 0.55},
    {"x": 0.55, "y": 0.75},
    {"x": 0.38, "y": 0.70},
    {"x": 0.30, "y": 0.50},
]


def _rows(document: str) -> list[str]:
    return [m.group(0) for m in _HISTORY_ITEM.finditer(document) if _attr(m.group(0), "darktable:operation")]


def _history_end(document: str) -> int:
    return int(re.search(r'darktable:history_end="(\d+)"', document).group(1))


def _current_forms(document: str) -> set[int]:
    """Form ids darktable loads: the snapshot of the last masked row below history_end."""

    end = _history_end(document)
    active = {int(_attr(tag, "darktable:num")) for tag in _rows(document)[:end]}
    records = read_mask_records(document)
    nums = sorted({r["num"] for r in records if r["num"] in active})
    if not nums:
        return set()
    return {r["id"] for r in records if r["num"] == nums[-1]}


def _blend_id(row: str) -> int:
    return decode_blend(_attr(row, "darktable:blendop_params"))["mask_id"]


def _subject(document: str) -> str:
    return splice_path_exposure(document, points=OUTLINE, exposure_ev=0.3, name="AI subject")


def _background(document: str) -> str:
    background = np.ones((90, 120), dtype=np.float32)
    background[25:70, 40:85] = 0.0
    return splice_shapes(
        document,
        forms=shapes.mask_to_forms(background),
        operation="exposure",
        params=exposure_params(-0.4),
        modversion=6,
        name="AI background",
    )


def _foreground(document: str) -> str:
    return splice_region_module(
        document,
        shapes=[{"cx": 0.5, "cy": 0.85, "rx": 0.3, "ry": 0.1}, {"cx": 0.2, "cy": 0.9, "rx": 0.1, "ry": 0.05}],
        params=_ZONES,
        operation="colorzones",
        modversion=5,
        name="AI foreground",
    )


class ChainedStepTests(unittest.TestCase):
    def _assert_top(self, document: str, name: str, rows: int) -> None:
        history = _rows(document)
        self.assertEqual(len(history), rows)
        self.assertEqual(_history_end(document), rows, "history_end must equal the row count")
        self.assertEqual(_attr(history[-1], "darktable:multi_name"), name, "the accepted step is the top row")
        verify_top(document, name)

    def _assert_every_mask_found(self, document: str) -> None:
        current = _current_forms(document)
        for row in _rows(document)[: _history_end(document)]:
            raw = _attr(row, "darktable:blendop_params")
            if not raw.startswith("gz") or not _attr(row, "darktable:multi_name").startswith("AI "):
                continue
            self.assertIn(_blend_id(row), current, f"{_attr(row, 'darktable:multi_name')} lost its mask")

    def test_three_masked_steps_each_become_the_active_top(self) -> None:
        start = _document()
        self.assertEqual(_history_end(start), 2)
        one = _subject(start)
        self._assert_top(one, "AI subject", 3)
        self._assert_every_mask_found(one)
        two = _background(one)
        self._assert_top(two, "AI background", 4)
        self._assert_every_mask_found(two)
        three = _foreground(two)
        self._assert_top(three, "AI foreground", 5)
        self._assert_every_mask_found(three)
        # The photographer's own shape is still current too.
        self.assertIn(7, _current_forms(three))

    def test_eyes_after_two_steps_keep_every_mask(self) -> None:
        document = _background(_subject(_document()))
        document = splice_ellipse_exposure(
            document, center_x=0.5, center_y=0.4, radius_x=0.05, radius_y=0.03, exposure_ev=0.2,
        )
        self._assert_top(document, "AI eyes", 5)
        self._assert_every_mask_found(document)

    def test_shapes_are_filed_under_their_own_row(self) -> None:
        # Before the fix the subject outline and the eye ellipse were filed
        # under "highest mask_num + 1", which named a different (often an
        # earlier) history row, not the new module.
        document = _subject(_document(masks=False))
        top = _rows(document)[-1]
        num = int(_attr(top, "darktable:num"))
        self.assertEqual({r["num"] for r in read_mask_records(document)}, {num})

    def test_a_stale_history_end_does_not_leave_rows_above_the_new_step(self) -> None:
        # The photographer clicked an earlier row in darktable's history,
        # then accepted the next step.
        document = _subject(_background(_document()))
        stale = re.sub(r'darktable:history_end="\d+"', 'darktable:history_end="3"', document, count=1)
        result = _foreground(stale)
        self._assert_top(result, "AI foreground", 4)
        self.assertNotIn("AI subject", result)
        self._assert_every_mask_found(result)

    def test_back_and_revert_after_a_chain(self) -> None:
        three = _foreground(_background(_subject(_document())))
        popped, _entry, count = pop_history_level(three)
        self.assertEqual(count, 1)
        self._assert_top(popped, "AI background", 4)
        self._assert_every_mask_found(popped)
        stripped, removed = strip_ai_history(three)
        self.assertEqual(removed, 3)
        self.assertEqual(_history_end(stripped), 2)
        self.assertEqual({r["id"] for r in read_mask_records(stripped)}, {7})

    def test_verify_top_refuses_an_inactive_or_misnamed_top(self) -> None:
        good = _subject(_document())
        lowered = re.sub(r'darktable:history_end="\d+"', 'darktable:history_end="2"', good, count=1)
        with self.assertRaisesRegex(MaskSpliceError, "active top"):
            verify_top(lowered, "AI subject")
        with self.assertRaisesRegex(MaskSpliceError, "top history row"):
            verify_top(good, "AI background")

    def test_the_script_chains_three_steps_through_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            xmp = base / "photo.xmp"
            xmp.write_text(_document(), encoding="utf-8")
            specs = [
                {"kind": "path", "points": OUTLINE, "exposure_ev": 0.3, "name": "AI subject"},
                {"kind": "shapes", "forms": [shapes.edge_gradient("top", 0.3)], "operation": "exposure",
                 "params": exposure_params(-0.3).hex(), "modversion": 6, "name": "AI sky"},
                {"kind": "regions", "shapes": [{"cx": 0.5, "cy": 0.8, "rx": 0.2, "ry": 0.1}],
                 "operation": "colorzones", "params": _ZONES.hex(), "modversion": 5,
                 "name": "AI foreground"},
            ]
            for index, spec in enumerate(specs, start=1):
                mask = base / f"mask{index}.json"
                mask.write_text(json.dumps(spec), encoding="utf-8")
                with redirect_stderr(StringIO()):
                    code = main(["--xmp", str(xmp), "--xmp-out", str(xmp), "--mask-json", str(mask)])
                self.assertEqual(code, 0)
                document = xmp.read_text(encoding="utf-8")
                self._assert_top(document, spec["name"], 2 + index)
                self._assert_every_mask_found(document)


if __name__ == "__main__":
    unittest.main()
