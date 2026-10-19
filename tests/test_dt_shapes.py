"""Path, gradient and group shapes for darktable 5.6 sidecars. No darktable needed.

The byte layouts follow ``src/develop/masks.h`` of darktable 5.6.0. Every
record written here is decoded again and compared with what went in, and
the combined mask is rendered with the same fold darktable's group uses.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dt import dt_shapes as shapes
from ai_dt.drawn_mask import (
    MaskSpliceError,
    decode_blend,
    read_mask_records,
    splice_shapes,
)
from ai_dt.mapping import colorbalance_params, exposure_params
from ai_dt.revert import pop_history_level, strip_ai_history
from tests.test_drawn_mask import PHOTO_COLOR, PHOTO_EXPOSURE, _document


def _disk(height: int, width: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return ((((xx + 0.5) / width - cx) / rx) ** 2 + (((yy + 0.5) / height - cy) / ry) ** 2 <= 1.0).astype(np.float32)


class RecordLayoutTests(unittest.TestCase):
    def test_struct_sizes_match_darktable(self) -> None:
        self.assertEqual(shapes.POINT_SIZES[shapes.ELLIPSE], 28)
        self.assertEqual(shapes.POINT_SIZES[shapes.PATH], 36)
        self.assertEqual(shapes.POINT_SIZES[shapes.GRADIENT], 28)
        self.assertEqual(shapes.POINT_SIZES[shapes.GROUP], 16)
        self.assertEqual((shapes.PATH, shapes.GROUP, shapes.GRADIENT, shapes.ELLIPSE), (2, 4, 16, 32))

    def test_path_round_trip(self) -> None:
        points = shapes.smooth_path([(0.2, 0.2), (0.8, 0.25), (0.7, 0.8), (0.3, 0.7)], border=0.02)
        blob = shapes.encode_path(points)
        self.assertEqual(len(blob), 4 * 36)
        decoded = shapes.decode_path(blob)
        for before, after in zip(points, decoded):
            self.assertAlmostEqual(before.x, after.x, places=6)
            self.assertAlmostEqual(before.ctrl2[1], after.ctrl2[1], places=6)
            self.assertAlmostEqual(after.border[0], 0.02, places=6)
            self.assertEqual(after.state, shapes.POINT_STATE_USER)
        # The first float pair is the corner, as in dt_masks_point_path_t.
        self.assertAlmostEqual(struct.unpack_from("<ff", blob, 0)[0], 0.2, places=6)

    def test_catmull_rom_handles_match_darktable(self) -> None:
        corners = [(0.2, 0.2), (0.8, 0.25), (0.7, 0.8), (0.3, 0.7)]
        points = shapes.smooth_path(corners)
        # darktable: ctrl1 = (p[i-1] + 6 p[i] - p[i+1]) / 6, ctrl2 = (-p[i-1] + 6 p[i] + p[i+1]) / 6
        prev, here, nxt = corners[0], corners[1], corners[2]
        self.assertAlmostEqual(points[1].ctrl1[0], (prev[0] + 6 * here[0] - nxt[0]) / 6, places=9)
        self.assertAlmostEqual(points[1].ctrl2[1], (-prev[1] + 6 * here[1] + nxt[1]) / 6, places=9)

    def test_corner_on_the_frame_edge_keeps_sharp_handles(self) -> None:
        points = shapes.smooth_path([(0.0, 0.0), (1.0, 0.0), (1.0, 0.4), (0.0, 0.4)])
        for point in points:
            self.assertEqual(point.ctrl1, (point.x, point.y))
            self.assertEqual(point.ctrl2, (point.x, point.y))

    def test_gradient_and_ellipse_round_trip(self) -> None:
        blob = shapes.encode_gradient(0.5, 0.3, 180.0, compression=0.2, curvature=0.1)
        self.assertEqual(len(blob), 28)
        decoded = shapes.decode_gradient(blob)
        self.assertAlmostEqual(decoded["anchor"][1], 0.3, places=6)
        self.assertAlmostEqual(decoded["rotation"], 180.0)
        self.assertEqual(decoded["state"], shapes.GRADIENT_SIGMOIDAL)
        ellipse = shapes.decode_ellipse(shapes.encode_ellipse(0.5, 0.4, 0.2, 0.1, border=0.05))
        self.assertAlmostEqual(ellipse["rx"], 0.2, places=6)
        self.assertEqual(ellipse["flags"], shapes.ELLIPSE_PROPORTIONAL)

    def test_group_states(self) -> None:
        self.assertEqual(shapes.member_state("union", first=True), 3)
        self.assertEqual(shapes.member_state("union"), 11)
        self.assertEqual(shapes.member_state("intersect"), 19)
        self.assertEqual(shapes.member_state("difference"), 35)
        self.assertEqual(shapes.member_state("union", first=True, invert=True), 7)
        members = shapes.decode_group(shapes.encode_group(9, [(5, 3, 1.0), (6, 19, 0.5)]))
        self.assertEqual([m["form_id"] for m in members], [5, 6])
        self.assertEqual({m["parent_id"] for m in members}, {9})
        self.assertEqual(members[1]["mode"], "intersect")
        self.assertAlmostEqual(members[1]["opacity"], 0.5)

    def test_bad_lengths_are_refused(self) -> None:
        with self.assertRaises(shapes.ShapeError):
            shapes.decode_path(b"\x00" * 35)
        with self.assertRaises(shapes.ShapeError):
            shapes.decode_gradient(b"\x00" * 27)
        with self.assertRaises(shapes.ShapeError):
            shapes.member_state("blend")


class GradientTests(unittest.TestCase):
    def test_each_edge_is_on_at_its_edge(self) -> None:
        cases = {
            "top": ((50, 3), (50, 90)),
            "bottom": ((50, 96), (50, 10)),
            "left": ((3, 50), (90, 50)),
            "right": ((96, 50), (10, 50)),
        }
        for edge, ((x_on, y_on), (x_off, y_off)) in cases.items():
            values = shapes.gradient_values(shapes.edge_gradient(edge, 0.3, compression=0.1), 100, 100)
            self.assertGreater(values[y_on, x_on], 0.8, edge)
            self.assertLess(values[y_off, x_off], 0.05, edge)

    def test_gradient_is_half_on_its_line(self) -> None:
        values = shapes.gradient_values(shapes.edge_gradient("bottom", 0.25), 80, 200)
        self.assertAlmostEqual(float(values[150, 40]), 0.5, delta=0.03)


class OutlineTests(unittest.TestCase):
    def test_disk_round_trip(self) -> None:
        mask = _disk(160, 240, 0.5, 0.5, 0.3, 0.35)
        forms = shapes.mask_to_forms(mask)
        self.assertEqual(len(forms), 1)
        self.assertLessEqual(len(forms[0]["points"]), 48)
        rendered = shapes.render_forms(forms, 240, 160)
        self.assertGreater(shapes.iou(rendered, mask), 0.95)

    def test_ring_has_a_hole(self) -> None:
        mask = _disk(200, 200, 0.5, 0.5, 0.4, 0.4) - _disk(200, 200, 0.5, 0.5, 0.15, 0.15)
        forms = shapes.mask_to_forms(mask)
        self.assertEqual([form["mode"] for form in forms], ["union", "difference"])
        rendered = shapes.render_forms(forms, 200, 200)
        self.assertGreater(shapes.iou(rendered, mask), 0.9)
        self.assertLess(float(rendered[100, 100]), 0.5)

    def test_two_blobs_are_two_paths_largest_first(self) -> None:
        mask = np.maximum(_disk(120, 300, 0.2, 0.5, 0.1, 0.2), _disk(120, 300, 0.7, 0.5, 0.2, 0.3))
        outlines = shapes.mask_to_outlines(mask)
        self.assertEqual(len(outlines), 2)
        self.assertGreater(outlines[0].area, outlines[1].area)

    def test_specks_and_max_paths(self) -> None:
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[10:12, 10:12] = 1.0
        self.assertEqual(shapes.mask_to_outlines(mask), [])
        for index in range(8):
            mask[20 + index * 22 : 36 + index * 22, 50:150] = 1.0
        self.assertEqual(len(shapes.mask_to_outlines(mask, max_paths=3)), 3)

    def test_sky_reaches_the_frame_edges(self) -> None:
        mask = np.zeros((100, 150), dtype=np.float32)
        mask[:40] = 1.0
        points = shapes.mask_to_outlines(mask)[0].points
        xs = {round(x, 6) for x, _y in points}
        ys = {round(y, 6) for _x, y in points}
        self.assertIn(0.0, xs)
        self.assertIn(1.0, xs)
        self.assertIn(0.0, ys)

    def test_point_budget_is_kept_on_a_ragged_outline(self) -> None:
        rng = np.random.default_rng(3)
        mask = _disk(256, 256, 0.5, 0.5, 0.35, 0.35)
        mask[rng.random(mask.shape) < 0.05] = 0.0
        mask = np.maximum(mask, _disk(256, 256, 0.5, 0.5, 0.3, 0.3))
        for outline in shapes.mask_to_outlines(mask, max_points=20, holes=False):
            self.assertLessEqual(len(outline.points), 20)

    def test_large_masks_are_downsampled(self) -> None:
        mask = _disk(1200, 1600, 0.4, 0.5, 0.2, 0.3)
        forms = shapes.mask_to_forms(mask, working_size=200)
        rendered = shapes.render_forms(forms, 400, 300)
        target = _disk(300, 400, 0.4, 0.5, 0.2, 0.3)
        self.assertGreater(shapes.iou(rendered, target), 0.93)

    def test_trace_is_clockwise_and_single_pixel_safe(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 3:8] = True
        traced = shapes.trace_boundary(mask)
        self.assertGreater(shapes.polygon_area([(float(x), float(y)) for x, y in traced]), 0.0)
        single = np.zeros((5, 5), dtype=bool)
        single[2, 2] = True
        self.assertEqual(shapes.trace_boundary(single), [(2, 2)])


class CombineTests(unittest.TestCase):
    def test_top_gradient_intersected_with_sky(self) -> None:
        sky = np.zeros((100, 100), dtype=np.float32)
        sky[:45, :60] = 1.0
        forms = shapes.mask_to_forms(sky)
        forms.append({**shapes.edge_gradient("top", 0.3, compression=0.05), "mode": "intersect"})
        rendered = shapes.render_forms(forms, 100, 100)
        self.assertGreater(rendered[5, 20], 0.8)
        self.assertEqual(rendered[5, 80], 0.0)  # top, but not sky
        self.assertLess(rendered[42, 20], 0.2)  # sky, past the gradient

    def test_inverse_member_and_inverted_group(self) -> None:
        subject = {"type": "ellipse", "cx": 0.5, "cy": 0.5, "rx": 0.2, "ry": 0.2}
        background = shapes.render_forms([{**subject, "invert": True}], 50, 50)
        self.assertEqual(background[25, 25], 0.0)
        self.assertEqual(background[2, 2], 1.0)
        flipped = shapes.render_forms([subject], 50, 50, invert=True)
        self.assertTrue(np.array_equal(background, flipped))


class SpliceShapesTests(unittest.TestCase):
    def _sky_forms(self) -> list[dict]:
        sky = np.zeros((90, 120), dtype=np.float32)
        sky[:35] = 1.0
        forms = shapes.mask_to_forms(sky)
        forms.append({**shapes.edge_gradient("top", 0.3), "mode": "intersect"})
        return forms

    def test_group_of_path_and_gradient(self) -> None:
        original = _document()
        forms = self._sky_forms()
        result = splice_shapes(
            original,
            forms=forms,
            operation="exposure",
            params=exposure_params(-0.4),
            modversion=6,
            name="AI sky",
        )
        self.assertIn(PHOTO_EXPOSURE, result)
        self.assertIn(PHOTO_COLOR, result)
        self.assertIn('darktable:history_end="3"', result)
        records = [r for r in read_mask_records(result) if r["name"] == "AI sky"]
        self.assertEqual([r["type"] for r in records], [shapes.PATH, shapes.GRADIENT, shapes.GROUP])
        self.assertEqual({r["num"] for r in records}, {2})
        self.assertEqual({r["version"] for r in records}, {6})
        path, gradient, group = records
        self.assertEqual(path["nb"], len(forms[0]["points"]))
        for written, point in zip(forms[0]["points"], path["points"]):
            self.assertAlmostEqual(point.x, written[0], places=6)
            self.assertAlmostEqual(point.y, written[1], places=6)
        self.assertAlmostEqual(gradient["points"]["anchor"][1], 0.3, places=6)
        self.assertEqual([m["form_id"] for m in group["points"]], [path["id"], gradient["id"]])
        self.assertEqual([m["state"] for m in group["points"]], [3, 19])
        tag_start = result.find('darktable:multi_name="AI sky"')
        open_at = result.rfind("<rdf:li", 0, tag_start)
        tag = result[open_at : result.find("/>", tag_start)]
        blend = decode_blend(tag.split('darktable:blendop_params="')[1].split('"')[0])
        self.assertEqual(blend["length"], 420)
        self.assertEqual(blend["mask_mode"], 3)
        self.assertEqual(blend["mask_id"], group["id"])
        self.assertFalse(blend["inverted"])
        self.assertIn('darktable:multi_priority="1"', tag)

    def test_inverted_background_on_color_balance(self) -> None:
        forms = shapes.mask_to_forms(_disk(80, 80, 0.5, 0.5, 0.25, 0.3))
        result = splice_shapes(
            _document(),
            forms=forms,
            operation="colorbalancergb",
            params=colorbalance_params({"saturation_global": -0.2}),
            modversion=5,
            name="AI background",
            invert=True,
        )
        tag_start = result.find('darktable:multi_name="AI background"')
        open_at = result.rfind("<rdf:li", 0, tag_start)
        tag = result[open_at : result.find("/>", tag_start)]
        blend = decode_blend(tag.split('darktable:blendop_params="')[1].split('"')[0])
        self.assertTrue(blend["inverted"])
        self.assertIn('darktable:operation="colorbalancergb"', tag)
        # colorbalancergb already has a row, so the new instance takes the next priority.
        self.assertIn('darktable:multi_priority="1"', tag)

    def test_refusals_leave_nothing(self) -> None:
        forms = self._sky_forms()
        with self.assertRaises(MaskSpliceError):
            splice_shapes(_document(), forms=forms, operation="exposure", params=b"\x00" * 23, modversion=6, name="AI sky")
        with self.assertRaises(MaskSpliceError):
            splice_shapes(_document(), forms=forms, operation="filmicrgb", params=b"", modversion=1, name="AI sky")
        with self.assertRaises(MaskSpliceError):
            splice_shapes(_document(), forms=forms, operation="exposure", params=exposure_params(0.1), modversion=6, name="sky")
        with self.assertRaises(MaskSpliceError):
            splice_shapes(_document(), forms=[], operation="exposure", params=exposure_params(0.1), modversion=6, name="AI sky")
        with self.assertRaises(MaskSpliceError):
            splice_shapes(
                _document(),
                forms=[{"type": "path", "points": [[0.1, 0.1], [1.4, 0.2], [0.5, 0.5]]}],
                operation="exposure",
                params=exposure_params(0.1),
                modversion=6,
                name="AI sky",
            )

    def test_revert_and_back_remove_every_shape(self) -> None:
        result = splice_shapes(
            _document(),
            forms=self._sky_forms(),
            operation="exposure",
            params=exposure_params(-0.4),
            modversion=6,
            name="AI sky",
        )
        stripped, removed = strip_ai_history(result)
        self.assertEqual(removed, 1)
        self.assertNotIn("AI sky", stripped)
        self.assertIn('mask_name="window"', stripped)
        popped, entry, count = pop_history_level(result)
        self.assertEqual(count, 1)
        # Three sky shapes plus the photographer's window, carried as
        # darktable's per-row snapshot.
        self.assertEqual(len(entry["masks"]), 4)
        self.assertNotIn("AI sky", popped)
        self.assertEqual(popped.count('mask_name="window"'), 1)
        self.assertEqual(stripped.count('mask_name="window"'), 1)

    def test_second_step_gets_fresh_ids(self) -> None:
        once = splice_shapes(
            _document(), forms=self._sky_forms(), operation="exposure",
            params=exposure_params(-0.4), modversion=6, name="AI sky",
        )
        twice = splice_shapes(
            once, forms=[shapes.edge_gradient("bottom", 0.3)], operation="exposure",
            params=exposure_params(-0.3), modversion=6, name="AI ground",
        )
        top = [r for r in read_mask_records(twice) if r["num"] == 3]
        ids = [r["id"] for r in top]
        self.assertEqual(len(ids), len(set(ids)))
        sky_ids = {r["id"] for r in read_mask_records(once) if r["name"] == "AI sky"}
        ground_ids = {r["id"] for r in top if r["name"] == "AI ground"}
        self.assertFalse(sky_ids & ground_ids)
        ground = [r for r in read_mask_records(twice) if r["name"] == "AI ground"]
        self.assertEqual({r["num"] for r in ground}, {3})
        self.assertIn('darktable:history_end="4"', twice)

    def test_script_by_path_reads_a_shapes_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / "photo.xmp"
            source.write_text(_document(), encoding="utf-8")
            spec = {
                "kind": "shapes",
                "name": "AI sky",
                "operation": "exposure",
                "modversion": 6,
                "params": exposure_params(-0.3).hex(),
                "forms": self._sky_forms(),
            }
            spec_path = directory / "mask.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            out = directory / "masked.xmp"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "ai_dt" / "drawn_mask.py"), "--xmp", str(source),
                 "--xmp-out", str(out), "--mask-json", str(spec_path)],
                capture_output=True,
                text=True,
                cwd=str(directory),
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = read_mask_records(out.read_text(encoding="utf-8"))
            self.assertIn(shapes.GROUP, [r["type"] for r in records])


if __name__ == "__main__":
    unittest.main()
