"""Drawn-ellipse history splicing. These tests do not open Darktable.

A sidecar that already has the photographer's modules must keep those
items byte for byte. The added exposure and its ellipse are the only
new records, and stripping the AI name puts the stack back.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dt.drawn_mask import (
    MaskSpliceError,
    eye_ellipse,
    main,
    splice_ellipse_exposure,
    splice_path_exposure,
    splice_region_module,
)
from ai_dt.features import commit_waiting_step
from ai_dt.mapping import DEFAULT_BLENDOP
from ai_dt.revert import strip_ai_history


PHOTO_EXPOSURE = (
    '<rdf:li darktable:num="0" darktable:operation="exposure" '
    'darktable:multi_name="" darktable:params="aa" '
    'darktable:blendop_params="gzPHOTO" darktable:multi_priority="0"/>'
)
PHOTO_COLOR = (
    '<rdf:li darktable:num="1" darktable:operation="colorbalancergb" '
    'darktable:multi_name="portrait" darktable:params="bb" '
    'darktable:blendop_params="gzPORTRAIT"/>'
)
WINDOW = (
    '<rdf:li darktable:mask_num="1" darktable:mask_id="7" '
    'darktable:mask_name="window" darktable:mask_points="abcd"/>'
)


def _document(history_end: str = "2", masks: bool = True) -> str:
    mask_block = ""
    if masks:
        mask_block = (
            "   <darktable:masks_history>\n"
            "    <rdf:Seq>\n"
            f"     {WINDOW}\n"
            "    </rdf:Seq>\n"
            "   </darktable:masks_history>\n"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<x:xmpmeta>\n"
        " <rdf:RDF>\n"
        f'  <rdf:Description darktable:history_end="{history_end}">\n'
        "   <darktable:history>\n"
        "    <rdf:Seq>\n"
        f"     {PHOTO_EXPOSURE}\n"
        f"     {PHOTO_COLOR}\n"
        "    </rdf:Seq>\n"
        "   </darktable:history>\n"
        f"{mask_block}"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
    )


def _decode_gz(encoded: str) -> bytes:
    return zlib.decompress(base64.b64decode(encoded[4:]))


def _added_item(document: str) -> str:
    start = document.find('darktable:multi_name="AI eyes"')
    if start < 0:
        raise AssertionError("the AI eyes history item is missing")
    open_at = document.rfind("<rdf:li", 0, start)
    close_at = document.find("/>", start)
    return document[open_at : close_at + 2]


class SpliceTests(unittest.TestCase):
    def test_photographer_history_stays_byte_for_byte(self) -> None:
        original = _document()
        result = splice_ellipse_exposure(
            original,
            center_x=0.48,
            center_y=0.42,
            radius_x=0.06,
            radius_y=0.04,
            exposure_ev=0.2,
        )
        self.assertEqual(result.count(PHOTO_EXPOSURE), 1)
        self.assertEqual(result.count(PHOTO_COLOR), 1)
        self.assertEqual(result.count(WINDOW), 1)
        self.assertEqual(result.count("gzPHOTO"), 1)
        self.assertEqual(result.count("gzPORTRAIT"), 1)
        self.assertIn('darktable:history_end="3"', result)
        self.assertNotIn('darktable:history_end="2"', result)
        added = _added_item(result)
        self.assertIn('darktable:operation="exposure"', added)
        self.assertIn('darktable:num="2"', added)
        self.assertIn('darktable:multi_priority="1"', added)
        self.assertNotIn("colorbalancergb", added)

    def test_a_subject_outline_is_a_closed_path(self) -> None:
        points = [
            {"x": 0.30, "y": 0.20},
            {"x": 0.70, "y": 0.22},
            {"x": 0.68, "y": 0.75},
            {"x": 0.40, "y": 0.80},
            {"x": 0.28, "y": 0.50},
        ]
        result = splice_path_exposure(
            _document(),
            points=points,
            exposure_ev=0.22,
            name="Subject lift",
        )
        self.assertIn('darktable:mask_type="2"', result)
        self.assertIn('darktable:mask_nb="5"', result)
        self.assertIn('darktable:multi_name="Subject lift"', result)
        self.assertIn('darktable:history_end="3"', result)
        encoded = result.split('darktable:mask_type="2"', 1)[1].split('darktable:mask_points="', 1)[1].split('"', 1)[0]
        blob = bytes.fromhex(encoded)
        self.assertEqual(len(blob), 5 * 36)
        x, y = struct.unpack("<ff", blob[:8])
        self.assertAlmostEqual(x, 0.30)
        self.assertAlmostEqual(y, 0.20)

    def test_the_first_eye_uses_the_open_exposure_slot(self) -> None:
        document = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<x:xmpmeta>\n"
            " <rdf:RDF>\n"
            '  <rdf:Description darktable:history_end="1" darktable:iop_order_version="4">\n'
            "   <darktable:history>\n"
            "    <rdf:Seq>\n"
            '     <rdf:li darktable:num="0" darktable:operation="highlights" '
            'darktable:multi_priority="0"/>\n'
            "    </rdf:Seq>\n"
            "   </darktable:history>\n"
            "  </rdf:Description>\n"
            " </rdf:RDF>\n"
            "</x:xmpmeta>\n"
        )
        result = splice_ellipse_exposure(
            document,
            center_x=0.77,
            center_y=0.68,
            radius_x=0.04,
            radius_y=0.03,
            exposure_ev=0.2,
        )
        added = _added_item(result)
        self.assertIn('darktable:multi_priority="0"', added)
        self.assertIn('darktable:operation="exposure"', added)
        self.assertIn('darktable:history_end="2"', result)

    def test_new_blend_binds_the_ellipse_and_leaves_the_other_blend_bytes(self) -> None:
        result = splice_ellipse_exposure(
            _document(),
            center_x=0.48,
            center_y=0.42,
            radius_x=0.06,
            radius_y=0.04,
            exposure_ev=0.2,
        )
        added = _added_item(result)
        encoded = added.split('darktable:blendop_params="', 1)[1].split('"', 1)[0]
        raw = _decode_gz(encoded)
        base = _decode_gz(DEFAULT_BLENDOP)
        self.assertEqual(len(raw), 420)
        mode = struct.unpack_from("<I", raw, 0)[0]
        opacity = struct.unpack_from("<f", raw, 16)[0]
        mask_id = struct.unpack_from("<I", raw, 24)[0]
        self.assertEqual(mode, 3)
        self.assertEqual(opacity, 100.0)
        self.assertGreaterEqual(mask_id, 0x10000000)
        self.assertEqual(raw[4:16], base[4:16])
        self.assertEqual(raw[20:24], base[20:24])
        self.assertEqual(raw[28:], base[28:])
        self.assertIn(f'darktable:mask_id="{mask_id}"', result)
        self.assertIn('darktable:mask_name="AI eyes"', result)
        self.assertIn('darktable:mask_type="32"', result)
        self.assertIn('darktable:mask_version="6"', result)
        points = result.split('darktable:mask_name="AI eyes"', 1)[1]
        points = points.split('darktable:mask_points="', 1)[1].split('"', 1)[0]
        blob = bytes.fromhex(points)
        center_x, center_y, radius_x, radius_y, rotation, border, flags = struct.unpack(
            "<ffffffI", blob
        )
        self.assertAlmostEqual(center_x, 0.48, places=5)
        self.assertAlmostEqual(center_y, 0.42, places=5)
        self.assertAlmostEqual(radius_x, 0.06, places=5)
        self.assertAlmostEqual(radius_y, 0.04, places=5)
        self.assertEqual(rotation, 0.0)
        self.assertGreater(border, 0.0)
        self.assertEqual(flags, 1)
        self.assertIn(WINDOW, result)

    def test_a_sidecar_without_a_mask_table_gains_one_and_keeps_history(self) -> None:
        original = _document(masks=False)
        result = splice_ellipse_exposure(
            original,
            center_x=0.5,
            center_y=0.5,
            radius_x=0.1,
            radius_y=0.1,
            exposure_ev=0.1,
        )
        self.assertEqual(result.count(PHOTO_EXPOSURE), 1)
        self.assertEqual(result.count(PHOTO_COLOR), 1)
        self.assertEqual(result.count("<darktable:masks_history"), 1)
        self.assertIn('darktable:history_end="3"', result)

    def test_stripping_the_ai_step_restores_the_photographer_stack(self) -> None:
        original = _document()
        spliced = splice_ellipse_exposure(
            original,
            center_x=0.48,
            center_y=0.42,
            radius_x=0.06,
            radius_y=0.04,
            exposure_ev=0.2,
        )
        cleaned, removed = strip_ai_history(spliced)
        self.assertEqual(removed, 1)
        self.assertIn(PHOTO_EXPOSURE, cleaned)
        self.assertIn(PHOTO_COLOR, cleaned)
        self.assertIn(WINDOW, cleaned)
        self.assertNotIn("AI eyes", cleaned)
        self.assertIn('darktable:history_end="2"', cleaned)
        self.assertNotIn('darktable:history_end="3"', cleaned)

    def test_an_undone_history_row_is_discarded_and_the_eye_module_is_the_top(self) -> None:
        # darktable discards rows past history_end on any new edit; keeping
        # them left the accepted step mid-stack and not highlighted.
        result = splice_ellipse_exposure(
            _document(history_end="1"),
            center_x=0.5,
            center_y=0.5,
            radius_x=0.1,
            radius_y=0.1,
            exposure_ev=0.1,
        )
        self.assertIn(PHOTO_EXPOSURE, result)
        self.assertNotIn('darktable:multi_name="portrait"', result)
        self.assertNotIn('darktable:mask_name="window"', result)
        self.assertIn('darktable:num="1" darktable:operation="exposure"', result)
        self.assertIn('darktable:multi_name="AI eyes"', result)
        self.assertIn('darktable:history_end="2"', result)

    def test_a_count_ahead_of_the_stack_still_receives_the_eye_module(self) -> None:
        result = splice_ellipse_exposure(
            _document(history_end="4"),
            center_x=0.5,
            center_y=0.5,
            radius_x=0.1,
            radius_y=0.1,
            exposure_ev=0.1,
        )
        self.assertIn(PHOTO_EXPOSURE, result)
        self.assertIn("portrait", result)
        self.assertIn('darktable:multi_name="AI eyes"', result)
        self.assertIn('darktable:history_end="3"', result)

    def test_a_document_without_history_is_refused(self) -> None:
        with self.assertRaises(MaskSpliceError):
            splice_ellipse_exposure(
                "<x:xmpmeta></x:xmpmeta>",
                center_x=0.5,
                center_y=0.5,
                radius_x=0.1,
                radius_y=0.1,
                exposure_ev=0.1,
            )

    def test_a_shape_outside_the_frame_is_refused(self) -> None:
        with self.assertRaises(MaskSpliceError):
            splice_ellipse_exposure(
                _document(),
                center_x=1.4,
                center_y=0.5,
                radius_x=0.1,
                radius_y=0.1,
                exposure_ev=0.1,
            )

    def test_a_refused_sidecar_is_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "photo.xmp"
            source.write_text("<x:xmpmeta></x:xmpmeta>", encoding="utf-8")
            destination = Path(tmp) / "out.xmp"
            errors = StringIO()
            with redirect_stderr(errors):
                code = main(
                    [
                        "--xmp",
                        str(source),
                        "--xmp-out",
                        str(destination),
                        "--cx",
                        "0.5",
                        "--cy",
                        "0.5",
                        "--rx",
                        "0.1",
                        "--ry",
                        "0.1",
                        "--ev",
                        "0.2",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertFalse(destination.exists())
            self.assertEqual(source.read_text(encoding="utf-8"), "<x:xmpmeta></x:xmpmeta>")


    def test_two_regions_become_one_masked_color_module(self) -> None:
        from ai_dt.modules_extra import colorzones_hue_params

        blob = colorzones_hue_params("red", center=0.075)
        self.assertIsNotNone(blob)
        assert blob is not None
        result = splice_region_module(
            _document(),
            shapes=[
                {"cx": 0.46, "cy": 0.48, "rx": 0.08, "ry": 0.07},
                {"cx": 0.43, "cy": 0.78, "rx": 0.07, "ry": 0.06},
            ],
            params=blob,
            operation="colorzones",
            modversion=5,
            name="Highlight the red",
        )
        self.assertEqual(result.count(PHOTO_EXPOSURE), 1)
        self.assertEqual(result.count(PHOTO_COLOR), 1)
        self.assertEqual(result.count(WINDOW), 1)
        self.assertIn('darktable:history_end="3"', result)
        self.assertIn('darktable:operation="colorzones"', result)
        self.assertIn('darktable:multi_name="Highlight the red"', result)
        self.assertEqual(result.count('darktable:mask_type="32"'), 2)
        self.assertIn('darktable:mask_type="4"', result)
        name_at = result.find('darktable:multi_name="Highlight the red"')
        open_at = result.rfind("<rdf:li", 0, name_at)
        added = result[open_at : result.find("/>", name_at) + 2]
        encoded = added.split('darktable:blendop_params="', 1)[1].split('"', 1)[0]
        raw = _decode_gz(encoded)
        mode = struct.unpack_from("<I", raw, 0)[0]
        mask_id = struct.unpack_from("<I", raw, 24)[0]
        self.assertEqual(mode, 3)
        self.assertIn(f'darktable:mask_id="{mask_id}" darktable:mask_type="4"', result)
        cleaned, removed = strip_ai_history(result)
        self.assertEqual(removed, 1)
        self.assertNotIn("Highlight the red", cleaned)
        self.assertIn(WINDOW, cleaned)
        self.assertIn('darktable:history_end="2"', cleaned)

    def test_the_first_color_module_uses_the_open_pipeline_slot(self) -> None:
        from ai_dt.modules_extra import colorzones_hue_params

        document = _document().replace(
            'darktable:history_end="2"',
            'darktable:history_end="2" darktable:iop_order_list="exposure,0,colorzones,0,sigmoid,0"',
        ).replace(
            'darktable:blendop_params="gzPORTRAIT"/>',
            'darktable:blendop_params="gzPORTRAIT" darktable:blendop_version="14"/>',
        )
        blob = colorzones_hue_params("red", center=0.075)
        assert blob is not None
        result = splice_region_module(
            document,
            shapes=[{"cx": 0.46, "cy": 0.48, "rx": 0.08, "ry": 0.07}],
            params=blob,
            operation="colorzones",
            modversion=5,
            name="Highlight the red",
        )
        name_at = result.find('darktable:multi_name="Highlight the red"')
        open_at = result.rfind("<rdf:li", 0, name_at)
        added = result[open_at : result.find("/>", name_at) + 2]
        self.assertIn('darktable:multi_priority="0"', added)
        self.assertIn('darktable:blendop_version="14"', added)
        self.assertEqual(result.count("colorzones,0"), 1)
        self.assertNotIn("colorzones,1", result)

    def test_a_second_color_module_is_added_beside_the_first_slot(self) -> None:
        from ai_dt.modules_extra import colorzones_hue_params

        document = _document(history_end="3").replace(
            'darktable:history_end="3"',
            'darktable:history_end="3" darktable:iop_order_list="exposure,0,colorzones,0"',
        )
        document = document.replace(
            f"     {PHOTO_COLOR}\n",
            f"     {PHOTO_COLOR}\n"
            '     <rdf:li darktable:num="2" darktable:operation="colorzones" '
            'darktable:multi_priority="0" darktable:params="cc"/>\n',
        )
        blob = colorzones_hue_params("red", center=0.075)
        assert blob is not None
        result = splice_region_module(
            document,
            shapes=[{"cx": 0.46, "cy": 0.48, "rx": 0.08, "ry": 0.07}],
            params=blob,
            operation="colorzones",
            modversion=5,
            name="Highlight the red",
        )
        self.assertIn('darktable:multi_priority="1"', result)
        self.assertIn("colorzones,0,colorzones,1", result)

    def test_a_count_ahead_of_the_stack_still_receives_the_color_module(self) -> None:
        from ai_dt.modules_extra import colorzones_hue_params

        blob = colorzones_hue_params("red", center=0.075)
        assert blob is not None
        result = splice_region_module(
            _document(history_end="3"),
            shapes=[{"cx": 0.46, "cy": 0.48, "rx": 0.08, "ry": 0.07}],
            params=blob,
            operation="colorzones",
            modversion=5,
            name="Highlight the red",
        )
        self.assertIn('darktable:operation="colorzones"', result)
        self.assertIn('darktable:history_end="3"', result)
        self.assertIn('darktable:num="2"', result)


    def test_darktable_can_launch_the_script_by_path(self) -> None:
        script = ROOT / "ai_dt" / "drawn_mask.py"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, "-I", str(script), "--help"],
            capture_output=True,
            text=True,
            env=env,
            cwd=tempfile.gettempdir(),
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("No module named", completed.stderr)

    def test_an_empty_mask_list_still_receives_the_shapes(self) -> None:
        from ai_dt.modules_extra import colorzones_hue_params

        document = _document().replace(
            "    <rdf:Seq>\n"
            f"     {WINDOW}\n"
            "    </rdf:Seq>\n",
            "    <rdf:Seq/>\n",
        )
        self.assertIn("<rdf:Seq/>", document)
        blob = colorzones_hue_params("red", center=0.075)
        assert blob is not None
        result = splice_region_module(
            document,
            shapes=[{"cx": 0.46, "cy": 0.48, "rx": 0.08, "ry": 0.07}],
            params=blob,
            operation="colorzones",
            modversion=5,
            name="Highlight the red",
        )
        self.assertIn('darktable:operation="colorzones"', result)
        self.assertIn('darktable:mask_type="32"', result)
        self.assertNotIn("<rdf:Seq/>", result)
        self.assertIn("</rdf:Seq>", result)


class EyeStepTests(unittest.TestCase):
    def test_two_eyes_become_one_ellipse(self) -> None:
        mask = eye_ellipse(
            [
                {
                    "category": "eye",
                    "bbox": [0.40, 0.30, 0.06, 0.04],
                    "exposure_ev": 0.12,
                },
                {
                    "category": "eye",
                    "bbox": [0.52, 0.31, 0.06, 0.04],
                    "exposure_ev": 0.2,
                },
            ]
        )
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(mask["name"], "AI eyes")
        self.assertAlmostEqual(mask["cx"], 0.49)
        self.assertAlmostEqual(mask["exposure_ev"], 0.2)

    def test_accepting_eyes_records_the_ellipse_and_skips_the_whole_frame_style(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            recipe = {
                "chat": True,
                "features_next": "eyes",
                "features_next_title": "Eyes",
                "features_next_action": "lift",
                "local_edits": [
                    {
                        "category": "eye",
                        "target": "eye",
                        "bbox": [0.42, 0.36, 0.08, 0.05],
                        "exposure_ev": 0.18,
                        "ellipse": {
                            "cx": 0.46,
                            "cy": 0.385,
                            "rx": 0.04,
                            "ry": 0.025,
                        },
                    }
                ],
                "plan": {
                    "features_done": [],
                    "features_path": ["eyes", "subject"],
                    "step_queue": [
                        {"id": "eyes", "action": "lift", "title": "Eyes"},
                        {"id": "subject", "action": "lift", "title": "Lift the subject"},
                    ],
                    "not_history": ["reflection"],
                },
            }
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            (directory / "step.dtstyle").write_text("old", encoding="utf-8")
            saved = commit_waiting_step(directory)
            self.assertEqual(saved["step_mask"]["name"], "AI eyes")
            self.assertNotIn("step_style", saved)
            self.assertFalse((directory / "step.dtstyle").exists())
            self.assertIn("ellipse", saved["status_text"])
            self.assertNotIn("color balance", saved["status_text"])
            self.assertEqual(saved["features_next"], "subject")


if __name__ == "__main__":
    unittest.main()
