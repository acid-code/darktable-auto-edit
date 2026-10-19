"""Segmentation backends. No model, network, or key is needed.

The heuristic backend is checked on synthetic scenes. The optional
backends are checked through their pure parts (label maps, Gemini replies,
alpha planes) and through ``available()`` staying false when the package
is missing. The chain is checked with fake segmenters.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dt.segment import SegmenterUnavailable, backend_order, segment
from ai_dt.segment.ade20k import Ade20kSegmenter, class_ids, masks_from_labels
from ai_dt.segment.gemini import GeminiSegmenter, masks_from_reply, prompt_for
from ai_dt.segment.heuristic import HeuristicSegmenter, horizon_row
from ai_dt.segment.rembg_backend import RembgSegmenter, alpha_to_mask
from tests.synthetic_scenes import landscape, region, subject_on_foliage


def _installed(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


class HeuristicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = segment(
            landscape(),
            ["sky", "architecture", "vegetation", "water", "ground"],
            backend="heuristic",
        )

    def test_every_landscape_class_is_found_where_it_is(self) -> None:
        masks = self.result.masks
        self.assertEqual(set(masks), {"sky", "architecture", "vegetation", "water", "ground"})
        for mask in masks.values():
            self.assertEqual(mask.shape, (240, 360))
            self.assertEqual(mask.dtype, np.float32)
        self.assertGreater(region(masks["sky"], slice(0, 85)), 0.9)
        self.assertGreater(region(masks["vegetation"], slice(95, 165), slice(0, 145)), 0.9)
        self.assertGreater(region(masks["architecture"], slice(95, 165), slice(160, 350)), 0.8)
        self.assertGreater(region(masks["water"], slice(175, 210)), 0.85)
        self.assertGreater(region(masks["ground"], slice(222, 240)), 0.85)

    def test_classes_do_not_spill_into_each_other(self) -> None:
        masks = self.result.masks
        self.assertLess(region(masks["sky"], slice(100, 240)), 0.02)
        self.assertLess(region(masks["vegetation"], slice(95, 165), slice(160, 360)), 0.02)
        self.assertLess(region(masks["architecture"], slice(175, 240)), 0.05)
        self.assertLess(region(masks["water"], slice(0, 165)), 0.02)
        self.assertLess(region(masks["ground"], slice(0, 205)), 0.02)
        self.assertEqual(set(self.result.sources.values()), {"heuristic"})

    def test_horizon_row_follows_the_sky(self) -> None:
        sky = np.zeros((100, 50), dtype=np.float32)
        sky[:40] = 1.0
        self.assertAlmostEqual(horizon_row(sky), 0.4, places=2)
        self.assertEqual(horizon_row(np.zeros((10, 10), dtype=np.float32)), 0.45)

    def test_no_sky_in_a_closed_frame(self) -> None:
        image, _box = subject_on_foliage()
        result = segment(image, ["sky", "water"], backend="heuristic")
        self.assertNotIn("sky", result.masks)

    def test_subject_from_a_box_and_its_inverse(self) -> None:
        image, box = subject_on_foliage()
        result = segment(
            image,
            ["subject", "background"],
            boxes=[{"category": "animal", "bbox": list(box)}],
            backend="heuristic",
        )
        subject = result.masks["subject"]
        self.assertGreater(region(subject, slice(80, 180), slice(130, 210)), 0.9)
        self.assertLess(region(subject, slice(0, 50)), 0.02)
        background = result.masks["background"]
        self.assertTrue(np.allclose(background, 1.0 - subject))
        self.assertIn("inverse subject", result.sources["background"])

    def test_subject_without_a_box_uses_saliency(self) -> None:
        image, _box = subject_on_foliage()
        result = segment(image, ["subject"], backend="heuristic")
        self.assertGreater(region(result.masks["subject"], slice(90, 170), slice(140, 200)), 0.8)

    def test_people_need_a_person_or_face_box(self) -> None:
        image, box = subject_on_foliage()
        none = segment(image, ["people"], backend="heuristic")
        self.assertNotIn("people", none.masks)
        found = segment(image, ["people"], boxes=[{"category": "person", "bbox": list(box)}], backend="heuristic")
        self.assertIn("people", found.masks)

    def test_unknown_classes_are_ignored(self) -> None:
        result = segment(landscape(), ["sky", "dragons"], backend="heuristic")
        self.assertEqual(set(result.masks), {"sky"})
        self.assertEqual(segment(landscape(), []).masks, {})


class _Fake:
    def __init__(self, name, classes, masks=None, *, available=True, error=None):
        self.name = name
        self.classes = tuple(classes)
        self._masks = masks or {}
        self._available = available
        self._error = error
        self.calls = []

    def available(self):
        return self._available

    def segment(self, image, classes, *, boxes):
        self.calls.append(list(classes))
        if self._error:
            raise self._error
        return {name: mask for name, mask in self._masks.items() if name in classes}


class ChainTests(unittest.TestCase):
    def _half(self, top: bool) -> np.ndarray:
        mask = np.zeros((60, 80), dtype=np.float32)
        if top:
            mask[:30] = 1.0
        else:
            mask[30:] = 1.0
        return mask

    def test_first_usable_backend_wins_and_the_rest_fill_gaps(self) -> None:
        image = Image.new("RGB", (160, 120), (128, 128, 128))
        model = _Fake("model", ["sky"], {"sky": self._half(True)})
        fallback = _Fake("fallback", ["sky", "water"], {"sky": self._half(False), "water": self._half(False)})
        result = segment(image, ["sky", "water"], segmenters=[model, fallback])
        self.assertEqual(result.sources, {"sky": "model", "water": "fallback"})
        self.assertEqual(fallback.calls, [["water"]])
        # Masks are resized to the image.
        self.assertEqual(result.masks["sky"].shape, (120, 160))
        self.assertGreater(region(result.masks["sky"], slice(0, 55)), 0.95)

    def test_broken_unavailable_and_empty_backends_are_skipped(self) -> None:
        image = Image.new("RGB", (80, 60))
        broken = _Fake("broken", ["sky"], error=RuntimeError("onnx exploded"))
        missing = _Fake("missing", ["sky"], available=False)
        gone = _Fake("gone", ["sky"], error=SegmenterUnavailable("no key"))
        empty = _Fake("empty", ["sky"], {"sky": np.zeros((60, 80), dtype=np.float32)})
        last = _Fake("last", ["sky"], {"sky": self._half(True)})
        result = segment(image, ["sky"], segmenters=[broken, missing, gone, empty, last])
        self.assertEqual(result.sources["sky"], "last")
        joined = " ".join(result.notes)
        self.assertIn("onnx exploded", joined)
        self.assertIn("missing: not installed", joined)
        self.assertIn("no key", joined)

    def test_background_is_asked_as_the_subject(self) -> None:
        image = Image.new("RGB", (80, 60))
        model = _Fake("model", ["subject"], {"subject": self._half(True)})
        result = segment(image, ["background"], segmenters=[model])
        self.assertEqual(set(result.masks), {"background"})
        self.assertEqual(model.calls, [["subject"]])
        self.assertGreater(region(result.masks["background"], slice(35, 60)), 0.95)

    def test_backend_order(self) -> None:
        with patch.dict(os.environ, {"AI_DT_SEGMENT_GEMINI": "", "AI_DT_SEGMENTER": ""}):
            self.assertEqual(backend_order("auto"), ["darktable-sam", "rembg", "ade20k", "heuristic"])
            self.assertEqual(backend_order("heuristic"), ["heuristic"])
            self.assertEqual(backend_order("rembg"), ["rembg", "heuristic"])
        with patch.dict(os.environ, {"AI_DT_SEGMENT_GEMINI": "1"}):
            self.assertIn("gemini", backend_order("auto"))
        with self.assertRaises(ValueError):
            segment(Image.new("RGB", (8, 8)), ["sky"], backend="magic")

    def test_auto_runs_without_optional_packages_or_keys(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "AI_DT_SEGMENT_GEMINI": "1", "LOCALAPPDATA": "", "XDG_CONFIG_HOME": "/nonexistent"}):
            result = segment(landscape(), ["sky"], backend="auto")
        self.assertIn("sky", result.masks)


class OptionalBackendTests(unittest.TestCase):
    def test_ade20k_label_map(self) -> None:
        id2label = {0: "wall", 1: "building, edifice", 2: "sky", 3: "tree", 4: "water", 5: "person;individual", 6: "road;route", 7: "car"}
        ids = class_ids(id2label)
        self.assertEqual(ids["architecture"], {0, 1})
        self.assertEqual(ids["people"], {5})
        self.assertEqual(ids["ground"], {6})
        labels = np.array([[2, 2, 2], [1, 3, 7], [4, 4, 6]])
        masks = masks_from_labels(labels, id2label, ["sky", "vegetation", "water", "ground", "people"])
        self.assertEqual(masks["sky"][0].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(masks["vegetation"][1].tolist(), [0.0, 1.0, 0.0])
        self.assertNotIn("people", masks)

    def test_gemini_polygon_and_box_mask_replies(self) -> None:
        patch_image = Image.new("L", (10, 10), 255)
        buffer = io.BytesIO()
        patch_image.save(buffer, format="PNG")
        encoded = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        reply = "```json\n" + json.dumps([
            {"label": "sky", "polygon": [[0, 0], [1000, 0], [1000, 400], [0, 400]]},
            {"label": "people", "box_2d": [500, 250, 1000, 500], "mask": encoded},
            {"label": "unicorn", "polygon": [[0, 0], [10, 0], [10, 10]]},
        ]) + "\n```"
        masks = masks_from_reply(reply, 100, 50)
        self.assertEqual(set(masks), {"sky", "people"})
        self.assertGreater(region(masks["sky"], slice(0, 18)), 0.95)
        self.assertLess(region(masks["sky"], slice(22, 50)), 0.05)
        self.assertGreater(region(masks["people"], slice(26, 50), slice(26, 49)), 0.95)
        self.assertEqual(masks_from_reply("not json", 10, 10), {})
        self.assertIn('"water"', prompt_for(["water", "background"]))

    def test_gemini_needs_a_key_and_never_calls_out_without_one(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            segmenter = GeminiSegmenter()
            self.assertFalse(segmenter.available())
            with self.assertRaises(SegmenterUnavailable):
                segmenter.segment(Image.new("RGB", (8, 8)), ["sky"], boxes=[])

    def test_gemini_request_is_parsed(self) -> None:
        reply = {"candidates": [{"content": {"parts": [{"text": json.dumps([
            {"label": "water", "polygon": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]]}
        ])}]}}]}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            segmenter = GeminiSegmenter()
            with patch.object(segmenter, "_post", return_value=reply) as post:
                masks = segmenter.segment(Image.new("RGB", (40, 20)), ["water"], boxes=[])
        self.assertIn("water", masks)
        body, key = post.call_args.args
        self.assertEqual(key, "test-key")
        self.assertIn("inline_data", body["contents"][0]["parts"][0])

    def test_rembg_alpha(self) -> None:
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        self.assertTrue(np.allclose(alpha_to_mask(rgba), 1.0))
        self.assertTrue(np.allclose(alpha_to_mask(Image.new("L", (3, 3), 0)), 0.0))

    @unittest.skipIf(_installed("rembg"), "rembg is installed here")
    def test_rembg_reports_missing(self) -> None:
        self.assertFalse(RembgSegmenter().available())

    @unittest.skipIf(_installed("transformers") and _installed("torch"), "transformers is installed here")
    def test_ade20k_reports_missing(self) -> None:
        self.assertFalse(Ade20kSegmenter().available())

    def test_heuristic_lists_its_classes(self) -> None:
        self.assertIn("sky", HeuristicSegmenter.classes)
        self.assertTrue(HeuristicSegmenter().available())


if __name__ == "__main__":
    unittest.main()


class SaliencyPrefersTheSubjectTests(unittest.TestCase):
    """A big patch running off the frame (water, foreground) must not beat a small subject."""

    def test_small_subject_beats_edge_patch(self) -> None:
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(5)
        pixels = np.full((240, 320, 3), (0.45, 0.45, 0.4), dtype=np.float32) + rng.random((240, 320, 3)) * 0.05
        pixels[40:240, 200:320] = (0.6, 0.6, 0.55)  # a lighter patch of water on the right edge
        pixels[100:150, 130:160] = (0.2, 0.5, 0.2)  # a small green bird left of centre
        image = Image.fromarray((np.clip(pixels, 0, 1) * 255).astype("uint8"))
        mask = segment(image, ["subject"], backend="heuristic").masks["subject"]
        self.assertGreater(region(mask, slice(105, 145), slice(135, 155)), 0.8)
        self.assertLess(region(mask, slice(60, 240), slice(230, 320)), 0.05)
