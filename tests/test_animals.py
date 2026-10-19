"""Animal masks: one outline per animal, an all-animals union, graceful misses."""

from __future__ import annotations

import unittest

import numpy as np

from ai_dt.dt_shapes import iou, render_forms
from ai_dt.schema import Detection
from ai_dt.segment import segment
from ai_dt.segment.animals import (
    CONFIDENCE_LARGE,
    CONFIDENCE_SMALL,
    animal_boxes,
    animal_forms,
    close_gaps,
    confidence_bar,
    segment_animals,
)
from tests.synthetic_scenes import animals_in_field, occluded_animal, tiny_bird


def _truth_for(animal, boxes, truths):
    for box, truth in zip(boxes, truths):
        if tuple(box["bbox"]) == tuple(animal.bbox):
            return truth
    raise AssertionError("no truth for that animal")


class BoxTests(unittest.TestCase):
    def test_small_animals_get_a_lower_confidence_bar(self):
        self.assertEqual(confidence_bar(0.002), CONFIDENCE_SMALL)
        self.assertEqual(confidence_bar(0.3), CONFIDENCE_LARGE)
        notes: list[str] = []
        kept = animal_boxes(
            [
                {"label": "bird", "category": "animal", "confidence": 0.17, "bbox": [0.5, 0.2, 0.04, 0.03]},
                {"label": "cow", "category": "animal", "confidence": 0.17, "bbox": [0.1, 0.4, 0.5, 0.5]},
            ],
            notes,
        )
        self.assertEqual([entry[0] for entry in kept], ["bird"])
        self.assertTrue(any("cow" in note for note in notes))

    def test_noise_boxes_and_duplicates_are_dropped(self):
        notes: list[str] = []
        kept = animal_boxes(
            [
                {"label": "dog", "category": "animal", "confidence": 0.9, "bbox": [0.2, 0.2, 0.4, 0.4]},
                {"label": "dog", "category": "animal", "confidence": 0.5, "bbox": [0.21, 0.2, 0.4, 0.41]},
                {"label": "cat", "category": "animal", "confidence": 0.9, "bbox": [0.7, 0.7, 0.004, 0.2]},
                {"label": "person", "category": "person", "confidence": 0.9, "bbox": [0.0, 0.0, 0.3, 0.6]},
            ],
            notes,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][1], 0.9)
        self.assertTrue(any("too small" in note for note in notes))

    def test_detection_objects_and_coco_labels_are_read(self):
        detection = Detection(label="giraffe", category="animal", confidence=0.4, bbox=(0.1, 0.1, 0.3, 0.6), source="yolo")
        self.assertEqual(animal_boxes([detection])[0][0], "giraffe")

    def test_close_gaps_bridges_thin_occluders(self):
        binary = np.zeros((30, 30), dtype=bool)
        binary[5:25, 5:25] = True
        binary[5:25, 14:16] = False  # a grass blade
        closed = close_gaps(binary, 2)
        self.assertTrue(closed[10:20, 14:16].all())
        self.assertFalse(closed[0:3, :].any())


class OutlineTests(unittest.TestCase):
    def test_multiple_animals_each_get_an_outline_and_a_union(self):
        image, boxes, truths = animals_in_field()
        found = segment_animals(image, boxes, backend="heuristic")
        self.assertEqual(len(found.animals), 3)
        self.assertEqual(found.animals[0].label, "horse")  # largest first
        for animal in found.animals:
            self.assertEqual(animal.source, "heuristic")
            self.assertGreater(iou(animal.mask, _truth_for(animal, boxes, truths)), 0.7, animal.label)
        union = found.union(image.size)
        self.assertGreater(iou(union, np.logical_or.reduce(truths)), 0.75)
        forms = animal_forms(found)
        self.assertGreaterEqual(len(forms), 3)
        self.assertEqual(forms[0]["mode"], "union")
        drawn = render_forms(forms, image.width, image.height)
        self.assertGreater(iou(drawn, np.logical_or.reduce(truths)), 0.7)
        main = render_forms(animal_forms(found, which="main"), image.width, image.height)
        self.assertGreater(iou(main, truths[0]), 0.7)

    def test_an_animal_behind_grass_keeps_one_outline(self):
        image, box, truth = occluded_animal()
        found = segment_animals(image, [box], backend="heuristic")
        self.assertEqual(len(found.animals), 1)
        self.assertGreater(iou(found.animals[0].mask, truth), 0.8)
        forms = animal_forms(found)
        outer = [form for form in forms if form["mode"] != "difference"]
        self.assertEqual(len(outer), 1)

    def test_a_tiny_distant_bird_is_outlined_on_its_own_crop(self):
        image, box, truth = tiny_bird()
        found = segment_animals(image, [box], backend="heuristic")
        self.assertEqual(len(found.animals), 1)
        forms = animal_forms(found)
        self.assertTrue(forms)
        self.assertGreater(iou(render_forms(forms, image.width, image.height), truth), 0.5)

    def test_no_animal_is_a_note_not_an_error(self):
        image, _box, _truth = tiny_bird()
        found = segment_animals(image, [], backend="heuristic")
        self.assertEqual(found.animals, [])
        self.assertIn("no animal was detected", found.notes)
        self.assertIsNone(found.union(image.size))
        result = segment(image, ["animal"], boxes=[], backend="heuristic")
        self.assertIsNone(result.get("animal"))

    def test_a_model_backend_is_used_first_and_a_bad_one_falls_back(self):
        image, box, truth = occluded_animal()

        class Exact:
            name = "fake-sam"
            classes = ("subject",)

            def available(self):
                return True

            def segment(self, crop, classes, *, boxes):
                self.boxes = boxes
                # The whole crop is "subject": not trusted (fills the box).
                return {"subject": np.ones((crop.height, crop.width), dtype=np.float32)}

        found = segment_animals(image, [box], segmenters=[Exact()])
        self.assertEqual(found.animals[0].source, "heuristic")

        class Good(Exact):
            def segment(self, crop, classes, *, boxes):
                x, y, w, h = boxes[0]["bbox"]
                yy, xx = np.mgrid[0:crop.height, 0:crop.width]
                cx, cy = (x + w / 2) * crop.width, (y + h / 2) * crop.height
                inside = ((xx - cx) / (w * crop.width / 2)) ** 2 + ((yy - cy) / (h * crop.height / 2)) ** 2 <= 1
                return {"subject": inside.astype(np.float32)}

        found = segment_animals(image, [box], segmenters=[Good()])
        self.assertEqual(found.animals[0].source, "fake-sam")
        self.assertGreater(iou(found.animals[0].mask, truth), 0.85)

    def test_segment_reports_animal_instances(self):
        image, boxes, _truths = animals_in_field()
        result = segment(image, ["animal", "sky"], boxes=boxes, backend="heuristic")
        self.assertIsNotNone(result.get("animal"))
        self.assertIsNotNone(result.get("sky"))
        self.assertEqual(len(result.instances["animal"]), 3)
        self.assertEqual(result.sources["animal"], "heuristic")


if __name__ == "__main__":
    unittest.main()
