"""Eyes on an animal become eye boxes the history ellipse can use."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from ai_dt.detectors.heuristic import animal_eye_boxes, attach_animal_eyes
from ai_dt.drawn_mask import eye_ellipse
from ai_dt.mapping import build_modules
from ai_dt.schema import Atmosphere, Detection


def _fur_with_two_pupils() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Light fur and two dark pupils sitting side by side, not in a human face."""

    height, width = 200, 320
    luma = np.full((height, width), 0.74, dtype=np.float32)
    luma[78:90, 150:162] = 0.06
    luma[80:92, 188:200] = 0.07
    box = (0.2, 0.15, 0.6, 0.7)
    return luma, box


class AnimalEyeTests(unittest.TestCase):
    def test_two_dark_pupils_on_fur_become_eye_boxes(self) -> None:
        luma, box = _fur_with_two_pupils()
        found = animal_eye_boxes(luma, box, luma.shape[1], luma.shape[0])
        self.assertEqual(len(found), 2)
        centers = sorted(item[0] + item[2] / 2.0 for item in found)
        self.assertLess(centers[0], 0.55)
        self.assertGreater(centers[1], 0.55)

    def test_plain_fur_has_no_eyes(self) -> None:
        luma = np.full((200, 320), 0.7, dtype=np.float32)
        found = animal_eye_boxes(luma, (0.2, 0.15, 0.6, 0.7), 320, 200)
        self.assertEqual(found, [])

    def test_one_dark_spot_is_not_a_pair_of_eyes(self) -> None:
        luma = np.full((200, 320), 0.74, dtype=np.float32)
        luma[80:92, 170:182] = 0.05
        found = animal_eye_boxes(luma, (0.2, 0.15, 0.6, 0.7), 320, 200)
        self.assertEqual(found, [])

    def test_an_animal_box_gains_eyes_and_a_face_keeps_its_own(self) -> None:
        height, width = 200, 320
        pixels = np.full((height, width, 3), 0.74, dtype=np.float32)
        pixels[78:90, 150:162] = 0.06
        pixels[80:92, 188:200] = 0.07
        animal = Detection("cat", "animal", 0.9, (0.2, 0.15, 0.6, 0.7), "yolo")
        face_eye = Detection("eye", "eye", 0.8, (0.05, 0.05, 0.04, 0.03), "mediapipe")
        found = attach_animal_eyes(pixels, [animal, face_eye])
        animal_eyes = [
            item
            for item in found
            if item.category == "eye" and item.bbox[0] > 0.2
        ]
        self.assertEqual(len(animal_eyes), 2)
        self.assertIn(face_eye, found)

    def test_orange_ear_tips_are_not_eyes(self) -> None:
        height, width = 200, 320
        pixels = np.full((height, width, 3), (0.72, 0.66, 0.55), dtype=np.float32)
        pixels[36:52, 130:148] = (0.92, 0.42, 0.08)
        pixels[40:48, 136:142] = (0.16, 0.08, 0.04)
        pixels[36:52, 190:208] = (0.90, 0.40, 0.08)
        pixels[40:48, 196:202] = (0.16, 0.08, 0.04)
        animal = Detection("cat", "animal", 0.8, (0.15, 0.1, 0.7, 0.75), "yolo")
        found = [item for item in attach_animal_eyes(pixels, [animal]) if item.category == "eye"]
        self.assertEqual(found, [])
        loose = Detection("animal", "animal", 0.95, (0.0, 0.0, 1.0, 1.0), "heuristic")
        found = [item for item in attach_animal_eyes(pixels, [loose, animal]) if item.category == "eye"]
        self.assertEqual(found, [])

    def test_a_sunlit_iris_in_a_dark_socket_is_still_an_eye(self) -> None:
        height, width = 200, 320
        pixels = np.full((height, width, 3), (0.74, 0.68, 0.58), dtype=np.float32)
        yy, xx = np.ogrid[0:height, 0:width]
        for cy, cx in ((100, 150), (102, 196)):
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            pixels[dist <= 14] = (0.16, 0.10, 0.07)
            pixels[dist <= 6] = (0.90, 0.74, 0.14)
            pixels[dist <= 1.5] = (0.05, 0.04, 0.03)
        animal = Detection("cat", "animal", 0.9, (0.2, 0.15, 0.6, 0.7), "yolo")
        found = [
            item
            for item in attach_animal_eyes(pixels, [animal])
            if item.category == "eye"
        ]
        self.assertEqual(len(found), 2)
        centers_y = [item.bbox[1] + item.bbox[3] / 2.0 for item in found]
        self.assertTrue(all(0.40 <= center <= 0.60 for center in centers_y))

    def test_colored_eyes_win_over_dark_ears(self) -> None:
        height, width = 200, 320
        pixels = np.full((height, width, 3), (0.72, 0.66, 0.55), dtype=np.float32)
        pixels[48:64, 140:156] = (0.16, 0.13, 0.11)
        pixels[48:64, 196:212] = (0.15, 0.12, 0.10)
        pixels[102:116, 142:156] = (0.86, 0.74, 0.12)
        pixels[106:112, 146:152] = (0.05, 0.04, 0.03)
        pixels[102:116, 198:212] = (0.84, 0.72, 0.14)
        pixels[106:112, 202:208] = (0.05, 0.04, 0.03)
        animal = Detection("cat", "animal", 0.9, (0.15, 0.1, 0.7, 0.75), "yolo")
        found = [
            item
            for item in attach_animal_eyes(pixels, [animal])
            if item.category == "eye"
        ]
        self.assertEqual(len(found), 2)
        centers_y = [item.bbox[1] + item.bbox[3] / 2.0 for item in found]
        self.assertTrue(all(center > 0.4 for center in centers_y))

    def test_the_box_sits_on_the_iris_below_the_bright_brow(self) -> None:
        height, width = 200, 320
        pixels = np.full((height, width, 3), (0.72, 0.66, 0.55), dtype=np.float32)
        pixels[88:100, 138:160] = (0.95, 0.40, 0.06)
        pixels[88:100, 196:218] = (0.95, 0.40, 0.06)
        pixels[98:116, 142:158] = (0.76, 0.60, 0.18)
        pixels[98:116, 200:216] = (0.76, 0.60, 0.18)
        pixels[104:112, 146:154] = (0.05, 0.04, 0.03)
        pixels[104:112, 204:212] = (0.05, 0.04, 0.03)
        animal = Detection("cat", "animal", 0.9, (0.15, 0.1, 0.7, 0.75), "yolo")
        found = [
            item
            for item in attach_animal_eyes(pixels, [animal])
            if item.category == "eye"
        ]
        self.assertEqual(len(found), 2)
        centers_y = [item.bbox[1] + item.bbox[3] / 2.0 for item in found]
        self.assertTrue(all(center >= 0.52 for center in centers_y))

    def test_a_shaded_iris_pulls_the_box_off_the_bright_brow(self) -> None:
        height, width = 200, 320
        pixels = np.full((height, width, 3), (0.72, 0.66, 0.55), dtype=np.float32)
        pixels[80:92, 144:156] = (0.88, 0.46, 0.10)
        pixels[80:92, 202:214] = (0.88, 0.46, 0.10)
        pixels[92:100, 142:158] = (0.24, 0.16, 0.10)
        pixels[92:100, 200:216] = (0.24, 0.16, 0.10)
        pixels[100:114, 142:158] = (0.40, 0.32, 0.20)
        pixels[100:114, 200:216] = (0.40, 0.32, 0.20)
        pixels[104:110, 146:152] = (0.05, 0.04, 0.03)
        pixels[104:110, 204:210] = (0.05, 0.04, 0.03)
        animal = Detection("cat", "animal", 0.9, (0.15, 0.1, 0.7, 0.75), "yolo")
        found = [
            item
            for item in attach_animal_eyes(pixels, [animal])
            if item.category == "eye"
        ]
        self.assertEqual(len(found), 2)
        centers_y = [item.bbox[1] + item.bbox[3] / 2.0 for item in found]
        self.assertTrue(all(0.50 <= center <= 0.58 for center in centers_y))

    def test_eye_boxes_stay_when_other_subjects_fill_the_edit_list(self) -> None:
        image = Image.new("RGB", (80, 80), (180, 170, 150))
        others = [
            Detection("thing", "subject", 0.4, (0.0, 0.0, 0.2, 0.2), "test")
            for _ in range(6)
        ]
        eyes = [
            Detection("eye", "eye", 0.7, (0.40, 0.30, 0.06, 0.05), "animal"),
            Detection("eye", "eye", 0.7, (0.55, 0.30, 0.06, 0.05), "animal"),
        ]
        _adjusted, _modules, edits, _blobs = build_modules(
            image,
            Atmosphere(),
            others + eyes,
        )
        mask = eye_ellipse(edits)
        self.assertIsNotNone(mask)
        self.assertEqual(mask["name"], "AI eyes")


class ProfileEyeTests(unittest.TestCase):
    """dd888ef made the eye search pairs-only. A bird side-on shows one eye,
    and a red-meets-yellow plumage edge came back as a level colour "pair"."""

    def _profile_bird(self, catchlight: bool = True) -> np.ndarray:
        height, width = 240, 360
        pixels = np.full((height, width, 3), (0.30, 0.55, 0.25), dtype=np.float32)
        pixels[60:220, 120:260] = (0.10, 0.10, 0.11)
        pixels[50:120, 140:250] = (0.78, 0.12, 0.08)  # red head
        pixels[125:200, 145:245] = (0.86, 0.70, 0.10)  # yellow breast, a colour edge
        pixels[70:84, 200:214] = (0.04, 0.03, 0.03)  # the one eye
        if catchlight:
            pixels[74:78, 205:209] = (0.92, 0.92, 0.90)
        return pixels

    def test_one_eye_with_a_catchlight_is_found_side_on(self) -> None:
        bird = Detection("animal", "animal", 0.6, (120 / 360, 50 / 240, 140 / 360, 170 / 240), "heuristic")
        found = [item for item in attach_animal_eyes(self._profile_bird(), [bird]) if item.category == "eye"]
        self.assertEqual(len(found), 1)
        x, y, w, h = found[0].bbox
        self.assertTrue(x <= 207 / 360 <= x + w and y <= 77 / 240 <= y + h)

    def test_a_dark_spot_without_an_enclosed_catchlight_is_not_an_eye(self) -> None:
        bird = Detection("animal", "animal", 0.6, (120 / 360, 50 / 240, 140 / 360, 170 / 240), "heuristic")
        found = [item for item in attach_animal_eyes(self._profile_bird(catchlight=False), [bird]) if item.category == "eye"]
        self.assertEqual(found, [])

    def test_a_colour_edge_is_not_a_pair_of_eyes(self) -> None:
        # The wildlife-guide bird: red meets yellow across the body. Its
        # edge gave a row of colour peaks and a level "pair" on the breast.
        from tests.test_guides import _bird

        pixels = np.rint(_bird() * 255.0) / 255.0
        luma = (0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]).astype(np.float32)
        found = animal_eye_boxes(luma, (0.436, 0.254, 0.239, 0.554), 720, 480, color=pixels.astype(np.float32))
        self.assertEqual(len(found), 1)
        x, y, w, h = found[0]
        self.assertTrue(x <= 409 / 720 <= x + w and y <= 197 / 480 <= y + h)
