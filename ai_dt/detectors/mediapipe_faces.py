"""Optional MediaPipe face and eye detector.

The legacy ``mp.solutions`` API and the newer Tasks API do not ship the same
call surface. This module uses whichever one is importable. Eye boxes come
from face landmarks. Install MediaPipe with ``requirements-ai.txt``. The Tasks
build downloads its two model files on first use.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from ai_dt.errors import DetectorUnavailable
from ai_dt.schema import Detection

_FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
)
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Face Mesh landmark pairs that bound the two eyes, including the iris set
# used when refine_landmarks is available.
_LEFT_EYE = (33, 133, 159, 145, 468)
_RIGHT_EYE = (362, 263, 386, 374, 473)


def _bbox_from_relative(origin_x: float, origin_y: float, width: float, height: float) -> tuple[float, float, float, float]:
    return (
        max(0.0, min(1.0, origin_x)),
        max(0.0, min(1.0, origin_y)),
        max(0.0, min(1.0, width)),
        max(0.0, min(1.0, height)),
    )


def _box_from_pixels(
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float] | None:
    if frame_w <= 0 or frame_h <= 0:
        return None
    x = max(0.0, min(1.0, float(origin_x) / float(frame_w)))
    y = max(0.0, min(1.0, float(origin_y) / float(frame_h)))
    box_w = max(0.0, min(1.0 - x, float(width) / float(frame_w)))
    box_h = max(0.0, min(1.0 - y, float(height) / float(frame_h)))
    if box_w < 0.01 or box_h < 0.01:
        return None
    return (x, y, box_w, box_h)


def faces_from_tasks(
    frame_w: int,
    frame_h: int,
    face_detections,
    landmark_sets,
    min_confidence: float = 0.5,
) -> list[Detection]:
    """Turn Tasks API faces and landmarks into normalized boxes."""

    found: list[Detection] = []
    for detection in face_detections or []:
        box = getattr(detection, "bounding_box", None)
        if box is None:
            continue
        normalized = _box_from_pixels(
            getattr(box, "origin_x", 0),
            getattr(box, "origin_y", 0),
            getattr(box, "width", 0),
            getattr(box, "height", 0),
            frame_w,
            frame_h,
        )
        if normalized is None:
            continue
        score = float(min_confidence)
        categories = getattr(detection, "categories", None) or []
        if categories:
            try:
                score = float(getattr(categories[0], "score", score) or score)
            except (TypeError, ValueError):
                score = float(min_confidence)
        found.append(
            Detection(
                label="face",
                category="face",
                confidence=score,
                bbox=normalized,
                source="mediapipe",
            )
        )
    for landmarks in landmark_sets or []:
        for label, indices in (("left eye", _LEFT_EYE), ("right eye", _RIGHT_EYE)):
            box = _eye_box(landmarks, indices)
            if box is None:
                continue
            found.append(
                Detection(
                    label=label,
                    category="eye",
                    confidence=0.6,
                    bbox=box,
                    source="mediapipe",
                )
            )
    return found


def _model_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    base = Path(root) if root else Path.home()
    path = base / "darktable" / "ai-editor-models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cached_model(name: str, url: str) -> Path:
    dest = _model_dir() / name
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    part = dest.with_name(name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            part.write_bytes(response.read())
        part.replace(dest)
    except Exception as exc:
        if part.exists():
            part.unlink()
        raise DetectorUnavailable(f"MediaPipe model {name} is not available: {exc}") from exc
    return dest


def _eye_box(landmarks, indices: tuple[int, ...]) -> tuple[float, float, float, float] | None:
    points = []
    for index in indices:
        if index >= len(landmarks):
            continue
        point = landmarks[index]
        points.append((float(point.x), float(point.y)))
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    pad = 0.012
    x1, x2 = max(0.0, min(xs) - pad), min(1.0, max(xs) + pad)
    y1, y2 = max(0.0, min(ys) - pad), min(1.0, max(ys) + pad)
    if x2 - x1 < 0.005 or y2 - y1 < 0.005:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


class MediaPipeFaceDetector:
    name = "mediapipe"

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence

    def detect(self, image: Image.Image) -> list[Detection]:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise DetectorUnavailable(
                "mediapipe is not installed. Install requirements-ai.txt or "
                "choose another detector."
            ) from exc

        rgb = np.asarray(image.convert("RGB"))
        if hasattr(mp, "solutions"):
            return self._detect_solutions(mp, rgb)
        return self._detect_tasks(mp, rgb)

    def _detect_tasks(self, mp, rgb: np.ndarray) -> list[Detection]:
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        pixels = np.ascontiguousarray(rgb)
        try:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=pixels)
            detector_path = _cached_model("blaze_face_full_range.tflite", _FACE_DETECTOR_URL)
            landmarker_path = _cached_model("face_landmarker.task", _FACE_LANDMARKER_URL)
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )
            from mediapipe.tasks.python.vision.face_detector import FaceDetector, FaceDetectorOptions
            from mediapipe.tasks.python.vision.face_landmarker import (
                FaceLandmarker,
                FaceLandmarkerOptions,
            )
        except DetectorUnavailable:
            raise
        except Exception as exc:
            raise DetectorUnavailable(f"MediaPipe face tasks could not start: {exc}") from exc

        detector = FaceDetector.create_from_options(
            FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(detector_path)),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_detection_confidence=self.min_confidence,
            )
        )
        try:
            detected = detector.detect(image)
            faces = list(detected.detections or [])
        finally:
            detector.close()

        landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(landmarker_path)),
                running_mode=VisionTaskRunningMode.IMAGE,
                num_faces=4,
                min_face_detection_confidence=self.min_confidence,
            )
        )
        try:
            meshed = landmarker.detect(image)
            landmarks = list(meshed.face_landmarks or [])
        finally:
            landmarker.close()
        return faces_from_tasks(width, height, faces, landmarks, self.min_confidence)

    def _detect_solutions(self, mp, rgb: np.ndarray) -> list[Detection]:
        found: list[Detection] = []
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.min_confidence,
        )
        try:
            result = face_detection.process(rgb)
        finally:
            face_detection.close()
        if result and result.detections:
            for detection in result.detections:
                score = 0.0
                if detection.score:
                    score = float(detection.score[0])
                box = detection.location_data.relative_bounding_box
                found.append(
                    Detection(
                        label="face",
                        category="face",
                        confidence=score or self.min_confidence,
                        bbox=_bbox_from_relative(box.xmin, box.ymin, box.width, box.height),
                        source=self.name,
                    )
                )

        if not hasattr(mp.solutions, "face_mesh"):
            return found
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=self.min_confidence,
        )
        try:
            meshed = mesh.process(rgb)
        except TypeError:
            mesh.close()
            mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=4,
                min_detection_confidence=self.min_confidence,
            )
            meshed = mesh.process(rgb)
        try:
            if meshed and meshed.multi_face_landmarks:
                for face in meshed.multi_face_landmarks:
                    for label, indices in (("left eye", _LEFT_EYE), ("right eye", _RIGHT_EYE)):
                        box = _eye_box(face.landmark, indices)
                        if box is None:
                            continue
                        found.append(
                            Detection(
                                label=label,
                                category="eye",
                                confidence=0.6,
                                bbox=box,
                                source=self.name,
                            )
                        )
        finally:
            mesh.close()
        return found
