"""Optional MediaPipe face and eye detector.

The legacy ``mp.solutions`` API and the newer Tasks API do not ship the same
call surface. This module uses whichever one is importable. Eye boxes come
from Face Mesh landmarks when that solution exists; otherwise only the face
box is returned. Install MediaPipe with ``requirements-ai.txt``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ai_dt.errors import DetectorUnavailable
from ai_dt.schema import Detection

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
        raise DetectorUnavailable(
            "This MediaPipe build has no mp.solutions face detector. "
            "Use the heuristic or YOLO detector, or install a MediaPipe "
            "build that still ships the solutions API."
        )

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
