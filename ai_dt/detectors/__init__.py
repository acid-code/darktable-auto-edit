"""Detector registry. Add a new backend by importing it here."""

from __future__ import annotations

from ai_dt.detectors.heuristic import HeuristicDetector
from ai_dt.detectors.mediapipe_faces import MediaPipeFaceDetector
from ai_dt.detectors.merge import merge_detections
from ai_dt.detectors.yolo import YoloDetector

__all__ = [
    "HeuristicDetector",
    "MediaPipeFaceDetector",
    "YoloDetector",
    "merge_detections",
]
