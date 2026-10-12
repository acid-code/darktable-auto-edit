"""Detector interface."""

from __future__ import annotations

from typing import Protocol

from PIL import Image

from ai_dt.schema import Detection


class Detector(Protocol):
    name: str

    def detect(self, image: Image.Image) -> list[Detection]:
        """Return subject regions. Coordinates are normalized to the full frame."""
