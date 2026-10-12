"""Advisor interface."""

from __future__ import annotations

from typing import Protocol

from PIL import Image

from ai_dt.schema import Atmosphere


class Advisor(Protocol):
    name: str

    def advise(self, image: Image.Image) -> Atmosphere:
        """Return global correction deltas for this developed preview."""
