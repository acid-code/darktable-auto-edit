"""Shared records passed between detectors, advisors, and the style writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Detection:
    """One subject region, in normalized top-left coordinates.

    ``bbox`` is ``(x, y, width, height)`` with each value in the range 0 to 1.
    ``category`` is one of ``face``, ``eye``, ``person``, ``animal``,
    ``architecture``, or ``subject``.
    """

    label: str
    category: str
    confidence: float
    bbox: tuple[float, float, float, float]
    source: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = [round(float(v), 5) for v in self.bbox]
        data["confidence"] = round(float(self.confidence), 4)
        return data


@dataclass
class Atmosphere:
    """Global corrections suggested for the developed preview.

    Numeric fields are deltas, not a replacement of the existing history.
    ``temperature_shift`` is positive to warm the photo and negative to cool it.
    ``tint`` is positive toward magenta and negative toward green.
    ``shadows`` and ``highlights`` are brilliance moves: positive lifts.
    """

    exposure_ev: float = 0.0
    contrast: float = 0.0
    temperature_shift: float = 0.0
    tint: float = 0.0
    saturation: float = 0.0
    shadows: float = 0.0
    highlights: float = 0.0
    vibrance: float = 0.0
    black: float = 0.0
    mood: str = ""
    rationale: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = round(value, 4)
        return data
