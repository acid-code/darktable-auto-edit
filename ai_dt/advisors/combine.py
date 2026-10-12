"""Blend a measured atmosphere with an optional model suggestion."""

from __future__ import annotations

from ai_dt.schema import Atmosphere

_NUMERIC = (
    "exposure_ev",
    "contrast",
    "temperature_shift",
    "tint",
    "saturation",
    "shadows",
    "highlights",
    "vibrance",
)

_FINAL_LIMITS = {
    "exposure_ev": (-2.0, 2.0),
    "contrast": (-0.45, 0.45),
    "temperature_shift": (-1.0, 1.0),
    "tint": (-1.0, 1.0),
    "saturation": (-0.4, 0.4),
    "shadows": (-0.45, 0.45),
    "highlights": (-0.45, 0.45),
    "vibrance": (-0.35, 0.35),
}


def combine_atmospheres(
    measured: Atmosphere,
    suggestion: Atmosphere | None,
    suggestion_weight: float = 0.35,
) -> Atmosphere:
    """Keep the histogram in charge and let the model steer within that.

    ``black`` is never taken from the model. It stays a measured value.
    """

    if suggestion is None:
        combined = Atmosphere(**measured.to_dict())
        combined.source = measured.source
        return combined

    weight = max(0.0, min(1.0, suggestion_weight))
    values = {}
    for key in _NUMERIC:
        measured_value = float(getattr(measured, key))
        suggested_value = float(getattr(suggestion, key))
        mixed = (1.0 - weight) * measured_value + weight * suggested_value
        low, high = _FINAL_LIMITS[key]
        values[key] = max(low, min(high, mixed))

    mood = suggestion.mood.strip() or measured.mood
    rationale_parts = [part for part in (measured.rationale, suggestion.rationale) if part]
    return Atmosphere(
        black=measured.black,
        mood=mood,
        rationale=" ".join(rationale_parts),
        source=f"{measured.source}+{suggestion.source}",
        **values,
    )
