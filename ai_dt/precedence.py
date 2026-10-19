"""One order for the stage plan.

The user's sentence wins. A subject that was actually found keeps the
stages that belong to it, unless the user skipped them. The model comes
after that. The guide supplied the starting list.
"""

from __future__ import annotations

from ai_dt.catalog import apply_operations, operation_features
from ai_dt.choices import apply_named_look
from ai_dt.features import _install_history_steps, cast_operations, hue_operation, select_mask_features
from ai_dt.review import review_plan
from ai_dt.runbook import lock_found_stages
from ai_dt.steer import apply_proposal, revise_plan


def resolve_plan(
    plan: dict,
    *,
    proposal: dict,
    steer: str,
    look: str,
    feature_look: str,
    picture: object,
    reading: object,
    detections: list,
    scene: str = "",
) -> dict:
    """Return the plan the renderer should paint."""

    chosen = proposal if isinstance(proposal, dict) else {}
    resolved = apply_proposal(plan, chosen)
    if str(steer or "").strip():
        resolved = revise_plan(resolved, steer)
    if str(look or "").strip():
        resolved = apply_named_look(resolved, look)
    resolved = review_plan(resolved, picture, reading, steer, scene)
    sentence = str(getattr(reading, "sentence", "") or "")
    if str(scene or "").strip():
        sentence = f"{sentence} {scene}".strip()
    operations = [item for item in (chosen.get("operations") or []) if isinstance(item, dict)]
    hue = hue_operation(steer, sentence)
    if hue is not None and not any(str(item.get("id") or "") == "colorzones" for item in operations):
        operations = [*operations, hue]
    operations = cast_operations(operations, steer)
    resolved = apply_operations(resolved, operations, list(chosen.get("dropped") or []))
    resolved = lock_found_stages(resolved, picture, detections)
    features = operation_features(operations)
    if features:
        merged = dict(chosen)
        existing = [item for item in (merged.get("features") or []) if isinstance(item, dict)]
        merged["features"] = existing + features
        chosen = merged
    resolved = select_mask_features(
        resolved,
        chosen,
        detections,
        steer,
        sentence,
        feature_look,
    )
    return _install_history_steps(resolved, operations)
