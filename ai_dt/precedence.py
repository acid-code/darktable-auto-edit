"""One order for the stage plan.

The user's sentence wins. A subject that was actually found keeps the
stages that belong to it, unless the user skipped them. The model comes
after that. The guide supplied the starting list.
"""

from __future__ import annotations

from ai_dt.choices import apply_named_look
from ai_dt.features import select_mask_features
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
) -> dict:
    """Return the plan the renderer should paint."""

    chosen = proposal if isinstance(proposal, dict) else {}
    resolved = apply_proposal(plan, chosen)
    if str(steer or "").strip():
        resolved = revise_plan(resolved, steer)
    if str(look or "").strip():
        resolved = apply_named_look(resolved, look)
    resolved = review_plan(resolved, picture, reading, steer)
    resolved = lock_found_stages(resolved, picture, detections)
    sentence = str(getattr(reading, "sentence", "") or "")
    return select_mask_features(
        resolved,
        chosen,
        detections,
        steer,
        sentence,
        feature_look,
    )
