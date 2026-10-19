"""A plan without a planner: guide steps plus request keywords.

With no Gemini key, no local model, or a Gemini call that failed, the
reasoner returns ``skipped``. The guide still knows what suits the picture
type, and the request grammar (``mask_grammar``) still reads plain asks such
as "make the water darker". Both become catalog operations, so the
Accept/Skip queue fills the same way a model plan does. Deterministic: the
same picture type and words give the same steps.
"""

from __future__ import annotations

import re

_EMPHASIS = re.compile(
    r"main (?:theme|subject|focus)|stand out|stands out|\bpop\b|focus|emphasi|hero|"
    r"draw the eye|prominent|the star\b|highlight the|isolate",
    re.IGNORECASE,
)


def planner_missing(proposal: object) -> bool:
    """True when no planner answered: skipped, failed, or nothing usable."""

    if not isinstance(proposal, dict):
        return True
    if str(proposal.get("status") or "") in {"skipped", "failed", "error"}:
        return True
    return not any(proposal.get(key) for key in ("stages", "features", "operations"))


def guide_steps(picture_type: str, *, animal: bool, ground: bool) -> list[str]:
    if animal:
        return ["mask_animal", "mask_animal_background"] + (["mask_bottom_gradient"] if ground else []) + ["mask_edge_burn"]
    from ai_dt.lessons import PRESETS

    return list(PRESETS.get(str(picture_type or ""), ()))


def offline_operations(
    steer: str,
    *,
    picture_type: str = "",
    animal: bool = False,
    ground: bool = False,
    first: bool = True,
) -> list[dict]:
    """Operations for the accept queue, request steps first.

    The guide steps join on a first plan, on a blank request, or when the
    request asks for emphasis ("make the duck the main theme"). A specific
    follow-up such as "make the water darker" keeps only its own step.
    """

    from ai_dt.mask_grammar import mask_operations

    found: list[dict] = []
    seen: set[str] = set()
    for item in mask_operations(steer):
        if item["id"] not in seen:
            seen.add(item["id"])
            found.append(dict(item))
    text = str(steer or "")
    emphasis = bool(_EMPHASIS.search(text))
    if first or not text.strip() or emphasis:
        reason = ""  # the step keeps its own title
        for step_id in guide_steps(picture_type, animal=animal, ground=ground):
            if step_id in seen:
                continue
            seen.add(step_id)
            found.append({"id": step_id, "action": "include", "reason": reason})
    return found


def apply_offline(plan: dict, operations: list[dict]) -> dict:
    """Put the offline operations on the accept queue (same path as a model plan)."""

    from ai_dt.catalog import apply_operations
    from ai_dt.features import _install_history_steps

    if not operations:
        return plan
    plan = apply_operations(plan, operations, [])
    plan = _install_history_steps(plan, operations)
    plan["offline_plan"] = True
    if str(plan.get("feature_note") or "").startswith(("This sentence could not", "This sentence did not")):
        plan["feature_note"] = ""
    return plan
