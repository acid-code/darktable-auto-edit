"""The six passes a finished edit has to complete.

Pass 1 names the picture. Pass 2 is the soft frame grade. Passes 3 to 5
build the keep-mask, mark the theme, and quiet what competes. Pass 6 is
an expert module only when its amount is already non-zero. A look is not
offered when a required stage never landed.
"""

from __future__ import annotations

import json
from pathlib import Path

# Local falloff belongs on these types. A landscape, a building, and a
# night frame do not get a subject vignette from the picture type. A person
# or an animal found in that frame is the separate feature instead.
FALLOFF_TYPES = frozenset(
    {"portrait", "group", "animal", "street", "food", "still-life", "macro"}
)

# The text model may not skip these when the subject was actually found.
# A skip the user typed still wins.
_LOCKED = {
    "portrait": frozenset({"subject", "eyes", "skin"}),
    "group": frozenset({"subject", "eyes", "skin"}),
    "animal": frozenset({"subject", "eyes"}),
}

_REQUIRED = {
    "portrait": ("subject", "eyes", "skin", "background"),
    "group": ("subject", "eyes", "skin", "background"),
    "animal": ("subject", "eyes", "background"),
    "street": ("subject", "background"),
    "food": ("subject", "background"),
    "still-life": ("subject", "background"),
    "macro": ("subject", "background"),
}


def subject_found(picture: object, detections: list) -> bool:
    """True when a box for this type's subject is already in hand."""

    picture_type = str(getattr(picture, "type", "") or "")
    categories = {str(getattr(item, "category", "") or "") for item in detections}
    box = getattr(picture, "subject_box", None)
    if picture_type in {"portrait", "group", "street"}:
        return bool(categories & {"face", "person", "eye"}) or box is not None
    if picture_type == "animal":
        return "animal" in categories or box is not None
    if picture_type in {"food", "still-life", "macro"}:
        return box is not None or bool(categories & {"subject", "food", "object"})
    return False


def lock_found_stages(plan: dict, picture: object, detections: list) -> dict:
    """Drop a reasoner skip of subject, eyes, or skin when that subject was found."""

    locked = _LOCKED.get(str(getattr(picture, "type", "") or ""), frozenset())
    if not locked or not subject_found(picture, detections):
        return plan
    skip_by = dict(plan.get("skip_by") or {})
    kept: list[str] = []
    removed: set[str] = set()
    for stage in plan.get("skip") or []:
        name = str(stage)
        if name in locked and skip_by.get(name) != "user":
            removed.add(name)
            continue
        kept.append(name)
    if not removed:
        return plan
    plan["skip"] = kept
    plan["proposed"] = [
        item
        for item in (plan.get("proposed") or [])
        if not (
            isinstance(item, dict)
            and item.get("action") == "skip"
            and str(item.get("stage") or "") in removed
        )
    ]
    plan["kept_stages"] = sorted(removed)
    return plan


def assess_run(picture_type: str, separation: dict) -> dict:
    """Whether this type's required stages landed on the rendered picture."""

    ran = {str(stage) for stage in (separation.get("stages_ran") or [])}
    skipped = {str(stage) for stage in (separation.get("skipped") or [])}
    name = str(picture_type or "general")
    missing: list[str] = []
    if name == "landscape":
        if "sky" not in ran and "horizon" not in ran:
            missing = ["sky or horizon"]
    elif name == "architecture":
        if "subject" in ran or separation.get("background") == "defocus":
            missing = ["structure"]
    elif name == "night":
        if "highlights" not in ran and "highlights" not in skipped:
            missing = ["highlights"]
    elif name not in {"general", ""}:
        required = _REQUIRED.get(name, ())
        missing = [stage for stage in required if stage not in ran]
    finished = not missing
    if finished:
        done = ", ".join(str(stage) for stage in (separation.get("stages_ran") or [])) or "grade"
        sentence = f"Finished {done}."
    else:
        sentence = "Not finished. Missing " + ", ".join(missing) + "."
    return {"finished": finished, "missing": missing, "sentence": sentence}


def original_behind_edit(path: Path, output_dir: Path | None = None) -> Path | None:
    """The camera file a follow-up should reread when the open image is a masked TIFF.

    ``edited.tif`` is the working render. A copy under ``opened/`` is the
    file Darktable imported. Both sit next to the ``output.json`` that
    recorded the camera path.
    """

    path = Path(path)
    recipes = []
    if output_dir is not None:
        recipes.append(Path(output_dir) / "output.json")
    recipes.append(path.parent / "output.json")
    if path.parent.name.lower() == "opened":
        recipes.append(path.parent.parent / "output.json")
    for recipe_path in recipes:
        original = _recorded_original(recipe_path, path)
        if original is not None:
            return original
    return None


def _recorded_original(recipe_path: Path, edited: Path) -> Path | None:
    if not recipe_path.is_file():
        return None
    try:
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    recorded = str((payload.get("input") or {}).get("path") or "")
    if not recorded:
        return None
    original = Path(recorded)
    try:
        if original.is_file() and original.resolve() != edited.resolve():
            return original
    except OSError:
        return None
    return None
