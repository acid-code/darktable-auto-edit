"""One version record per Darktable duplicate.

The work folder is ``<root>/<image id>/v<duplicate index>/``. A second
duplicate of the same file gets its own folder, so a later steer reads that
duplicate's plan instead of the one overwritten on the original.
"""

from __future__ import annotations

import json
from pathlib import Path


def version_work_dir(root: Path, image_id: int, duplicate_index: int) -> Path:
    """Directory for one image id and one duplicate index."""

    return Path(root) / str(int(image_id)) / f"v{int(duplicate_index)}"


def version_sentence(recipe: dict) -> str:
    """One sentence: the look, what stayed, what was set aside, and the plan."""

    features = [str(item) for item in (recipe.get("features") or []) if str(item).strip()]
    kept = "Kept " + ", ".join(features) + "." if features else "Kept the frame."
    aside_titles = []
    reasoning = recipe.get("reasoning") if isinstance(recipe.get("reasoning"), dict) else {}
    for item in reasoning.get("set_aside") or []:
        if isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            if title:
                aside_titles.append(title)
    aside = (
        "Set aside " + ", ".join(aside_titles) + "."
        if aside_titles
        else "Nothing set aside."
    )
    match = recipe.get("recipe_match") if isinstance(recipe.get("recipe_match"), dict) else {}
    title = str(match.get("title") or "").strip() or "Faithful"
    plan = recipe.get("plan") if isinstance(recipe.get("plan"), dict) else {}
    plan_text = str(plan.get("label") or plan.get("text") or "").strip()
    sentence = f"{title}. {kept} {aside}"
    if plan_text:
        sentence = f"{sentence} {plan_text}"
    sentence = " ".join(sentence.split())
    if len(sentence) > 240:
        sentence = sentence[:237].rstrip() + "..."
    return sentence


def build_version_record(
    recipe: dict,
    directory: Path,
    *,
    image_id: int | None,
    duplicate_index: int | None,
    parent_id: int | None,
    parent_index: int | None,
    look_id: str,
) -> dict:
    """The JSON object stored as ``version.json`` for this duplicate."""

    directory = Path(directory)
    plan = recipe.get("plan") if isinstance(recipe.get("plan"), dict) else {}
    modules = []
    for item in recipe.get("modules") or []:
        if not isinstance(item, dict):
            continue
        modules.append(
            {
                "operation": str(item.get("operation") or ""),
                "multi_name": str(item.get("multi_name") or ""),
            }
        )
    parent = None
    if parent_id is not None:
        parent = {
            "image_id": int(parent_id),
            "duplicate_index": int(parent_index or 0),
        }
    sentence = version_sentence(recipe)
    return {
        "version": 1,
        "image_id": None if image_id is None else int(image_id),
        "duplicate_index": None if duplicate_index is None else int(duplicate_index),
        "parent": parent,
        "look_id": look_id or "faithful",
        "plan_text": str(plan.get("text") or ""),
        "sentence": sentence,
        "modules": modules,
        "work": {
            "dir": str(directory),
            "output": str(directory / "output.json"),
            "style": str(directory / "edit.dtstyle"),
            "tiff": str(directory / "edited.tif"),
        },
    }


def write_version(directory: Path, record: dict) -> Path:
    path = Path(directory) / "version.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
