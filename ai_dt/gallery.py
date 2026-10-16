"""Pictures of every look, and the pick the panel and the preview share."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from PIL import Image

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_RENDER_LOCK = threading.Lock()


def work_is_gallery(path: Path) -> bool:
    """True when this folder is an editor work dir with a recipe in it."""

    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    if "ai-editor-work" not in resolved.parts:
        return False
    return (resolved / "output.json").is_file()


def list_cards(recipe: dict) -> list[dict]:
    """Every choice on the sheet, then Accept and Skip when a feature is waiting."""

    cards: list[dict] = []
    for item in recipe.get("choices") or []:
        if not isinstance(item, dict):
            continue
        identity = str(item.get("id") or "").strip()
        if not _SAFE_ID.match(identity):
            continue
        cards.append(
            {
                "id": identity,
                "title": str(item.get("title") or identity),
                "why": str(item.get("why") or ""),
                "applied": bool(item.get("applied")),
                "kind": "look",
            }
        )
    waiting = str(recipe.get("features_next") or "").strip()
    if waiting:
        title = str(recipe.get("features_next_title") or waiting).strip() or waiting
        cards.append(
            {
                "id": "accept-feature",
                "title": f"Accept {title}",
                "why": "Paint this feature onto the edit.",
                "applied": False,
                "kind": "feature",
            }
        )
        cards.append(
            {
                "id": "skip-feature",
                "title": f"Skip {title}",
                "why": "Leave this feature off and show the one after it.",
                "applied": False,
                "kind": "feature",
            }
        )
    return cards


def load_cards(work: Path) -> list[dict]:
    """Cards for a work folder. Raises when the folder is not a gallery."""

    recipe = _load_recipe(work)
    return list_cards(recipe)


def write_pick(
    work: Path,
    look_id: str,
    *,
    source: str,
    title: str = "",
    applied: bool = False,
) -> dict:
    """Write the shared pick. A later write replaces an earlier one."""

    identity = str(look_id or "").strip()
    if not _SAFE_ID.match(identity):
        raise ValueError("That look id cannot be stored")
    origin = "editor" if source == "editor" else "preview"
    record = {
        "id": identity,
        "title": str(title or identity)[:80],
        "source": origin,
        "applied": bool(applied),
        "at": time.time_ns(),
    }
    folder = Path(work)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "picked.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record) + "\n", encoding="utf-8")
    temporary.replace(target)
    return record


def read_pick(work: Path) -> dict | None:
    """The latest pick, or None when the file is missing or unreadable."""

    path = Path(work) / "picked.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    identity = str(payload.get("id") or "").strip()
    if not _SAFE_ID.match(identity):
        return None
    payload["id"] = identity
    payload["applied"] = bool(payload.get("applied"))
    payload["source"] = "editor" if payload.get("source") == "editor" else "preview"
    return payload


def wait_for_pick(work: Path, *, since: int = 0, timeout: float = 30) -> int:
    """Return 0 when a newer unapplied pick appears, or 2 when time runs out."""

    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        pick = read_pick(work)
        if pick is not None and int(pick.get("at") or 0) > int(since) and not pick.get("applied"):
            return 0
        if time.monotonic() >= deadline:
            return 2
        time.sleep(0.05)


def render_option(work: Path, source: Path, look_id: str) -> Path:
    """A JPEG of one look on the developed picture.

    The card starts from the photo already open and adds only that look's
    grade. It does not repaint the camera file or the background blur.
    """

    identity = str(look_id or "").strip()
    if not _SAFE_ID.match(identity):
        raise ValueError("That look id cannot be painted")
    folder = Path(work)
    recipe = _load_recipe(folder)
    destination = folder / "options" / identity
    destination.mkdir(parents=True, exist_ok=True)
    jpeg = destination / "look.jpg"
    parent_json = folder / "output.json"
    with _RENDER_LOCK:
        if jpeg.is_file() and jpeg.stat().st_mtime >= parent_json.stat().st_mtime:
            return jpeg
        base = folder / "preview.jpg"
        if not base.is_file():
            base = Path(source)
        with Image.open(base) as handle:
            image = handle.convert("RGB")
        graded = _grade_look(recipe, identity, image)
        graded.thumbnail((960, 960))
        destination.mkdir(parents=True, exist_ok=True)
        graded.save(jpeg, format="JPEG", quality=85)
        os.utime(jpeg, None)
    return jpeg


def _grade_look(recipe: dict, look_id: str, image: Image.Image) -> Image.Image:
    """The developed picture plus the grade this look would add."""

    from ai_dt.choices import apply_named_look
    from ai_dt.pipeline import apply_expert_requests
    from ai_dt.schema import Atmosphere
    from ai_dt.simulate import approximate, vignette_from_modules

    raw = {}
    analysis = recipe.get("analysis") if isinstance(recipe.get("analysis"), dict) else {}
    if isinstance(analysis.get("atmosphere"), dict):
        raw = analysis["atmosphere"]
    fields = {name: raw[name] for name in Atmosphere.__dataclass_fields__ if name in raw}
    atmosphere = Atmosphere(**fields)
    if look_id not in {"", "faithful"}:
        plan = apply_named_look({"asks": [], "skip": []}, look_id)
        atmosphere = apply_expert_requests(atmosphere, plan)
    vignette = vignette_from_modules(list(recipe.get("modules") or []))
    return approximate(image, atmosphere, vignette)


def _load_recipe(work: Path) -> dict:
    if not work_is_gallery(work):
        raise ValueError("That folder is not an editor work folder")
    payload = json.loads((Path(work).resolve() / "output.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("output.json is not a recipe")
    return payload
