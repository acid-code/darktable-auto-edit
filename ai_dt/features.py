"""Mask features chosen for one frame.

``separate`` is committed only when a person or an animal box was found.
The other four are tools. A run paints one of them after it is accepted.
A preview can show the next one first. Pixel measurements do not invent
a path, a gap, or a trunk.
"""

from __future__ import annotations

import json
import re

FEATURE_IDS = ("separate", "foreground", "opening", "shafts", "highlights", "eyes")
OPTIONAL_IDS = ("foreground", "opening", "shafts", "highlights", "eyes")
FEATURE_TITLES = {
    "separate": "Separate the subject",
    "foreground": "Darken the foreground",
    "opening": "Darken the bright gap",
    "shafts": "Darken the shafts",
    "highlights": "Warm the highlights",
    "subject": "Lift the subject",
    "eyes": "Eyes",
    "recolor": "Recolor",
}
# Drawn-mask steps (ai_dt/mask_steps.py) carry their own titles.
from ai_dt.mask_steps import EXPANDS as _MASK_EXPANDS, STEPS as _MASK_STEPS  # noqa: E402

FEATURE_TITLES.update({ident: step.title for ident, step in _MASK_STEPS.items()})
FEATURE_TITLES["mask_landscape"] = "Grade each part of the landscape"
FEATURE_TITLES.update(
    {
        "split_tone": "Split tone: warm highlights, cool shadows",
        "palander": "Warm-only look",
        "keep_color": "Keep only one color",
    }
)
_STEP_FROM_STAGE = {
    "subject": "subject",
    "eyes": "eyes",
    "eye": "eyes",
    "foreground": "foreground",
    "surround": "foreground",
    "background": "foreground",
    "opening": "opening",
    "shafts": "shafts",
    "highlights": "highlights",
}
_BODY = frozenset({"face", "person", "animal"})
_ASKS = frozenset({"darken", "lift", "hold"})
_LESS = frozenset({"less on the background", "a little less on the background"})
_MORE = frozenset({"more on the background", "a little more on the background"})


def body_found(detections: list) -> bool:
    """A person or an animal box, including a face. A window is not a body."""

    categories = {str(getattr(item, "category", "") or "") for item in detections}
    return bool(categories & _BODY)


def feature_title(feature_id: str) -> str:
    return FEATURE_TITLES.get(str(feature_id or ""), str(feature_id or ""))


def _clauses(sentence: str) -> list[str]:
    text = " ".join(str(sentence or "").lower().split())
    if not text:
        return []
    parts = re.split(r"[,.;!?]|\band\b", text)
    return [" ".join(part.split()) for part in parts if part.strip()]


def background_amount(steer: str) -> float | None:
    """Rescale the feature that just landed. Known blur phrases stay elsewhere."""

    for clause in _clauses(steer):
        if clause in _LESS:
            return 0.7
        if clause in _MORE:
            return 1.35
    return None


def _user_lift_steps(plan: dict, skipped: list[str], done: list[str]) -> list[dict]:
    """Subject and eye lifts the user named, in the order the sentence said them."""

    steps: list[dict] = []
    blocked = set(skipped) | set(done)
    for item in plan.get("asks") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("source") or "") not in {"user", ""}:
            continue
        verb = str(item.get("verb") or item.get("action") or "")
        target = str(item.get("target") or item.get("stage") or "")
        if verb != "lift" or target not in {"subject", "eyes"} or target in blocked:
            continue
        if any(str(step.get("id") or "") == target for step in steps):
            continue
        title = str(item.get("reason") or "").strip() or feature_title(target)
        steps.append({"id": target, "action": "lift", "title": title})
    return steps


def select_mask_features(
    plan: dict,
    proposal: dict | None,
    detections: list,
    steer: str,
    reading_sentence: str,
    feature_look: str,
) -> dict:
    """Commit separate when a body exists. Leave every other feature as a preview.

    Accept paints the suggestion that was waiting. Skip moves on without it.
    An ask the grammar already understands, such as darken the foreground,
    becomes that preview. An open sentence is not guessed from keywords.
    """

    from ai_dt.review import review_feature_path

    committed = [name for name in (plan.get("features_committed") or []) if name in FEATURE_IDS]
    skipped = [
        name
        for name in (plan.get("features_skipped") or [])
        if name in FEATURE_IDS or name == "subject"
    ]
    done = [name for name in (plan.get("features_done") or []) if name == "subject" or name in FEATURE_IDS]
    if body_found(detections) and "separate" not in committed:
        committed.append("separate")

    waiting = str(plan.get("features_next") or "")
    if feature_look == "accept-feature" and waiting == "subject" and "subject" not in done:
        done.append("subject")
    elif feature_look == "accept-feature" and waiting in OPTIONAL_IDS and waiting not in committed:
        committed.append(waiting)
    elif feature_look == "skip-feature" and waiting in OPTIONAL_IDS and waiting not in skipped:
        skipped.append(waiting)
    elif feature_look == "skip-feature" and waiting == "subject":
        skipped.append("subject")

    asked: list[str] = []
    for item in plan.get("asks") or []:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or item.get("target") or "")
        action = str(item.get("action") or item.get("verb") or "")
        if stage in OPTIONAL_IDS and action in _ASKS and stage not in asked:
            asked.append(stage)
    skip_by = plan.get("skip_by") if isinstance(plan.get("skip_by"), dict) else {}
    for stage in plan.get("skip") or []:
        name = str(stage)
        if name in OPTIONAL_IDS and skip_by.get(name) == "user" and name not in skipped:
            skipped.append(name)

    reviewed = review_feature_path(
        (proposal or {}).get("features") or [],
        steer,
        reading_sentence,
    )
    if feature_look in {"accept-feature", "skip-feature"}:
        queue = [item for item in (plan.get("step_queue") or []) if isinstance(item, dict)]
        if queue and str(queue[0].get("id") or "") == waiting:
            queue = queue[1:]
        else:
            queue = [item for item in queue if str(item.get("id") or "") != waiting]
        proposed_ids: list[str] = []
    else:
        queue = _steps_from_proposal(proposal, steer, reading_sentence)
        proposed_ids = [str(item.get("id") or "") for item in queue]
    user_steps = _user_lift_steps(plan, skipped, done)
    if feature_look not in {"accept-feature", "skip-feature"} and not queue and user_steps:
        queue = user_steps
    path: list[str] = [str(item.get("id") or "") for item in queue if str(item.get("id") or "")]
    for name in proposed_ids + asked + reviewed:
        if name in OPTIONAL_IDS and name not in path and name not in committed and name not in skipped:
            if name in done and name not in proposed_ids:
                continue
            path.append(name)
            if not any(str(item.get("id") or "") == name for item in queue):
                queue.append({"id": name, "action": "lift", "title": feature_title(name)})

    amount = background_amount(steer)
    note = ""
    lift = _grounded_subject_lift(proposal, steer, reading_sentence)
    if amount is not None and committed:
        last = committed[-1]
        scales = dict(plan.get("scales") or {})
        current = float(scales.get(last) or 1.0)
        scales[last] = max(0.25, min(1.5, current * amount))
        plan["scales"] = scales
    elif _unmapped(steer, plan, asked, reviewed) and not lift:
        note = "This sentence could not be mapped, so nothing extra was painted."
    elif not path and "separate" not in committed and not str(steer or "").strip() and not lift:
        note = "No person or animal was found, and no mask feature was chosen, so this stays the measured grade."

    nxt = str(queue[0]["id"]) if queue else (path[0] if path else "")
    reopen = (
        feature_look == ""
        and bool(str(steer or "").strip())
        and (lift or _another_subject_lift(proposal, steer))
        and "subject" not in skipped
        and nxt not in OPTIONAL_IDS
    )
    if nxt in done:
        done = [name for name in done if name != nxt]
    if not nxt and lift and "subject" not in skipped and "subject" not in done:
        nxt = "subject"
    elif reopen:
        # "Another lift" read like a repeat on a first plan. Say it only when
        # a subject lift is already on the photo, and keep a flag for Accept.
        again = "subject" in done or "subject" in [str(name) for name in (plan.get("features_done") or [])]
        done = [name for name in done if name != "subject"]
        nxt = "subject"
        plan["subject_again"] = again
        note = (
            "A second, gentler subject lift is planned, because this request asks for more on the subject."
            if again
            else ""
        )
    elif feature_look == "accept-feature" and not nxt and waiting == "subject":
        note = "The subject lift is now on this photo."
    elif feature_look == "accept-feature" and not nxt and waiting:
        note = "This step is on this photo."
    elif "subject" in done and not nxt and not str(steer or "").strip():
        note = "The subject lift is now on this photo."
    elif str(steer or "").strip() and not nxt and not note:
        note = "This sentence did not become a step."
    plan["features_committed"] = committed
    plan["features_skipped"] = skipped
    plan["features_done"] = done
    if nxt == "subject" and not any(str(item.get("id") or "") == "subject" for item in queue):
        queue.insert(0, {"id": "subject", "action": "lift", "title": feature_title("subject")})
    elif nxt and queue and str(queue[0].get("id") or "") != nxt:
        queue.insert(0, {"id": nxt, "action": "lift", "title": feature_title(nxt)})
    current = queue[0] if queue else {}
    plan["step_queue"] = queue
    plan["features_path"] = [str(item.get("id") or "") for item in queue if str(item.get("id") or "")]
    plan["features_next"] = str(current.get("id") or "")
    plan["features_next_title"] = str(current.get("title") or "")
    plan["features_next_action"] = str(current.get("action") or "")
    plan["feature_note"] = note
    return plan


def _load_recipe(output_dir):
    import json
    from pathlib import Path

    directory = Path(output_dir)
    path = directory / "output.json"
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise ValueError("output.json is not a plan")
    plan = recipe.get("plan") if isinstance(recipe.get("plan"), dict) else {}
    recipe["plan"] = plan
    return directory, path, recipe, plan


def _save_recipe(path, recipe) -> None:
    import json

    from ai_dt.status_text import accept_skip_lines, panel_status

    recipe["chat"] = True
    recipe["offer_looks"] = False
    recipe["choices"] = []
    recipe["status_text"] = panel_status(recipe)
    recipe["accept_skip"] = accept_skip_lines(recipe)
    path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")


def _promote(plan: dict, recipe: dict, leaving: str) -> None:
    queue = [item for item in (plan.get("step_queue") or []) if isinstance(item, dict)]
    if queue and str(queue[0].get("id") or "") == leaving:
        queue = queue[1:]
    else:
        queue = [item for item in queue if str(item.get("id") or "") != leaving]
    follow = queue[0] if queue else {}
    plan["step_queue"] = queue
    plan["features_path"] = [str(item.get("id") or "") for item in queue if str(item.get("id") or "")]
    plan["features_next"] = str(follow.get("id") or "")
    recipe["features_next"] = str(follow.get("id") or "")
    recipe["features_next_title"] = str(follow.get("title") or "")
    recipe["features_next_action"] = str(follow.get("action") or "")


def _refresh_handoff(recipe: dict) -> None:
    if recipe.get("features_next"):
        sentence = "This plan starts from the photo you are viewing. Nothing is added until you press Accept."
    else:
        sentence = "This plan starts from the photo you are viewing."
    recipe["handoff"] = {"sentence": sentence}


def _drop_undone_steps(plan: dict) -> None:
    plan["step_log"] = [
        item
        for item in (plan.get("step_log") or [])
        if isinstance(item, dict) and item.get("active")
    ]


def _remember_applied_step(plan: dict, recipe: dict, feature_id: str, name: str) -> None:
    if not feature_id or not name:
        return
    log = [item for item in (plan.get("step_log") or []) if isinstance(item, dict)]
    log.append(
        {
            "id": feature_id,
            "name": name,
            "title": str(recipe.get("features_next_title") or feature_id),
            "action": str(recipe.get("features_next_action") or "lift"),
            "active": True,
        }
    )
    plan["step_log"] = log


def _queue_from_log(plan: dict, recipe: dict) -> None:
    log = [item for item in (plan.get("step_log") or []) if isinstance(item, dict)]
    done = [str(item.get("id") or "") for item in log if item.get("active") and str(item.get("id") or "")]
    seen = set(done)
    queue: list[dict] = []
    for item in log:
        ident = str(item.get("id") or "")
        if item.get("active") or not ident or ident in seen:
            continue
        queue.append(
            {
                "id": ident,
                "action": str(item.get("action") or "lift"),
                "title": str(item.get("title") or ident),
            }
        )
        seen.add(ident)
    for item in plan.get("step_queue") or []:
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or "")
        if not ident or ident in seen:
            continue
        queue.append(item)
        seen.add(ident)
    follow = queue[0] if queue else {}
    plan["features_done"] = done
    plan["step_queue"] = queue
    plan["features_path"] = [str(item.get("id") or "") for item in queue if str(item.get("id") or "")]
    plan["features_next"] = str(follow.get("id") or "")
    recipe["features_next"] = str(follow.get("id") or "")
    recipe["features_next_title"] = str(follow.get("title") or "")
    recipe["features_next_action"] = str(follow.get("action") or "")


def align_plan_to_history(output_dir, name: str, direction: str) -> dict:
    """Move the plan with the history cursor.

    Back past a step this editor applied makes that step the waiting one.
    Next puts it back among the done steps. A module the photographer
    added does not change the plan.
    """

    _directory, path, recipe, plan = _load_recipe(output_dir)
    log = [item for item in (plan.get("step_log") or []) if isinstance(item, dict)]
    plan["step_log"] = log
    matched = None
    if direction == "back":
        for item in reversed(log):
            if item.get("active") and item.get("name") == name:
                item["active"] = False
                matched = item
                break
    elif direction == "next":
        for item in log:
            if not item.get("active") and item.get("name") == name:
                item["active"] = True
                matched = item
                break
    if matched is None:
        return recipe
    _queue_from_log(plan, recipe)
    title = str(matched.get("title") or matched.get("id") or "This step")
    if direction == "back":
        recipe["feature_note"] = (
            f"Back moved off {title}. Next puts that step back. "
            "Accept writes it again and drops the later history."
        )
    else:
        recipe["feature_note"] = f"Next put {title} back on this photo."
    _refresh_handoff(recipe)
    _save_recipe(path, recipe)
    return recipe


def _plan_hue(recipe: dict) -> str | None:
    plan = recipe.get("plan") if isinstance(recipe.get("plan"), dict) else {}
    parts = [str(recipe.get("features_next_title") or "")]
    for key in ("operations", "step_queue", "step_log"):
        for item in plan.get(key) or []:
            if isinstance(item, dict):
                parts.append(str(item.get("reason") or ""))
                parts.append(str(item.get("title") or ""))
                parts.append(str(item.get("name") or ""))
    return _named_hue(" ".join(parts).lower())


def _names_a_thing(title: str) -> bool:
    from ai_dt.steer import NAMED_SUBJECTS

    words = set(re.findall(r"[a-z]+", str(title or "").lower()))
    return bool(words & NAMED_SUBJECTS)


def _body_boxes(recipe: dict) -> list[tuple[float, float, float, float]]:
    analysis = recipe.get("analysis") if isinstance(recipe.get("analysis"), dict) else {}
    title = str(recipe.get("features_next_title") or "").lower()
    boxes = []
    for item in analysis.get("subjects") or []:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "")
        label = str(item.get("label") or "").lower()
        named = bool(label) and label in title
        if category not in {"animal", "person", "face"} and not (category == "subject" and named):
            continue
        raw = item.get("bbox")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        try:
            box = tuple(float(part) for part in raw)
        except (TypeError, ValueError):
            continue
        if min(box[2], box[3]) <= 0.0:
            continue
        boxes.append(box)
    return boxes


def _overlaps(box: tuple[float, float, float, float], other: tuple[float, float, float, float], pad: float = 0.18) -> bool:
    ax, ay, aw, ah = box
    bx, by, bw, bh = other
    return not (
        bx + bw < ax - pad
        or bx > ax + aw + pad
        or by + bh < ay - pad
        or by > ay + ah + pad
    )


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    x0 = max(0.0, x0 - 0.03)
    y0 = max(0.0, y0 - 0.03)
    x1 = min(1.0, x1 + 0.03)
    y1 = min(1.0, y1 + 0.03)
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def _free_spans(blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not blocked:
        return [(0.0, 1.0)]
    merged: list[list[float]] = []
    for start, end in sorted(blocked):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    free: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            free.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < 1.0:
        free.append((cursor, 1.0))
    return free


def lower_band_ellipse(bodies: list[tuple[float, float, float, float]]) -> dict | None:
    """A shallow ellipse on the bottom, kept off a body that reaches into it."""

    cy = 0.86
    ry = 0.12
    blocked = []
    for box in bodies:
        x, y, width, height = box
        if y + height < cy - ry:
            continue
        blocked.append((max(0.0, x - 0.04), min(1.0, x + width + 0.04)))
    spans = _free_spans(blocked)
    if not spans:
        return None
    start, end = max(spans, key=lambda span: span[1] - span[0])
    width = end - start
    if width < 0.18:
        return None
    cx = start + width / 2.0
    rx = min(width / 2.0, 0.46, cx, 1.0 - cx)
    if rx < 0.08:
        return None
    return {
        "kind": "ellipse",
        "cx": round(cx, 5),
        "cy": cy,
        "rx": round(rx, 5),
        "ry": ry,
        "exposure_ev": -0.18,
        "name": "AI foreground",
    }


def _shape_box(shape: dict) -> tuple[float, float, float, float]:
    return (
        float(shape["cx"]) - float(shape["rx"]),
        float(shape["cy"]) - float(shape["ry"]),
        float(shape["rx"]) * 2.0,
        float(shape["ry"]) * 2.0,
    )


def _open_preview(directory):
    path = directory / "preview.jpg"
    if not path.is_file():
        return None
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _hue_on_preview(recipe: dict, directory) -> dict | None:
    hue = _plan_hue(recipe)
    if not hue:
        return None
    preview = _open_preview(directory)
    if preview is None:
        return None
    from ai_dt.hue_mask import hue_sample

    return hue_sample(preview, hue)


def _subject_outline(recipe: dict, directory, *, again: bool) -> dict | None:
    """A path around the subject from Darktable's SAM model, when it can run.

    A preview under 256px is the small picture the tests draw. The model
    encodes the long side at 1024, so that picture stays on the vignette.
    """

    preview = _open_preview(directory)
    box = _subject_box(recipe, directory)
    if preview is None or box is None or max(preview.size) < 256:
        return None
    from ai_dt.mapping import step_history_name
    from ai_dt.object_segment import outline_box

    points = outline_box(preview, box)
    if not points:
        return None
    name = step_history_name("subject", again=again)
    spec = {
        "kind": "path",
        "name": name,
        "exposure_ev": 0.22,
        "points": [{"x": x, "y": y} for x, y in points],
    }
    path = directory / "mask.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return {"kind": "path", "file": str(path), "name": name}


def _subject_box(recipe: dict, directory) -> tuple[float, float, float, float] | None:
    bodies = _body_boxes(recipe)
    sample = _hue_on_preview(recipe, directory)
    shapes = list(sample["shapes"]) if sample else []
    chosen = list(bodies)
    for shape in shapes:
        box = _shape_box(shape)
        if bodies and any(_overlaps(body, box) for body in bodies):
            chosen.append(box)
        elif not bodies and _named_hue(str(recipe.get("features_next_title") or "")):
            chosen.append(box)
    if not chosen:
        return None
    box = _union(chosen)
    if box[2] <= 0.0 or box[3] <= 0.0:
        return None
    return box


def _frame_size(recipe: dict, directory) -> tuple[int, int] | None:
    preview = _open_preview(directory)
    if preview is not None:
        return preview.size
    try:
        width = int(recipe.get("width") or 0)
        height = int(recipe.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if width > 1 and height > 1:
        return width, height
    return None


def restore_waiting_step(output_dir) -> dict:
    """Put the last accepted step back. Its history write did not land."""

    _directory, path, recipe, plan = _load_recipe(output_dir)
    log = [item for item in (plan.get("step_log") or []) if isinstance(item, dict)]
    left = plan.pop("left_off", None)
    if (not log or not log[-1].get("active")) and isinstance(left, dict) and left.get("id"):
        # The step made no mask (no animal outline, an empty region). It was
        # left off; put it back as the waiting step so the plan does not advance.
        ident = str(left["id"])
        plan["features_skipped"] = [name for name in (plan.get("features_skipped") or []) if name != ident]
        queue = [item for item in (plan.get("step_queue") or []) if isinstance(item, dict) and item.get("id") != ident]
        item = {"id": ident, "action": str(left.get("action") or "lift"), "title": str(left.get("title") or ident)}
        queue.insert(0, item)
        plan["step_queue"] = queue
        plan["features_path"] = [str(entry.get("id") or "") for entry in queue]
        plan["features_next"] = ident
        recipe["features_next"] = ident
        recipe["features_next_title"] = item["title"]
        recipe["features_next_action"] = item["action"]
        recipe["feature_note"] = f"This step was not added: {left.get('reason') or 'it made no mask'} Skip moves past it."
        recipe["plan"] = plan
        _refresh_handoff(recipe)
        _save_recipe(path, recipe)
        return recipe
    if not log or not log[-1].get("active"):
        return recipe
    last = log.pop()
    plan["step_log"] = log
    ident = str(last.get("id") or "")
    done = [str(item) for item in (plan.get("features_done") or []) if str(item).strip()]
    for index in range(len(done) - 1, -1, -1):
        if done[index] == ident:
            del done[index]
            break
    plan["features_done"] = done
    queue = [item for item in (plan.get("step_queue") or []) if isinstance(item, dict)]
    restored = {
        "id": ident,
        "action": str(last.get("action") or "lift"),
        "title": str(last.get("title") or ident),
    }
    if ident and (not queue or str(queue[0].get("id") or "") != ident):
        queue.insert(0, restored)
    plan["step_queue"] = queue
    plan["features_path"] = [str(item.get("id") or "") for item in queue if str(item.get("id") or "")]
    plan["features_next"] = ident
    recipe["features_next"] = ident
    recipe["features_next_title"] = restored["title"]
    recipe["features_next_action"] = restored["action"]
    recipe["feature_note"] = "This step was not written. Accept tries it again."
    _refresh_handoff(recipe)
    _save_recipe(path, recipe)
    return recipe


def _leave_off(directory, path, recipe, plan, nxt: str, reason: str) -> dict:
    """Move past a step that has no region. Do not write a whole-frame module."""

    step_path = directory / "step.dtstyle"
    if step_path.is_file():
        step_path.unlink()
    recipe.pop("step_style", None)
    recipe.pop("step_mask", None)
    skipped = [str(item) for item in (plan.get("features_skipped") or []) if str(item).strip()]
    if nxt and nxt not in skipped:
        skipped.append(nxt)
    plan["features_skipped"] = skipped
    # Kept so --restore-step can put it back: Lua reports a step that wrote
    # nothing as failed, and the plan must not move past it.
    plan["left_off"] = {
        "id": nxt,
        "title": str(recipe.get("features_next_title") or nxt),
        "action": str(recipe.get("features_next_action") or ""),
        "reason": reason,
    }
    if nxt:
        _promote(plan, recipe, nxt)
    recipe["feature_note"] = reason
    _refresh_handoff(recipe)
    _save_recipe(path, recipe)
    return recipe


def commit_waiting_step(output_dir) -> dict:
    """Write one history module for the waiting step. Do not rebuild the look."""

    from ai_dt.drawn_mask import eye_ellipse
    from ai_dt.mapping import step_history_name, step_style
    from ai_dt.modules_extra import colorzones_hue_params

    directory, path, recipe, plan = _load_recipe(output_dir)
    nxt = str(recipe.get("features_next") or "").strip()
    step_path = directory / "step.dtstyle"
    again = bool((recipe.get("plan") or {}).get("subject_again")) or "Another" in str(recipe.get("feature_note") or "")
    title = str(recipe.get("features_next_title") or "")
    if nxt == "subject" and _names_a_thing(title) and _subject_box(recipe, directory) is None:
        return _leave_off(
            directory,
            path,
            recipe,
            plan,
            nxt,
            "No box was found for that subject, so this step was left off.",
        )
    if nxt in _MASK_STEPS:
        from ai_dt.mask_steps import missing_sentence, write_step_mask

        action = str(recipe.get("features_next_action") or "")
        drawn = write_step_mask(recipe, directory, nxt, action=action)
        if drawn is None:
            return _leave_off(directory, path, recipe, plan, nxt, missing_sentence(nxt))
        if step_path.is_file():
            step_path.unlink()
        recipe.pop("step_style", None)
        recipe["step_mask"] = drawn
        _drop_undone_steps(plan)
        _remember_applied_step(plan, recipe, nxt, str(drawn["name"]))
        done = [str(item) for item in (plan.get("features_done") or []) if str(item).strip()]
        if nxt not in done:
            done.append(nxt)
        plan["features_done"] = done
        _promote(plan, recipe, nxt)
        source = str(drawn.get("source") or "")
        found_by = f" The region came from the {source} segmenter." if source else ""
        if recipe["features_next"]:
            recipe["feature_note"] = "This step is on this photo." + found_by
        else:
            recipe["feature_note"] = (
                "This step is on this photo." + found_by + " The plan is finished. "
                "Type another change and press Ask."
            )
        _refresh_handoff(recipe)
        _save_recipe(path, recipe)
        return recipe
    mask = eye_ellipse(list(recipe.get("local_edits") or [])) if nxt == "eyes" else None
    if nxt == "eyes" and mask is None:
        return _leave_off(
            directory,
            path,
            recipe,
            plan,
            nxt,
            "No eye was found, so this step was left off.",
        )
    if nxt == "foreground":
        band = lower_band_ellipse(_body_boxes(recipe))
        if band is None:
            return _leave_off(
                directory,
                path,
                recipe,
                plan,
                nxt,
                "The foreground is not clear of the subject, so this step was left off.",
            )
        mask = band
    if nxt == "subject" and mask is None:
        outline = _subject_outline(recipe, directory, again=again)
        if outline is not None:
            mask = outline
    if (
        nxt == "subject"
        and mask is None
        and _subject_box(recipe, directory) is None
        and (directory / "preview.jpg").is_file()
    ):
        # No box and no outline: never lift the whole frame. Draw the main
        # subject (the animal detector, then the general subject segmenter).
        from ai_dt.mask_steps import write_step_mask

        drawn = write_step_mask(recipe, directory, "mask_animal", action="lift")
        if drawn is None:
            return _leave_off(
                directory,
                path,
                recipe,
                plan,
                nxt,
                "No subject was found on the photo, so this step was left off.",
            )
        mask = drawn
    recoloring = nxt == "recolor" or (nxt == "colorzones" and _cast_hue(title) is not None)
    style_id = "recolor" if recoloring else nxt
    if nxt == "colorzones" and mask is None and not recoloring:
        sample = _hue_on_preview(recipe, directory)
        hue = _named_hue(title.lower()) or _plan_hue(recipe) or "red"
        center = float(sample["center"]) if sample else None
        blob = colorzones_hue_params(hue, center=center) if sample else None
        if not sample or blob is None:
            return _leave_off(
                directory,
                path,
                recipe,
                plan,
                nxt,
                "No color patch was found, so this step was left off.",
            )
        spec = {
            "operation": "colorzones",
            "modversion": 5,
            "name": step_history_name("colorzones", title=title),
            "params": blob.hex(),
            "shapes": sample["shapes"],
        }
        spec_path = directory / "mask.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        mask = {"kind": "regions", "file": str(spec_path), "name": spec["name"]}
    if mask:
        history_name = str(mask.get("name") or "AI eyes")
    else:
        history_name = step_history_name(style_id, again=again, title=title)
    if mask:
        if step_path.is_file():
            step_path.unlink()
        recipe.pop("step_style", None)
        recipe["step_mask"] = mask
    else:
        recipe.pop("step_mask", None)
        placed = _subject_box(recipe, directory) if nxt == "subject" else None
        hue_center = None
        if nxt == "colorzones":
            hue_center = (_hue_on_preview(recipe, directory) or {}).get("center")
        xml = (
            step_style(
                style_id,
                again=again,
                title=title,
                action="darken" if recoloring and _cast_hue(title) else str(
                    recipe.get("features_next_action") or "lift"
                ),
                box=placed,
                size=_frame_size(recipe, directory) if placed else None,
                center=hue_center,
                color=_color_advice(recipe) if recoloring else None,
            )
            if nxt
            else None
        )
        if xml and "<operation>vignette</operation>" in xml:
            history_name = "Vignette the subject"
        if not xml:
            # Nothing can be written for this step. Say so instead of
            # keeping the previous note, which read as if it was applied.
            label = title or nxt
            return _leave_off(
                directory,
                path,
                recipe,
                plan,
                nxt,
                f"No darktable module could be written for \u201c{label}\u201d, so this step was left off.",
            )
        step_path.write_text(xml, encoding="utf-8")
        recipe["step_style"] = {
            "name": "AI accepted step",
            "file": str(step_path),
            "feature": nxt,
        }
    _drop_undone_steps(plan)
    _remember_applied_step(plan, recipe, nxt, history_name)
    done = [str(item) for item in (plan.get("features_done") or []) if str(item).strip()]
    if nxt not in done:
        done.append(nxt)
    plan["features_done"] = done
    _promote(plan, recipe, nxt)
    if recipe["features_next"]:
        recipe["feature_note"] = "This step is on this photo."
    else:
        recipe["feature_note"] = (
            "This step is on this photo. The plan is finished. "
            "Type another change and press Ask."
        )
    _refresh_handoff(recipe)
    _save_recipe(path, recipe)
    return recipe


def skip_waiting_step(output_dir) -> dict:
    """Move to the next step without writing a history module."""

    directory, path, recipe, plan = _load_recipe(output_dir)
    nxt = str(recipe.get("features_next") or "").strip()
    step_path = directory / "step.dtstyle"
    if step_path.is_file():
        step_path.unlink()
    recipe.pop("step_style", None)
    if nxt:
        skipped = [str(item) for item in (plan.get("features_skipped") or []) if str(item).strip()]
        if nxt not in skipped:
            skipped.append(nxt)
        plan["features_skipped"] = skipped
        _promote(plan, recipe, nxt)
    recipe["feature_note"] = "Skipped. The look was left as it is."
    _refresh_handoff(recipe)
    _save_recipe(path, recipe)
    return recipe


def plan_chat(output_dir, steer: str, scene_text: str = "", history_xmp=None) -> dict:
    """Turn one sentence into the next steps. Do not grade the photo again.

    Steps already on the photo (``history_xmp`` rows and the plan's log) and
    steps skipped this session are told to the planner, and a repeated one
    is dropped from the new queue unless the request asks to redo it.
    """

    import json

    from ai_dt.applied import (
        applied_prompt_text,
        applied_steps,
        history_names_from_xmp,
        skipped_steps,
    )

    from ai_dt.catalog import apply_operations, operation_features, technical_brief
    from ai_dt.reasoner import propose_gemini, propose_ollama
    from ai_dt.status_text import panel_status, plan_backend_line
    from ai_dt.steer import apply_proposal, revise_plan

    directory, path, recipe, plan = _load_recipe(output_dir)
    sentence = ""
    stage_path = directory / "stage.json"
    if stage_path.is_file():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if isinstance(stage, dict):
            sentence = str(stage.get("sentence") or "")
    if not sentence:
        reading = recipe.get("reading") if isinstance(recipe.get("reading"), dict) else {}
        sentence = str(reading.get("sentence") or "")
    vibe = ""
    preset = ""
    if stage_path.is_file():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if isinstance(stage, dict):
            vibe = str(stage.get("vibe") or "")
            preset = str(stage.get("preset") or "")
    scene = recipe.get("scene") if isinstance(recipe.get("scene"), dict) else {}
    typed = " ".join(str(scene_text or "").split())
    if typed:
        scene = {
            "description": " ".join(str(scene.get("description") or "").split()) or typed,
            "summary": typed,
            "source": "user",
        }
        recipe["scene"] = scene
    scene_for_model = "\n".join(
        part
        for part in (scene.get("summary"), scene.get("description"))
        if str(part or "").strip()
    )
    analysis = recipe.get("analysis") if isinstance(recipe.get("analysis"), dict) else {}
    atmosphere = analysis.get("atmosphere") if isinstance(analysis.get("atmosphere"), dict) else {}
    ran = [
        str(item)
        for item in ((recipe.get("separation") or {}).get("stages_ran") or [])
        if str(item).strip()
    ]
    brief = technical_brief(atmosphere, stages=ran)
    from ai_dt.animal_plan import animal_noun, subject_is_animal, subject_line

    picture_record = recipe.get("picture") if isinstance(recipe.get("picture"), dict) else {}
    subject_words = f"{scene_for_model} {steer}"
    animal_subject = plan.get("subject_kind") == "animal" or subject_is_animal(
        str(picture_record.get("type") or ""), [], subject_words
    )
    ground_bright = bool(plan.get("bright_ground"))
    water_seen = bool(plan.get("subject_water")) or bool(
        re.search(r"\b(water|pond|lake|river|sea|stream|canal)\b", scene_for_model, re.IGNORECASE)
    )
    if animal_subject:
        brief = f"{brief}\n" + subject_line(animal_noun(steer, scene_for_model), ground_bright, water_seen)
    recipe["technical_brief"] = brief
    history_names = history_names_from_xmp(history_xmp)
    applied = applied_steps(plan, history_names)
    if history_names is not None:
        # Done means "in the sidecar history". Drop saved done marks and
        # log entries the sidecar does not hold.
        landed = {str(item.get("id") or "") for item in applied}
        plan["features_done"] = [name for name in (plan.get("features_done") or []) if str(name) in landed]
        plan["step_log"] = [
            item for item in (plan.get("step_log") or [])
            if isinstance(item, dict) and (not item.get("active") or str(item.get("name") or "") in set(history_names))
        ]
    skipped_before = skipped_steps(plan)
    edit_state = applied_prompt_text(applied, skipped_before)
    recipe["applied_steps"] = applied
    asked = propose_gemini(
        sentence, vibe, preset, steer, scene=scene_for_model, brief=brief, applied=edit_state
    )
    proposal = asked.get("proposal") if isinstance(asked.get("proposal"), dict) else None
    if asked.get("ask_ollama"):
        asked = propose_ollama(
            sentence,
            vibe,
            preset,
            steer,
            failure=str(asked.get("failure") or ""),
            gemini_tried=bool(asked.get("gemini_tried")),
            scene=scene_for_model,
            brief=brief,
            applied=edit_state,
        )
        proposal = asked.get("proposal") if isinstance(asked.get("proposal"), dict) else None
    (directory / "proposal.json").write_text(json.dumps(asked, indent=2) + "\n", encoding="utf-8")
    previous = {
        key: plan.get(key)
        for key in ("step_queue", "features_path", "features_next", "features_next_title", "features_next_action")
    }
    if not previous["features_next"]:
        previous["features_next"] = str(recipe.get("features_next") or "")
        previous["features_next_title"] = str(recipe.get("features_next_title") or "")
        previous["features_next_action"] = str(recipe.get("features_next_action") or "")
    plan["features_next"] = ""
    plan["features_path"] = []
    if isinstance(proposal, dict):
        plan = apply_proposal(plan, proposal)
    if str(steer or "").strip():
        plan = revise_plan(plan, steer)
    operations = [item for item in ((proposal or {}).get("operations") or []) if isinstance(item, dict)]
    grounded = f"{sentence} {scene_for_model}".strip()
    hue = hue_operation(steer, grounded)
    if hue is not None and not any(str(item.get("id") or "") == "colorzones" for item in operations):
        operations = [*operations, hue]
    operations = cast_operations(operations, steer)
    from ai_dt.mask_grammar import mask_operations

    claimed = " ".join(
        str(item.get("reason") or "")
        for key in ("features", "stages")
        for item in ((proposal or {}).get(key) or [])
        if isinstance(item, dict)
    )
    for extra_operation in mask_operations(steer, claimed):
        if not any(str(item.get("id") or "") == extra_operation["id"] for item in operations):
            operations = [*operations, extra_operation]
    plan = apply_operations(plan, operations, list((proposal or {}).get("dropped") or []))
    merged = dict(proposal) if isinstance(proposal, dict) else {}
    extra = operation_features(operations)
    if extra:
        merged["features"] = [item for item in (merged.get("features") or []) if isinstance(item, dict)] + extra
    plan = select_mask_features(plan, merged, [], steer, grounded, "")
    plan = _install_history_steps(plan, operations)
    from ai_dt.offline_plan import apply_offline, offline_operations, planner_missing

    if planner_missing(proposal):
        plan = apply_offline(
            plan,
            offline_operations(
                steer,
                picture_type=str(picture_record.get("type") or ""),
                animal=animal_subject,
                ground=ground_bright,
                first=False,
            ),
        )
    plan = _drop_applied_steps(plan, applied, skipped_before, steer)
    if animal_subject:
        from ai_dt.animal_plan import animal_plan

        animal_count = sum(
            1 for item in ((recipe.get("analysis") or {}).get("subjects") or [])
            if isinstance(item, dict) and str(item.get("category") or "") == "animal"
        )
        plan = animal_plan(
            plan,
            steer=steer,
            description=scene_for_model,
            animals=max(1, animal_count),
            ground=ground_bright,
            water=water_seen,
            applied=[str(item.get("id") or "") for item in applied if isinstance(item, dict)] + list(skipped_before or []),
        )
        plan["subject_water"] = water_seen
    plan = _keep_current_plan(plan, previous, applied)
    recipe["plan"] = plan
    nxt = str(plan.get("features_next") or "")
    recipe["features_next"] = nxt
    recipe["features_next_title"] = str(plan.get("features_next_title") or "")
    recipe["features_next_action"] = str(plan.get("features_next_action") or "")
    recipe["feature_note"] = str(plan.get("feature_note") or "")
    reasoning = plan.get("reasoning") if isinstance(plan.get("reasoning"), dict) else {}
    recipe["plan_backend"] = plan_backend_line(reasoning, description=str(scene.get("source") or "") == "user")
    if nxt:
        follow = "Nothing is added until you press Accept."
    else:
        follow = "This sentence did not add a step."
    recipe["handoff"] = {
        "sentence": f"This plan starts from the photo you are viewing. {follow}"
    }
    step_path = directory / "step.dtstyle"
    if step_path.is_file():
        step_path.unlink()
    recipe.pop("step_style", None)
    recipe["plan_request"] = " ".join(str(steer or "").split())
    recipe["plan_description"] = " ".join(str(scene.get("summary") or "").split())
    recipe["plan_description_source"] = str(scene.get("source") or "")
    recipe["status_text"] = panel_status(recipe)
    _save_recipe(path, recipe)
    return recipe


def _keep_current_plan(plan: dict, previous: dict, applied: list[dict]) -> dict:
    """A request that adds no step leaves the waiting plan in place.

    The plan on screen stays the current plan until Ask returns a new one,
    so Accept and Skip keep applying to the step they named.
    """

    from ai_dt.applied import step_kind

    if plan.get("features_next") or not previous.get("features_next"):
        return plan
    waiting = str(previous["features_next"])
    if step_kind(waiting) in {step_kind(item["id"]) for item in applied}:
        return plan
    for key, value in previous.items():
        plan[key] = value
    title = str(previous.get("features_next_title") or "") or feature_title(waiting)
    note = str(plan.get("feature_note") or "").strip()
    keep = f"This request did not add a new step, so the current plan stays: {title}."
    plan["feature_note"] = f"{note} {keep}".strip() if note else keep
    return plan


def _drop_applied_steps(plan: dict, applied: list[dict], skipped: list[str], steer: str) -> dict:
    """Remove steps already on the photo (or skipped) from a freshly proposed queue."""

    from ai_dt.applied import drop_repeated

    queue = [dict(item) for item in (plan.get("step_queue") or []) if isinstance(item, dict)]
    nxt = str(plan.get("features_next") or "")
    if not queue and nxt:
        queue = [
            {
                "id": nxt,
                "title": str(plan.get("features_next_title") or ""),
                "action": str(plan.get("features_next_action") or ""),
            }
        ]
    kept, dropped = drop_repeated(queue, applied, skipped, steer)
    done = [str(item) for item in (plan.get("features_done") or []) if str(item).strip()]
    for item in applied:
        if item["id"] and item["id"] not in done and item["id"] not in {str(q.get("id")) for q in kept}:
            done.append(item["id"])
    plan["features_done"] = done
    if not dropped:
        return plan
    follow = kept[0] if kept else {}
    plan["step_queue"] = kept
    plan["features_path"] = [str(item.get("id") or "") for item in kept if str(item.get("id") or "")]
    plan["features_next"] = str(follow.get("id") or "")
    plan["features_next_title"] = str(follow.get("title") or "")
    plan["features_next_action"] = str(follow.get("action") or "")
    plan["features_repeated"] = dropped
    names = ", ".join(dict.fromkeys(dropped))
    if kept:
        plan["feature_note"] = f"Already on this photo, left out of the new plan: {names}."
    else:
        plan["feature_note"] = (
            f"Every proposed step is already on this photo ({names}). "
            "Ask to redo or strengthen one, or type another change."
        )
    return plan


def cast_operations(operations: list[dict], steer: str) -> list[dict]:
    """A color change for the whole picture is one recolor step, not a hue mask."""

    from ai_dt.mapping import color_thought, wants_recolor

    hue = _cast_hue(steer)
    asked = wants_recolor(steer)
    rewritten: list[dict] = []
    found = False
    for item in operations:
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or "").strip().lower()
        reason = " ".join(str(item.get("reason") or "").split())
        action = str(item.get("action") or "include").strip().lower()
        if ident in {"recolor", "colorbalancergb", "channelmixerrgb", "colorzones"} and (
            asked or color_thought(reason)
        ):
            if ident == "colorzones" and not asked and not color_thought(reason):
                rewritten.append(item)
                continue
            found = True
            verb = action if action in {"lift", "darken", "hold"} else "include"
            if hue or _cast_hue(reason):
                verb = "darken"
            rewritten.append(
                {
                    "id": "recolor",
                    "action": verb,
                    "reason": reason or (f"reduce the {hue}" if hue else "recolor"),
                }
            )
            continue
        rewritten.append(item)
    if asked and not found:
        rewritten.append(
            {
                "id": "recolor",
                "action": "darken" if hue else "lift",
                "reason": f"reduce the {hue}" if hue else "recolor",
            }
        )
    return rewritten


def _cast_hue(text: str) -> str | None:
    from ai_dt.mapping import cast_hue

    return cast_hue(text)


def _color_advice(recipe: dict) -> dict:
    analysis = recipe.get("analysis") if isinstance(recipe.get("analysis"), dict) else {}
    for key in ("llm", "atmosphere"):
        block = analysis.get(key)
        if isinstance(block, dict):
            return block
    return {}


def hue_operation(steer: str, picture: str = "") -> dict | None:
    """A color-zones step when the request highlights one hue, such as red aprons."""

    typed = str(steer or "").lower()
    if re.search(r"\blanterns?\b", typed) and any(
        word in typed for word in ("highlight", "brighten", "brighter", "lift", "saturate", "pop", "outshine")
    ):
        return {"id": "colorzones", "action": "lift", "reason": "highlight the orange lanterns"}
    if not any(word in typed for word in ("highlight", "outshine", "saturate")):
        return None
    hue = _named_hue(typed)
    if hue is None and any(word in typed for word in ("apron", "bib")):
        hue = _named_hue(str(picture or "").lower())
    if hue is None:
        return None
    return {"id": "colorzones", "action": "lift", "reason": f"highlight the {hue}"}


def _named_hue(text: str) -> str | None:
    for name in ("red", "orange", "yellow", "green", "blue"):
        if re.search(rf"\b{name}\b", text):
            return name
    return None


def _step_title(reason: str, step: str) -> str:
    title = " ".join(str(reason or "").split())
    if len(title) > 80:
        title = title[:80].rstrip()
    return title or feature_title(step)


_HISTORY_STEP = {
    "subject": "subject",
    "eyes": "eyes",
    "eye": "eyes",
    "foreground": "foreground",
    "surround": "foreground",
    "opening": "opening",
    "shafts": "shafts",
    "highlights": "highlights",
    "colorzones": "colorzones",
    "recolor": "recolor",
    **{ident: ident for ident in _MASK_STEPS},
    **{ident: ident for ident in _MASK_EXPANDS},
    # Global looks from the lessons (one module each, see mapping.step_style).
    "split_tone": "split_tone",
    "splittone": "split_tone",
    "palander": "palander",
    "warm_only": "palander",
    "keep_color": "keep_color",
    "selective_color": "keep_color",
}
_NOT_HISTORY = frozenset({"reflection", "background", "skin", "sky"})


def history_steps(operations: list[dict] | None) -> tuple[list[dict], list[str]]:
    """Accept steps a history module can hold, in the model's order.

    A reflection or a background stays out. Those grades are painted into
    the preview, and a style cannot store the mask.
    """

    from ai_dt.mapping import step_style

    found: list[dict] = []
    seen: set[str] = set()
    held: list[str] = []
    for item in operations or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("id") or "").strip().lower()
        action = str(item.get("action") or "include").strip().lower()
        if not name or action == "skip":
            continue
        step = _HISTORY_STEP.get(name)
        if step is None:
            if name in _NOT_HISTORY and name not in held:
                held.append(name)
            continue
        if step in _MASK_EXPANDS or step in _MASK_STEPS:
            reason = " ".join(str(item.get("reason") or "").split())
            verb = action if action in {"lift", "darken", "hold"} else ""
            for part in _MASK_EXPANDS.get(step, (step,)):
                if part in seen:
                    continue
                seen.add(part)
                found.append(
                    {
                        "id": part,
                        "action": verb or _mask_default_action(part),
                        "title": reason if step == part and reason else feature_title(part),
                    }
                )
            continue
        if step in seen or step_style(step) is None:
            if name in _NOT_HISTORY and name not in held:
                held.append(name)
            continue
        seen.add(step)
        reason = " ".join(str(item.get("reason") or "").split())
        verb = action if action in {"lift", "darken", "hold"} else "lift"
        if step == "recolor" and _cast_hue(reason):
            verb = "darken"
        found.append({"id": step, "action": verb, "title": reason or feature_title(step)})
    from ai_dt.lessons import order_steps

    return order_steps(found), held


def _mask_default_action(step_id: str) -> str:
    from ai_dt.mask_steps import DEFAULT_ACTION

    return DEFAULT_ACTION.get(step_id, "lift")


def _install_history_steps(plan: dict, operations: list[dict]) -> dict:
    """Put the model's history steps on the accept queue, ahead of leftovers."""

    chosen, held = history_steps(operations)
    if held:
        plan["not_history"] = held
    if not chosen:
        return plan
    queue = [dict(item) for item in chosen]
    seen = {str(item.get("id") or "") for item in queue}
    for item in plan.get("step_queue") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("id") or "")
        if name and name not in seen:
            queue.append(dict(item))
            seen.add(name)
    done = [name for name in (plan.get("features_done") or []) if name not in seen]
    current = queue[0]
    plan["step_queue"] = queue
    plan["features_path"] = [str(item.get("id") or "") for item in queue]
    plan["features_done"] = done
    plan["features_next"] = str(current.get("id") or "")
    plan["features_next_title"] = str(current.get("title") or "")
    plan["features_next_action"] = str(current.get("action") or "")
    if str(plan.get("feature_note") or "").startswith("This sentence could not"):
        plan["feature_note"] = ""
    return plan


def _steps_from_proposal(proposal: dict | None, steer: str, reading_sentence: str) -> list[dict]:
    """Every stage the model named, in order, when its reason rests on this request."""

    from ai_dt.review import _content_words

    source = _content_words(steer) or _content_words(reading_sentence)
    if not source:
        return []
    found: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in (proposal or {}).get("stages") or []:
        if not isinstance(item, dict):
            continue
        step = _STEP_FROM_STAGE.get(str(item.get("stage") or "").strip().lower())
        action = str(item.get("action") or "").strip().lower()
        if not step or action not in {"lift", "darken", "hold"}:
            continue
        if step == "subject" and action != "lift":
            continue
        reason = str(item.get("reason") or "")
        if not (_content_words(reason) & source):
            continue
        title = _step_title(reason, step)
        key = (step, action, title.lower())
        if key in seen:
            continue
        seen.add(key)
        found.append({"id": step, "action": action, "title": title})
    return found


def _another_subject_lift(proposal: dict | None, steer: str) -> bool:
    """A new sentence whose words also appear in a model promise to lift the subject."""

    from ai_dt.review import _content_words

    words = _content_words(steer)
    if not words:
        return False
    paragraph = str((proposal or {}).get("paragraph") or "")
    lowered = paragraph.lower()
    if "lift" not in lowered or "subject" not in lowered:
        return False
    return bool(words & _content_words(paragraph))


def _grounded_subject_lift(proposal: dict | None, steer: str, reading_sentence: str) -> bool:
    """The model asked to lift the subject, and that reason rests on the user's words."""

    from ai_dt.review import _content_words

    source = _content_words(steer) or _content_words(reading_sentence)
    if not source:
        return False
    for item in (proposal or {}).get("stages") or []:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or "")
        action = str(item.get("action") or "")
        if stage != "subject" or action != "lift":
            continue
        if _content_words(str(item.get("reason") or "")) & source:
            return True
    return False


def _unmapped(steer: str, plan: dict, asked: list[str], reviewed: list[str]) -> bool:
    if not str(steer or "").strip() or asked or reviewed:
        return False
    summary = str(plan.get("summary") or "")
    if summary.startswith("Ignored:"):
        return True
    if summary:
        return False
    if plan.get("scales") or plan.get("asks") or plan.get("skip") or plan.get("strong"):
        return False
    return True
