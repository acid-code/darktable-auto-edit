"""Mask features chosen for one frame.

``separate`` is committed only when a person or an animal box was found.
The other four are tools. A run paints one of them after it is accepted.
A preview can show the next one first. Pixel measurements do not invent
a path, a gap, or a trunk.
"""

from __future__ import annotations

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
}
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
        stage = str(item.get("stage") or "")
        action = str(item.get("action") or "")
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
        done = [name for name in done if name != "subject"]
        nxt = "subject"
        note = "Another lift, for this request."
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

    from ai_dt.status_text import panel_status

    recipe["chat"] = True
    recipe["offer_looks"] = False
    recipe["choices"] = []
    recipe["status_text"] = panel_status(recipe)
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


def commit_waiting_step(output_dir) -> dict:
    """Write one history module for the waiting step. Do not rebuild the look."""

    from ai_dt.mapping import step_style

    directory, path, recipe, plan = _load_recipe(output_dir)
    nxt = str(recipe.get("features_next") or "").strip()
    again = "Another" in str(recipe.get("feature_note") or "")
    xml = (
        step_style(
            nxt,
            again=again,
            title=str(recipe.get("features_next_title") or ""),
            action=str(recipe.get("features_next_action") or "lift"),
        )
        if nxt
        else None
    )
    step_path = directory / "step.dtstyle"
    if not xml:
        if step_path.is_file():
            step_path.unlink()
        recipe.pop("step_style", None)
        _save_recipe(path, recipe)
        return recipe
    step_path.write_text(xml, encoding="utf-8")
    recipe["step_style"] = {
        "name": "AI accepted step",
        "file": str(step_path),
        "feature": nxt,
    }
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
    _save_recipe(path, recipe)
    return recipe


def plan_chat(output_dir, steer: str) -> dict:
    """Turn one sentence into the next steps. Do not grade the photo again."""

    import json

    from ai_dt.reasoner import propose_gemini, propose_ollama
    from ai_dt.status_text import plan_backend_line
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
    asked = propose_gemini(sentence, vibe, preset, steer)
    proposal = asked.get("proposal") if isinstance(asked.get("proposal"), dict) else None
    if asked.get("ask_ollama"):
        asked = propose_ollama(
            sentence,
            vibe,
            preset,
            steer,
            failure=str(asked.get("failure") or ""),
            gemini_tried=bool(asked.get("gemini_tried")),
        )
        proposal = asked.get("proposal") if isinstance(asked.get("proposal"), dict) else None
    (directory / "proposal.json").write_text(json.dumps(asked, indent=2) + "\n", encoding="utf-8")
    plan["features_next"] = ""
    plan["features_path"] = []
    if isinstance(proposal, dict):
        plan = apply_proposal(plan, proposal)
    if str(steer or "").strip():
        plan = revise_plan(plan, steer)
    plan = select_mask_features(plan, proposal if isinstance(proposal, dict) else {}, [], steer, sentence, "")
    recipe["plan"] = plan
    nxt = str(plan.get("features_next") or "")
    recipe["features_next"] = nxt
    recipe["features_next_title"] = str(plan.get("features_next_title") or "")
    recipe["features_next_action"] = str(plan.get("features_next_action") or "")
    recipe["feature_note"] = str(plan.get("feature_note") or "")
    reasoning = plan.get("reasoning") if isinstance(plan.get("reasoning"), dict) else {}
    recipe["plan_backend"] = plan_backend_line(reasoning)
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
    _save_recipe(path, recipe)
    return recipe


def _step_title(reason: str, step: str) -> str:
    title = " ".join(str(reason or "").split())
    if len(title) > 80:
        title = title[:80].rstrip()
    return title or feature_title(step)


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
