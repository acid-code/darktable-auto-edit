"""Run detection, lighting advice, and Darktable style generation."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from PIL import Image

from ai_dt.catalog import technical_brief
from ai_dt.advisors.combine import combine_atmospheres
from ai_dt.advisors.histogram import HistogramAdvisor, refine_for_subjects
from ai_dt.advisors.mistral_vision import MistralVisionAdvisor
from ai_dt.advisors.ollama_llava import OllamaLLaVAAdvisor
from ai_dt.detectors.heuristic import HeuristicDetector, attach_animal_eyes, gate_faces
from ai_dt.detectors.mediapipe_faces import MediaPipeFaceDetector
from ai_dt.detectors.merge import merge_detections
from ai_dt.detectors.yolo import YoloDetector
from ai_dt.errors import AdvisorUnavailable, DetectorUnavailable, RawReadError, RawSupportUnavailable
from ai_dt.io_image import LoadedImage, is_raw, load_image
from ai_dt.choices import collect_choices, mark_recommendations
from ai_dt.handoff import build_handoff
from ai_dt.history_ops import history_operations
from ai_dt.mapping import build_modules, render_style, step_style
from ai_dt.masks_xmp import MASKS_REASON, ellipse_sidecar, masks_can_attach
from ai_dt.modules_extra import append_extra_modules, retouch_decision
from ai_dt.guides.decide import apply_steer_scene, decide_picture
from ai_dt.guides.library import match_guide
from ai_dt.reading import _core_box, _luma, _mean, linear_falloff, read_frame
from ai_dt.recipes.library import blend_look, match_vibe, scene_from_detections
from ai_dt.recipes.reason import reason_about
from ai_dt.recipes.recommend import recommend_looks
from ai_dt.reasoner import propose_stages
from ai_dt.features import FEATURE_IDS, feature_title
from ai_dt.precedence import resolve_plan
from ai_dt.review import guard_faithful_grade, guard_warm_grade
from ai_dt.runbook import assess_run, original_behind_edit
from ai_dt.schema import Atmosphere
from ai_dt.steer import (
    build_run_lines,
    empty_plan,
    format_plan_label,
    format_plan_text,
    load_stored,
    stage_intents,
)
from ai_dt.schema import Atmosphere as _Atmosphere
from ai_dt.status_text import accept_skip_lines, panel_status, picture_backend_line, plan_backend_line
from ai_dt.separate import render_finish
from ai_dt.simulate import vignette_from_modules
from ai_dt.tiff16 import write_rgb16
from ai_dt.versions import build_version_record, write_version

DETECTORS = ("auto", "yolo", "mediapipe", "heuristic")
ADVISORS = ("auto", "ollama", "histogram")
# The darkroom opens edited.tif. The style is the global grade only.
# Half-resolution RAW is several times this edge, and the paint scales with
# the pixel count, so the looks stay on a preview-sized frame.
_RENDER_EDGE = 1280


def _finish_intents(bundle: dict) -> list[dict]:
    """Turn the steer plan's asks into the renderer's stage and action records."""

    intents: list[dict] = []
    for item in bundle.get("asks") or []:
        if not isinstance(item, dict):
            continue
        verb = str(item.get("verb") or "")
        target = str(item.get("target") or "")
        if verb == "gradient":
            intents.append({"stage": "gradient", "action": "darken"})
            continue
        if verb in {"darken", "lift", "hold"} and target:
            intents.append({"stage": target, "action": verb})
    for stage in bundle.get("skip") or []:
        name = str(stage)
        if name:
            intents.append({"stage": name, "action": "skip"})
    return intents


def _guide_record(guide_match, separation: dict) -> dict:
    record = guide_match.to_dict()
    record["stages_ran"] = list(separation.get("stages_ran") or [])
    record["skipped"] = list(separation.get("skipped") or [])
    return record


def _cap_fuji_color(atmosphere: Atmosphere, picture) -> Atmosphere:
    """Pull a Fuji grade back inside the measured color window.

    The histogram advisor will not push temperature past ±0.28 or tint
    past ±0.2, and it will not cool a warm face or add green to it.
    A film simulation is added on top of that measurement, so the sum is
    held inside the same bounds. Exposure is the measurement: the shipped
    Fuji recipes store no exposure offset.
    """

    temperature = min(max(float(atmosphere.temperature_shift), -0.28), 0.28)
    tint = min(max(float(atmosphere.tint), -0.2), 0.2)
    if picture.warmth > 0.05 and picture.type in {"portrait", "group", "street"}:
        temperature = max(temperature, 0.0)
        tint = max(tint, 0.0)
    atmosphere.temperature_shift = temperature
    atmosphere.tint = tint
    return atmosphere


def apply_expert_requests(atmosphere: object, plan: dict) -> object:
    """Push a sentence or a picked look onto the modules that can carry it.

    Tone equalizer, color calibration, and color zones stay out of the style
    when their number is zero. A direct request sets that number.
    """

    lift = float(plan.get("shadow_lift") or 0)
    if lift > 0:
        # The request is stored as 0.5. That number is not a slider. Writing
        # it into color-balance brilliance shows as +50% and turns warm
        # shadows yellow. The tone equalizer opens the dark bands by a step
        # on top of the measurement.
        measured = float(atmosphere.shadows)
        opened = bool(plan.get("caps_open") or plan.get("strong"))
        if opened:
            atmosphere.shadows = min(0.45, max(measured, min(lift, 0.45)))
        else:
            # About four tenths of a stop on the tone equalizer. The color
            # balance shadow slider stays capped, so this is not +50% brilliance.
            atmosphere.shadows = max(measured, min(0.45, measured + 0.32))
        if float(atmosphere.temperature_shift) > 0.02:
            atmosphere.temperature_shift = max(0.0, float(atmosphere.temperature_shift) - 0.06)
            plan["shadow_note"] = (
                "Shadows open from the measurement. The warm cast is pulled back "
                "so the lift does not go yellow."
            )
        else:
            plan["shadow_note"] = "Shadows open from the measurement, not by a fixed amount."
    if plan.get("highlight_hold"):
        measured_high = float(atmosphere.highlights)
        opened = bool(plan.get("caps_open") or plan.get("strong"))
        if opened:
            atmosphere.highlights = min(measured_high, -0.35)
        else:
            atmosphere.highlights = min(measured_high - 0.12, -0.08)
        atmosphere.exposure_ev = min(float(atmosphere.exposure_ev), 0.15)
    warmth = float(plan.get("warmth") or 0)
    if warmth:
        # A picked look is not "strong". The full 0.4 would push color
        # balance into an additive wash. Closed caps keep a short step.
        opened = bool(plan.get("caps_open") or plan.get("strong"))
        step = float(warmth) if opened else max(-0.16, min(0.16, float(warmth)))
        limit = 1.0 if opened else 0.28
        atmosphere.temperature_shift = max(-limit, min(limit, float(atmosphere.temperature_shift) + step))
    color = float(plan.get("color") or 0)
    if color:
        atmosphere.saturation = max(-1.0, min(1.0, float(atmosphere.saturation) + color))
    contrast = float(plan.get("contrast_push") or 0)
    if contrast:
        atmosphere.contrast = max(-1.0, min(1.0, float(atmosphere.contrast) + contrast))
    return atmosphere


def _style_name(explicit: str | None) -> str:
    if explicit and explicit.strip():
        cleaned = " ".join(explicit.split())
        return cleaned[:120]
    return f"AI Edit {os.getpid()}-{time.time_ns()}"


def _detect(image: Image.Image, choice: str, yolo_model: str) -> tuple[list, list[str]]:
    if choice not in DETECTORS:
        raise DetectorUnavailable(f"Unknown detector {choice!r}")
    notes: list[str] = []
    found = []

    if choice in ("auto", "yolo"):
        try:
            found.extend(YoloDetector(yolo_model).detect(image))
        except DetectorUnavailable as exc:
            if choice == "yolo":
                raise
            notes.append(str(exc))

    if choice in ("auto", "mediapipe"):
        try:
            found.extend(MediaPipeFaceDetector().detect(image))
        except DetectorUnavailable as exc:
            if choice == "mediapipe":
                raise
            notes.append(str(exc))

    if choice in ("auto", "heuristic"):
        found.extend(HeuristicDetector().detect(image))

    return merge_detections(found), notes


def _advise(
    image: Image.Image,
    image_path: Path,
    choice: str,
    ollama_model: str,
    prompt_path: Path | None,
    context: str = "",
) -> tuple[Atmosphere, Atmosphere | None, list[str]]:
    if choice not in ADVISORS:
        raise AdvisorUnavailable(f"Unknown advisor {choice!r}")
    measured = HistogramAdvisor().advise(image)
    notes: list[str] = []
    suggestion = None
    # auto prefers Mistral vision when the key is set. A failed call, or no
    # key, leaves the histogram and then tries LLaVA. histogram and ollama
    # stay explicit: neither sends the picture to Mistral.
    if choice == "auto" and os.environ.get("MISTRAL_API_KEY", "").strip():
        try:
            suggestion = MistralVisionAdvisor(prompt_path).advise(image, image_path, context)
        except AdvisorUnavailable as exc:
            notes.append(str(exc))
    if suggestion is None and choice in ("auto", "ollama"):
        try:
            suggestion = OllamaLLaVAAdvisor(ollama_model, prompt_path).advise(image, image_path)
        except AdvisorUnavailable as exc:
            if choice == "ollama":
                raise
            notes.append(str(exc))
    return measured, suggestion, notes


def _atmosphere_from(data: object) -> Atmosphere | None:
    if not isinstance(data, dict):
        return None
    kwargs = {}
    for name in Atmosphere.__dataclass_fields__:
        if name in data:
            kwargs[name] = data[name]
    return Atmosphere(**kwargs)


def _stored_recipe(output_dir: Path | None) -> dict:
    if output_dir is None:
        return {}
    path = Path(output_dir) / "output.json"
    if not path.is_file():
        return {}
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


_IMPERATIVE = re.compile(
    r"^(adjust|enhance|keep|keeping|maintain|focus|focusing|make|increase|decrease|"
    r"reduce|lift|darken|brighten|preserve|ensure|add|remove|boost|soften|sharpen|"
    r"hold|leave|avoid)\b",
    re.IGNORECASE,
)
_INSTRUCTION = re.compile(
    r"\b(should|while keeping|maintain a|focusing on|minimal distraction|"
    r"adjust the|enhance the)\b",
    re.IGNORECASE,
)


def _sentence_is_instruction(sentence: str) -> bool:
    text = sentence.strip()
    if not text:
        return False
    return bool(_IMPERATIVE.match(text) or _INSTRUCTION.search(text))


def _observations(text: str) -> str:
    """Sentences that say what is in the picture. Edit orders are left out."""

    parts = re.split(r"(?<=[.!?])\s+", " ".join(str(text or "").split()))
    return " ".join(part for part in parts if part and not _sentence_is_instruction(part))


def _crop_suggestion(image, detections) -> dict | None:
    """A rule-of-thirds 16:9 crop (photoserge examples 1 and 8). Advisory; never applied."""

    try:
        from ai_dt.lessons import crop_suggestion
        from ai_dt.segment.heuristic import _small, horizon_row, sky_mask

        subject = None
        best = 0.0
        for item in detections or []:
            if getattr(item, "category", "") in {"animal", "person", "face", "subject"}:
                x, y, w, h = item.bbox
                if w * h > best:
                    best, subject = w * h, (x, y, w, h)
        sky = sky_mask(_small(image))
        horizon = horizon_row(sky) if float(sky.mean()) > 0.08 else None
        return crop_suggestion(image.size, subject=subject, horizon=horizon)
    except Exception:  # advisory only; never block a run
        return None


def _vision_context(brief: str, scene_text: str, *, describe: bool) -> str:
    parts = []
    if brief.strip():
        parts.append("Technical snapshot:\n" + brief.strip())
    if describe:
        parts.append(
            "Describe what is already in the photograph: the subject, where it is, the light, "
            "and the background. State what you see. Do not write an edit order such as adjust, "
            "enhance, keep, or maintain. Put that in description and summary, and also return the numeric keys."
        )
    else:
        if scene_text.strip():
            parts.append(
                "Photo description, already confirmed. Do not replace it:\n" + scene_text.strip()
            )
        parts.append(
            "Return mood, rationale, and the numeric keys. Leave description and summary empty."
        )
    from ai_dt.lessons import general_tips

    tips = general_tips()
    if tips:
        parts.append("Editing order from the lessons (global first):\n" + "\n".join(f"- {tip}" for tip in tips))
    return "\n\n".join(parts)


def _write_description(
    output_dir: Path,
    scene: dict,
    brief: str,
    picture_line: str,
    analysis: dict,
) -> dict:
    """Store the photo description. An existing plan keeps its steps."""

    import json

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "output.json"
    existing: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = {}
        if isinstance(loaded, dict):
            existing = loaded
    has_plan = isinstance(existing.get("plan"), dict) and bool(existing.get("plan"))
    if has_plan or existing.get("modules") or existing.get("features_next"):
        existing["scene"] = scene
        existing["technical_brief"] = brief
        if picture_line and not existing.get("picture_backend"):
            existing["picture_backend"] = picture_line
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        return existing
    recipe = {
        "version": 1,
        "scene": scene,
        "technical_brief": brief,
        "picture_backend": picture_line,
        "offer_looks": False,
        "choices": [],
        "plan": {},
        "analysis": analysis,
        "status_text": (
            (picture_line + "\n" if picture_line else "")
            + "Description is ready. Edit it if the read is wrong, then press Propose looks."
        ),
    }
    path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    return recipe


def _scene_record(
    suggestion: Atmosphere | None,
    sentence: str,
    user_text: str,
    stored: dict,
    *,
    rewrite: bool,
) -> dict:
    user = " ".join(str(user_text or "").split())
    saved = stored if isinstance(stored, dict) else {}
    if user and not rewrite:
        observed = _observations(user)
        if observed:
            description = _observations(str(saved.get("description") or "")) or observed
            return {"description": description, "summary": observed, "source": "user"}
    stored_summary = " ".join(str(saved.get("summary") or "").split())
    stored_ok = _observations(stored_summary) if str(saved.get("source") or "") != "user" else stored_summary
    if str(saved.get("source") or "") == "user" and stored_summary and not rewrite:
        kept = _observations(stored_summary)
        if kept:
            description = _observations(str(saved.get("description") or "")) or kept
            return {"description": description, "summary": kept, "source": "user"}
    if stored_ok and stored_ok == stored_summary and not rewrite:
        description = " ".join(str(saved.get("description") or "").split()) or stored_summary
        description = _observations(description) or stored_summary
        return {
            "description": description,
            "summary": stored_summary,
            "source": str(saved.get("source") or "mistral"),
        }
    description = ""
    summary = ""
    if suggestion is not None:
        description = _observations(suggestion.scene_description)
        summary = _observations(suggestion.scene_summary)
    saw_model = bool(summary or description)
    lead = " ".join(str(sentence or "").split())
    if not summary:
        summary = description or lead
    if not description:
        description = summary
    source = "mistral" if suggestion is not None and saw_model else "reading"
    return {"description": description, "summary": summary, "source": source}


def _frame_levels(pixels: object, picture: object) -> tuple[float | None, float | None]:
    import numpy as np

    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] < 3:
        return None, None
    view = np.clip(array.astype(np.float32, copy=False), 0.0, 1.0)
    frame = float(_luma(view).mean())
    box = getattr(picture, "subject_box", None)
    subject = _mean(view, None if box is None else _core_box(tuple(box)))
    return subject, frame


def _status(output_dir: Path | None, text: str) -> None:
    """One line the panel can show while this process is still running."""

    if output_dir is None:
        return
    try:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "status.txt").write_text(text.strip() + "\n", encoding="utf-8")
    except OSError:
        return


def analyze(
    image_path: Path,
    *,
    detector: str = "auto",
    advisor: str = "auto",
    yolo_model: str = "yolov8n.pt",
    ollama_model: str = "llava",
    prompt_path: Path | None = None,
    style_name: str | None = None,
    output_dir: Path | None = None,
    vibe: str = "",
    library_dir: Path | None = None,
    guides_dir: Path | None = None,
    source_path: Path | None = None,
    full_res: bool = False,
    steer: str = "",
    look: str = "",
    image_id: int | None = None,
    duplicate_index: int | None = None,
    parent_id: int | None = None,
    parent_index: int | None = None,
    library_path: Path | None = None,
    picture_only: bool = False,
    saved_advice: dict | None = None,
    proposal: dict | None = None,
    scene_text: str = "",
    rewrite_scene: bool = False,
    describe_only: bool = False,
) -> dict:
    """Analyze one image and return the recipe written to ``output.json``.

    ``source_path`` is the camera file when ``image_path`` is the developed
    preview. The preview is the picture that gets graded. The camera file
    is decoded only when there is no separate developed preview.
    """

    path = Path(image_path)
    _status(output_dir, "Reading the picture…")
    loaded, input_notes = _open_input(
        path, source_path, full_res=full_res, output_dir=output_dir
    )
    image = loaded.image
    detections, detector_notes = _detect(image, detector, yolo_model)
    detections = gate_faces(loaded.pixels, detections)
    detections = attach_animal_eyes(loaded.pixels, detections)
    stored = _stored_recipe(output_dir)
    stored_scene = stored.get("scene") if isinstance(stored.get("scene"), dict) else {}
    stored_stages = [
        str(item)
        for item in ((stored.get("separation") or {}).get("stages_ran") or [])
        if str(item).strip()
    ]
    if saved_advice is not None:
        measured = HistogramAdvisor().advise(image)
        suggestion = _atmosphere_from(saved_advice.get("suggestion"))
        advisor_notes = [str(item) for item in (saved_advice.get("advisor_notes") or [])]
    else:
        measured = HistogramAdvisor().advise(image)
        suggestion = None
        advisor_notes = []
    # The picture type picks the guide. A written description then joins the
    # recipes whose explanations belong: one grade on the frame, and the
    # others as masks on a subject or a line. Drop a false animal label
    # before exposure follows that box, so a train is not lifted as an animal.
    picture = decide_picture(loaded.pixels, detections)
    picture, detections, subject_name, scene_override = apply_steer_scene(picture, detections, steer)
    measured = refine_for_subjects(image, measured, detections)
    subject_level, frame_level = _frame_levels(loaded.pixels, picture)
    measured_brief = technical_brief(
        measured,
        subject_brightness=subject_level,
        frame_brightness=frame_level,
        stages=stored_stages,
    )
    user_scene = " ".join(str(scene_text or "").split())
    stored_summary = " ".join(str(stored_scene.get("summary") or "").split())
    if str(stored_scene.get("source") or "") == "user":
        stored_usable = bool(stored_summary)
    else:
        stored_usable = bool(stored_summary) and _observations(stored_summary) == stored_summary
    describe = bool(rewrite_scene) or not (_observations(user_scene) or stored_usable)
    held_scene = user_scene or str(stored_scene.get("summary") or "")
    if saved_advice is None:
        _measured, suggestion, advisor_notes = _advise(
            image,
            path,
            advisor,
            ollama_model,
            prompt_path,
            _vision_context(measured_brief, held_scene, describe=describe),
        )
        measured.source = _measured.source if suggestion is None else measured.source
    atmosphere = combine_atmospheres(measured, suggestion)
    guide_match = match_guide(picture.type, directory=guides_dir)
    reading = read_frame(loaded.pixels, picture, detections, subject_name=subject_name)
    recommendations = mark_recommendations(recommend_looks(picture, reading), look)
    reflection = "lift"
    theme = ""
    for region in reading.regions:
        if region.get("role") == "reflection":
            reflection = str(region.get("intention") or reflection)
        elif region.get("role") == "theme" and not theme:
            theme = str(region.get("kind") or "")
    guide_stages = list(guide_match.guide.stages)
    # Eyes are a stage only when a face or an eye is actually there.
    # A wildlife picture can still use a dark spot inside the animal.
    saw_face = any(item.category in {"face", "eye", "person"} for item in detections)
    if picture.type not in {"portrait", "group", "animal"} and not saw_face:
        guide_stages = [stage for stage in guide_stages if stage != "eyes"]
    plan = empty_plan(guide_stages, reflection=reflection, theme=theme)
    # A steer, or a look picked onto a duplicate, continues the plan already
    # written in this folder. A blank first run has nothing to continue.
    raw_look = (look or "").strip()
    feature_look = raw_look if raw_look in {"accept-feature", "skip-feature"} else ""
    picked = "" if feature_look else raw_look
    if output_dir is not None and (steer.strip() or picked or feature_look):
        stored = load_stored(output_dir)
        if stored is not None:
            previous = stored["plan"]
            plan["skip"] = list(previous.get("skip") or [])
            plan["scales"] = dict(previous.get("scales") or {})
            plan["pinned"] = str(previous.get("pinned") or "")
            plan["asks"] = [dict(item) for item in (previous.get("asks") or []) if isinstance(item, dict)]
            plan["strong"] = bool(previous.get("strong"))
            plan["strong_word"] = str(previous.get("strong_word") or "")
            plan["gradient"] = bool(previous.get("gradient"))
            plan["proposed"] = [dict(item) for item in (previous.get("proposed") or []) if isinstance(item, dict)]
            plan["features_committed"] = [
                str(item) for item in (previous.get("features_committed") or [])
            ]
            plan["features_skipped"] = [str(item) for item in (previous.get("features_skipped") or [])]
            plan["features_done"] = [str(item) for item in (previous.get("features_done") or [])]
            plan["features_next"] = str(previous.get("features_next") or "")
            plan["features_path"] = [str(item) for item in (previous.get("features_path") or [])]
    scene = scene_from_detections(detections)
    if scene_override and picture.type == "night":
        scene = "night"
    elif scene_override:
        scene = "general"
    # A pinned preset is the user choosing a look. A recommendation is not.
    vibe_for_reason = plan["pinned"] or vibe
    recipe_id = "" if picked in {"", "faithful"} else picked
    match = match_vibe(
        scene,
        vibe_for_reason,
        directory=library_dir,
        recipe_id=recipe_id or None,
    )
    preset_name = match.recipe.title if match.recipe is not None else ""
    if picture_only:
        return {
            "sentence": reading.sentence,
            "preset": preset_name,
            "vibe": vibe,
            "steer": steer,
            "suggestion": suggestion.to_dict() if suggestion is not None else None,
            "advisor_notes": advisor_notes,
        }
    scene = _scene_record(
        suggestion,
        reading.sentence,
        scene_text,
        stored_scene,
        rewrite=rewrite_scene,
    )
    grade_brief = technical_brief(
        atmosphere,
        subject_brightness=subject_level,
        frame_brightness=frame_level,
        stages=stored_stages,
    )
    scene_for_model = "\n".join(
        part for part in (scene.get("summary"), scene.get("description")) if str(part or "").strip()
    )
    if describe_only:
        picture_line = picture_backend_line(str(atmosphere.source or ""), advisor_notes)
        analysis = {
            "mood": atmosphere.mood,
            "rationale": atmosphere.rationale,
            "atmosphere": atmosphere.to_dict(),
            "measured": measured.to_dict(),
            "llm": suggestion.to_dict() if suggestion is not None else None,
        }
        if output_dir is None:
            return {
                "scene": scene,
                "technical_brief": grade_brief,
                "picture_backend": picture_line,
                "offer_looks": False,
                "analysis": analysis,
            }
        return _write_description(output_dir, scene, grade_brief, picture_line, analysis)
    if proposal is None:
        _status(output_dir, "Asking Gemini…")
    chosen = (
        proposal
        if proposal is not None
        else propose_stages(
            reading.sentence,
            vibe,
            preset_name,
            steer,
            scene=scene_for_model,
            brief=grade_brief,
        )
    )
    # User sentence, then locks from a found subject, then the model, then the guide.
    waiting_step = str(plan.get("features_next") or "")
    plan = resolve_plan(
        plan,
        proposal=chosen if isinstance(chosen, dict) else {},
        steer=steer,
        look=picked,
        feature_look=feature_look,
        picture=picture,
        reading=reading,
        detections=detections,
        scene=scene_for_model,
    )
    reasoning = reason_about(
        picture,
        detections,
        vibe_for_reason,
        guide_match.guide,
        match,
        directory=library_dir,
    )
    vignette_strength = 1.0
    if reasoning.frame_stages is not None and match.recipe is not None:
        atmosphere = blend_look(atmosphere, reasoning.frame_stages)
        if match.recipe.id.startswith("fuji-"):
            atmosphere = _cap_fuji_color(atmosphere, picture)
        vignette_strength = float(match.recipe.stages.get("vignette_strength", 1.0))
        atmosphere.source = f"{atmosphere.source}+recipe:{match.recipe.id}"
    elif reasoning.nudges:
        names = ",".join(reasoning.nudges)
        atmosphere.source = f"{atmosphere.source}+masks:{names}"
    if reasoning.decision:
        atmosphere.rationale = f"{atmosphere.rationale} {reasoning.decision}".strip()
    # Caps follow the user's epic or strong word. A blank steer stays closed.
    # stage_intents never reports a flag the reasoner set on its own.
    bundle = stage_intents(plan)
    caps_open = bool(bundle["caps_open"]) if steer.strip() else False
    intents = _finish_intents(bundle)
    include_vignette = "vignette" not in set(bundle["skip"])
    history_skipped = {str(item) for item in (plan.get("history_skipped") or [])}
    include_exposure = "exposure" not in history_skipped
    include_color = "colorbalancergb" not in history_skipped
    if "vignette" in history_skipped:
        include_vignette = False
    atmosphere = apply_expert_requests(atmosphere, plan)
    recipe_id = match.recipe.id if match.recipe is not None else ""
    faithful = picked in {"", "faithful"} and (not recipe_id or "faithful" in recipe_id)
    atmosphere = guard_faithful_grade(atmosphere, measured, picture, faithful=faithful)
    atmosphere = guard_warm_grade(atmosphere, picture, plan, faithful=faithful)
    falloff = linear_falloff(loaded.pixels)
    if falloff is not None:
        plan["falloff"] = falloff
    committed_features = [str(item) for item in (plan.get("features_committed") or [])]
    next_feature = str(plan.get("features_next") or "")
    _status(output_dir, "Writing the looks…")
    render_pixels = _pixels_for_render(loaded.pixels)
    adjusted, modules, local_edits, plugins = build_modules(
        image,
        atmosphere,
        detections,
        vignette_strength=vignette_strength,
        picture=picture,
        include_vignette=include_vignette,
        include_exposure=include_exposure,
        include_color=include_color,
        opened=caps_open,
    )
    def _render(mask_features: list[str]) -> tuple[object, dict]:
        return render_finish(
            render_pixels,
            adjusted,
            detections,
            vignette_from_modules(modules),
            guide=guide_match.guide,
            picture=picture,
            guide_match=guide_match,
            nudges=reasoning.nudges,
            stages=bundle["stages"],
            scales=plan["scales"],
            intentions={
                "reflection": plan["reflection"],
                "theme": plan["theme"],
                "regions": reading.to_dict().get("regions") or [],
            },
            omit=set(bundle["skip"]),
            asks=bundle["asks"],
            caps_open=caps_open,
            intents=intents,
            mask_features=mask_features,
        )

    rendered, separation = _render(committed_features)
    next_rendered = None
    next_info: dict = {}
    # A subject suggestion is already in this paint. Only a catalog tool
    # changes the next picture, so that case is the only second pass.
    if next_feature in FEATURE_IDS and next_feature not in committed_features:
        next_rendered, next_info = _render(committed_features + [next_feature])
    elif next_feature:
        next_rendered = rendered
    feature_notes = list(separation.get("feature_notes") or [])
    for note in next_info.get("feature_notes") or []:
        if note not in feature_notes:
            feature_notes.append(note)
    feature_note = str(plan.get("feature_note") or "").strip()
    if feature_notes:
        feature_note = f"{feature_note} {' '.join(feature_notes)}".strip()
    plan["feature_note"] = feature_note
    plan["lines"] = build_run_lines(plan, separation)
    plan["text"] = format_plan_text(plan)
    plan["label"] = format_plan_label(plan)
    known_history = None
    if library_path is not None and image_id is not None:
        known_history = history_operations(Path(library_path), int(image_id))
    modules, plugins = append_extra_modules(
        modules,
        plugins,
        atmosphere=adjusted,
        picture_type=picture.type,
        reading=reading,
        plan=plan,
        separation=separation,
        guide=guide_match.guide,
        history_ops=known_history,
        look=picked,
        vibe=vibe,
        masks=masks_can_attach(),
    )
    if plan.get("gradient_note") or plan.get("warmth_note"):
        plan["text"] = format_plan_text(plan)
        plan["label"] = format_plan_label(plan)
    stacked = [
        stage
        for stage in separation.get("stages_ran", [])
        if stage in {"subject", "eyes", "reflection"}
    ]
    if len(stacked) >= 2:
        names = {"subject": "the subject", "eyes": "the eyes", "reflection": "the reflection"}
        together = ", ".join(names[stage] for stage in stacked)
        reasoning.description = (
            reasoning.description
            + f" Lifted together: {together}. "
            + "The history vignette only darkens around the subject. "
            + "The brightness of these layers is stacked in edited.tif."
        )

    name = _style_name(style_name)
    subject_labels = ", ".join(item.label for item in detections[:6]) or "none"
    look_note = ""
    if match.recipe is not None:
        look_note = f" Look: {match.recipe.title}."
    kept_features = list(separation.get("features") or [])
    kept = ""
    if kept_features:
        kept = " Kept " + ", ".join(kept_features) + "."
    description = (
        f"{reading.sentence} Local AI edit. {picture.type} — {guide_match.guide.title} guide.{kept} {adjusted.mood}.{look_note} "
        f"Exposure {adjusted.exposure_ev:+.2f} EV, "
        f"contrast {adjusted.contrast:+.2f}, "
        f"warmth {adjusted.temperature_shift:+.2f}. "
        f"Subjects: {subject_labels}. "
        "Appended as new module instances and does not replace existing history."
    )
    style_file = ""
    if output_dir is not None:
        style_file = str(Path(output_dir) / "edit.dtstyle")

    sources = sorted({item.source for item in detections})
    recipe = {
        "version": 1,
        "image": str(loaded.path),
        "width": image.size[0],
        "height": image.size[1],
        "input": {
            "path": str(loaded.path),
            "kind": loaded.kind,
            "notes": input_notes,
            "resolution": "full" if full_res else "half",
        },
        "separation": separation,
        "backends": {
            "detector": "+".join(sources) if sources else detector,
            "advisor": adjusted.source,
            "detector_notes": detector_notes,
            "advisor_notes": advisor_notes,
            "yolo_model": yolo_model,
            "ollama_model": ollama_model,
            "recipe_notes": match.warnings,
            "guide_notes": guide_match.warnings,
        },
        "picture": picture.to_dict(),
        "reading": reading.to_dict(),
        "recommendations": recommendations,
        "choices": collect_choices(
            recommendations,
            (match.to_dict().get("neighbors") or []),
            picked,
            subject_name,
        ),
        "plan": plan,
        "masks": {
            "attached": masks_can_attach(),
            "reason": "" if masks_can_attach() else MASKS_REASON,
        },
        "retouch": retouch_decision(detections, masks=masks_can_attach()),
        "guide": _guide_record(guide_match, separation),
        "features": list(separation.get("features") or []),
        "features_committed": committed_features,
        "features_next": next_feature,
        "features_next_title": str(plan.get("features_next_title") or "")
        or (feature_title(next_feature) if next_feature else ""),
        "features_next_action": str(plan.get("features_next_action") or ""),
        "feature_note": feature_note,
        "scene": scene,
        "technical_brief": grade_brief,
        "subject_name": subject_name,
        "look_id": picked or "faithful",
        "run": assess_run(picture.type, separation),
        "recipe_match": match.to_dict(),
        "reasoning": reasoning.to_dict(),
        "analysis": {
            "mood": adjusted.mood,
            "rationale": adjusted.rationale,
            "subjects": [item.to_dict() for item in detections],
            "crop_suggestion": _crop_suggestion(image, detections),
            "atmosphere": adjusted.to_dict(),
            "measured": measured.to_dict(),
            "llm": suggestion.to_dict() if suggestion is not None else None,
        },
        "modules": modules,
        "local_edits": local_edits,
        "style": {
            "name": name,
            "description": description,
            "file": style_file,
        },
    }
    recipe["offer_looks"] = bool((recipe.get("run") or {}).get("finished"))
    if not recipe["offer_looks"]:
        recipe["choices"] = []
    recipe["picture_backend"] = picture_backend_line(str(adjusted.source or ""), advisor_notes)
    recipe["plan_backend"] = plan_backend_line((plan.get("reasoning") if isinstance(plan, dict) else {}) or {})
    if output_dir is not None:
        recipe["render"] = {"file": str(Path(output_dir) / "edited.tif"), "format": "tiff16"}
    accepted_step = waiting_step if feature_look == "accept-feature" and waiting_step else ""
    if output_dir is not None:
        step_path = Path(output_dir) / "step.dtstyle"
        again = "Another lift" in str(plan.get("feature_note") or "")
        xml = step_style(accepted_step, again=again) if accepted_step else None
        if xml:
            step_path.write_text(xml, encoding="utf-8")
            recipe["step_style"] = {
                "name": "AI accepted step",
                "file": str(step_path),
                "feature": accepted_step,
            }
        elif step_path.is_file():
            step_path.unlink()
    recipe["handoff"] = build_handoff(recipe)
    recipe["plan_request"] = " ".join(str(steer or "").split())
    recipe["status_text"] = panel_status(recipe)
    recipe["accept_skip"] = accept_skip_lines(recipe)
    recipe["_plugins"] = plugins
    recipe["_rendered"] = rendered
    if next_rendered is not None:
        recipe["_next_rendered"] = next_rendered
    recipe["_sidecar"] = ellipse_sidecar(local_edits)
    recipe["_version_meta"] = {
        "image_id": image_id,
        "duplicate_index": duplicate_index,
        "parent_id": parent_id,
        "parent_index": parent_index,
        "look_id": picked or "faithful",
    }
    return recipe


def _developed_preview(preview: Path, original: Path | None) -> Path | None:
    """The camera path when ``preview`` is the edited picture, else None."""

    if original is None or not preview.is_file() or not original.is_file():
        return None
    if preview.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
        return None
    try:
        if preview.resolve() == original.resolve():
            return None
    except OSError:
        return original
    return original


def _open_input(
    preview: Path,
    original: Path | None,
    *,
    full_res: bool,
    output_dir: Path | None = None,
) -> tuple[LoadedImage, list[str]]:
    """Use the developed preview when it is a different file from the camera original.

    That preview is the picture currently open, including edits already made.
    The camera file is decoded only when it is the file being analyzed, or
    when the preview cannot be read.
    """

    notes: list[str] = []
    developed = _developed_preview(preview, original)
    if developed is not None:
        try:
            loaded = load_image(preview, full_res=False)
        except OSError:
            notes.append("The developed picture could not be read. The camera file was used.")
        else:
            notes.append(
                f"Used the developed picture. The camera file {developed.name} was not decoded."
            )
            return loaded, notes
    if original is not None:
        source = Path(original)
        continued = original_behind_edit(source, output_dir)
        if continued is not None:
            notes.append("Continued from the edited TIFF back to the original file.")
            source = continued
        same = False
        try:
            same = source.is_file() and preview.is_file() and source.resolve() == preview.resolve()
        except OSError:
            same = False
        if source.is_file() and not same:
            try:
                loaded = load_image(source, full_res=full_res)
            except (RawSupportUnavailable, RawReadError) as exc:
                notes.append(f"{exc} Fell back to the developed preview.")
            else:
                if loaded.kind == "raw" and not full_res:
                    notes.append("Decoded the RAW at half resolution. Pass --full-res for every pixel.")
                return loaded, notes
    loaded = load_image(preview, full_res=full_res and is_raw(preview))
    if loaded.kind == "raw" and not full_res:
        notes.append("Decoded the RAW at half resolution. Pass --full-res for every pixel.")
    return loaded, notes


def _pixels_for_render(pixels: object) -> object:
    """Shrink a large frame to the preview edge before the mask paint."""

    import numpy as np

    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] != 3:
        return pixels
    height, width = int(array.shape[0]), int(array.shape[1])
    longest = max(height, width)
    if longest <= _RENDER_EDGE:
        return pixels
    scale = _RENDER_EDGE / float(longest)
    image = Image.fromarray(np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")
    image = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.BOX,
    )
    return np.asarray(image).astype(np.float32) / np.float32(255.0)


def _write_look_jpeg(path: Path, pixels: object) -> None:
    """A JPEG of the separated render, for the preview page and the panel."""

    import numpy as np

    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] != 3:
        return
    scaled = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(np.rint(scaled * 255.0).astype(np.uint8), "RGB")
    image.thumbnail((1280, 1280))
    image.save(path, format="JPEG", quality=90)


def write_outputs(recipe: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    """Write ``output.json``, ``edit.dtstyle``, and ``edited.tif``."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plugins = recipe.pop("_plugins")
    rendered = recipe.pop("_rendered")
    next_rendered = recipe.pop("_next_rendered", None)
    sidecar = recipe.pop("_sidecar", "")
    meta = recipe.pop("_version_meta", None) or {}
    style_path = directory / "edit.dtstyle"
    json_path = directory / "output.json"
    edited_path = directory / "edited.tif"
    recipe["style"]["file"] = str(style_path)
    recipe["render"] = {"file": str(edited_path), "format": "tiff16"}
    recipe["handoff"] = build_handoff(recipe)
    recipe["status_text"] = panel_status(recipe)
    recipe["accept_skip"] = accept_skip_lines(recipe)
    record = build_version_record(
        recipe,
        directory,
        image_id=meta.get("image_id"),
        duplicate_index=meta.get("duplicate_index"),
        parent_id=meta.get("parent_id"),
        parent_index=meta.get("parent_index"),
        look_id=str(meta.get("look_id") or "faithful"),
    )
    recipe["version_record"] = record
    style_xml = render_style(
        recipe["style"]["name"],
        recipe["style"]["description"],
        plugins,
    )
    style_path.write_text(style_xml, encoding="utf-8")
    write_rgb16(edited_path, rendered)
    _write_look_jpeg(directory / "look.jpg", rendered)
    if next_rendered is not None:
        _write_look_jpeg(directory / "next.jpg", next_rendered)
    if sidecar:
        (directory / "masks.xmp").write_text(sidecar, encoding="utf-8")
    write_version(directory, record)
    import json

    json_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    return json_path, style_path, edited_path
