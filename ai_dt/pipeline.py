"""Run detection, lighting advice, and Darktable style generation."""

from __future__ import annotations

import os
import time
from pathlib import Path

from PIL import Image

from ai_dt.advisors.combine import combine_atmospheres
from ai_dt.advisors.histogram import HistogramAdvisor, refine_for_subjects
from ai_dt.advisors.mistral_vision import MistralVisionAdvisor
from ai_dt.advisors.ollama_llava import OllamaLLaVAAdvisor
from ai_dt.detectors.heuristic import HeuristicDetector, gate_faces
from ai_dt.detectors.mediapipe_faces import MediaPipeFaceDetector
from ai_dt.detectors.merge import merge_detections
from ai_dt.detectors.yolo import YoloDetector
from ai_dt.errors import AdvisorUnavailable, DetectorUnavailable, RawReadError, RawSupportUnavailable
from ai_dt.io_image import LoadedImage, is_raw, load_image
from ai_dt.choices import apply_named_look, collect_choices, mark_recommendations
from ai_dt.history_ops import history_operations
from ai_dt.mapping import build_modules, render_style
from ai_dt.masks_xmp import MASKS_REASON, ellipse_sidecar, masks_can_attach
from ai_dt.modules_extra import append_extra_modules, retouch_decision
from ai_dt.guides.decide import apply_steer_scene, decide_picture
from ai_dt.guides.library import match_guide
from ai_dt.reading import linear_falloff, read_frame
from ai_dt.recipes.library import blend_look, match_vibe, scene_from_detections
from ai_dt.recipes.reason import reason_about
from ai_dt.recipes.recommend import recommend_looks
from ai_dt.reasoner import propose_stages
from ai_dt.review import guard_faithful_grade, guard_warm_grade, review_plan
from ai_dt.schema import Atmosphere
from ai_dt.steer import (
    apply_proposal,
    build_run_lines,
    empty_plan,
    format_plan_label,
    format_plan_text,
    load_stored,
    revise_plan,
    stage_intents,
)
from ai_dt.schema import Atmosphere as _Atmosphere
from ai_dt.status_text import panel_status, picture_backend_line, plan_backend_line
from ai_dt.separate import render_finish
from ai_dt.simulate import vignette_from_modules
from ai_dt.tiff16 import write_rgb16
from ai_dt.versions import build_version_record, write_version

DETECTORS = ("auto", "yolo", "mediapipe", "heuristic")
ADVISORS = ("auto", "ollama", "histogram")


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
            suggestion = MistralVisionAdvisor(prompt_path).advise(image, image_path)
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
) -> dict:
    """Analyze one image and return the recipe written to ``output.json``.

    ``source_path`` is the original file when ``image_path`` is only a
    developed preview. A camera RAW at that path is decoded and used.
    """

    path = Path(image_path)
    loaded, input_notes = _open_input(path, source_path, full_res=full_res)
    image = loaded.image
    detections, detector_notes = _detect(image, detector, yolo_model)
    detections = gate_faces(loaded.pixels, detections)
    if saved_advice is not None:
        measured = HistogramAdvisor().advise(image)
        suggestion = _atmosphere_from(saved_advice.get("suggestion"))
        advisor_notes = [str(item) for item in (saved_advice.get("advisor_notes") or [])]
    else:
        measured, suggestion, advisor_notes = _advise(
            image, path, advisor, ollama_model, prompt_path
        )
    # The picture type picks the guide. A written description then joins the
    # recipes whose explanations belong: one grade on the frame, and the
    # others as masks on a subject or a line. Drop a false animal label
    # before exposure follows that box, so a train is not lifted as an animal.
    picture = decide_picture(loaded.pixels, detections)
    picture, detections, subject_name, scene_override = apply_steer_scene(picture, detections, steer)
    measured = refine_for_subjects(image, measured, detections)
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
    picked = (look or "").strip()
    if output_dir is not None and (steer.strip() or picked):
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
    chosen = proposal if proposal is not None else propose_stages(reading.sentence, vibe, preset_name, steer)
    plan = apply_proposal(plan, chosen)
    if steer.strip():
        plan = revise_plan(plan, steer)
    if picked:
        plan = apply_named_look(plan, picked)
    # Before the choice buttons exist: drop claims the picture and the
    # steer do not support. This does not call another model.
    plan = review_plan(plan, picture, reading, steer)
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
    atmosphere = apply_expert_requests(atmosphere, plan)
    recipe_id = match.recipe.id if match.recipe is not None else ""
    faithful = picked in {"", "faithful"} and (not recipe_id or "faithful" in recipe_id)
    atmosphere = guard_faithful_grade(atmosphere, measured, picture, faithful=faithful)
    atmosphere = guard_warm_grade(atmosphere, picture, plan, faithful=faithful)
    falloff = linear_falloff(loaded.pixels)
    if falloff is not None:
        plan["falloff"] = falloff
    adjusted, modules, local_edits, plugins = build_modules(
        image,
        atmosphere,
        detections,
        vignette_strength=vignette_strength,
        picture=picture,
        include_vignette=include_vignette,
        opened=caps_open,
    )
    rendered, separation = render_finish(
        loaded.pixels,
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
    )
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
        "subject_name": subject_name,
        "look_id": picked or "faithful",
        "recipe_match": match.to_dict(),
        "reasoning": reasoning.to_dict(),
        "analysis": {
            "mood": adjusted.mood,
            "rationale": adjusted.rationale,
            "subjects": [item.to_dict() for item in detections],
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
    recipe["picture_backend"] = picture_backend_line(str(adjusted.source or ""), advisor_notes)
    recipe["plan_backend"] = plan_backend_line((plan.get("reasoning") if isinstance(plan, dict) else {}) or {})
    recipe["status_text"] = panel_status(recipe)
    recipe["_plugins"] = plugins
    recipe["_rendered"] = rendered
    recipe["_sidecar"] = ellipse_sidecar(local_edits)
    recipe["_version_meta"] = {
        "image_id": image_id,
        "duplicate_index": duplicate_index,
        "parent_id": parent_id,
        "parent_index": parent_index,
        "look_id": picked or "faithful",
    }
    return recipe


def _open_input(
    preview: Path,
    original: Path | None,
    *,
    full_res: bool,
) -> tuple[LoadedImage, list[str]]:
    """Prefer the original file. A RAW that will not decode falls back to the preview."""

    notes: list[str] = []
    if original is not None:
        source = Path(original)
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


def write_outputs(recipe: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    """Write ``output.json``, ``edit.dtstyle``, and ``edited.tif``."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plugins = recipe.pop("_plugins")
    rendered = recipe.pop("_rendered")
    sidecar = recipe.pop("_sidecar", "")
    meta = recipe.pop("_version_meta", None) or {}
    style_path = directory / "edit.dtstyle"
    json_path = directory / "output.json"
    edited_path = directory / "edited.tif"
    recipe["style"]["file"] = str(style_path)
    recipe["render"] = {"file": str(edited_path), "format": "tiff16"}
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
    if sidecar:
        (directory / "masks.xmp").write_text(sidecar, encoding="utf-8")
    write_version(directory, record)
    import json

    json_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    return json_path, style_path, edited_path
