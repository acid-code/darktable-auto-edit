"""Run detection, lighting advice, and Darktable style generation."""

from __future__ import annotations

import os
import time
from pathlib import Path

from PIL import Image

from ai_dt.advisors.combine import combine_atmospheres
from ai_dt.advisors.histogram import HistogramAdvisor, refine_for_subjects
from ai_dt.advisors.ollama_llava import OllamaLLaVAAdvisor
from ai_dt.detectors.heuristic import HeuristicDetector
from ai_dt.detectors.mediapipe_faces import MediaPipeFaceDetector
from ai_dt.detectors.merge import merge_detections
from ai_dt.detectors.yolo import YoloDetector
from ai_dt.errors import AdvisorUnavailable, DetectorUnavailable, RawReadError, RawSupportUnavailable
from ai_dt.io_image import LoadedImage, is_raw, load_image
from ai_dt.mapping import build_modules, render_style
from ai_dt.guides.decide import decide_picture
from ai_dt.guides.library import match_guide
from ai_dt.recipes.library import blend_look, match_vibe, scene_from_detections
from ai_dt.recipes.reason import reason_about
from ai_dt.schema import Atmosphere
from ai_dt.separate import render_finish
from ai_dt.simulate import vignette_from_modules
from ai_dt.tiff16 import write_rgb16

DETECTORS = ("auto", "yolo", "mediapipe", "heuristic")
ADVISORS = ("auto", "ollama", "histogram")


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
    if choice in ("auto", "ollama"):
        try:
            suggestion = OllamaLLaVAAdvisor(ollama_model, prompt_path).advise(image, image_path)
        except AdvisorUnavailable as exc:
            if choice == "ollama":
                raise
            notes.append(str(exc))
    return measured, suggestion, notes


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
) -> dict:
    """Analyze one image and return the recipe written to ``output.json``.

    ``source_path`` is the original file when ``image_path`` is only a
    developed preview. A camera RAW at that path is decoded and used.
    """

    path = Path(image_path)
    loaded, input_notes = _open_input(path, source_path, full_res=full_res)
    image = loaded.image
    detections, detector_notes = _detect(image, detector, yolo_model)
    measured, suggestion, advisor_notes = _advise(
        image, path, advisor, ollama_model, prompt_path
    )
    measured = refine_for_subjects(image, measured, detections)
    atmosphere = combine_atmospheres(measured, suggestion)
    # The picture type picks the guide. A written description then joins the
    # recipes whose explanations belong: one grade on the frame, and the
    # others as masks on a subject or a line.
    picture = decide_picture(loaded.pixels, detections)
    guide_match = match_guide(picture.type, directory=guides_dir)
    scene = scene_from_detections(detections)
    match = match_vibe(scene, vibe, directory=library_dir)
    reasoning = reason_about(
        picture,
        detections,
        vibe,
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
    adjusted, modules, local_edits, plugins = build_modules(
        image, atmosphere, detections, vignette_strength=vignette_strength
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
    )
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
    look = ""
    if match.recipe is not None:
        look = f" Look: {match.recipe.title}."
    kept_features = list(separation.get("features") or [])
    kept = ""
    if kept_features:
        kept = " Kept " + ", ".join(kept_features) + "."
    description = (
        f"Local AI edit. {picture.type} — {guide_match.guide.title} guide.{kept} {adjusted.mood}.{look} "
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
        "guide": _guide_record(guide_match, separation),
        "features": list(separation.get("features") or []),
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
    recipe["_plugins"] = plugins
    recipe["_rendered"] = rendered
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
    style_path = directory / "edit.dtstyle"
    json_path = directory / "output.json"
    edited_path = directory / "edited.tif"
    recipe["style"]["file"] = str(style_path)
    recipe["render"] = {"file": str(edited_path), "format": "tiff16"}
    style_xml = render_style(
        recipe["style"]["name"],
        recipe["style"]["description"],
        plugins,
    )
    style_path.write_text(style_xml, encoding="utf-8")
    write_rgb16(edited_path, rendered)
    import json

    json_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    return json_path, style_path, edited_path
