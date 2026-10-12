"""Match an open-text vibe to a folder of recipe documents.

Each recipe is a JSON file. Its folder is only a place to keep the file.
The searchable document is the title, the scene tags, the vibe words, and
the description, turned into a term-frequency vector. The query is the
detected scene plus the user's sentence, turned into the same kind of
vector. Cosine similarity picks the nearest neighbor.

A sentence-embedding model can replace ``term_frequency`` later. The
score stays a cosine over vectors, so a vector database is not required
for a library this size. Nothing here reads the web.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from ai_dt.guides.support import is_face_bar
from ai_dt.schema import Atmosphere, Detection

SCENES = ("portrait", "architecture", "landscape", "animal", "night", "general")

# Where a recipe is allowed to land. ``frame`` is the global grade.
# The other places are masks: a face, the eyes, a sunset line, the sky,
# the lines of a building, or the background behind a subject.
PLACES = ("frame", "subject", "eyes", "horizon", "sky", "lines", "background")

STAGE_KEYS = (
    "exposure_ev",
    "contrast",
    "temperature_shift",
    "tint",
    "saturation",
    "shadows",
    "highlights",
    "vibrance",
    "vignette_strength",
)

# How much of a recipe offset is allowed on top of the measurement.
# Exposure stays mostly measured. Creative fields can move further.
LOOK_WEIGHTS = {
    "exposure_ev": 0.15,
    "contrast": 0.55,
    "temperature_shift": 0.55,
    "tint": 0.50,
    "saturation": 0.65,
    "shadows": 0.45,
    "highlights": 0.45,
    "vibrance": 0.60,
}

_STAGE_LIMITS = {
    "exposure_ev": (-1.5, 1.5),
    "contrast": (-0.6, 0.6),
    "temperature_shift": (-1.0, 1.0),
    "tint": (-1.0, 1.0),
    "saturation": (-0.6, 0.6),
    "shadows": (-0.6, 0.6),
    "highlights": (-0.6, 0.6),
    "vibrance": (-0.5, 0.5),
    "vignette_strength": (0.0, 1.6),
}

_FINAL_LIMITS = {
    "exposure_ev": (-2.0, 2.0),
    "contrast": (-0.45, 0.45),
    "temperature_shift": (-1.0, 1.0),
    "tint": (-1.0, 1.0),
    "saturation": (-0.4, 0.4),
    "shadows": (-0.45, 0.45),
    "highlights": (-0.45, 0.45),
    "vibrance": (-0.35, 0.35),
}

_STOP = frozenset(
    "a an the of and to with for in on is it its this that from by as or be are was not".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Below this cosine the nearest file is a weak guess. Use the faithful
# recipe for the detected scene instead of applying a stray grade.
_MIN_COSINE = 0.12
_SCENE_BONUS = 0.18

DEFAULT_LIBRARY = Path(__file__).resolve().parents[2] / "recipes"


@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    scenes: tuple[str, ...]
    vibes: tuple[str, ...]
    description: str
    stages: dict[str, float]
    path: str
    document: str
    place: str = "frame"
    explains: str = ""

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "scenes": list(self.scenes),
            "vibes": list(self.vibes),
            "description": self.description,
            "place": self.place,
            "explains": self.explains,
            "stages": {key: round(float(value), 4) for key, value in self.stages.items()},
        }


@dataclass
class Neighbor:
    id: str
    title: str
    score: float
    cosine: float
    scene_bonus: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "score": round(self.score, 4),
            "cosine": round(self.cosine, 4),
            "scene_bonus": round(self.scene_bonus, 4),
        }


@dataclass
class Match:
    scene: str
    vibe: str
    query: str
    method: str
    vectorizer: str
    applied: bool
    reason: str
    recipe: Recipe | None
    score: float = 0.0
    cosine: float = 0.0
    neighbors: list[Neighbor] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        recipe = self.recipe.to_public() if self.recipe is not None else None
        return {
            "id": recipe["id"] if recipe else None,
            "title": recipe["title"] if recipe else None,
            "score": round(self.score, 4),
            "cosine": round(self.cosine, 4),
            "scene": self.scene,
            "vibe": self.vibe,
            "query": self.query,
            "method": self.method,
            "vectorizer": self.vectorizer,
            "applied": self.applied,
            "reason": self.reason,
            "stages": recipe["stages"] if recipe else None,
            "neighbors": [item.to_dict() for item in self.neighbors[:3]],
            "warnings": list(self.warnings),
        }


def scene_from_detections(detections: list[Detection]) -> str:
    """Name the scene the detectors actually saw.

    Landscape and night are not detected. They are reached only when the
    vibe sentence itself is close to a landscape or night recipe.
    """

    categories = {
        item.category
        for item in detections
        if not (item.category in {"face", "person"} and is_face_bar(item.bbox))
    }
    if "face" in categories or "person" in categories:
        return "portrait"
    if "animal" in categories:
        return "animal"
    if "architecture" in categories:
        return "architecture"
    return "general"


def tokenize(text: str) -> list[str]:
    words = _TOKEN.findall(text.lower())
    return [word for word in words if len(word) >= 3 and word not in _STOP]


def term_frequency(text: str) -> dict[str, float]:
    """Bag-of-words vector. Swap this function for an embedding model later."""

    tokens = tokenize(text)
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    scale = float(len(tokens))
    return {token: count / scale for token, count in counts.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = 0.0
    for key, value in left.items():
        other = right.get(key)
        if other:
            dot += value * other
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def document_text(
    title: str,
    scenes: list[str],
    vibes: list[str],
    description: str,
    explains: str = "",
) -> str:
    """Build the text that becomes the vector.

    Title, scenes, and vibe words are repeated so a short query is not
    drowned by the description. The directory name is intentionally absent.
    The explanation is included once, so a recipe can say where it belongs.
    """

    scene_line = " ".join(scenes)
    vibe_line = " ".join(vibes)
    return " ".join(
        (
            title,
            title,
            scene_line,
            scene_line,
            vibe_line,
            vibe_line,
            vibe_line,
            description,
            explains,
        )
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def validate_recipe(data: object, source: str = "recipe") -> dict:
    """Check one recipe document. Raises ``ValueError`` with a usable reason."""

    if not isinstance(data, dict):
        raise ValueError(f"{source} must be a JSON object")
    extra = sorted(set(data) - {"id", "title", "scenes", "vibes", "description", "stages", "place", "explains"})
    if extra:
        raise ValueError(f"{source} has unknown fields: {', '.join(extra)}")
    recipe_id = data.get("id")
    if not isinstance(recipe_id, str) or _ID.fullmatch(recipe_id) is None:
        raise ValueError(f"{source} id must be lowercase words separated by hyphens")
    title = data.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{source} needs a title")
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes or not all(isinstance(item, str) for item in scenes):
        raise ValueError(f"{source} needs a scenes list")
    unknown_scenes = [item for item in scenes if item not in SCENES]
    if unknown_scenes:
        raise ValueError(
            f"{source} has unknown scenes: {', '.join(unknown_scenes)}. "
            f"Use one of: {', '.join(SCENES)}"
        )
    vibes = data.get("vibes")
    if (
        not isinstance(vibes, list)
        or not vibes
        or not all(isinstance(item, str) and item.strip() for item in vibes)
    ):
        raise ValueError(f"{source} needs at least one vibe word")
    description = data.get("description")
    if not isinstance(description, str) or len(description.strip()) < 20:
        raise ValueError(f"{source} description should say what the look does, in a sentence")
    place = data.get("place", "frame")
    if not isinstance(place, str) or place not in PLACES:
        raise ValueError(f"{source} place must be one of: {', '.join(PLACES)}")
    explains = data.get("explains", "")
    if not isinstance(explains, str):
        raise ValueError(f"{source} explains must be a sentence")
    explains = " ".join(explains.split())
    if explains and len(explains) < 20:
        raise ValueError(f"{source} explains should say where the recipe belongs, in a sentence")
    stages = data.get("stages")
    if not isinstance(stages, dict):
        raise ValueError(f"{source} needs a stages object")
    missing = [key for key in STAGE_KEYS if key not in stages]
    if missing:
        raise ValueError(f"{source} stages missing: {', '.join(missing)}")
    unknown_stages = [key for key in stages if key not in STAGE_KEYS]
    if unknown_stages:
        raise ValueError(f"{source} has unknown stages: {', '.join(unknown_stages)}")
    clean_stages: dict[str, float] = {}
    for key in STAGE_KEYS:
        value = stages[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{source} stage {key} must be a number")
        number = float(value)
        low, high = _STAGE_LIMITS[key]
        if number < low or number > high:
            raise ValueError(f"{source} stage {key} must be between {low} and {high}")
        clean_stages[key] = number
    return {
        "id": recipe_id,
        "title": " ".join(title.split()),
        "scenes": list(scenes),
        "vibes": [" ".join(item.split()) for item in vibes],
        "description": " ".join(description.split()),
        "place": place,
        "explains": explains,
        "stages": clean_stages,
    }


def _recipe_from_payload(payload: dict, path: Path) -> Recipe:
    return Recipe(
        id=payload["id"],
        title=payload["title"],
        scenes=tuple(payload["scenes"]),
        vibes=tuple(payload["vibes"]),
        description=payload["description"],
        stages=dict(payload["stages"]),
        path=str(path),
        document=document_text(
            payload["title"],
            payload["scenes"],
            payload["vibes"],
            payload["description"],
            payload.get("explains", ""),
        ),
        place=payload.get("place", "frame"),
        explains=payload.get("explains", ""),
    )


def load_library(directory: Path | None = None) -> tuple[list[Recipe], list[str]]:
    """Read every recipe JSON under ``directory``. Bad files are skipped."""

    root = Path(directory) if directory is not None else DEFAULT_LIBRARY
    if not root.is_dir():
        return [], [f"Recipe library not found: {root}"]
    recipes: list[Recipe] = []
    warnings: list[str] = []
    seen: dict[str, str] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cleaned = validate_recipe(payload, source=str(path.name))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"{path.name}: {exc}")
            continue
        previous = seen.get(cleaned["id"])
        if previous is not None:
            warnings.append(f"{path.name}: duplicate id {cleaned['id']} (also in {previous})")
            continue
        seen[cleaned["id"]] = path.name
        recipes.append(_recipe_from_payload(cleaned, path))
    return recipes, warnings


def _query_text(scene: str, vibe: str) -> str:
    words = " ".join(vibe.split())
    if not tokenize(words):
        return f"faithful neutral correction {scene}"
    return f"{scene} {words}"


def _rank(recipes: list[Recipe], scene: str, query: str) -> list[tuple[Recipe, Neighbor]]:
    query_vector = term_frequency(query)
    ranked: list[tuple[Recipe, Neighbor]] = []
    for recipe in recipes:
        similarity = cosine(query_vector, term_frequency(recipe.document))
        bonus = _SCENE_BONUS if scene in recipe.scenes else 0.0
        neighbor = Neighbor(
            id=recipe.id,
            title=recipe.title,
            score=similarity + bonus,
            cosine=similarity,
            scene_bonus=bonus,
        )
        ranked.append((recipe, neighbor))
    ranked.sort(key=lambda item: (-item[1].score, -item[1].cosine, item[0].id))
    return ranked


def _faithful(recipes: list[Recipe], scene: str) -> Recipe | None:
    for candidate in (f"{scene}-faithful", "general-faithful"):
        for recipe in recipes:
            if recipe.id == candidate:
                return recipe
    for recipe in recipes:
        if scene in recipe.scenes and "faithful" in recipe.vibes:
            return recipe
    return None


def match_vibe(
    scene: str,
    vibe: str = "",
    directory: Path | None = None,
    recipes: list[Recipe] | None = None,
) -> Match:
    """Return the nearest recipe and whether it should actually be applied."""

    if scene not in SCENES:
        scene = "general"
    warnings: list[str] = []
    if recipes is None:
        recipes, warnings = load_library(directory)
    query = _query_text(scene, vibe)
    if not recipes:
        return Match(
            scene=scene,
            vibe=vibe.strip(),
            query=query,
            method="nearest-neighbor",
            vectorizer="term-frequency",
            applied=False,
            reason="no recipes are loaded",
            recipe=None,
            warnings=warnings,
        )
    ranked = _rank(recipes, scene, query)
    neighbors = [item[1] for item in ranked if item[1].score > 0.02][:3]
    best_recipe, best = ranked[0]
    # The scene name is part of the query, so a nonsense sentence can still
    # share that one word with every recipe for the scene. The user's own
    # words have to land somewhere before a creative look is applied.
    content_cosine = best.cosine
    if tokenize(vibe):
        user_vector = term_frequency(vibe)
        content_cosine = max(
            cosine(user_vector, term_frequency(recipe.document)) for recipe, _neighbor in ranked
        )
    if best.cosine >= _MIN_COSINE and content_cosine >= _MIN_COSINE:
        chosen = best_recipe
        applied = True
        reason = "nearest recipe document"
    else:
        chosen = _faithful(recipes, scene)
        applied = chosen is not None
        reason = "no close recipe; using the faithful correction for this scene"
    chosen_score = 0.0
    chosen_cosine = 0.0
    if chosen is not None:
        for recipe, neighbor in ranked:
            if recipe.id == chosen.id:
                chosen_score = neighbor.score
                chosen_cosine = neighbor.cosine
                break
    return Match(
        scene=scene,
        vibe=" ".join(vibe.split()),
        query=query,
        method="nearest-neighbor",
        vectorizer="term-frequency",
        applied=applied,
        reason=reason,
        recipe=chosen,
        score=chosen_score,
        cosine=chosen_cosine,
        neighbors=neighbors,
        warnings=warnings,
    )


def blend_look(atmosphere: Atmosphere, stages: dict[str, float]) -> Atmosphere:
    """Add a fraction of the recipe offsets on top of the measured atmosphere.

    A faithful recipe stores zeros, so this leaves the measurement unchanged.
    ``black`` is never taken from a recipe.
    """

    values = {}
    for key, weight in LOOK_WEIGHTS.items():
        measured = float(getattr(atmosphere, key))
        offset = float(stages.get(key, 0.0)) * weight
        low, high = _FINAL_LIMITS[key]
        values[key] = _clamp(measured + offset, low, high)
    return Atmosphere(
        black=atmosphere.black,
        mood=atmosphere.mood,
        rationale=atmosphere.rationale,
        source=atmosphere.source,
        **values,
    )
