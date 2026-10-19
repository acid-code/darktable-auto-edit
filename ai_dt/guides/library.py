"""Match a picture type to a folder of guide documents.

Each guide is a JSON or plain-text file. Its folder is only a place to keep
the file. The searchable document is the title, the picture types, the
words, and the sentences, turned into a term-frequency vector. The query is
the picture type the frame was given, not the vibe and not the folder name.
Cosine similarity picks the nearest neighbor.

A weak or unknown type uses the faithful general guide. This module never
writes a new guide. Adding one is ``python3 -m ai_dt.guides.ingest``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ai_dt.recipes.library import cosine, term_frequency, tokenize

PICTURE_TYPES = (
    "portrait",
    "group",
    "animal",
    "landscape",
    "architecture",
    "night",
    "food",
    "still-life",
    "street",
    "macro",
    "general",
)

STAGES = (
    "grade",
    "subject",
    "eyes",
    "skin",
    "sky",
    "horizon",
    "reflection",
    "background",
    "foreground",
    "highlights",
    # Segmenter classes (ai_dt/segment). Optional in a guide.
    "people",
    "architecture",
    "vegetation",
    "water",
    "ground",
)

MASKS = (
    "subject",
    "eyes",
    "skin",
    "sky",
    "background",
    "foreground",
)

OPTIONAL_MASKS = (
    "highlights",
    "horizon",
    "reflection",
    "people",
    "architecture",
    "vegetation",
    "water",
    "ground",
)

BUILDS = (
    "never",
    "face",
    "animal",
    "subject",
    "eyes",
    "skin",
    "sky",
    "background",
    "foreground",
    "structure",
    "highlights",
    "horizon",
    "reflection",
    "people",
    "architecture",
    "vegetation",
    "water",
    "ground",
)

GRADE_KEYS = (
    "exposure",
    "contrast",
    "saturation",
    "clarity",
    "defocus",
    "mute",
)

# What a guide may name as something to keep sharp or to quiet.
# The mask that stays sharp is the union of the features that were found.
FEATURE_IDS = (
    "animal",
    "perch",
    "person",
    "seat",
    "held",
    "building",
    "sky",
    "land",
    "dish",
    "object",
    "background",
    "table",
    "highlights",
    "street",
)

# Modest on purpose. A guide is not allowed to blow a photo out.
GRADE_LIMITS = {
    "exposure": (-0.35, 0.35),
    "contrast": (-0.25, 0.25),
    "saturation": (-0.35, 0.35),
    "clarity": (-0.45, 0.9),
    "defocus": (0.0, 1.0),
    "mute": (0.0, 0.55),
}

# Below this cosine, and with no picture-type word in common, the nearest
# file is a weak guess. Use the faithful general guide.
_MIN_COSINE = 0.12
_TYPE_BONUS = 0.18
_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

DEFAULT_LIBRARY = Path(__file__).resolve().parents[2] / "guides"

_FAITHFUL_REASON = "No guide matched this picture, so the faithful general guide was used."


@dataclass(frozen=True)
class MaskPlan:
    build: str
    skip: str
    exposure: float
    contrast: float
    saturation: float
    clarity: float
    defocus: float
    mute: float

    def to_dict(self) -> dict:
        return {
            "build": self.build,
            "skip": self.skip,
            "exposure": self.exposure,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "clarity": self.clarity,
            "defocus": self.defocus,
            "mute": self.mute,
        }


@dataclass(frozen=True)
class Guide:
    id: str
    title: str
    types: tuple[str, ...]
    words: tuple[str, ...]
    description: str
    reason: str
    stages: tuple[str, ...]
    masks: dict[str, MaskPlan]
    keep: tuple[str, ...]
    quiet: tuple[str, ...]
    features: dict[str, str]
    path: str
    document: str

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "types": list(self.types),
            "words": list(self.words),
            "description": self.description,
            "reason": self.reason,
            "keep": list(self.keep),
            "quiet": list(self.quiet),
            "features": dict(self.features),
            "stages": list(self.stages),
            "masks": {name: plan.to_dict() for name, plan in self.masks.items()},
        }


@dataclass
class Neighbor:
    id: str
    title: str
    score: float
    cosine: float
    type_bonus: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "score": round(self.score, 4),
            "cosine": round(self.cosine, 4),
            "type_bonus": round(self.type_bonus, 4),
        }


@dataclass
class GuideMatch:
    picture_type: str
    query: str
    method: str
    vectorizer: str
    fallback: bool
    reason: str
    guide: Guide
    score: float = 0.0
    cosine: float = 0.0
    neighbors: list[Neighbor] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        guide = self.guide
        return {
            "id": guide.id,
            "title": guide.title,
            "file": _display_path(guide.path),
            "reason": self.reason,
            "about": guide.reason,
            "fallback": self.fallback,
            "picture_type": self.picture_type,
            "query": self.query,
            "method": self.method,
            "vectorizer": self.vectorizer,
            "score": round(self.score, 4),
            "cosine": round(self.cosine, 4),
            "stages": list(guide.stages),
            "types": list(guide.types),
            "neighbors": [item.to_dict() for item in self.neighbors[:3]],
            "warnings": list(self.warnings),
        }


def document_text(
    title: str,
    types: list[str],
    words: list[str],
    description: str,
    reason: str,
    features: dict[str, str] | None = None,
) -> str:
    """Build the text that becomes the vector.

    Title, picture types, and words are repeated so a short type name is not
    drowned by the sentences. The directory name is intentionally absent.
    Feature sentences are included so the file says what stays sharp.
    """

    type_line = " ".join(types)
    word_line = " ".join(words)
    feature_line = " ".join(features.values()) if features else ""
    return " ".join(
        (
            title,
            title,
            type_line,
            type_line,
            word_line,
            word_line,
            word_line,
            description,
            reason,
            feature_line,
        )
    )


def format_picture_line(picture_type: str, title: str, features: list[str] | None = None) -> str:
    """One readable line: the type, the guide, and the features kept sharp."""

    line = f"{picture_type} — {title} guide"
    names = [str(item) for item in (features or []) if str(item).strip()]
    if names:
        line += ", kept " + ", ".join(names)
    return line


def _display_path(path: str) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(DEFAULT_LIBRARY.parent.resolve()))
    except (OSError, ValueError):
        return str(candidate)


def _clamp_number(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# Open only when the user wrote strong or epic. A guide file still uses the modest limits.
OPEN_LIMITS = {
    "exposure": (-1.6, 1.0),
    "contrast": (-0.45, 0.45),
    "saturation": (-0.5, 0.5),
    "clarity": (-0.45, 0.9),
    "defocus": (0.0, 1.0),
    "mute": (0.0, 0.7),
}


def clamp_grades(plan: MaskPlan, *, opened: bool = False) -> MaskPlan:
    """Force one mask's numbers back inside the modest limits.

    ``opened`` is the strong or epic switch. Without it the tight caps stay.
    """

    limits = OPEN_LIMITS if opened else GRADE_LIMITS
    values = {}
    for key in GRADE_KEYS:
        low, high = limits[key]
        values[key] = _clamp_number(float(getattr(plan, key)), low, high)
    if plan.build != "background":
        values["defocus"] = 0.0
    return MaskPlan(
        build=plan.build,
        skip=plan.skip,
        exposure=values["exposure"],
        contrast=values["contrast"],
        saturation=values["saturation"],
        clarity=values["clarity"],
        defocus=values["defocus"],
        mute=values["mute"],
    )


def _grade_numbers(raw: dict, source: str, mask_name: str) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key in GRADE_KEYS:
        if key not in raw:
            raise ValueError(f"{source} mask {mask_name} needs {key}")
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{source} mask {mask_name} {key} must be a number")
        number = float(value)
        low, high = GRADE_LIMITS[key]
        if number < low or number > high:
            raise ValueError(
                f"{source} mask {mask_name} {key} must be between {low} and {high}"
            )
        clean[key] = number
    if mask_name != "background" and clean["defocus"] > 0:
        raise ValueError(
            f"{source} mask {mask_name} cannot defocus. Only the background mask can blur."
        )
    return clean


def validate_guide(data: object, source: str = "guide") -> dict:
    """Check one guide document. Raises ``ValueError`` with a usable reason."""

    if not isinstance(data, dict):
        raise ValueError(f"{source} must be a JSON object or a text guide")
    allowed = {
        "id",
        "title",
        "types",
        "words",
        "description",
        "reason",
        "keep",
        "quiet",
        "features",
        "stages",
        "masks",
    }
    extra = sorted(set(data) - allowed)
    if extra:
        raise ValueError(f"{source} has unknown fields: {', '.join(extra)}")
    guide_id = data.get("id")
    if not isinstance(guide_id, str) or _ID.fullmatch(guide_id) is None:
        raise ValueError(f"{source} id must be lowercase words separated by hyphens")
    title = data.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{source} needs a title")
    types = data.get("types")
    if not isinstance(types, list) or not types or not all(isinstance(item, str) and item.strip() for item in types):
        raise ValueError(f"{source} needs a types list")
    words = data.get("words")
    if not isinstance(words, list) or not words or not all(isinstance(item, str) and item.strip() for item in words):
        raise ValueError(f"{source} needs at least one word a person might say for this picture")
    description = data.get("description")
    if not isinstance(description, str) or len(description.strip()) < 20:
        raise ValueError(f"{source} description should say what this picture type is, in a sentence")
    reason = data.get("reason")
    if not isinstance(reason, str) or len(reason.strip()) < 20:
        raise ValueError(f"{source} reason should say why these stages belong to this picture, in a sentence")
    stages = data.get("stages")
    if not isinstance(stages, list) or not stages or not all(isinstance(item, str) for item in stages):
        raise ValueError(f"{source} needs a stages list")
    unknown_stages = [item for item in stages if item not in STAGES]
    if unknown_stages:
        raise ValueError(
            f"{source} has unknown stages: {', '.join(unknown_stages)}. "
            f"Use: {', '.join(STAGES)}"
        )
    if "grade" not in stages:
        raise ValueError(f"{source} stages must include grade")
    if stages[0] != "grade":
        raise ValueError(f"{source} stages must start with grade")
    keep, quiet, feature_text = _feature_fields(data, source)
    masks = data.get("masks")
    if not isinstance(masks, dict):
        raise ValueError(f"{source} needs a masks object")
    missing = [name for name in MASKS if name not in masks]
    if missing:
        raise ValueError(f"{source} masks missing: {', '.join(missing)}")
    unknown_masks = [name for name in masks if name not in MASKS and name not in OPTIONAL_MASKS]
    if unknown_masks:
        raise ValueError(f"{source} has unknown masks: {', '.join(unknown_masks)}")
    for name in OPTIONAL_MASKS:
        if name in stages and name not in masks:
            raise ValueError(f"{source} stages include {name} but there is no {name} mask")
    clean_masks: dict[str, dict] = {}
    for name, raw in masks.items():
        if not isinstance(raw, dict):
            raise ValueError(f"{source} mask {name} must be an object")
        extra_mask = sorted(set(raw) - {"build", "skip", *GRADE_KEYS})
        if extra_mask:
            raise ValueError(f"{source} mask {name} has unknown fields: {', '.join(extra_mask)}")
        build = raw.get("build")
        if not isinstance(build, str) or build not in BUILDS:
            raise ValueError(
                f"{source} mask {name} build must be one of: {', '.join(BUILDS)}"
            )
        skip = raw.get("skip")
        if not isinstance(skip, str) or len(skip.strip()) < 12:
            raise ValueError(f"{source} mask {name} needs a skip sentence a person can read")
        grades = _grade_numbers(raw, source, name)
        clean_masks[name] = {"build": build, "skip": " ".join(skip.split()), **grades}
    ordered = [name for name in (*MASKS, *OPTIONAL_MASKS) if name in clean_masks]
    return {
        "id": guide_id,
        "title": " ".join(title.split()),
        "types": [" ".join(item.split()) for item in types],
        "words": [" ".join(item.split()) for item in words],
        "description": " ".join(description.split()),
        "reason": " ".join(reason.split()),
        "keep": keep,
        "quiet": quiet,
        "features": feature_text,
        "stages": list(stages),
        "masks": {name: clean_masks[name] for name in ordered},
    }


def _feature_fields(data: dict, source: str) -> tuple[list[str], list[str], dict[str, str]]:
    """Which features stay sharp, which fall quiet, and the sentence for each."""

    keep = data.get("keep", [])
    quiet = data.get("quiet", [])
    features = data.get("features", {})
    if keep is None:
        keep = []
    if quiet is None:
        quiet = []
    if features is None:
        features = {}
    if not isinstance(keep, list) or not all(isinstance(item, str) and item.strip() for item in keep):
        raise ValueError(f"{source} keep must be a list of feature names")
    if not isinstance(quiet, list) or not all(isinstance(item, str) and item.strip() for item in quiet):
        raise ValueError(f"{source} quiet must be a list of feature names")
    if not isinstance(features, dict):
        raise ValueError(f"{source} features must be an object of feature name to sentence")
    keep = [" ".join(item.split()) for item in keep]
    quiet = [" ".join(item.split()) for item in quiet]
    unknown = [item for item in keep + quiet if item not in FEATURE_IDS]
    if unknown:
        raise ValueError(
            f"{source} has unknown features: {', '.join(unknown)}. "
            f"Use: {', '.join(FEATURE_IDS)}"
        )
    overlap = sorted(set(keep) & set(quiet))
    if overlap:
        raise ValueError(f"{source} lists a feature as both kept and quiet: {', '.join(overlap)}")
    if len(keep) != len(set(keep)) or len(quiet) != len(set(quiet)):
        raise ValueError(f"{source} repeats a feature")
    clean: dict[str, str] = {}
    for name, sentence in features.items():
        if not isinstance(name, str) or name not in FEATURE_IDS:
            raise ValueError(f"{source} feature {name!r} is not a known feature")
        if not isinstance(sentence, str) or len(sentence.strip()) < 20:
            raise ValueError(f"{source} feature {name} needs a sentence saying what happens to it")
        clean[name] = " ".join(sentence.split())
    missing = [name for name in keep + quiet if name not in clean]
    if missing:
        raise ValueError(f"{source} needs a sentence for: {', '.join(missing)}")
    extra = [name for name in clean if name not in keep and name not in quiet]
    if extra:
        raise ValueError(f"{source} feature sentences are not listed in keep or quiet: {', '.join(extra)}")
    ordered = {name: clean[name] for name in (*keep, *quiet)}
    return keep, quiet, ordered


def _guide_from_payload(payload: dict, path: Path | None) -> Guide:
    masks = {
        name: MaskPlan(
            build=raw["build"],
            skip=raw["skip"],
            exposure=float(raw["exposure"]),
            contrast=float(raw["contrast"]),
            saturation=float(raw["saturation"]),
            clarity=float(raw["clarity"]),
            defocus=float(raw["defocus"]),
            mute=float(raw["mute"]),
        )
        for name, raw in payload["masks"].items()
    }
    features = dict(payload.get("features") or {})
    return Guide(
        id=payload["id"],
        title=payload["title"],
        types=tuple(payload["types"]),
        words=tuple(payload["words"]),
        description=payload["description"],
        reason=payload["reason"],
        stages=tuple(payload["stages"]),
        masks=masks,
        keep=tuple(payload.get("keep") or ()),
        quiet=tuple(payload.get("quiet") or ()),
        features=features,
        path=str(path) if path is not None else "",
        document=document_text(
            payload["title"],
            payload["types"],
            payload["words"],
            payload["description"],
            payload["reason"],
            features,
        ),
    )


def parse_text_guide(text: str, source: str = "guide") -> dict:
    """Read a hand-written text guide. Grades left out are zero."""

    data: dict = {"masks": {}}
    section: str | None = None
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            data["masks"].setdefault(section, {})
            continue
        parts = line.split()
        if parts and parts[0].lower() == "mask" and ":" not in parts[0] and len(parts) >= 2:
            section = parts[1].strip().lower()
            data["masks"].setdefault(section, {})
            continue
        if ":" not in line:
            raise ValueError(f"{source} line {line_number} is not a field: {line}")
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        target = data if section is None else data["masks"].setdefault(section, {})
        if key in {"types", "words", "stages", "keep", "quiet"} and section is None:
            target[key] = [item.strip() for item in value.split(",") if item.strip()]
            continue
        if section is None and key.startswith("feature_"):
            feature_id = key.split("_", 1)[1].strip()
            data.setdefault("features", {})[feature_id] = value
            continue
        if section is not None and key in GRADE_KEYS:
            try:
                target[key] = float(value)
            except ValueError as exc:
                raise ValueError(f"{source} line {line_number}: {key} must be a number") from exc
            continue
        target[key] = value
    for raw in data["masks"].values():
        for key in GRADE_KEYS:
            raw.setdefault(key, 0.0)
    return data


def _read_guide_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc
    elif path.suffix.lower() == ".txt":
        payload = parse_text_guide(text, source=path.name)
    else:
        raise ValueError(f"{path.name} must be .json or .txt")
    return validate_guide(payload, source=path.name)


def load_guides(directory: Path | None = None) -> tuple[list[Guide], list[str]]:
    """Read every guide under ``directory``. Bad files are skipped."""

    root = Path(directory) if directory is not None else DEFAULT_LIBRARY
    if not root.is_dir():
        return [], [f"Guide library not found: {root}"]
    guides: list[Guide] = []
    warnings: list[str] = []
    seen: dict[str, str] = {}
    paths = sorted(root.rglob("*.json")) + sorted(root.rglob("*.txt"))
    for path in paths:
        try:
            cleaned = _read_guide_file(path)
        except (OSError, ValueError) as exc:
            warnings.append(f"{path.name}: {exc}")
            continue
        previous = seen.get(cleaned["id"])
        if previous is not None:
            warnings.append(f"{path.name}: duplicate id {cleaned['id']} (also in {previous})")
            continue
        seen[cleaned["id"]] = path.name
        guides.append(_guide_from_payload(cleaned, path))
    return guides, warnings


def _aliases(text: str) -> set[str]:
    raw = " ".join(text.strip().lower().split())
    if not raw:
        return set()
    return {raw, raw.replace("-", " "), raw.replace(" ", "-")}


def _type_hit(picture_type: str, guide: Guide) -> bool:
    wanted = _aliases(picture_type)
    if not wanted:
        return False
    listed: set[str] = set()
    for item in guide.types:
        listed |= _aliases(item)
    return bool(wanted & listed)


def _query_text(picture_type: str) -> str:
    words = " ".join(picture_type.replace("-", " ").split())
    if not tokenize(words):
        return "general faithful"
    return words


def _rank(guides: list[Guide], picture_type: str, query: str) -> list[tuple[Guide, Neighbor]]:
    query_vector = term_frequency(query)
    ranked: list[tuple[Guide, Neighbor]] = []
    for guide in guides:
        similarity = cosine(query_vector, term_frequency(guide.document))
        bonus = _TYPE_BONUS if _type_hit(picture_type, guide) else 0.0
        neighbor = Neighbor(
            id=guide.id,
            title=guide.title,
            score=similarity + bonus,
            cosine=similarity,
            type_bonus=bonus,
        )
        ranked.append((guide, neighbor))
    ranked.sort(key=lambda item: (-item[1].score, -item[1].cosine, item[0].id))
    return ranked


def _general_guide(guides: list[Guide]) -> Guide:
    for guide in guides:
        if guide.id == "general":
            return guide
    for guide in guides:
        if _type_hit("general", guide):
            return guide
    return _guide_from_payload(_builtin_general_payload(), None)


def _builtin_general_payload() -> dict:
    """Used only when the library has no general guide. Nothing is written."""

    skip = "Skip this mask. A frame with no clear subject keeps the global grade only."
    empty = {
        "build": "never",
        "skip": skip,
        "exposure": 0.0,
        "contrast": 0.0,
        "saturation": 0.0,
        "clarity": 0.0,
        "defocus": 0.0,
        "mute": 0.0,
    }
    return validate_guide(
        {
            "id": "general",
            "title": "Faithful general",
            "types": ["general", "faithful"],
            "words": ["general", "faithful", "neutral", "correction"],
            "description": "A frame with no clear subject. The edit is the measured global grade and nothing is cut out.",
            "reason": "Nothing in the frame should be cut out, so the edit stays the global grade.",
            "keep": [],
            "quiet": [],
            "features": {},
            "stages": ["grade"],
            "masks": {name: dict(empty) for name in MASKS},
        },
        source="built-in general guide",
    )


def match_guide(
    picture_type: str = "",
    directory: Path | None = None,
    guides: list[Guide] | None = None,
) -> GuideMatch:
    """Return the guide for this picture type.

    The vibe is not part of the query. A sentence such as "wildlife" must
    not pull a portrait onto the animal guide; the picture type already
    decided which words to look up.
    """

    picture_type = " ".join(str(picture_type).split())
    warnings: list[str] = []
    if guides is None:
        guides, warnings = load_guides(directory)
    query = _query_text(picture_type)
    if not guides:
        guide = _general_guide([])
        return GuideMatch(
            picture_type=picture_type or "general",
            query=query,
            method="nearest-neighbor",
            vectorizer="term-frequency",
            fallback=True,
            reason=_FAITHFUL_REASON,
            guide=guide,
            warnings=warnings or ["no guides are loaded"],
        )
    ranked = _rank(guides, picture_type, query)
    neighbors = [item[1] for item in ranked if item[1].score > 0.02][:3]
    best_guide, best = ranked[0]
    hit = _type_hit(picture_type, best_guide)
    if best.cosine >= _MIN_COSINE or hit:
        chosen = best_guide
        fallback = False
        reason = chosen.reason
    else:
        chosen = _general_guide(guides)
        fallback = True
        reason = _FAITHFUL_REASON
    chosen_score = 0.0
    chosen_cosine = 0.0
    for guide, neighbor in ranked:
        if guide.id == chosen.id:
            chosen_score = neighbor.score
            chosen_cosine = neighbor.cosine
            break
    return GuideMatch(
        picture_type=picture_type or "general",
        query=query,
        method="nearest-neighbor",
        vectorizer="term-frequency",
        fallback=fallback,
        reason=reason,
        guide=chosen,
        score=chosen_score,
        cosine=chosen_cosine,
        neighbors=neighbors,
        warnings=warnings,
    )
