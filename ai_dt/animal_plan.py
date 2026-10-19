"""Animal subjects: routing, the step plan, and the wording shown for it.

The duck report: a mallard on a bright ledge by green water was planned as
a portrait (subject, skin, eyes, background; "Painted grades for separate,
person") with the old painted ``subject`` lift titled with the request
("make the duck the main theme"), and the reasoning called the mallard an
eider and said "darken the background keeping the water untouched" while
the background was the water.

* ``retype_from_text`` moves a person-type picture to ``animal`` when the
  reading, the (possibly edited) description or the request names an
  animal and no person, so an edited description routes too.
* ``animal_plan`` rewrites the accept queue onto the animal steps: outline
  lift -> background darken+mute (inverse animal) -> bottom gradient when a
  bright ground or ledge sits under the animal -> edge burn minus the
  subject; a subtle glow only when it was proposed. Skin, person and face
  steps are dropped. Titles are concrete actions on a generic noun (the
  user's word, else "the bird" / "the animal").
* ``generic_species`` and ``consistent_regions`` clean the plan text.
"""

from __future__ import annotations

import re
from dataclasses import replace

# Bird family first: the noun for a bird is "the bird" unless the user named it.
BIRD_WORDS = (
    "duck", "ducks", "duckling", "ducklings", "drake", "mallard", "eider", "teal", "goose", "geese",
    "swan", "swans", "gull", "seagull", "heron", "egret", "stork", "crane", "pelican", "cormorant",
    "coot", "moorhen", "grebe", "merganser", "pigeon", "dove", "sparrow", "robin", "owl", "eagle",
    "hawk", "falcon", "parrot", "penguin", "flamingo", "kingfisher", "woodpecker", "crow", "raven",
    "magpie", "finch", "tit", "chicken", "hen", "rooster", "bird", "birds",
)
MAMMAL_WORDS = (
    "dog", "dogs", "puppy", "cat", "cats", "kitten", "horse", "horses", "pony", "cow", "cows", "sheep",
    "lamb", "goat", "deer", "fox", "foxes", "bear", "lion", "tiger", "elephant", "giraffe", "zebra",
    "squirrel", "rabbit", "hare", "monkey", "seal", "otter", "wolf", "animal", "animals", "pet",
)
OTHER_ANIMAL_WORDS = ("frog", "lizard", "turtle", "snake", "butterfly", "bee", "fish", "insect")
ANIMAL_WORDS = BIRD_WORDS + MAMMAL_WORDS + OTHER_ANIMAL_WORDS
# Words a reading uses for a person. "male"/"female" are not here: a male eider is a bird.
PERSON_WORDS = (
    "person", "people", "man", "men", "woman", "women", "girl", "girls", "boy", "boys", "child",
    "children", "kid", "kids", "baby", "portrait", "face", "faces", "selfie", "bride", "groom",
    "lady", "gentleman", "skin",
)
# Species a vision model may guess. Shown generically unless the user wrote them.
SPECIES_WORDS = (
    "mallard", "eider", "teal", "wigeon", "pintail", "shoveler", "gadwall", "merganser", "goldeneye",
    "scaup", "pochard", "coot", "moorhen", "grebe", "goose", "swan", "drake",
)
PERSON_TYPES = frozenset({"portrait", "group", "street"})
PERSON_STEPS = frozenset({"skin", "face", "person", "mask_people", "people"})

# Generic step names a model proposes, mapped onto the animal steps.
_ANIMAL_MAP = {
    "subject": "mask_animal",
    "separate": "mask_animal",
    "mask_animal": "mask_animal",
    "mask_animals": "mask_animals",
    "background": "mask_animal_background",
    "surround": "mask_animal_background",
    "mask_background": "mask_animal_background",
    "mask_animal_background": "mask_animal_background",
    "foreground": "mask_bottom_gradient",
    "mask_ground": "mask_bottom_gradient",
    "mask_bottom_gradient": "mask_bottom_gradient",
}
_ACTIONS = {
    "mask_animal": "lift",
    "mask_animals": "lift",
    "mask_animal_background": "darken",
    "mask_bottom_gradient": "darken",
    "mask_edge_burn": "darken",
    "mask_glow": "lift",
}
_ORDER = ("mask_animal", "mask_animals", "mask_animal_background", "mask_bottom_gradient", "mask_edge_burn", "mask_glow")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text or "").lower())


def names_animal(text: str) -> str:
    """The first animal word in ``text`` ("" when none)."""

    words = _words(text)
    # A stone fox or a wooden duck is not an animal subject.
    if any(word in _NOT_ALIVE for word in words):
        return ""
    for word in words:
        if word in ANIMAL_WORDS:
            return word
    return ""


_NOT_ALIVE = frozenset(
    {"statue", "statues", "sculpture", "sculptures", "figurine", "figurines", "carving", "carvings",
     "toy", "toys", "plush", "ornament", "ornaments", "decoy", "decoys", "stuffed"}
)


def names_person(text: str) -> bool:
    return any(word in PERSON_WORDS for word in _words(text))


def is_bird(word: str) -> bool:
    return word in BIRD_WORDS


def animal_noun(steer: str, description: str = "") -> str:
    """How plan text names the animal: the user's word, else "the bird" / "the animal".

    A species from the reading ("eider") is not repeated: a vision model's
    species guess is often wrong (the mallard was called an eider).
    """

    typed = names_animal(steer)
    if typed and typed not in {"animal", "animals", "pet"}:
        return "the " + typed
    seen = names_animal(description)
    # A family word the reading uses ("duck", "cat") is safe; a species
    # ("eider", "mallard") is not.
    if seen:
        plain = [
            word for word in _words(description)
            if word in ANIMAL_WORDS and word not in SPECIES_WORDS
            and word not in {"animal", "animals", "pet", "bird", "birds", "male", "female"}
            and not word.endswith("s")
        ]
        if plain:
            return "the " + plain[0]
    if seen and is_bird(seen):
        return "the bird"
    return "the animal"


def subject_is_animal(picture_type: str, detections: list, text: str) -> bool:
    """An animal subject: the picture type, a detector animal, or the words, with no person named."""

    if picture_type == "animal":
        return True
    real_people = any(
        str(getattr(item, "category", "")) in {"face", "person"} and str(getattr(item, "source", "")) == "mediapipe"
        for item in detections or []
    )
    if real_people:
        return False
    detected = any(
        str(getattr(item, "category", "")) == "animal" and str(getattr(item, "source", "")) != "heuristic"
        for item in detections or []
    )
    named = bool(names_animal(text)) and not names_person(text)
    return detected or named


def retype_from_text(picture, detections: list, text: str):
    """Move a person-type picture to ``animal`` when the words name only an animal.

    Returns ``(picture, detections, changed)``. Person and face boxes become
    the animal (the box the detector put on the head or body); eyes stay.
    A real MediaPipe face keeps the portrait.
    """

    kind = getattr(picture, "type", "")
    if kind not in PERSON_TYPES and kind != "general":
        return picture, list(detections or []), False
    word = names_animal(text)
    if not word or names_person(text):
        return picture, list(detections or []), False
    if any(getattr(item, "category", "") in {"face", "person"} and getattr(item, "source", "") == "mediapipe" for item in detections or []):
        return picture, list(detections or []), False
    from ai_dt.schema import Detection

    found = list(detections or [])
    animals = [item for item in found if item.category == "animal"]
    people = [item for item in found if item.category in {"face", "person"}]
    kept = [item for item in found if item.category not in {"face", "person"}]
    if not animals and people:
        biggest = max(people, key=lambda item: float(item.bbox[2]) * float(item.bbox[3]))
        kept.append(Detection("bird" if is_bird(word) else word, "animal", float(biggest.confidence), biggest.bbox, "reading"))
    elif not animals and not people:
        # No detector box at all (no YOLO): the heuristic subject box is the
        # best place to look for the animal the words name.
        boxes = [item for item in found if item.category == "subject"]
        if not boxes:
            return picture, list(detections or []), False
        biggest = max(boxes, key=lambda item: float(item.confidence))
        kept.append(Detection("bird" if is_bird(word) else word, "animal", float(biggest.confidence), biggest.bbox, "reading"))
    box = None
    pool = animals or [item for item in kept if item.category == "animal"]
    if pool:
        box = max(pool, key=lambda item: float(item.bbox[2]) * float(item.bbox[3])).bbox
    updated = replace(
        picture,
        type="animal",
        reason=(
            f"The description names {animal_noun('', text).replace('the ', 'a ', 1)} and no person, so this is an animal"
            + (", not a portrait." if kind in PERSON_TYPES else ".")
        ),
        subject_box=box if box is not None else getattr(picture, "subject_box", None),
    )
    return updated, kept, True


def bright_ground(pixels) -> bool:
    """A bright band (ledge, sand, snow) along the bottom of the frame."""

    import numpy as np

    view = np.asarray(pixels, dtype=np.float32)
    if view.ndim != 3 or view.shape[0] < 20:
        return False
    if view.size and float(view.max()) > 1.5:
        view = view / np.float32(255.0)
    luma = 0.2126 * view[..., 0] + 0.7152 * view[..., 1] + 0.0722 * view[..., 2]
    rows = luma.shape[0]
    bottom = float(luma[int(rows * 0.82):].mean())
    rest = float(luma[: int(rows * 0.82)].mean())
    # A pale ledge in front of bright water is only ~1.2x the rest (the duck
    # photo: 0.67 vs 0.55), so a clearly bright band needs less contrast.
    return bottom > 0.45 and (bottom > rest * 1.25 or (bottom > 0.6 and bottom > rest * 1.15))


def generic_species(text: str, noun: str, steer: str = "") -> str:
    """Replace a species guess with the generic noun, unless the user wrote that species."""

    typed = set(_words(steer))
    base = noun.split()[-1] if noun else "bird"

    def swap(match: re.Match[str]) -> str:
        article, word = match.group(1), match.group(2)
        if word.lower() in typed:
            return match.group(0)
        if article:
            lead = article.lower()
            if lead in {"a", "an"}:
                lead = "an" if base[:1] in "aeiou" else "a"
            if article[:1].isupper():
                lead = lead.capitalize()
            return f"{lead} {base}"
        return base

    pattern = r"\b(?:(the|a|an|this|that)\s+)?((?:common\s+)?(?:" + "|".join(SPECIES_WORDS) + r")s?)\b"
    cleaned = re.sub(pattern, swap, str(text or ""), flags=re.IGNORECASE)
    return re.sub(r"\bmale\s+(?=" + re.escape(base) + r"\b)", "", cleaned, flags=re.IGNORECASE)


def consistent_regions(text: str, *, water: bool, ground: bool) -> str:
    """Drop a clause that keeps untouched the very region a step darkens.

    For an animal on a ledge by water the background step darkens the water,
    so "darken the background, keeping the water and ledge untouched" is
    rewritten to say what the steps touch.
    """

    sentence = str(text or "")
    if not sentence:
        return sentence
    touched = []
    if water:
        touched.append("the water behind it is the background that gets darker and muted")
    if ground:
        touched.append("the bright ledge or ground below is darkened with a bottom gradient")
    pattern = re.compile(
        r",?\s*(?:while\s+)?(?:keeping|leaving)\s+(?:the\s+)?[a-z ,]*?(?:water|ledge|ground|pond|lake|river|concrete)[a-z ,]*?(?:untouched|alone|as it is|unchanged)",
        re.IGNORECASE,
    )
    if not pattern.search(sentence):
        return sentence
    fixed = pattern.sub("", sentence).rstrip(" ,.")
    if touched:
        fixed = f"{fixed}; " + " and ".join(touched)
    return fixed + "."


def _titles(noun: str, bird: bool) -> dict[str, str]:
    coat = "feathers" if bird else "fur"
    return {
        "mask_animal": f"Lift {noun} a little and crisp its {coat}",
        "mask_animals": "Lift every animal a little",
        "mask_animal_background": f"Darken and mute everything around {noun}",
        "mask_bottom_gradient": "Darken the bright ground at the bottom",
        "mask_edge_burn": f"Burn the edges, keeping {noun} clear",
        "mask_glow": f"Add a subtle glow behind {noun}",
    }


def animal_plan(
    plan: dict,
    *,
    steer: str = "",
    description: str = "",
    animals: int = 1,
    ground: bool = False,
    water: bool = False,
    applied: list | None = None,
) -> dict:
    """Rewrite the accept queue for an animal subject (see the module doc)."""

    noun = animal_noun(steer, description)
    bird = is_bird(names_animal(steer)) or is_bird(names_animal(description))
    titles = _titles(noun, bird)
    wants_eyes = bool(re.search(r"\beyes?\b", str(steer or ""), re.IGNORECASE))
    queue = [dict(item) for item in (plan.get("step_queue") or []) if isinstance(item, dict)]
    nxt = str(plan.get("features_next") or "")
    if not queue and nxt:
        queue = [{"id": nxt, "title": str(plan.get("features_next_title") or ""), "action": str(plan.get("features_next_action") or "")}]
    # Steps already on the photo, or skipped, in either naming.
    blocked = set()
    for name in list(plan.get("features_done") or []) + list(plan.get("features_skipped") or []) + list(applied or []):
        name = str(name or "")
        blocked.add(name)
        blocked.add(_ANIMAL_MAP.get(name, name))
    if "mask_animal" in blocked or "mask_animals" in blocked:
        blocked.update({"mask_animal", "mask_animals"})
    lead = "mask_animals" if animals > 1 else "mask_animal"
    mapped_queue: list[dict] = []
    seen: set[str] = set()
    for item in queue:
        ident = str(item.get("id") or "")
        if ident in PERSON_STEPS or (ident == "eyes" and not wants_eyes):
            continue
        mapped = _ANIMAL_MAP.get(ident, ident)
        if mapped in {"mask_animal", "mask_animals"}:
            mapped = lead
        if mapped in seen:
            continue
        seen.add(mapped)
        if mapped in _ORDER:
            mapped_queue.append({"id": mapped, "action": _ACTIONS[mapped], "title": titles[mapped]})
            continue
        title = " ".join(str(item.get("title") or "").split())
        if _echoes(title, steer):
            from ai_dt.features import feature_title

            item["title"] = feature_title(ident)
        mapped_queue.append(item)
    # A general "make it the main theme" plan gets the whole animal recipe:
    # lift, quiet around it, the bright ground, the edges. A specific request
    # (the eyes, one colour) keeps the planner's own steps and order.
    emphasis = any(item["id"] in {lead, "mask_animal_background"} for item in mapped_queue)
    general = not wants_eyes and (not str(steer or "").strip() or bool(_GENERAL.search(str(steer or ""))))
    if emphasis and general:
        wanted = [lead, "mask_animal_background"] + (["mask_bottom_gradient"] if ground else []) + ["mask_edge_burn"]
        present = {item["id"] for item in mapped_queue}
        for ident in wanted:
            if ident not in present and ident not in blocked:
                mapped_queue.append({"id": ident, "action": _ACTIONS[ident], "title": titles[ident]})
        animal_part = sorted((item for item in mapped_queue if item["id"] in _ORDER), key=lambda item: _ORDER.index(item["id"]))
        rebuilt = animal_part + [item for item in mapped_queue if item["id"] not in _ORDER]
    else:
        rebuilt = mapped_queue
    ordered = [str(item["id"]) for item in rebuilt]
    follow = rebuilt[0] if rebuilt else {}
    plan["step_queue"] = rebuilt
    plan["features_path"] = [str(item.get("id") or "") for item in rebuilt]
    plan["features_next"] = str(follow.get("id") or "")
    plan["features_next_title"] = str(follow.get("title") or "")
    plan["features_next_action"] = str(follow.get("action") or "")
    plan["subject_kind"] = "animal"
    plan["subject_noun"] = noun
    plan["bright_ground"] = bool(ground)
    for key in ("guide_stages", "stages"):
        if isinstance(plan.get(key), list):
            plan[key] = [stage for stage in plan[key] if stage not in PERSON_STEPS]
    held = [name for name in (plan.get("not_history") or []) if name not in PERSON_STEPS and not (name == "background" and "mask_animal_background" in ordered)]
    plan["not_history"] = held
    done = [name for name in (plan.get("features_done") or []) if name not in PERSON_STEPS]
    plan["features_done"] = done
    reasoning = plan.get("reasoning") if isinstance(plan.get("reasoning"), dict) else None
    if reasoning is not None and str(reasoning.get("paragraph") or ""):
        paragraph = generic_species(str(reasoning["paragraph"]), noun, steer)
        reasoning["paragraph"] = consistent_regions(paragraph, water=water, ground=ground)
    return plan


_GENERAL = re.compile(
    r"main (theme|subject)|stand out|\bpop\b|focus|emphasi|hero|\bfinish\b|subject|prominent|draw the eye",
    re.IGNORECASE,
)


def _echoes(title: str, steer: str) -> bool:
    """A step title that only repeats the request, not an action."""

    a = " ".join(_words(title))
    b = " ".join(_words(steer))
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def subject_line(noun: str, ground: bool, water: bool) -> str:
    """The brief line that tells the planner the subject is an animal."""

    parts = [
        f"Subject: an animal ({noun}). Do not use skin, face, person or mask_people steps.",
        "Use mask_animal (small lift), then mask_animal_background (darken and mute around it)",
    ]
    if ground:
        parts.append("then mask_bottom_gradient for the bright ground or ledge under it")
    parts.append("then mask_edge_burn; mask_glow only as a subtle option.")
    regions = []
    if water:
        regions.append("the background is the water")
    if ground:
        regions.append("the ground/foreground is the bright ledge")
    text = " ".join(parts[:1]) + " " + ", ".join(parts[1:])
    if regions:
        text += " Region words: " + "; ".join(regions) + ". Do not say a step leaves a region untouched when that step darkens it."
    return text
