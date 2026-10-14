"""Name the picture in front of the processor.

Detectors contribute faces, eyes, animals, and architecture. The pixels
contribute the rest: how bright the frame is, whether the upper part reads
as sky, whether a compact body stands apart from the background, and whether
the frame is a grid of lines. YOLO is not required.

The vibe is not an input. A sentence cannot turn a portrait into wildlife.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageFilter

from ai_dt.guides.support import boxes_overlap, creature_box, is_elongated, is_face_bar
from ai_dt.schema import Detection

# A warm compact body has to be this solid before it beats a night reading.
# Scattered window lights fail the fill test and never reach this bar.
_BODY_AREA = 0.03
_BODY_FILL = 0.34
# Share of pixels that are hard edges. A facade is full of them. A solid
# subject only has an outline, so its fraction stays low.
_LINE_FRACTION = 0.055
_NIGHT_BRIGHTNESS = 0.36
_NIGHT_WARMTH = -0.055
_CITY_BRIGHTNESS = 0.48
_CITY_WARMTH = -0.04
_SKY_MIN = 0.5


@dataclass
class Blob:
    """One compact region, in normalized top-left coordinates."""

    box: tuple[float, float, float, float]
    fill: float
    area: float


@dataclass
class Picture:
    """The type decided for one frame, and the measurements behind it."""

    type: str
    reason: str
    brightness: float
    warmth: float
    saturation: float
    edges: float
    line_fraction: float
    sky: float
    skin: float
    highlight_specks: float
    compact_area: float
    compact_fill: float
    subject_box: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "reason": self.reason,
            "measurements": {
                "brightness": round(float(self.brightness), 4),
                "warmth": round(float(self.warmth), 4),
                "saturation": round(float(self.saturation), 4),
                "edges": round(float(self.edges), 4),
                "line_fraction": round(float(self.line_fraction), 4),
                "sky": round(float(self.sky), 4),
                "skin": round(float(self.skin), 4),
                "highlight_specks": round(float(self.highlight_specks), 4),
                "compact_area": round(float(self.compact_area), 4),
                "compact_fill": round(float(self.compact_fill), 4),
            },
        }


def _working(pixels: np.ndarray) -> np.ndarray:
    """Shrink a large frame so the type decision stays cheap.

    Test-sized pictures are left alone, so a subject box matches the one
    the renderer builds. A half-resolution RAW is reduced to its long side.
    """

    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    longest = max(height, width)
    if longest <= 960:
        return pixels
    scale = 960.0 / float(longest)
    image = Image.fromarray(np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BOX)
    return np.asarray(image).astype(np.float32) / np.float32(255.0)


def _luma(pixels: np.ndarray) -> np.ndarray:
    return 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]


def _blur(values: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    image = image.filter(ImageFilter.GaussianBlur(radius=max(0.4, radius)))
    return np.asarray(image).astype(np.float32) / np.float32(255.0)


def _blur_radius(height: int, width: int) -> float:
    # 2.2 at the bird test's short side, so that mask does not move.
    return 2.2 * max(1.0, min(height, width) / 480.0)


def _box_from_core(core: np.ndarray) -> Blob | None:
    count = int(core.sum())
    height, width = core.shape
    area = height * width
    if count < 40 or count > int(0.4 * area):
        return None
    ys, xs = np.nonzero(core)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    box_area = max(1, box_w * box_h)
    fill = count / box_area
    fraction = box_area / area
    if fill < 0.18 or fraction > 0.4 or fraction < 0.008:
        return None
    pad_x = int(box_w * 0.4)
    pad_y = int(box_h * 0.45)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width - 1, x1 + pad_x)
    y1 = min(height - 1, y1 + pad_y)
    box = (x0 / width, y0 / height, (x1 - x0 + 1) / width, (y1 - y0 + 1) / height)
    return Blob(box=box, fill=float(fill), area=float(fraction))


def warm_subject_box(pixels: np.ndarray) -> Blob | None:
    """A compact red, yellow, orange, or brown body.

    Green is ignored so foliage does not become the subject. The box is the
    whole warm core: scattered lights make a large, empty box and are rejected.
    """

    red = pixels[..., 0] - np.maximum(pixels[..., 1], pixels[..., 2])
    yellow = np.minimum(pixels[..., 0], pixels[..., 1]) - pixels[..., 2]
    green = (pixels[..., 1] > pixels[..., 0] + 0.04) & (pixels[..., 1] > pixels[..., 2] + 0.02)
    score = np.clip(red, 0.0, 1.0) * np.float32(1.3) + np.clip(yellow, 0.0, 1.0)
    score = np.where(green, np.float32(0.0), score)
    height, width = score.shape
    radius = _blur_radius(height, width)
    core = _blur((score > np.float32(0.2)).astype(np.float32), radius) > np.float32(0.35)
    return _box_from_core(core)


def plain_subject_box(pixels: np.ndarray) -> Blob | None:
    """A compact body of any color, measured against the border.

    The border stands in for the background. A subject that fills the frame,
    or a grid of lines, does not survive the fill and area gates.
    """

    height, width = pixels.shape[:2]
    band_y = max(1, height // 16)
    band_x = max(1, width // 16)
    border = np.concatenate(
        (
            pixels[:band_y].reshape(-1, 3),
            pixels[-band_y:].reshape(-1, 3),
            pixels[:, :band_x].reshape(-1, 3),
            pixels[:, -band_x:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(border, axis=0).astype(np.float32)
    distance = np.sqrt(np.sum((pixels - background) ** 2, axis=2))
    core = distance > np.float32(0.18)
    blob = _box_from_core(core)
    if blob is None:
        return None
    if _interior_edge(pixels, blob) > 0.1:
        return None
    return blob


def skin_subject_box(pixels: np.ndarray) -> Blob | None:
    """A compact skin-toned region, used when no face box was returned."""

    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    spread = np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)
    # Blue has to stay up. A yellow belly passes a loose skin test and must not.
    skin = (
        (red > 0.38)
        & (green > 0.18)
        & (blue > 0.28)
        & (red > green + 0.04)
        & (red > blue)
        & (blue > red * 0.45)
        & (spread > 0.08)
    )
    return _box_from_core(skin)


def _interior_edge(pixels: np.ndarray, blob: Blob) -> float:
    height, width = pixels.shape[:2]
    x, y, bw, bh = blob.box
    x0 = min(width - 1, max(0, int(x * width)))
    y0 = min(height - 1, max(0, int(y * height)))
    x1 = min(width, max(x0 + 1, int((x + bw) * width)))
    y1 = min(height, max(y0 + 1, int((y + bh) * height)))
    luma = _luma(pixels[y0:y1, x0:x1])
    if luma.size < 16:
        return 0.0
    gy = np.zeros_like(luma)
    gx = np.zeros_like(luma)
    gy[1:, :] = np.abs(luma[1:, :] - luma[:-1, :])
    gx[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
    return float((gx + gy).mean())


def _edges(luma: np.ndarray) -> tuple[float, float]:
    gy = np.zeros_like(luma)
    gx = np.zeros_like(luma)
    gy[1:, :] = np.abs(luma[1:, :] - luma[:-1, :])
    gx[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
    edge = gx + gy
    return float(edge.mean()), float((edge > np.float32(0.18)).mean())


def _sky_score(pixels: np.ndarray, luma: np.ndarray) -> float:
    height = luma.shape[0]
    if height < 12:
        return 0.0
    split = max(1, height // 3)
    upper_l = luma[:split]
    lower_l = luma[-split:]
    upper = pixels[:split]
    upper_blue = float((upper[..., 2] - upper[..., 0]).mean())
    upper_edge, _ = _edges(upper_l)
    lower_edge, _ = _edges(lower_l)
    score = 0.0
    if upper_blue > 0.04 and upper_edge + 0.008 < lower_edge:
        score += 0.5
    if float(upper_l.mean()) > float(lower_l.mean()) + 0.05 and upper_edge < 0.06:
        score += 0.4
    if upper_blue > 0.08 and float(upper_l.mean()) > 0.45 and upper_edge < 0.05:
        score += 0.25
    return min(1.0, score)


def _specks(luma: np.ndarray) -> float:
    bright = luma > np.float32(0.75)
    # A broad highlight is a sky or a wall, not a window.
    if float(bright.mean()) > 0.12:
        return 0.0
    return float(bright.mean())


def _skin_fraction(pixels: np.ndarray) -> float:
    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    spread = np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)
    skin = (
        (red > 0.38)
        & (green > 0.18)
        & (blue > 0.28)
        & (red > green + 0.04)
        & (red > blue)
        & (blue > red * 0.45)
        & (spread > 0.08)
    )
    return float(skin.mean())


def _crop(pixels: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = pixels.shape[:2]
    x, y, bw, bh = box
    x0 = min(width - 1, max(0, int(x * width)))
    y0 = min(height - 1, max(0, int(y * height)))
    x1 = min(width, max(x0 + 1, int((x + bw) * width)))
    y1 = min(height, max(y0 + 1, int((y + bh) * height)))
    return pixels[y0:y1, x0:x1]


def _face_ok(pixels: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    """Skin, a quiet interior, and two separated dark spots.

    A warm textured cluster fails this, and so does a facade or a field.
    The check does not name what the cluster is.
    """

    from ai_dt.detectors.heuristic import heuristic_face_ok

    return heuristic_face_ok(pixels, box)


def _face_is_skin(pixels: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    """Reject a yellow body that the loose detector called a face.

    The heuristic skin test accepts a yellow belly. A real face keeps blue
    in the skin. If the box fails that, it is not a person.
    """

    region = _crop(pixels, box)
    if region.size == 0:
        return False
    if _skin_fraction(region) < 0.34:
        return False
    # Green packed through the box is not a cheek, for example a flower cluster.
    red = region[..., 0]
    green = region[..., 1]
    blue = region[..., 2]
    leafy = (green > red + 0.02) & (green > blue + 0.02)
    return float(leafy.mean()) < 0.18


def _people(detections: list[Detection]) -> list[Detection]:
    return [item for item in detections if item.category in {"face", "person"} and _usable(item.bbox)]


def _usable(bbox: tuple[float, float, float, float]) -> bool:
    return bbox[2] > 0.01 and bbox[3] > 0.01


def _area(bbox: tuple[float, float, float, float]) -> float:
    return float(bbox[2] * bbox[3])


def _has(detections: list[Detection], category: str) -> bool:
    return any(item.category == category and _usable(item.bbox) for item in detections)


def _is_body(blob: Blob | None) -> bool:
    return blob is not None and blob.area >= _BODY_AREA and blob.fill >= _BODY_FILL


def _open_land(pixels: np.ndarray, brightness: float, line_fraction: float) -> bool:
    """A field or other broad green land. Crop rows are not a facade."""

    if brightness < 0.28:
        return False
    if line_fraction >= _LINE_FRACTION * 0.45:
        return False
    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    leafy = (green > red + np.float32(0.02)) & (green > blue + np.float32(0.02))
    return float(leafy.mean()) >= 0.55


def _is_night(brightness: float, warmth: float, specks: float) -> bool:
    if brightness < _NIGHT_BRIGHTNESS and warmth < _NIGHT_WARMTH:
        return True
    return (
        brightness < _CITY_BRIGHTNESS
        and warmth < _CITY_WARMTH
        and 0.0015 < specks < 0.08
    )


def _foodish(pixels: np.ndarray, blob: Blob) -> bool:
    x, y, bw, bh = blob.box
    height, width = pixels.shape[:2]
    x0 = min(width - 1, max(0, int(x * width)))
    y0 = min(height - 1, max(0, int(y * height)))
    x1 = min(width, max(x0 + 1, int((x + bw) * width)))
    y1 = min(height, max(y0 + 1, int((y + bh) * height)))
    region = pixels[y0:y1, x0:x1]
    if region.size == 0 or bw < bh * 1.12:
        return False
    luma = _luma(region)
    chroma = np.max(np.abs(region - luma[..., None]), axis=2)
    warmth = float((region[..., 0] - region[..., 2]).mean())
    return float(chroma.mean()) > 0.12 and warmth > 0.03


def _macroish(pixels: np.ndarray, blob: Blob) -> bool:
    if blob.area < 0.18:
        return False
    luma = _luma(pixels)
    height, width = luma.shape
    corner = np.concatenate(
        (
            luma[: height // 6, : width // 6].ravel(),
            luma[: height // 6, -width // 6 :].ravel(),
            luma[-height // 6 :, : width // 6].ravel(),
            luma[-height // 6 :, -width // 6 :].ravel(),
        )
    )
    x, y, bw, bh = blob.box
    x0 = min(width - 1, max(0, int((x + bw * 0.3) * width)))
    y0 = min(height - 1, max(0, int((y + bh * 0.3) * height)))
    x1 = min(width, max(x0 + 1, int((x + bw * 0.7) * width)))
    y1 = min(height, max(y0 + 1, int((y + bh * 0.7) * height)))
    center = luma[y0:y1, x0:x1]
    if center.size < 16 or corner.size < 16:
        return False
    # Soft corners and a busier center read as a close subject.
    return float(np.std(center)) > float(np.std(corner)) * 1.4


def _blob_chroma(pixels: np.ndarray, blob: Blob) -> float:
    region = _crop(pixels, blob.box)
    if region.size == 0:
        return 0.0
    luma = _luma(region)
    chroma = np.max(np.abs(region - luma[..., None]), axis=2)
    return float(chroma.mean())


def _still(pixels: np.ndarray, blob: Blob, saturation: float) -> bool:
    # Color inside the object matters. A gray frame around a blue subject
    # must not hide that color and turn a creature into a still life.
    if saturation > 0.14 or _blob_chroma(pixels, blob) > 0.1 or blob.area < 0.02:
        return False
    luma = _luma(pixels)
    return float(luma.std()) < 0.22


def _picture(
    picture_type: str,
    reason: str,
    *,
    brightness: float,
    warmth: float,
    saturation: float,
    edges: float,
    line_fraction: float,
    sky: float,
    skin: float,
    specks: float,
    blob: Blob | None,
) -> Picture:
    return Picture(
        type=picture_type,
        reason=reason,
        brightness=brightness,
        warmth=warmth,
        saturation=saturation,
        edges=edges,
        line_fraction=line_fraction,
        sky=sky,
        skin=skin,
        highlight_specks=specks,
        compact_area=0.0 if blob is None else blob.area,
        compact_fill=0.0 if blob is None else blob.fill,
        subject_box=None if blob is None else blob.box,
    )


def decide_picture(pixels: np.ndarray, detections: list[Detection] | None = None) -> Picture:
    """Choose the picture type that should pick the guide.

    Faces win, then a real compact body, then night, then a line-filled
    facade, then sky, then a quieter object. Anything unresolved stays general.
    """

    found = list(detections or [])
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    empty = _picture(
        "general",
        "The image is too small to judge a picture type.",
        brightness=0.0,
        warmth=0.0,
        saturation=0.0,
        edges=0.0,
        line_fraction=0.0,
        sky=0.0,
        skin=0.0,
        specks=0.0,
        blob=None,
    )
    if min(height, width) < 8:
        return empty

    view = _working(np.clip(pixels.astype(np.float32, copy=False), 0.0, 1.0))
    luma = _luma(view)
    brightness = float(luma.mean())
    warmth = float((view[..., 0] - view[..., 2]).mean())
    chroma = np.max(np.abs(view - luma[..., None]), axis=2)
    saturation = float(chroma.mean())
    edges, line_fraction = _edges(luma)
    sky = _sky_score(view, luma)
    skin = _skin_fraction(view)
    specks = _specks(luma)
    warm = warm_subject_box(view)
    # A full-width branch is warm and woody. It is the perch, not the animal.
    if warm is not None and is_elongated(warm.box, view.shape):
        warm = None
    # A textured cluster is not a compact creature. The interior is busy.
    if warm is not None and _interior_edge(view, warm) > 0.1:
        warm = None
    plain = plain_subject_box(view)
    skin_box = skin_subject_box(view)
    # A warm horizon can fill a wide skin box. That strip is not a head.
    if skin_box is not None and (is_face_bar(skin_box.box) or is_elongated(skin_box.box, view.shape)):
        skin_box = None
    creature_blob: Blob | None | bool = None

    def _creature() -> Blob | None:
        nonlocal creature_blob
        if creature_blob is False:
            return None
        if isinstance(creature_blob, Blob):
            return creature_blob
        found_region = creature_box(view)
        if found_region is None:
            creature_blob = False
            return None
        box, fill, area = found_region
        creature_blob = Blob(box=box, fill=fill, area=area)
        return creature_blob

    facts = dict(
        brightness=brightness,
        warmth=warmth,
        saturation=saturation,
        edges=edges,
        line_fraction=line_fraction,
        sky=sky,
        skin=skin,
        specks=specks,
    )

    people = [item for item in _people(found) if not is_face_bar(item.bbox)]
    real_faces = [item for item in people if _face_ok(view, item.bbox)]
    creature_now = _creature()
    if creature_now is not None:
        # A patch away from the compact body is not the subject, for example a flower cluster.
        people = [item for item in people if boxes_overlap(item.bbox, creature_now.box)]
        real_faces = [item for item in real_faces if boxes_overlap(item.bbox, creature_now.box)]
    # A warm body can fool the face detector. Trust the face only when the
    # box is actually skin; otherwise a bird shaped like a face stays an animal.
    if real_faces:
        people = real_faces
    elif people and warm is not None:
        return _picture(
            "animal",
            "A warm compact subject was read as a face, but the color is an animal, not skin.",
            blob=warm,
            **facts,
        )
    elif people and _creature() is not None:
        # A brown branch can pass the loose skin test. If a real creature is
        # in the frame, that branch is not a person.
        people = []
    if len(people) >= 2:
        return _picture(
            "group",
            "Several faces share the frame, so this is a group.",
            blob=_largest_blob(people),
            **facts,
        )
    if len(people) == 1:
        area = _area(people[0].bbox)
        structured = line_fraction >= _LINE_FRACTION * 0.85 or _has(found, "architecture")
        if area < 0.1 and structured:
            return _picture(
                "street",
                "A person is small in a structured place, so this is a street picture.",
                blob=Blob(people[0].bbox, fill=1.0, area=area),
                **facts,
            )
        return _picture(
            "portrait",
            "A face is the clear subject, so this is a portrait.",
            blob=Blob(people[0].bbox, fill=1.0, area=area),
            **facts,
        )

    if skin_box is not None and _is_body(skin_box) and _face_ok(view, skin_box.box):
        return _picture(
            "portrait",
            "A skin-toned face fills part of the frame, so this is a portrait.",
            blob=skin_box,
            **facts,
        )

    if creature_now is not None and not real_faces and _face_ok(view, creature_now.box):
        return _picture(
            "portrait",
            "A skin-toned face with a quiet interior and two dark spots fills the frame, so this is a portrait.",
            blob=creature_now,
            **facts,
        )

    # A compact body of its own color is the animal. A separate warm patch
    # is not that body, and a mirror in the water is not one either.
    if creature_now is not None and not real_faces:
        return _picture(
            "animal",
            "A compact subject of its own color stands apart, so this is an animal.",
            blob=creature_now,
            **facts,
        )

    if _has(found, "animal") or _is_body(warm):
        return _picture(
            "animal",
            "A compact subject stands apart from the background, so this is an animal.",
            blob=warm,
            **facts,
        )

    if _is_night(brightness, warmth, specks):
        return _picture(
            "night",
            "The frame is dark and cool, with no compact subject, so this is night.",
            blob=None,
            **facts,
        )

    # A smaller warm subject still counts once night has been ruled out.
    # Window lights never get here: their box is too empty to be returned.
    if warm is not None:
        return _picture(
            "animal",
            "A warm compact subject stands apart from the background, so this is an animal.",
            blob=warm,
            **facts,
        )

    if line_fraction >= _LINE_FRACTION or (
        _has(found, "architecture") and line_fraction >= _LINE_FRACTION * 0.45
    ):
        # Needles and window grids are full of edges. A compact creature in
        # front of them is still the subject; the lines are the background.
        creature = _creature()
        if creature is not None:
            return _picture(
                "animal",
                "A compact subject stands apart from a busy background, so this is an animal.",
                blob=creature,
                **facts,
            )
        return _picture(
            "architecture",
            "Straight edges fill the frame and nothing stands apart, so this is architecture.",
            blob=None,
            **facts,
        )

    if sky >= _SKY_MIN and plain is None:
        return _picture(
            "landscape",
            "The upper frame reads as sky over a wider scene, so this is a landscape.",
            blob=None,
            **facts,
        )

    if plain is not None and _macroish(view, plain):
        return _picture(
            "macro",
            "A close subject fills the middle and the corners fall away, so this is a macro.",
            blob=plain,
            **facts,
        )
    if plain is not None and _foodish(view, plain):
        return _picture(
            "food",
            "A wide, warm dish sits in the frame, so this is food.",
            blob=plain,
            **facts,
        )
    if plain is not None and _still(view, plain, saturation):
        return _picture(
            "still-life",
            "One quiet object sits against an even background, so this is a still life.",
            blob=plain,
            **facts,
        )
    if plain is not None:
        return _picture(
            "animal",
            "A compact subject of its own color stands apart, so this is an animal.",
            blob=plain,
            **facts,
        )

    if sky >= _SKY_MIN:
        return _picture(
            "landscape",
            "The upper frame reads as sky over a wider scene, so this is a landscape.",
            blob=None,
            **facts,
        )
    if _open_land(view, brightness, line_fraction) and _creature() is None:
        return _picture(
            "landscape",
            "Open land fills the frame, so this is a landscape.",
            blob=None,
            **facts,
        )
    if _has(found, "architecture") and line_fraction >= _LINE_FRACTION * 0.45:
        creature = _creature()
        if creature is not None:
            return _picture(
                "animal",
                "A compact subject stands apart from a busy background, so this is an animal.",
                blob=creature,
                **facts,
            )
        return _picture(
            "architecture",
            "The detector found structure and no separate subject, so this is architecture.",
            blob=None,
            **facts,
        )
    creature = _creature()
    if creature is not None:
        return _picture(
            "animal",
            "A compact subject of its own color stands apart, so this is an animal.",
            blob=creature,
            **facts,
        )
    return _picture(
        "general",
        "Nothing in the frame is a clear subject, so this stays a general picture.",
        blob=None,
        **facts,
    )


def _largest_blob(people: list[Detection]) -> Blob:
    box = max(people, key=lambda item: _area(item.bbox)).bbox
    return Blob(box=box, fill=1.0, area=_area(box))


# A steer can name the thing the compact-blob detector called an animal.
_VEHICLE_WORDS = ("train", "tram", "metro", "subway", "locomotive", "railway")
_URBAN_WORDS = frozenset({"urban", "city", "street", "downtown", "night"})


def steer_words(steer: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (steer or "").lower()))


def named_vehicle(steer: str) -> str:
    """The vehicle the user named, when the detector did not find one."""

    words = steer_words(steer)
    for word in _VEHICLE_WORDS:
        if word in words:
            return word
    return ""


def apply_steer_scene(
    picture: Picture,
    detections: list[Detection] | None,
    steer: str,
) -> tuple[Picture, list[Detection], str, bool]:
    """Let a train or urban steer replace a false animal label.

    A compact light shape on a night street, such as a train, is filed as
    an animal before the night reading runs. Words the user already typed
    win over that label. The picture stays on an existing night or street
    type. A real detector animal is kept when the steer does not name a
    vehicle. Returns the picture, the detections, the subject name, and
    whether the animal label was replaced.
    """

    found = list(detections or [])
    subject = named_vehicle(steer)
    words = steer_words(steer)
    urban = bool(subject) or bool(words & _URBAN_WORDS)
    if picture.type != "animal" or not urban:
        keep_name = subject if picture.type in {"night", "street", "architecture", "general"} else ""
        return picture, found, keep_name, False
    real_animal = any(item.category == "animal" and item.source != "heuristic" for item in found)
    if real_animal and not subject:
        return picture, found, "", False
    dark = picture.brightness < 0.45 or "night" in words
    if dark:
        new_type = "night"
        reason = "The request names a night urban scene"
    else:
        new_type = "street"
        reason = "The request names an urban scene"
    if subject:
        reason += f", and the {subject} is the subject the detector did not find"
    reason += ", so this is not an animal."
    updated = replace(
        picture,
        type=new_type,
        reason=reason,
        subject_box=None,
        compact_area=0.0,
        compact_fill=0.0,
    )
    kept = [item for item in found if item.category != "animal"]
    return updated, kept, subject, True
