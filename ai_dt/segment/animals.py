"""One outline per animal, plus an ``all animals`` union.

Boxes come from the detector (YOLO COCO: bird, cat, dog, horse, sheep, cow,
elephant, bear, zebra, giraffe all map to category ``animal``). Each box is
refined to an outline on its own crop, so a small or distant animal gets
the same pixel budget as a big one:

1. Keep a box if its confidence clears a size-aware bar (a small animal
   scores lower, so it gets a lower bar) and it is not a duplicate of a
   stronger box.
2. Crop the box with a margin, upscale tiny crops, and ask the model
   backends (darktable's SAM 2.1, rembg, Gemini when opted in) for the
   ``subject`` of that crop.
3. Without a model, a color model of the box against its ring decides.
   A morphological close bridges thin occluders (grass, branches, fence
   wires) so a partly hidden animal keeps one outline; fragments far from
   the box core are dropped.
4. A refined mask that is empty, a sliver, or fills the whole box is not
   trusted: a soft ellipse inside the box stands in, and says so.
5. A mask with almost no pixels is skipped with a note.

The pose does not matter to any of this: standing, lying, running, birds
perched or in flight are all "the thing in the box that differs from its
surroundings".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ai_dt.segment.base import box_of

ANIMAL_LABELS = frozenset(
    {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal"}
)

# Size-aware confidence bars. The detector already passes boxes down to 0.15.
SMALL_AREA = 0.01
MEDIUM_AREA = 0.05
CONFIDENCE_SMALL = 0.15
CONFIDENCE_MEDIUM = 0.2
CONFIDENCE_LARGE = 0.25
MIN_BOX_SIDE = 0.008  # under this the box is detector noise
DUPLICATE_IOU = 0.6
MIN_PIXELS = 24  # an animal mask with fewer preview pixels is skipped
MAX_ANIMALS = 12
CROP_MARGIN = 0.2
CROP_MIN = 96
CROP_MAX = 320


@dataclass
class Animal:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    mask: np.ndarray  # full frame, 0..1
    source: str
    crop: tuple[float, float, float, float]  # normalized (x, y, w, h) of crop_mask
    crop_mask: np.ndarray

    @property
    def area(self) -> float:
        return float(self.bbox[2] * self.bbox[3])


@dataclass
class AnimalResult:
    animals: list[Animal] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def union(self, size: tuple[int, int]) -> np.ndarray | None:
        if not self.animals:
            return None
        width, height = size
        out = np.zeros((height, width), dtype=np.float32)
        for animal in self.animals:
            out = np.maximum(out, animal.mask)
        return out


def _label_of(item) -> str:
    if isinstance(item, dict):
        return str(item.get("label") or "").strip().lower()
    return str(getattr(item, "label", "") or "").strip().lower()


def _confidence_of(item) -> float | None:
    raw = item.get("confidence") if isinstance(item, dict) else getattr(item, "confidence", None)
    try:
        return None if raw is None else float(raw)
    except (TypeError, ValueError):
        return None


def confidence_bar(area: float) -> float:
    if area < SMALL_AREA:
        return CONFIDENCE_SMALL
    if area < MEDIUM_AREA:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LARGE


def _box_iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def animal_boxes(boxes: list, notes: list[str] | None = None) -> list[tuple[str, float, tuple]]:
    """``(label, confidence, (x, y, w, h))`` for the animal boxes worth refining."""

    notes = notes if notes is not None else []
    kept: list[tuple[str, float, tuple]] = []
    for item in boxes or []:
        parsed = box_of(item)
        if not parsed:
            continue
        category, box = parsed
        label = _label_of(item) or category
        if category != "animal" and label not in ANIMAL_LABELS:
            continue
        x, y, w, h = box
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        w, h = min(w, 1.0 - x), min(h, 1.0 - y)
        if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
            notes.append(f"a {label} box was too small to outline")
            continue
        confidence = _confidence_of(item)
        score = 1.0 if confidence is None else confidence
        if score < confidence_bar(w * h):
            notes.append(f"a {label} at {score:.2f} confidence was below the bar for its size")
            continue
        kept.append((label, score, (x, y, w, h)))
    kept.sort(key=lambda entry: entry[1], reverse=True)
    unique: list[tuple[str, float, tuple]] = []
    for entry in kept:
        if any(_box_iou(entry[2], other[2]) > DUPLICATE_IOU for other in unique):
            continue
        unique.append(entry)
    if len(unique) > MAX_ANIMALS:
        notes.append(f"{len(unique) - MAX_ANIMALS} more animals were left out")
        unique = sorted(unique, key=lambda entry: entry[2][2] * entry[2][3], reverse=True)[:MAX_ANIMALS]
    return unique


def _crop_rect(box, size) -> tuple[int, int, int, int]:
    width, height = size
    x, y, w, h = box
    mx = max(CROP_MARGIN * w, 4.0 / width)
    my = max(CROP_MARGIN * h, 4.0 / height)
    x0 = int(max(0, np.floor((x - mx) * width)))
    y0 = int(max(0, np.floor((y - my) * height)))
    x1 = int(min(width, np.ceil((x + w + mx) * width)))
    y1 = int(min(height, np.ceil((y + h + my) * height)))
    return x0, y0, max(x1, x0 + 2), max(y1, y0 + 2)


def _scaled(crop: Image.Image) -> Image.Image:
    short = min(crop.size)
    long = max(crop.size)
    scale = 1.0
    if short < CROP_MIN:
        scale = CROP_MIN / float(short)
    if long * scale > CROP_MAX:
        scale = CROP_MAX / float(long)
    if abs(scale - 1.0) < 1e-3:
        return crop
    size = (max(8, int(round(crop.width * scale))), max(8, int(round(crop.height * scale))))
    return crop.resize(size, Image.BILINEAR)


def _binary_dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    out = binary.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            shifted = np.zeros_like(binary)
            ys = slice(max(0, dy), binary.shape[0] + min(0, dy))
            yd = slice(max(0, -dy), binary.shape[0] + min(0, -dy))
            xs = slice(max(0, dx), binary.shape[1] + min(0, dx))
            xd = slice(max(0, -dx), binary.shape[1] + min(0, -dx))
            shifted[ys, xs] = binary[yd, xd]
            out |= shifted
    return out


def close_gaps(binary: np.ndarray, radius: int) -> np.ndarray:
    """Dilate then erode: bridges gaps up to about ``2 * radius`` pixels."""

    if radius < 1:
        return binary
    grown = _binary_dilate(binary, radius)
    return ~_binary_dilate(~grown, radius)


def _keep_near_core(binary: np.ndarray, inner: tuple[int, int, int, int]) -> np.ndarray:
    from ai_dt.dt_shapes import label_components

    labels, count = label_components(binary)
    if count == 0:
        return binary
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    biggest = int(sizes.max())
    x0, y0, x1, y1 = inner
    cw, ch = x1 - x0, y1 - y0
    core = np.zeros_like(binary)
    core[y0 + ch // 4:y1 - ch // 4 + 1, x0 + cw // 4:x1 - cw // 4 + 1] = True
    touching = set(np.unique(labels[core & binary]).tolist()) - {0}
    keep = np.zeros(count + 1, dtype=bool)
    for index in range(1, count + 1):
        keep[index] = index in touching or sizes[index] >= 0.15 * biggest
    return keep[labels]


def heuristic_outline(pixels: np.ndarray, inner: tuple[int, int, int, int]) -> np.ndarray | None:
    """Color model of the box against its ring, occlusion-tolerant."""

    from ai_dt.segment.heuristic import _blur, _color_codes

    height, width = pixels.shape[:2]
    x0, y0, x1, y1 = inner
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    codes = _color_codes(pixels)
    box = np.zeros((height, width), dtype=bool)
    box[y0:y1, x0:x1] = True
    core = np.zeros_like(box)
    bw, bh = x1 - x0, y1 - y0
    core[y0 + bh // 5:y1 - bh // 5, x0 + bw // 5:x1 - bw // 5] = True
    ring = ~box
    if ring.sum() < 8 or core.sum() < 4:
        return None
    bg = np.bincount(codes[ring], minlength=512).astype(np.float64) + 0.5
    bg /= bg.sum()
    # Colors that are common outside the box are background even inside the
    # core (grass in front of a deer), so the core histogram is cleaned first.
    fg = np.bincount(codes[core], minlength=512).astype(np.float64) + 0.5
    fg /= fg.sum()
    fg = np.where(bg >= 0.8 * fg, 0.0, fg)
    likely = (fg[codes] > bg[codes] * 1.5) & box
    blob = _blur(likely.astype(np.float32), 1) > 0.45
    if not blob.any():
        return None
    radius = max(1, int(round(min(bw, bh) * 0.06)))
    blob = close_gaps(blob, radius) & box
    return _keep_near_core(blob, inner)


def _ellipse(shape: tuple[int, int], inner: tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = inner
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = (x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0
    rx, ry = max(1.0, (x1 - x0) / 2.0), max(1.0, (y1 - y0) / 2.0)
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def _trusted(binary: np.ndarray, inner: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = inner
    area = max(1, (x1 - x0) * (y1 - y0))
    fill = float(binary[y0:y1, x0:x1].sum()) / area
    return 0.08 <= fill <= 0.97


def _model_chain(backend: str, segmenters):
    if segmenters is not None:
        return [item for item in segmenters if getattr(item, "name", "") != "heuristic"]
    from ai_dt.segment import _make, backend_order

    chain = []
    for name in backend_order(backend):
        if name in ("heuristic", "ade20k"):
            continue
        try:
            chain.append(_make(name))
        except Exception:  # an optional backend that cannot load
            continue
    return chain


def _model_outline(crop: Image.Image, box_in_crop, label: str, chain) -> tuple[np.ndarray | None, str]:
    for segmenter in chain:
        if "subject" not in getattr(segmenter, "classes", ()):
            continue
        try:
            if not segmenter.available():
                continue
            found = segmenter.segment(
                crop, ["subject"], boxes=[{"category": "animal", "label": label, "bbox": box_in_crop}]
            )
        except Exception:  # a model that fails at run time is skipped
            continue
        mask = (found or {}).get("subject")
        if mask is None:
            continue
        values = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        if values.ndim != 2:
            continue
        if values.shape != (crop.height, crop.width):
            values = np.asarray(
                Image.fromarray(values, mode="F").resize(crop.size, Image.BILINEAR), dtype=np.float32
            )
        return values >= 0.5, segmenter.name
    return None, ""


def segment_animals(
    image: Image.Image,
    boxes: list,
    *,
    backend: str = "auto",
    segmenters=None,
) -> AnimalResult:
    """Outline every animal box. Never raises for a bad box; notes it instead."""

    rgb = image.convert("RGB")
    result = AnimalResult()
    chosen = animal_boxes(boxes, result.notes)
    if not chosen:
        if not any(note for note in result.notes):
            result.notes.append("no animal was detected")
        return result
    chain = _model_chain(backend, segmenters)
    width, height = rgb.size
    for label, confidence, box in chosen:
        x0, y0, x1, y1 = _crop_rect(box, rgb.size)
        crop = _scaled(rgb.crop((x0, y0, x1, y1)))
        sx = crop.width / float(x1 - x0)
        sy = crop.height / float(y1 - y0)
        bx, by, bw, bh = box
        inner = (
            int(max(0, np.floor((bx * width - x0) * sx))),
            int(max(0, np.floor((by * height - y0) * sy))),
            int(min(crop.width, np.ceil((bx * width + bw * width - x0) * sx))),
            int(min(crop.height, np.ceil((by * height + bh * height - y0) * sy))),
        )
        box_in_crop = (
            inner[0] / crop.width,
            inner[1] / crop.height,
            (inner[2] - inner[0]) / crop.width,
            (inner[3] - inner[1]) / crop.height,
        )
        binary, source = _model_outline(crop, box_in_crop, label, chain)
        if binary is not None:
            # A model outline may run past the box a little, never far.
            limit = np.zeros_like(binary)
            px = max(2, (inner[2] - inner[0]) // 10)
            py = max(2, (inner[3] - inner[1]) // 10)
            limit[max(0, inner[1] - py):inner[3] + py, max(0, inner[0] - px):inner[2] + px] = True
            binary = binary & limit
            if not _trusted(binary, inner):
                binary, source = None, ""
        if binary is None:
            pixels = np.asarray(crop, dtype=np.float32) / 255.0
            binary = heuristic_outline(pixels, inner)
            source = "heuristic"
            if binary is None or not _trusted(binary, inner):
                binary = _ellipse(binary.shape if binary is not None else (crop.height, crop.width), inner)
                source = "box ellipse"
        crop_mask = binary.astype(np.float32)
        full = np.zeros((height, width), dtype=np.float32)
        back = np.asarray(
            Image.fromarray(crop_mask, mode="F").resize((x1 - x0, y1 - y0), Image.BILINEAR), dtype=np.float32
        )
        full[y0:y1, x0:x1] = back
        if int((full >= 0.5).sum()) < MIN_PIXELS:
            result.notes.append(f"the {label} mask was too small to keep")
            continue
        result.animals.append(
            Animal(
                label=label,
                confidence=confidence,
                bbox=box,
                mask=full,
                source=source,
                crop=(x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height),
                crop_mask=crop_mask,
            )
        )
    result.animals.sort(key=lambda animal: animal.area, reverse=True)
    return result


# darktable feathers a path *outward* by ``border * min(width, height)``.
# With the outline on the animal's edge, the lift spilled that far onto the
# water and an inverted (background) mask, or a gradient minus the animal,
# only reached full strength that far out: a bright ring round the duck and
# its feet. The outline is pulled in by the feather width, so the feather
# ends on the real edge for both uses (the lift fades out and the background
# darken fades in across the same thin band inside the edge).
ANIMAL_BORDER = 0.005


def animal_forms(
    result: AnimalResult,
    *,
    which: str = "all",
    border: float = ANIMAL_BORDER,
    inset: float | None = None,
    max_forms: int = 24,
    max_points: int = 64,
    working_size: int = 512,
) -> list[dict]:
    """Path forms for one darktable group. ``which`` is ``all`` or ``main``.

    Each animal is vectorised on its own crop, so a distant bird keeps its
    shape, then mapped back to frame coordinates. All animals join by union;
    holes are differences that follow their animal. The outline sits
    ``inset`` (default: the feather width) inside the segmented edge; parts
    too thin to survive that (legs, a bill) keep their own outline.
    """

    from ai_dt.dt_shapes import mask_to_forms

    pull = border if inset is None else max(0.0, float(inset))
    animals = result.animals[:1] if which == "main" else result.animals
    forms: list[dict] = []
    for animal in animals:
        cx, cy, cw, ch = animal.crop
        frame_h, frame_w = animal.mask.shape
        crop_h = animal.crop_mask.shape[0]
        # Crop-mask pixels per frame pixel, then the feather in crop pixels.
        scale = crop_h / max(1.0, ch * frame_h)
        radius = pull * min(frame_w, frame_h) * scale
        binary = animal.crop_mask >= 0.5
        # A distant bird a few pixels across would vanish; its ring is a
        # pixel at most, so cap the pull at a tenth of its size.
        radius = min(radius, 0.1 * float(np.sqrt(binary.sum() / np.pi)))
        shape = inset_mask(binary, radius)
        local = mask_to_forms(
            shape.astype(np.float32), border=border, max_paths=3, max_points=max_points, min_area=0.01, working_size=working_size,
        )
        for form in local:
            points = [
                [min(max(cx + x * cw, 0.0), 1.0), min(max(cy + y * ch, 0.0), 1.0)] for x, y in form["points"]
            ]
            # Sharp turns (feet, tail tip) keep sharp handles: no swing
            # out past the edge beside them.
            forms.append(dict(form, points=points, sharp_turns=True))
        if len(forms) >= max_forms:
            break
    if forms:
        forms[0]["mode"] = "union"
    return forms[:max_forms]


def inset_mask(binary: np.ndarray, radius: float) -> np.ndarray:
    """``binary`` shrunk by about ``radius`` pixels, keeping thin parts.

    A blur thresholded at 0.84 moves a straight edge in by one sigma, with a
    round (not square) structuring element. Parts the shrink removes
    entirely, such as legs thinner than twice the radius, are put back as
    they were so the outline does not lose them.
    """

    mask = np.asarray(binary, dtype=bool)
    if radius < 0.5 or not mask.any():
        return mask
    from PIL import Image, ImageFilter

    def blur(values: np.ndarray) -> np.ndarray:
        image = Image.fromarray((values.astype(np.float32) * 255.0).astype(np.uint8))
        return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.float32) / 255.0

    # Holes (water between the legs) stay as they are: they are cut out with
    # a difference, whose feather already runs into the animal, so growing
    # them would eat the legs. Only the outside edge moves in.
    holes = _holes(mask)
    filled = mask | holes
    shrunk = filled & (blur(filled) >= 0.84)
    # About two sigma: a 0.16 cut (one sigma) fell just short of the old
    # edge and kept a loose one-pixel ring round the whole animal as "thin".
    grown = blur(shrunk) >= 0.02
    thin = filled & ~grown
    return (shrunk | thin) & ~holes


def _holes(mask: np.ndarray) -> np.ndarray:
    """Background pixels not connected to the frame edge."""

    from PIL import Image, ImageDraw

    padded = np.pad(mask, 1, constant_values=False)
    # A copy: an image over a numpy buffer is read-only and the fill is lost.
    image = Image.fromarray(np.where(padded, 255, 0).astype(np.uint8)).copy()
    ImageDraw.floodfill(image, (0, 0), 128)
    return (np.asarray(image) == 0)[1:-1, 1:-1]
