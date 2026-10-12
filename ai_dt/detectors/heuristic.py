"""Subject finder that needs only Pillow and NumPy.

Replace or precede this with :class:`ai_dt.detectors.yolo.YoloDetector` when a
detection model is installed. The heuristic is the fallback so a click still
produces a recipe on a machine with no weights downloaded.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ai_dt.guides.support import boxes_overlap, is_face_bar, reflected_subject
from ai_dt.schema import Detection


def _downsample(image: Image.Image, max_side: int = 240) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BILINEAR,
    )


def _luma(rgb: np.ndarray) -> np.ndarray:
    color = rgb.astype(np.float32) / 255.0
    return 0.2126 * color[:, :, 0] + 0.7152 * color[:, :, 1] + 0.0722 * color[:, :, 2]


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    spread = np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)
    return (
        (red > 95)
        & (green > 40)
        & (blue > 20)
        & (red > green)
        & (red > blue)
        & (spread > 15)
    )


def _largest_bbox(mask: np.ndarray, min_fraction: float) -> tuple[int, int, int, int] | None:
    """Bounding box of the largest connected component, as x, y, w, h in pixels."""

    height, width = mask.shape
    if height == 0 or width == 0 or not mask.any():
        return None
    visited = np.zeros(mask.shape, dtype=bool)
    seeds = np.argwhere(mask)
    best_area = 0
    best: tuple[int, int, int, int] | None = None
    for seed_y, seed_x in seeds:
        if visited[seed_y, seed_x]:
            continue
        stack = [(int(seed_y), int(seed_x))]
        visited[seed_y, seed_x] = True
        area = 0
        min_x = max_x = int(seed_x)
        min_y = max_y = int(seed_y)
        while stack:
            y, x = stack.pop()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
        if area > best_area:
            best_area = area
            best = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    if best is None or best_area < min_fraction * height * width:
        return None
    return best


def _normalize(box: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (
        max(0.0, x / width),
        max(0.0, y / height),
        min(1.0, w / width),
        min(1.0, h / height),
    )


def _box_blur(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    total = np.zeros_like(values)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
    return total / 9.0


def _iou_norm(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


class HeuristicDetector:
    """Find a skin-toned face, dark eye spots inside it, and a line-heavy region."""

    name = "heuristic"

    def detect(self, image: Image.Image) -> list[Detection]:
        small = _downsample(image.convert("RGB"))
        rgb = np.asarray(small)
        height, width = rgb.shape[:2]
        luma = _luma(rgb)
        skin = _skin_mask(rgb)

        faces = self._faces(skin, luma, width, height)
        full = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        mirrored = reflected_subject(full)
        # Pale flowers can pass a skin test. They are not the body standing
        # over its reflection, so a face that misses that body is dropped.
        if mirrored is not None:
            body, _reflection = mirrored
            faces = [face for face in faces if boxes_overlap(face.bbox, body)]
        eyes: list[Detection] = []
        for face in faces:
            eyes.extend(self._eyes(luma, face.bbox, width, height))
        architecture = self._architecture(rgb, luma, skin, faces, width, height)

        found = faces + eyes + architecture
        if mirrored is not None:
            body, _reflection = mirrored
            found.append(
                Detection(
                    label="animal",
                    category="animal",
                    confidence=0.74,
                    bbox=body,
                    source=self.name,
                )
            )
        if found:
            return found
        salient = self._salient(luma, width, height)
        return [salient] if salient is not None else []

    def _faces(self, skin: np.ndarray, _luma: np.ndarray, width: int, height: int) -> list[Detection]:
        box = _largest_bbox(skin, min_fraction=0.01)
        if box is None:
            return []
        x, y, w, h = box
        # Grow the box a little so hair and the chin stay inside the region.
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.08)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(width - x, w + pad_x * 2)
        h = min(height - y, h + pad_y * 2)
        area_fraction = (w * h) / float(width * height)
        skin_fraction = float(skin[y : y + h, x : x + w].mean()) if w and h else 0.0
        if skin_fraction < 0.35 or area_fraction > 0.9:
            return []
        bbox = _normalize((x, y, w, h), width, height)
        # A warm horizon is a short band across the frame. It is not a head.
        if is_face_bar(bbox):
            return []
        confidence = 0.45 + 0.4 * min(1.0, skin_fraction)
        # A face-like region is taller than it is wide, but a close crop may not be.
        if h >= w * 0.75:
            confidence = min(0.9, confidence + 0.05)
        return [
            Detection(
                label="face",
                category="face",
                confidence=confidence,
                bbox=bbox,
                source=self.name,
            )
        ]

    def _eyes(
        self,
        luma: np.ndarray,
        face: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> list[Detection]:
        fx, fy, fw, fh = face
        x0 = int(fx * width)
        y0 = int(fy * height)
        x1 = int((fx + fw) * width)
        upper = y0 + max(1, int(fh * height * 0.48))
        if x1 - x0 < 8 or upper - y0 < 6:
            return []
        region = luma[y0:upper, x0:x1]
        blurred = _box_blur(region)
        face_mean = float(blurred.mean())
        mid = blurred.shape[1] // 2
        eyes: list[Detection] = []
        halves = (
            (blurred[:, :mid], x0),
            (blurred[:, mid:], x0 + mid),
        )
        for half, origin_x in halves:
            if half.size == 0 or half.shape[0] < 2 or half.shape[1] < 2:
                continue
            flat = int(np.argmin(half))
            iy, ix = divmod(flat, half.shape[1])
            darkness = face_mean - float(half[iy, ix])
            if darkness < 0.08:
                continue
            eye_w = max(2, int(fw * width * 0.16))
            eye_h = max(2, int(fh * height * 0.1))
            cx = origin_x + ix
            cy = y0 + iy
            ex = max(0, cx - eye_w // 2)
            ey = max(0, cy - eye_h // 2)
            ex = min(ex, width - eye_w)
            ey = min(ey, height - eye_h)
            confidence = min(0.8, 0.35 + darkness)
            eyes.append(
                Detection(
                    label="eye",
                    category="eye",
                    confidence=confidence,
                    bbox=_normalize((ex, ey, eye_w, eye_h), width, height),
                    source=self.name,
                )
            )
        return eyes

    def _architecture(
        self,
        rgb: np.ndarray,
        luma: np.ndarray,
        skin: np.ndarray,
        faces: list[Detection],
        width: int,
        height: int,
    ) -> list[Detection]:
        gy = np.zeros_like(luma)
        gx = np.zeros_like(luma)
        gy[1:, :] = np.abs(luma[1:, :] - luma[:-1, :])
        gx[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
        edges = gx + gy
        rows, cols = 6, 6
        cell_h = max(1, height // rows)
        cell_w = max(1, width // cols)
        scores: list[tuple[float, int, int]] = []
        for row in range(rows):
            for col in range(cols):
                y0, x0 = row * cell_h, col * cell_w
                if y0 >= height or x0 >= width:
                    continue
                y1 = min(height, height if row == rows - 1 else (row + 1) * cell_h)
                x1 = min(width, width if col == cols - 1 else (col + 1) * cell_w)
                if y1 <= y0 or x1 <= x0:
                    continue
                if float(skin[y0:y1, x0:x1].mean()) > 0.3:
                    continue
                scores.append((float(edges[y0:y1, x0:x1].mean()), row, col))
        if not scores:
            return []
        strongest = max(score for score, _, _ in scores)
        if strongest < 0.08:
            return []
        chosen = [(row, col) for score, row, col in scores if score >= strongest * 0.72]
        if not chosen:
            return []
        min_row = min(row for row, _ in chosen)
        max_row = max(row for row, _ in chosen)
        min_col = min(col for _, col in chosen)
        max_col = max(col for _, col in chosen)
        x = min_col * cell_w
        y = min_row * cell_h
        x2 = width if max_col == cols - 1 else (max_col + 1) * cell_w
        y2 = height if max_row == rows - 1 else (max_row + 1) * cell_h
        box = _normalize((x, y, max(1, x2 - x), max(1, y2 - y)), width, height)
        area = box[2] * box[3]
        if area < 0.04 or area > 0.85:
            return []
        region = rgb[y:y2, x:x2]
        if region.size:
            red = region[:, :, 0].astype(np.int16)
            green = region[:, :, 1].astype(np.int16)
            blue = region[:, :, 2].astype(np.int16)
            leafy = (green > red + 8) & (green > blue + 8)
            # Furrows in a field are edges, and they are green. A facade is not.
            if float(leafy.mean()) > 0.45:
                return []
        if any(_iou_norm(box, face.bbox) > 0.25 for face in faces):
            return []
        return [
            Detection(
                label="structure",
                category="architecture",
                confidence=min(0.8, 0.4 + strongest),
                bbox=box,
                source=self.name,
            )
        ]

    def _salient(
        self,
        luma: np.ndarray,
        width: int,
        height: int,
    ) -> Detection | None:
        rows, cols = 6, 6
        cell_h = max(1, height // rows)
        cell_w = max(1, width // cols)
        best = 0.0
        best_at = (0, 0)
        found = False
        for row in range(rows):
            for col in range(cols):
                y0, x0 = row * cell_h, col * cell_w
                if y0 >= height or x0 >= width:
                    continue
                y1 = min(height, height if row == rows - 1 else (row + 1) * cell_h)
                x1 = min(width, width if col == cols - 1 else (col + 1) * cell_w)
                if y1 <= y0 or x1 <= x0:
                    continue
                score = float(luma[y0:y1, x0:x1].var())
                if score > best:
                    best = score
                    best_at = (row, col)
                    found = True
        if not found or best < 0.002:
            return None
        row, col = best_at
        row0, col0 = max(0, row - 1), max(0, col - 1)
        row1, col1 = min(rows, row + 2), min(cols, col + 2)
        x = col0 * cell_w
        y = row0 * cell_h
        x2 = width if col1 == cols else col1 * cell_w
        y2 = height if row1 == rows else row1 * cell_h
        return Detection(
            label="subject",
            category="subject",
            confidence=0.34,
            bbox=_normalize((x, y, max(1, x2 - x), max(1, y2 - y)), width, height),
            source=self.name,
        )
