"""Outline one detected box with Darktable's downloaded object-mask model.

Darktable's AI object mask encodes the picture once, then answers a click
or a box with a silhouette. This uses the same SAM 2.1 files from
``models/mask-object-sam21-small``: longest side scaled to 1024, ImageNet
normalization, a box prompt, and the mask with the highest predicted
overlap. The silhouette comes back as a short closed path in the same
0–1 coordinates as an ellipse. A missing model or runtime returns no path,
and the caller keeps the oval it already knows how to write.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from PIL import Image

_INPUT = 1024
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def model_dir() -> str | None:
    """The SAM 2.1 small folder Darktable stores after the AI download."""

    root = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CONFIG_HOME")
    if not root:
        return None
    folder = os.path.join(root, "darktable", "models", "mask-object-sam21-small")
    encoder = os.path.join(folder, "encoder.onnx")
    decoder = os.path.join(folder, "decoder.onnx")
    if os.path.isfile(encoder) and os.path.isfile(decoder):
        return folder
    return None


def outline_box(image: Image.Image, box: tuple[float, float, float, float]) -> list[tuple[float, float]] | None:
    """A closed path around the object inside ``box``, or None.

    ``box`` is ``x, y, width, height`` in 0–1, origin at the top left.
    """

    try:
        x, y, w, h = (float(part) for part in box)
    except (TypeError, ValueError):
        return None
    if min(w, h) <= 0.0 or image.width < 2 or image.height < 2:
        return None
    mask = _segment(image, (x, y, w, h))
    if mask is None:
        return None
    return _path(mask, (x, y, w, h))


def _segment(image: Image.Image, box: tuple[float, float, float, float]) -> np.ndarray | None:
    sessions = _sessions()
    if sessions is None:
        return None
    encoder, decoder = sessions
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    scale = _INPUT / float(max(width, height))
    prepared = _prepare(rgb, scale)
    encoded = encoder.run(None, {encoder.get_inputs()[0].name: prepared})
    feeds = _decoder_feeds(decoder, encoder, encoded, box, width, height, scale)
    if feeds is None:
        return None
    outputs = decoder.run(None, feeds)
    logits, scores = outputs[0], outputs[1]
    best = int(np.argmax(np.asarray(scores).reshape(-1)))
    plane = np.asarray(logits[0, best], dtype=np.float32)
    return _back_to_image(plane, scale, width, height)


def _prepare(rgb: np.ndarray, scale: float) -> np.ndarray:
    import cv2

    height, width = rgb.shape[:2]
    new_w = min(int(width * scale + 0.5), _INPUT)
    new_h = min(int(height * scale + 0.5), _INPUT)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((_INPUT, _INPUT, 3), dtype=np.float32)
    canvas[:new_h, :new_w] = resized
    canvas = (canvas - _MEAN) / _STD
    return np.transpose(canvas, (2, 0, 1))[None, ...].astype(np.float32)


def _decoder_feeds(decoder, encoder, encoded, box, width, height, scale):
    encoded_by_name = {item.name: value for item, value in zip(encoder.get_outputs(), encoded)}
    x, y, w, h = box
    coords = np.array(
        [[
            [x * width * scale, y * height * scale],
            [(x + w) * width * scale, (y + h) * height * scale],
            [0.0, 0.0],
        ]],
        dtype=np.float32,
    )
    labels = np.array([[2.0, 3.0, -1.0]], dtype=np.float32)
    feeds = {}
    for item in decoder.get_inputs():
        if item.name in encoded_by_name:
            feeds[item.name] = encoded_by_name[item.name]
        elif item.name == "point_coords":
            feeds[item.name] = coords
        elif item.name == "point_labels":
            feeds[item.name] = labels
        elif item.name == "has_mask_input":
            feeds[item.name] = np.zeros((1,), dtype=np.float32)
        elif item.name == "mask_input":
            feeds[item.name] = np.zeros((1, 1, 256, 256), dtype=np.float32)
        else:
            return None
    return feeds


def _back_to_image(plane: np.ndarray, scale: float, width: int, height: int) -> np.ndarray:
    import cv2

    dec_h, dec_w = plane.shape
    new_w = min(int(width * scale + 0.5), _INPUT)
    new_h = min(int(height * scale + 0.5), _INPUT)
    crop_w = max(1, int(round(new_w * dec_w / _INPUT)))
    crop_h = max(1, int(round(new_h * dec_h / _INPUT)))
    crop = plane[:crop_h, :crop_w]
    probability = 1.0 / (1.0 + np.exp(-np.clip(crop, -20.0, 20.0)))
    return cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)


def _path(mask: np.ndarray, box: tuple[float, float, float, float]) -> list[tuple[float, float]] | None:
    import cv2

    binary = (mask >= 0.5).astype(np.uint8)
    covered = float(binary.mean())
    if covered < 0.005 or covered > 0.85:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    height, width = binary.shape
    cx = (box[0] + box[2] * 0.5) * width
    cy = (box[1] + box[3] * 0.5) * height
    chosen = None
    for contour in contours:
        if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
            chosen = contour
            break
    if chosen is None:
        chosen = max(contours, key=cv2.contourArea)
    if cv2.contourArea(chosen) < width * height * 0.005:
        return None
    length = cv2.arcLength(chosen, True)
    simplified = cv2.approxPolyDP(chosen, 0.012 * length, True)
    if len(simplified) > 48:
        simplified = cv2.approxPolyDP(chosen, 0.03 * length, True)
    if len(simplified) < 5:
        return None
    points = []
    for item in simplified[:48]:
        px = float(np.clip(item[0][0] / width, 0.002, 0.998))
        py = float(np.clip(item[0][1] / height, 0.002, 0.998))
        points.append((px, py))
    return points


@lru_cache(maxsize=1)
def _sessions():
    folder = model_dir()
    if folder is None:
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    providers = ["CPUExecutionProvider"]
    encoder = ort.InferenceSession(
        os.path.join(folder, "encoder.onnx"),
        sess_options=options,
        providers=providers,
    )
    decoder = ort.InferenceSession(
        os.path.join(folder, "decoder.onnx"),
        sess_options=options,
        providers=providers,
    )
    return encoder, decoder
