"""Merge overlapping detections from more than one backend."""

from __future__ import annotations

from ai_dt.schema import Detection


def iou(a: Detection, b: Detection) -> float:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def merge_detections(items: list[Detection], overlap: float = 0.5) -> list[Detection]:
    """Keep the higher-confidence box when two regions of the same category overlap.

    Eyes are compared only with other eyes, so a face does not swallow them.
    """

    ranked = sorted(items, key=lambda item: item.confidence, reverse=True)
    kept: list[Detection] = []
    for item in ranked:
        if any(
            item.category == previous.category and iou(item, previous) >= overlap
            for previous in kept
        ):
            continue
        kept.append(item)
    return kept
