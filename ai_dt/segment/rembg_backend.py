"""Subject (and people) masks from rembg. ``pip install rembg`` (CPU ONNX).

The model is ``AI_DT_REMBG_MODEL`` (default ``isnet-general-use``; try
``birefnet-general`` for finer edges, slower). People use
``u2net_human_seg``. The first run downloads the model into ``~/.u2net``.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from ai_dt.segment.base import SegmenterUnavailable


@lru_cache(maxsize=4)
def _session(model: str):
    from rembg import new_session

    return new_session(model, providers=["CPUExecutionProvider"])


def alpha_to_mask(alpha) -> np.ndarray:
    values = np.asarray(alpha, dtype=np.float32)
    if values.ndim == 3:
        values = values[..., -1]
    if values.max() > 1.0:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0)


class RembgSegmenter:
    name = "rembg"
    classes = ("subject", "people")

    def available(self) -> bool:
        try:
            import rembg  # noqa: F401
        except Exception:
            return False
        return True

    def segment(self, image, classes, *, boxes: list) -> dict[str, np.ndarray]:
        try:
            from rembg import remove
        except Exception as exc:
            raise SegmenterUnavailable(f"rembg is not installed ({exc})") from exc
        out: dict[str, np.ndarray] = {}
        if "subject" in classes:
            model = os.environ.get("AI_DT_REMBG_MODEL", "isnet-general-use")
            out["subject"] = alpha_to_mask(remove(image, session=_session(model), only_mask=True))
        if "people" in classes and any(
            str(getattr(item, "category", None) or (item.get("category") if isinstance(item, dict) else "")) in {"person", "face"}
            for item in boxes
        ):
            out["people"] = alpha_to_mask(remove(image, session=_session("u2net_human_seg"), only_mask=True))
        return out
