"""Subject and people from darktable 5.6's own SAM 2.1 files and a detector box."""

from __future__ import annotations

import numpy as np

from ai_dt.segment.base import PEOPLE_CATEGORIES, SUBJECT_CATEGORIES, SegmenterUnavailable, boxes_for


class DarktableSamSegmenter:
    name = "darktable-sam"
    classes = ("subject", "people")

    def available(self) -> bool:
        from ai_dt import object_segment

        if object_segment.model_dir() is None:
            return False
        try:
            import cv2  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def segment(self, image, classes, *, boxes: list) -> dict[str, np.ndarray]:
        from ai_dt import object_segment

        if max(image.size) < 256:
            raise SegmenterUnavailable("the preview is too small for SAM")
        out: dict[str, np.ndarray] = {}
        for name, categories in (("subject", SUBJECT_CATEGORIES), ("people", PEOPLE_CATEGORIES)):
            if name not in classes:
                continue
            combined = None
            for box in boxes_for(boxes, categories)[:4]:
                mask = object_segment._segment(image, box)
                if mask is None:
                    continue
                combined = mask if combined is None else np.maximum(combined, mask)
            if combined is not None:
                out[name] = combined.astype(np.float32)
        return out
