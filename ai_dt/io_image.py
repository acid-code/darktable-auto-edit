"""Open a picture file, including camera RAW.

JPEG, PNG, TIFF, and WebP go through Pillow. Camera RAW goes through rawpy,
which is LibRaw. The working image is float RGB in the range 0 to 1, plus an
8-bit Pillow image for the detectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from ai_dt.errors import RawReadError, RawSupportUnavailable

RAW_EXTENSIONS = frozenset(
    {
        ".3fr",
        ".arw",
        ".cr2",
        ".cr3",
        ".dcr",
        ".dng",
        ".erf",
        ".fff",
        ".gpr",
        ".iiq",
        ".kdc",
        ".mef",
        ".mos",
        ".mrw",
        ".nef",
        ".nrw",
        ".orf",
        ".pef",
        ".raf",
        ".raw",
        ".rw2",
        ".rwl",
        ".sr2",
        ".srf",
        ".srw",
        ".x3f",
    }
)

RASTER_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})


@dataclass
class LoadedImage:
    """One decoded picture.

    ``pixels`` is float32 RGB, shape ``(height, width, 3)``, range 0 to 1.
    ``image`` is the 8-bit view the detectors already know how to read.
    """

    image: Image.Image
    pixels: np.ndarray
    kind: str
    path: Path


def is_raw(path: Path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def is_picture(path: Path) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in RAW_EXTENSIONS or suffix in RASTER_EXTENSIONS


def load_image(path: Path, *, full_res: bool = False) -> LoadedImage:
    """Decode ``path``. RAW files use the camera white balance.

    ``full_res`` asks LibRaw for every pixel. The default is half resolution,
    which is the size to use when a folder of RAW files is the input.
    """

    source = Path(path)
    if is_raw(source):
        return _load_raw(source, full_res=full_res)
    try:
        with Image.open(source) as handle:
            image = handle.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image {source.name}: {exc}") from exc
    pixels = np.asarray(image).astype(np.float32) / np.float32(255.0)
    return LoadedImage(image=image, pixels=pixels, kind="raster", path=source)


def _load_raw(path: Path, *, full_res: bool) -> LoadedImage:
    try:
        import rawpy
    except ImportError as exc:
        raise RawSupportUnavailable(
            "This is a camera RAW file. Install rawpy to open it: pip install rawpy"
        ) from exc
    try:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=not full_res,
                output_bps=16,
                no_auto_bright=False,
                output_color=rawpy.ColorSpace.sRGB,
                user_flip=-1,
            )
    except RawSupportUnavailable:
        raise
    except Exception as exc:
        raise RawReadError(f"Could not decode RAW file {path.name}: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
        raise RawReadError(f"RAW file {path.name} decoded to an empty image")
    if rgb.dtype != np.uint16:
        rgb = np.clip(rgb, 0, 65535).astype(np.uint16)
    pixels = rgb.astype(np.float32) / np.float32(65535.0)
    preview = Image.fromarray(np.rint(pixels * 255.0).astype(np.uint8), mode="RGB")
    return LoadedImage(image=preview, pixels=pixels, kind="raw", path=path)
