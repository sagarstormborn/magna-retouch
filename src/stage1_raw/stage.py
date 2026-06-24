from pathlib import Path

import numpy as np
import structlog

from .decode import decode_raw
from .lens_correction import correct_lens

log = structlog.get_logger(__name__)


def process(raw_path: Path, cfg: dict) -> tuple[np.ndarray, dict]:
    """
    RAW → linear uint16 RGB with lens correction applied.
    Returns (img_uint16, exif).
    """
    s1 = cfg["stage1_raw"]
    log.info("stage1.start", path=str(raw_path))

    img, exif = decode_raw(raw_path)

    lc = s1["lens_correction"]
    if lc["enabled"]:
        img = correct_lens(
            img,
            exif,
            interpolation=lc["interpolation"],
            loose_search_fallback=lc["loose_search_fallback"],
        )

    log.info("stage1.done", shape=img.shape, dtype=str(img.dtype))
    return img, exif
