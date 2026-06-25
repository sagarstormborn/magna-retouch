from pathlib import Path

import numpy as np
import structlog

from src.common.io import uint16_to_float
from .estimator import shades_of_gray, full_illuminant, apply_wb, series_illuminant

log = structlog.get_logger(__name__)


def process_series(
    images: list[np.ndarray],
    cfg: dict,
    exif_list: list[dict] | None = None,
    per_room_overrides: dict | None = None,
) -> list[np.ndarray]:
    """
    Apply WB to an entire property series with a single locked illuminant.
    exif_list: per-image EXIF dicts (containing daylight_wb); if None, falls back
               to Shades-of-Gray only.
    per_room_overrides: {index: illuminant_3vec} to override specific rooms.
    Returns list of uint16 WB-corrected images.
    """
    s3 = cfg["stage3_wb"]
    exif_list = exif_list or [{}] * len(images)

    illuminants = [
        full_illuminant(uint16_to_float(img), exif.get("daylight_wb"))
        for img, exif in zip(images, exif_list)
    ]

    if s3["series_lock"]:
        lock = series_illuminant(illuminants)
        illuminants = [lock] * len(images)

    if per_room_overrides and s3["per_room_override"]:
        for idx, override in per_room_overrides.items():
            illuminants[idx] = np.array(override, dtype=np.float32)

    corrected = [apply_wb(img, ill) for img, ill in zip(images, illuminants)]
    log.info("stage3.done", n_images=len(corrected), series_lock=s3["series_lock"])
    return corrected


def process_single(img: np.ndarray, cfg: dict, exif: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Single-image WB. Pass exif dict with 'daylight_wb' for sensor-calibrated correction."""
    daylight_wb = (exif or {}).get("daylight_wb")
    illuminant = full_illuminant(uint16_to_float(img), daylight_wb)
    return apply_wb(img, illuminant), illuminant
