from pathlib import Path

import numpy as np
import structlog

from src.common.io import uint16_to_float
from .estimator import shades_of_gray, apply_wb, series_illuminant

log = structlog.get_logger(__name__)


def process_series(
    images: list[np.ndarray],
    cfg: dict,
    per_room_overrides: dict | None = None,
) -> list[np.ndarray]:
    """
    Apply WB to an entire property series with a single locked illuminant.
    per_room_overrides: {index: illuminant_3vec} to override specific rooms.
    Returns list of uint16 WB-corrected images.
    """
    s3 = cfg["stage3_wb"]

    illuminants = [shades_of_gray(uint16_to_float(img)) for img in images]

    if s3["series_lock"]:
        lock = series_illuminant(illuminants)
        illuminants = [lock] * len(images)

    if per_room_overrides and s3["per_room_override"]:
        for idx, override in per_room_overrides.items():
            illuminants[idx] = np.array(override, dtype=np.float32)

    corrected = [apply_wb(img, ill) for img, ill in zip(images, illuminants)]
    log.info("stage3.done", n_images=len(corrected), series_lock=s3["series_lock"])
    return corrected


def process_single(img: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Single-image WB (useful in the pipeline without a full series)."""
    illuminant = shades_of_gray(uint16_to_float(img))
    return apply_wb(img, illuminant), illuminant
