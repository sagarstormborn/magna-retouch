"""
Stage 6: SAM-based zone segmentation + per-zone colour correction.

process() follows the same (img_u8, cfg) → img_u8 convention as every other stage.
If SAM is disabled in config or unavailable, returns the input unchanged.
"""
from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger(__name__)


def process(img: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Args:
        img: HxWx3 uint8 RGB (Stage 4 LUT output)
        cfg: full pipeline config
    Returns:
        HxWx3 uint8 RGB with zone corrections applied
    """
    s6 = cfg.get("stage6_zones", {})
    if not s6.get("enabled", True):
        return img

    try:
        from .zone_seg import get_segmenter
        from .zone_correct import apply_zone_corrections
    except ImportError as e:
        log.warning("stage6.skipped", reason=str(e))
        return img

    try:
        segmenter = get_segmenter(cfg)
        zones, scene_type = segmenter.segment(img)

        img_f32 = img.astype(np.float32) / 255.0
        corrected_f32 = apply_zone_corrections(img_f32, zones, scene_type=scene_type, cfg=cfg)
        result = (np.clip(corrected_f32, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        log.info("stage6.done", shape=result.shape, scene_type=scene_type)
        return result

    except Exception as e:
        log.warning("stage6.error", error=str(e), fallback="passthrough")
        return img
