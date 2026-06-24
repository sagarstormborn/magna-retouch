"""
HDR merge for tripod/static brackets.

Alignment: MTB (micro-vibration) → ECC escalation if residual shows.
Fusion: Mertens with configurable weights. Weights default 1/1/1 — the
exposure weight of 1.0 is critical (OpenCV default is 0.0 → flat output).
"""
from pathlib import Path

import cv2
import numpy as np
import structlog

from src.common.io import float_to_uint16, uint16_to_float

log = structlog.get_logger(__name__)


def process(images: list[np.ndarray], cfg: dict) -> np.ndarray:
    """
    images: list of uint16 HxWx3 at the same exposure series.
    Returns: uint16 HxWx3 fused image.
    """
    s2 = cfg["stage2_hdr"]

    if len(images) == 1:
        log.info("stage2.single_bracket_passthrough")
        return images[0]

    aligned = _align(images, s2["align_method"])
    fused_f32 = _mertens_fuse(aligned, s2["mertens_weights"])
    result = float_to_uint16(fused_f32)

    _check_highlight_protection(result, cfg["stage5_benchmark"]["highlight_protection_stops"])
    log.info("stage2.done", shape=result.shape)
    return result


def _align(images: list[np.ndarray], method: str) -> list[np.ndarray]:
    if method == "MTB":
        aligner = cv2.createAlignMTB()
        aligned = list(images)
        aligner.process(aligned, aligned)
        return aligned

    if method == "ECC":
        return _align_ecc(images)

    raise ValueError(f"Unknown align_method: {method}")


def _align_ecc(images: list[np.ndarray]) -> list[np.ndarray]:
    """ECC alignment — more robust than MTB for larger vibration / window frames."""
    reference = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
    aligned = [images[0]]
    for img in images[1:]:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                reference.astype(np.float32),
                gray.astype(np.float32),
                warp,
                cv2.MOTION_EUCLIDEAN,
            )
        except cv2.error:
            log.warning("stage2.ecc_failed_fallback_passthrough")
        h, w = img.shape[:2]
        aligned.append(cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
    return aligned


def _mertens_fuse(images: list[np.ndarray], weights: dict) -> np.ndarray:
    merge = cv2.createMergeMertens(
        contrast_weight=float(weights["contrast"]),
        saturation_weight=float(weights["saturation"]),
        exposure_weight=float(weights["exposure"]),
    )
    # Mertens expects uint8 or float32 0-1; feed float32 from uint16
    f32_list = [uint16_to_float(img) for img in images]
    fused = merge.process(f32_list)   # returns float32 [0,1]
    return fused


def _check_highlight_protection(img: np.ndarray, stops: int) -> None:
    """
    Verify that the top `stops` EV worth of headroom is not uniformly clipped.
    Threshold: values > (1 - 2^-stops) of full scale must not be all 65535.
    Logs a warning if more than 0.1% of pixels are blown in all 3 channels.
    """
    threshold = int(65535 * (1.0 - 2.0 ** -stops))
    blown_mask = np.all(img >= threshold, axis=2)
    blown_pct = blown_mask.mean() * 100.0
    if blown_pct > 0.1:
        log.warning(
            "stage2.highlight_protection_violated",
            blown_pct=round(blown_pct, 3),
            threshold_dn=threshold,
            stops=stops,
        )
    else:
        log.info("stage2.highlight_protection_ok", blown_pct=round(blown_pct, 4))
