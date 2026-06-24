"""
Illuminant estimation for white balance.

Default: Shades-of-Gray (generalised Grey World, p=6).
Series lock: take the robust median illuminant across a property's image set
and apply it uniformly — directly addresses Matt's criterion 5 (no WB drift).

Mixed-illuminant path (Afifi mixedillWB): DISABLED — research-only license.
Requires explicit license clearance before enabling.
"""
import numpy as np
import structlog

log = structlog.get_logger(__name__)

_SHADES_P = 6  # p=6 is the standard Shades-of-Gray tuning point


def shades_of_gray(img_f32: np.ndarray) -> np.ndarray:
    """
    Returns the illuminant estimate as a 3-vector (R, G, B) normalised to mean 1.
    Input: float32 HxWx3 in [0, 1].
    """
    p = _SHADES_P
    norms = np.mean(img_f32 ** p, axis=(0, 1)) ** (1.0 / p)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return norms / norms.mean()


def apply_wb(img: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
    """
    Divide each channel by the illuminant estimate.
    Works on uint16 or float32; always returns the same dtype.
    """
    dtype = img.dtype
    f32 = img.astype(np.float32)
    corrected = f32 / illuminant[np.newaxis, np.newaxis, :]
    if dtype == np.uint16:
        return np.clip(corrected, 0, 65535).astype(np.uint16)
    return np.clip(corrected, 0, 1).astype(np.float32)


def series_illuminant(illuminants: list[np.ndarray]) -> np.ndarray:
    """Robust median illuminant across a property series."""
    stack = np.stack(illuminants, axis=0)      # N x 3
    median = np.median(stack, axis=0)
    log.info("stage3.series_illuminant", illuminant=median.tolist())
    return median


def wb_cct_variance(illuminants: list[np.ndarray]) -> float:
    """
    Proxy for series WB consistency: std-dev of the R/B ratio (correlates with CCT).
    Lower is more consistent.
    """
    rb_ratios = [il[0] / (il[2] + 1e-8) for il in illuminants]
    return float(np.std(rb_ratios))
