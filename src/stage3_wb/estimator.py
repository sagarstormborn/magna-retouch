"""
Illuminant estimation for white balance.

Default: Shades-of-Gray (generalised Grey World, p=6) applied on top of a
sensor-calibration prior derived from the camera's daylight WB multipliers.

Two-stage correction:
  1. Apply daylight WB (sensor → D65-neutral): removes the Bayer channel
     imbalance baked in by user_wb=[1,1,1,1] decode.
  2. Run Shades-of-Gray on the D65-balanced image: estimates the scene-specific
     deviation from D65 (e.g. tungsten warm cast) and removes it.

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


def full_illuminant(img_f32: np.ndarray, daylight_wb: list | None) -> np.ndarray:
    """
    Combined illuminant = daylight sensor calibration × scene deviation.

    daylight_wb: [R_gain, 1.0, B_gain] from EXIF daylight_whitebalance,
                 normalised so G=1. None → fall back to Shades-of-Gray only.

    Returns a 3-vector to divide channels by (same contract as shades_of_gray).
    """
    if daylight_wb is None:
        return shades_of_gray(img_f32)

    # daylight_wb are MULTIPLIERS: multiply raw R by 2.076, raw B by 1.471 to reach D65.
    prior = np.array(daylight_wb, dtype=np.float32)

    # Step 1: multiply raw image by daylight gains → D65-balanced working space
    d65 = img_f32 * prior[np.newaxis, np.newaxis, :]
    d65 = np.clip(d65, 0, 1)

    # Step 2: Shades-of-Gray on D65 image → scene deviation from D65 (warm cast etc.)
    scene_dev = shades_of_gray(d65)

    # Combined illuminant for apply_wb (which divides):
    #   apply_wb(img, combined) = img / combined
    #   = img × prior / scene_dev  (the desired two-stage correction)
    #   → combined = scene_dev / prior
    combined = scene_dev / prior
    combined /= combined.mean()
    log.debug("wb.illuminant", prior=prior.tolist(), scene_dev=scene_dev.tolist(), combined=combined.tolist())
    return combined


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
