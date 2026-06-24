"""
Camera-agnostic lens correction via lensfunpy + Lensfun.

Physically-mandated order: vignetting → TCA → geometry.
Vignetting MUST run before HDR merge (uneven corners corrupt Mertens weights).

Fallback chain:
  1. Exact EXIF match
  2. loose_search=True (same focal length, different variants)
  3. Log miss + pass through uncorrected — never guess.
"""
from pathlib import Path
from typing import Optional

import cv2
import lensfunpy
import numpy as np
import structlog

log = structlog.get_logger(__name__)

_INTERP = {
    "LANCZOS4": cv2.INTER_LANCZOS4,
    "LINEAR": cv2.INTER_LINEAR,
    "CUBIC": cv2.INTER_CUBIC,
}


def correct_lens(
    img: np.ndarray,
    exif: dict,
    interpolation: str = "LANCZOS4",
    loose_search_fallback: bool = True,
) -> np.ndarray:
    """
    Apply vignetting → TCA → geometry correction.
    Returns the corrected image (same dtype as input) or the original on lookup failure.
    lensfunpy's apply_color_modification only accepts float32/float64/uint8,
    so we work in float32 throughout and round-trip back to the original dtype.
    """
    db = lensfunpy.Database()
    cam, lens = _lookup(db, exif, loose_search_fallback)
    if cam is None or lens is None:
        log.warning("lens_correction.miss", make=exif.get("camera_make"), model=exif.get("camera_model"))
        return img

    h, w = img.shape[:2]
    focal = exif.get("focal_length") or 0.0
    aperture = exif.get("aperture") or 0.0

    mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
    mod.initialize(focal, aperture, distance=10.0)

    interp_flag = _INTERP.get(interpolation, cv2.INTER_LANCZOS4)

    # Work in float32 — lensfunpy does not support uint16
    original_dtype = img.dtype
    scale = 65535.0 if original_dtype == np.uint16 else 1.0
    work = img.astype(np.float32) / scale

    # 1. Vignetting (in-place on float32)
    mod.apply_color_modification(work)

    # 2. TCA (per-channel remap)
    tca_coords = mod.apply_subpixel_distortion()
    if tca_coords is not None:
        work = _remap_subpixel(work, tca_coords, interp_flag)

    # 3. Geometry distortion
    geo_coords = mod.apply_geometry_distortion()
    if geo_coords is not None:
        work = cv2.remap(work, geo_coords, None, interp_flag)

    log.info("lens_correction.ok", camera=exif.get("camera_model"), lens=str(lens))

    result = np.clip(work * scale, 0, scale).astype(original_dtype)
    return result


def _lookup(db, exif: dict, loose: bool) -> tuple[Optional[object], Optional[object]]:
    make = exif.get("camera_make", "")
    model = exif.get("camera_model", "")

    # Don't attempt a lookup with empty strings — lensfunpy returns arbitrary results
    if not make or not model:
        return None, None

    cams = db.find_cameras(make, model)
    if not cams:
        return None, None
    cam = cams[0]

    lenses = db.find_lenses(cam)
    if not lenses and loose:
        lenses = db.find_lenses(cam, loose_search=True)
    if not lenses:
        return None, None

    return cam, lenses[0]


def _remap_subpixel(img: np.ndarray, coords: np.ndarray, interp: int) -> np.ndarray:
    """Apply per-channel (R/G/B) remap for TCA correction."""
    out = np.empty_like(img)
    for c in range(3):
        out[:, :, c] = cv2.remap(img[:, :, c], coords[:, :, c * 2: c * 2 + 2], None, interp)
    return out
