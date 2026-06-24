"""
RAW → linear 16-bit RGB.

user_wb=[1,1,1,1] forces neutral — LibRaw otherwise silently falls back to
its built-in daylight WB which skews the baseline before Stage 3 runs.
"""
import multiprocessing as mp
from pathlib import Path

import numpy as np
import rawpy


def _configure_spawn():
    """rawpy + OpenMP deadlocks under fork; must be called before any Pool."""
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")


def decode_raw(path: Path, half_size: bool = False) -> tuple[np.ndarray, dict]:
    """
    Returns (rgb_uint16 HxWx3, exif_dict).
    Linear, no auto-bright, no camera WB — all correction deferred to later stages.
    """
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            no_auto_bright=True,
            gamma=(1, 1),
            output_bps=16,
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            half_size=half_size,
        )
        exif = _extract_exif(raw)
    return rgb, exif


def _extract_exif(raw: rawpy.RawPy) -> dict:
    exif: dict = {}
    try:
        # Camera make/model live in raw.lens (rawpy ≥ 0.20)
        exif["camera_make"] = (raw.lens.make or "").strip()
        exif["camera_model"] = (raw.lens.model or "").strip()
    except Exception:
        pass
    try:
        # Exposure metadata in raw.other
        other = raw.other
        exif["focal_length"] = getattr(other, "focal_length", None)
        exif["aperture"] = getattr(other, "aperture", None)
        exif["shutter_speed"] = getattr(other, "shutter_speed", None)
        exif["iso"] = getattr(other, "iso_speed", None)
    except Exception:
        pass
    return exif
