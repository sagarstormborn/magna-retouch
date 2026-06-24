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
    try:
        d = raw.raw_image_visible  # noqa: access just to confirm readable
    except Exception:
        pass

    exif: dict = {}
    try:
        exif["camera_make"] = raw.metadata.make.strip()
        exif["camera_model"] = raw.metadata.model.strip()
        exif["focal_length"] = raw.metadata.focal_len
        exif["aperture"] = raw.metadata.aperture
        exif["shutter_speed"] = raw.metadata.shutter
        exif["iso"] = raw.metadata.iso_speed
    except Exception:
        pass
    return exif
