"""
RAW → linear 16-bit RGB.

user_wb=[1,1,1,1] forces neutral — LibRaw otherwise silently falls back to
its built-in daylight WB which skews the baseline before Stage 3 runs.
"""
import multiprocessing as mp
import shutil
import subprocess
from pathlib import Path

import numpy as np
import rawpy


def _configure_spawn():
    """rawpy + OpenMP deadlocks under fork; must be called before any Pool."""
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")


def decode_raw(path: Path, wb_mode: str = "camera", half_size: bool = False) -> tuple[np.ndarray, dict]:
    """
    Returns (rgb_uint16 HxWx3, exif_dict).

    wb_mode="camera"  — use_camera_wb=True: camera's measured WB applied during
                        demosaic. Correct for single-shot real estate work.
    wb_mode="neutral" — user_wb=[1,1,1,1]: neutral Bayer gains, relative exposures
                        preserved across brackets. Required for HDR bracket sets.
    """
    use_cwb = wb_mode == "camera"
    with rawpy.imread(str(path)) as raw:
        kwargs = dict(
            no_auto_bright=True,
            gamma=(1, 1),
            output_bps=16,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            half_size=half_size,
        )
        if use_cwb:
            kwargs["use_camera_wb"] = True
        else:
            kwargs["use_camera_wb"] = False
            kwargs["use_auto_wb"] = False
            kwargs["user_wb"] = [1.0, 1.0, 1.0, 1.0]
        rgb = raw.postprocess(**kwargs)
        exif = _extract_exif(raw, path)
    return rgb, exif


def _extract_exif(raw: rawpy.RawPy, path: Path) -> dict:
    exif: dict = {}

    # ── Camera body (Make / Model) — requires exiftool ─────────────────────────
    # rawpy.lens.model returns the LENS name, not the camera body (Fujifilm stores
    # them separately in EXIF). exiftool reads IFD0 Make/Model correctly for all
    # manufacturers including Fujifilm RAF.
    cam_make, cam_body = _exiftool_camera_body(path)
    exif["camera_make"] = cam_make
    exif["camera_body"] = cam_body

    # ── Lens name — rawpy.lens is reliable for all manufacturers ───────────────
    try:
        exif["lens_make"] = (raw.lens.make or "").strip()
        exif["lens_model"] = (raw.lens.model or "").strip()
    except Exception:
        exif["lens_make"] = ""
        exif["lens_model"] = ""

    # ── WB multipliers — needed by Stage 3 as sensor-calibration prior ────────
    # daylight_whitebalance: [R, G1, B, G2] gains that map sensor→D65 neutral.
    # camera_whitebalance: [R, G1, B, G2] gains the camera metered for this scene.
    # We normalise by G so downstream code gets (R_gain, G_gain=1, B_gain).
    try:
        dwb = raw.daylight_whitebalance   # D65 sensor calibration
        g = dwb[1] if dwb[1] > 0 else 1.0
        exif["daylight_wb"] = [dwb[0]/g, 1.0, dwb[2]/g]
    except Exception:
        exif["daylight_wb"] = None

    try:
        cwb = raw.camera_whitebalance     # scene-metered WB (cross-check only)
        g = cwb[1] if cwb[1] > 0 else 1.0
        exif["camera_wb"] = [cwb[0]/g, 1.0, cwb[2]/g]
    except Exception:
        exif["camera_wb"] = None

    # ── Exposure metadata ──────────────────────────────────────────────────────
    try:
        other = raw.other
        exif["focal_length"] = getattr(other, "focal_length", None)
        exif["aperture"] = getattr(other, "aperture", None)
        exif["shutter_speed"] = getattr(other, "shutter_speed", None)
        exif["iso"] = getattr(other, "iso_speed", None)
    except Exception:
        pass

    return exif


def _exiftool_camera_body(path: Path) -> tuple[str, str]:
    """
    Use exiftool to read IFD0 Make and Model (camera body).
    Falls back to ('', '') if exiftool is not installed.
    """
    if not shutil.which("exiftool"):
        return "", ""
    try:
        result = subprocess.run(
            ["exiftool", "-Make", "-Model", "-s3", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.strip().splitlines()
        make = lines[0].strip() if len(lines) > 0 else ""
        model = lines[1].strip() if len(lines) > 1 else ""
        return make, model
    except Exception:
        return "", ""
