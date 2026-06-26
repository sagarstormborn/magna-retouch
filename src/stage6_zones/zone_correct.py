"""
Per-zone colour corrections applied on top of the Stage 4 LUT output.

Corrections are loaded from models/sam/zone_corrections.json (produced by
calibrate.py). This file contains the mean Lab(target) - Lab(stage4) per zone
measured on the training set — i.e., what Matt's grade actually adds per zone.

Fallback to zero-corrections if the calibration file is absent (safe pass-through).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import structlog

log = structlog.get_logger(__name__)

_CALIBRATION_PATH = "models/sam/zone_corrections.json"
_calib_cache: dict | None = None


def _load_calibration(cfg: dict | None) -> dict:
    global _calib_cache
    if _calib_cache is not None:
        return _calib_cache
    path = Path((cfg or {}).get("stage6_zones", {}).get(
        "calibration_path", _CALIBRATION_PATH))
    if not path.exists():
        log.warning("zone_correct.no_calibration", path=str(path),
                    note="run: python -m src.stage6_zones.calibrate")
        _calib_cache = {}
        return _calib_cache
    _calib_cache = json.loads(path.read_text())
    log.info("zone_correct.calibration_loaded", path=str(path),
             keys=[k for k in _calib_cache if k != "scene_counts"])
    return _calib_cache


def apply_zone_corrections(
    image_rgb: np.ndarray,          # HxWx3 float32 [0, 1]
    zones: Dict[str, np.ndarray],   # from ZoneSegmenter.segment()
    scene_type: str = "exterior",   # "exterior" | "interior"
    cfg: dict | None = None,
) -> np.ndarray:
    """
    Returns float32 [0, 1] RGB with calibrated zone corrections blended in.
    If no calibration file exists, returns image unchanged.
    """
    calib = _load_calibration(cfg)
    if not calib:
        return image_rgb

    blend_sigma = int((cfg or {}).get("stage6_zones", {})
                      .get("corrections", {}).get("blend_sigma", 41))

    lab = cv2.cvtColor(np.clip(image_rgb, 0.0, 1.0), cv2.COLOR_RGB2Lab)
    out = lab.copy()
    applied = []

    for zone_name, mask in zones.items():
        if mask is None or mask.sum() < 100:
            continue

        key = f"{scene_type}/{zone_name}"
        entry = calib.get(key)
        if entry is None:
            continue

        dL = float(entry.get("dL", 0.0))
        da = float(entry.get("da", 0.0))
        db = float(entry.get("db", 0.0))

        # Skip negligible corrections (< 0.5 Lab unit) to avoid touching things unnecessarily
        if abs(dL) < 0.5 and abs(da) < 0.5 and abs(db) < 0.5:
            continue

        k = blend_sigma | 1
        w = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)

        out[:, :, 0] += w * dL
        out[:, :, 1] += w * da
        out[:, :, 2] += w * db
        applied.append(zone_name)

    out[:, :, 0] = np.clip(out[:, :, 0], 0.0, 100.0)
    out[:, :, 1] = np.clip(out[:, :, 1], -127.0, 127.0)
    out[:, :, 2] = np.clip(out[:, :, 2], -127.0, 127.0)

    result = np.clip(cv2.cvtColor(out, cv2.COLOR_Lab2RGB), 0.0, 1.0)
    log.info("zone_correct.done", scene_type=scene_type, zones_applied=applied)
    return result
