"""
Per-zone colour corrections applied on top of the Stage 4 LUT output.

All corrections operate in float32 CIE Lab space and are blended with a
Gaussian-blurred mask so zone boundaries are never sharp (avoids halos).

Tuned for Danish real estate:
    sky     — slight cool shift + highlight protection (overcast sky shouldn't be warm)
    floor   — gentle warmth (wood/carpet tones)
    windows — soft highlight clip (prevent blown interiors)
    walls   — pass-through (LUT already handles the main grade)
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
import structlog

log = structlog.get_logger(__name__)


def apply_zone_corrections(
    image_rgb: np.ndarray,               # HxWx3 float32 [0, 1]
    zones: Dict[str, np.ndarray],        # from ZoneSegmenter.segment()
    cfg: dict | None = None,
) -> np.ndarray:
    """
    Returns float32 [0, 1] RGB with zone corrections blended in.
    cfg key: stage6_zones.corrections (all optional, falls back to defaults).
    """
    c = (cfg or {}).get("stage6_zones", {}).get("corrections", {})
    blend_sigma    = int(c.get("blend_sigma", 41))

    sky_da         = float(c.get("sky_a_shift",     0.0))   # Lab a* shift
    sky_db         = float(c.get("sky_b_shift",    -4.0))   # cool (negative b* = more blue)
    sky_clip_L     = float(c.get("sky_highlight_L", 92.0))  # L* soft-clip threshold

    floor_da       = float(c.get("floor_a_shift",   1.0))   # slight red
    floor_db       = float(c.get("floor_b_shift",   3.5))   # warm yellow

    win_clip_L     = float(c.get("win_highlight_L", 90.0))  # highlight clip for windows
    win_clip_range = float(c.get("win_clip_range",  15.0))  # tanh compression width

    H, W = image_rgb.shape[:2]

    # Float32 LAB: cv2 expects [0,1] input → L in [0,100], a/b in [-127,127]
    lab = cv2.cvtColor(np.clip(image_rgb, 0.0, 1.0), cv2.COLOR_RGB2Lab)
    out = lab.copy()

    def _soft_mask(name: str) -> np.ndarray | None:
        m = zones.get(name)
        if m is None or m.sum() == 0:
            return None
        blurred = cv2.GaussianBlur(m.astype(np.float32),
                                   (blend_sigma | 1, blend_sigma | 1), 0)
        return blurred

    # ── Sky ───────────────────────────────────────────────────────────────────
    sky_w = _soft_mask("sky")
    if sky_w is not None:
        out[:, :, 1] += sky_w * sky_da
        out[:, :, 2] += sky_w * sky_db
        # Soft-clip sky highlights: compress L* above threshold with tanh
        L = out[:, :, 0]
        over = np.maximum(L - sky_clip_L, 0.0)
        L_clipped = sky_clip_L + np.tanh(over / 12.0) * 8.0
        out[:, :, 0] = L * (1.0 - sky_w) + L_clipped * sky_w

    # ── Floor ─────────────────────────────────────────────────────────────────
    floor_w = _soft_mask("floor")
    if floor_w is not None:
        out[:, :, 1] += floor_w * floor_da
        out[:, :, 2] += floor_w * floor_db

    # ── Windows / light sources ───────────────────────────────────────────────
    win_w = _soft_mask("windows")
    if win_w is not None:
        L = out[:, :, 0]
        over = np.maximum(L - win_clip_L, 0.0)
        L_clipped = win_clip_L + np.tanh(over / win_clip_range) * (win_clip_range * 0.6)
        out[:, :, 0] = L * (1.0 - win_w) + L_clipped * win_w
        # Gently desaturate blown highlights
        out[:, :, 1] = out[:, :, 1] * (1.0 - win_w * 0.25)
        out[:, :, 2] = out[:, :, 2] * (1.0 - win_w * 0.25)

    # ── Clip to valid Lab range and convert back ──────────────────────────────
    out[:, :, 0] = np.clip(out[:, :, 0], 0.0, 100.0)
    out[:, :, 1] = np.clip(out[:, :, 1], -127.0, 127.0)
    out[:, :, 2] = np.clip(out[:, :, 2], -127.0, 127.0)

    result = cv2.cvtColor(out, cv2.COLOR_Lab2RGB)
    result = np.clip(result, 0.0, 1.0)

    log.info("zone_correct.done", zones_applied=[k for k, w in
             [("sky", sky_w), ("floor", floor_w), ("windows", win_w)] if w is not None])
    return result
