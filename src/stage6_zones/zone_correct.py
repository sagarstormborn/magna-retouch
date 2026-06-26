"""
Per-zone colour corrections applied on top of the Stage 4 LUT output.

The corrections are completely different for interior vs exterior shots:

EXTERIOR — camera looking out at a property:
    sky      — cool shift (-b*) + highlight soft-clip (overcast Danish sky)
    ground   — subtle warmth, preserve natural tones
    walls    — pass-through (LUT grade)

INTERIOR — camera inside a room:
    ceiling  — mild warmth anchor (LEDs read slightly blue on camera WB)
    windows  — aggressive highlight clip (blown window = lost detail)
    floor    — warm wood/carpet preservation
    walls    — pass-through (LUT grade)

All corrections operate in float32 CIE Lab space and are blended with
a Gaussian-blurred mask so zone boundaries are never visibly sharp.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
import structlog

log = structlog.get_logger(__name__)


def apply_zone_corrections(
    image_rgb: np.ndarray,          # HxWx3 float32 [0, 1]
    zones: Dict[str, np.ndarray],   # from ZoneSegmenter.segment()
    scene_type: str = "exterior",   # "exterior" | "interior"
    cfg: dict | None = None,
) -> np.ndarray:
    """
    Returns float32 [0, 1] RGB with scene-aware zone corrections blended in.
    """
    c = (cfg or {}).get("stage6_zones", {}).get("corrections", {})
    blend_sigma = int(c.get("blend_sigma", 41))

    lab = cv2.cvtColor(np.clip(image_rgb, 0.0, 1.0), cv2.COLOR_RGB2Lab)
    out = lab.copy()

    def soft(name: str) -> np.ndarray | None:
        m = zones.get(name)
        if m is None or m.sum() == 0:
            return None
        k = blend_sigma | 1
        return cv2.GaussianBlur(m.astype(np.float32), (k, k), 0)

    zones_applied = []

    if scene_type == "exterior":
        out, zones_applied = _correct_exterior(out, zones, soft, c)
    else:
        out, zones_applied = _correct_interior(out, zones, soft, c)

    out[:, :, 0] = np.clip(out[:, :, 0], 0.0, 100.0)
    out[:, :, 1] = np.clip(out[:, :, 1], -127.0, 127.0)
    out[:, :, 2] = np.clip(out[:, :, 2], -127.0, 127.0)

    result = np.clip(cv2.cvtColor(out, cv2.COLOR_Lab2RGB), 0.0, 1.0)
    log.info("zone_correct.done", scene_type=scene_type, zones_applied=zones_applied)
    return result


# ── Exterior corrections ───────────────────────────────────────────────────────

def _correct_exterior(
    out: np.ndarray,
    zones: dict,
    soft,
    c: dict,
) -> tuple[np.ndarray, list]:
    """
    Exterior: sky cool-shift + highlight protect; ground subtle warmth.
    The building facade (walls) is already handled by the global LUT.
    """
    applied = []

    # Sky: pull cooler (overcast Danish sky should not read warm)
    sky_w = soft("sky")
    if sky_w is not None:
        sky_da = float(c.get("sky_a_shift",     0.0))
        sky_db = float(c.get("sky_b_shift",    -4.0))
        clip_L = float(c.get("sky_highlight_L", 92.0))

        out[:, :, 1] += sky_w * sky_da
        out[:, :, 2] += sky_w * sky_db

        # Soft-clip highlights (blown sky detail)
        L = out[:, :, 0]
        over = np.maximum(L - clip_L, 0.0)
        L_c = clip_L + np.tanh(over / 12.0) * 8.0
        out[:, :, 0] = L * (1.0 - sky_w) + L_c * sky_w
        applied.append("sky")

    # Ground/garden: slight warmth to keep grass/paving natural
    floor_w = soft("floor")
    if floor_w is not None:
        out[:, :, 2] += floor_w * float(c.get("ground_b_shift", 2.0))
        applied.append("ground")

    return out, applied


# ── Interior corrections ───────────────────────────────────────────────────────

def _correct_interior(
    out: np.ndarray,
    zones: dict,
    soft,
    c: dict,
) -> tuple[np.ndarray, list]:
    """
    Interior: ceiling warmth anchor, aggressive window highlight management,
    floor warmth to preserve wood/carpet tones.
    No sky correction — bright upper regions are ceiling, not sky.
    """
    applied = []

    # Ceiling: mild warmth — camera WB overcorrects mixed LED+window light
    ceiling_w = soft("ceiling")
    if ceiling_w is not None:
        ceil_da = float(c.get("ceiling_a_shift",  0.5))   # slight red
        ceil_db = float(c.get("ceiling_b_shift",  2.5))   # slight warm
        out[:, :, 1] += ceiling_w * ceil_da
        out[:, :, 2] += ceiling_w * ceil_db
        applied.append("ceiling")

    # Windows: the most critical interior correction
    # Blown windows destroy the sense of space — soft-clip aggressively
    win_w = soft("windows")
    if win_w is not None:
        clip_L     = float(c.get("win_highlight_L",    88.0))  # interior clips earlier
        clip_range = float(c.get("win_clip_range",     12.0))
        desat      = float(c.get("win_desaturate",      0.35))

        L = out[:, :, 0]
        over = np.maximum(L - clip_L, 0.0)
        L_c = clip_L + np.tanh(over / clip_range) * (clip_range * 0.5)
        out[:, :, 0] = L * (1.0 - win_w) + L_c * win_w

        # Desaturate near-blown window pixels (white light, not coloured)
        out[:, :, 1] *= 1.0 - win_w * desat
        out[:, :, 2] *= 1.0 - win_w * desat
        applied.append("windows")

    # Floor: wood/carpet warmth — more aggressive than exterior ground
    floor_w = soft("floor")
    if floor_w is not None:
        floor_da = float(c.get("floor_a_shift", 1.0))
        floor_db = float(c.get("floor_b_shift", 3.5))
        out[:, :, 1] += floor_w * floor_da
        out[:, :, 2] += floor_w * floor_db
        applied.append("floor")

    return out, applied
