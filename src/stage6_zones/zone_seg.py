"""
SAM ViT-B zone segmenter for real estate photos.

Runs SAM's automatic mask generator then classifies each mask into one of four
semantic zones using position + colour heuristics tuned for Danish real estate:
    sky      — overcast/blue upper regions (exterior)
    floor    — warm lower regions (wood/carpet)
    windows  — very bright interior rectangles
    walls    — everything else (main LUT correction zone)
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import structlog

log = structlog.get_logger(__name__)

_SAM_REGISTRY = "vit_b"
_SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
_DEFAULT_CKPT = "models/sam/sam_vit_b_01ec64.pth"

# Running instance cache (one per checkpoint path)
_instance_cache: dict[str, "ZoneSegmenter"] = {}


def get_segmenter(cfg: dict) -> "ZoneSegmenter":
    ckpt = cfg.get("stage6_zones", {}).get("sam_checkpoint", _DEFAULT_CKPT)
    if ckpt not in _instance_cache:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _instance_cache[ckpt] = ZoneSegmenter(device=device, checkpoint=ckpt)
    return _instance_cache[ckpt]


class ZoneSegmenter:
    """Lazy-loading SAM ViT-B wrapper."""

    def __init__(self, device: str = "cuda", checkpoint: str = _DEFAULT_CKPT):
        self.device = device
        self.checkpoint = Path(checkpoint)
        self._generator = None

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init(self):
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError as e:
            raise ImportError(
                "segment-anything not installed. Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from e

        if not self.checkpoint.exists():
            _download(self.checkpoint)

        sam = sam_model_registry[_SAM_REGISTRY](checkpoint=str(self.checkpoint))
        sam.to(self.device)
        self._generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,           # 256 prompts — fast, sufficient for zone masks
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=500,
        )
        log.info("sam.loaded", device=self.device, checkpoint=str(self.checkpoint))

    # ── Public API ────────────────────────────────────────────────────────────

    def segment(self, image_rgb_u8: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Args:
            image_rgb_u8: HxWx3 uint8 RGB
        Returns:
            dict with keys sky, floor, windows, walls — each HxW bool mask
        """
        if self._generator is None:
            self._init()

        H, W = image_rgb_u8.shape[:2]

        # SAM works best at ~1024px long side
        scale = min(1024.0 / max(H, W), 1.0)
        if scale < 1.0:
            h2, w2 = int(H * scale), int(W * scale)
            thumb = cv2.resize(image_rgb_u8, (w2, h2), interpolation=cv2.INTER_AREA)
        else:
            thumb, h2, w2 = image_rgb_u8, H, W

        masks_data = self._generator.generate(thumb)
        # Largest masks first — dominant zones before small details
        masks_data.sort(key=lambda m: m["area"], reverse=True)

        sky   = _detect_sky(image_rgb_u8, masks_data, H, W, scale)
        floor = _detect_floor(image_rgb_u8, masks_data, H, W, scale)
        wins  = _detect_windows(image_rgb_u8, masks_data, H, W, scale, exclude=sky)

        walls = ~(sky | floor | wins)

        log.info(
            "zone_seg.done",
            sky_pct=round(sky.mean() * 100, 1),
            floor_pct=round(floor.mean() * 100, 1),
            windows_pct=round(wins.mean() * 100, 1),
            walls_pct=round(walls.mean() * 100, 1),
        )
        return {"sky": sky, "floor": floor, "windows": wins, "walls": walls}


# ── Zone heuristics ───────────────────────────────────────────────────────────

def _up(mask_small: np.ndarray, H: int, W: int) -> np.ndarray:
    return cv2.resize(mask_small.astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def _detect_sky(img: np.ndarray, masks: list, H: int, W: int, scale: float) -> np.ndarray:
    """
    Sky: bright or blue/grey, centroid in top 40% of frame.
    Danish overcast sky = high luminance, very low saturation.
    Clear sky = high blue, low saturation.
    """
    top_cut = int(H * 0.40)
    sky = np.zeros((H, W), bool)

    for m in masks:
        seg = _up(m["segmentation"], H, W)
        if seg.sum() == 0:
            continue
        cy = np.argwhere(seg).mean(axis=0)[0]   # centroid row
        if cy > top_cut:
            continue
        top_frac = seg[:top_cut].sum() / seg.sum()
        if top_frac < 0.55:
            continue

        region = img[seg].astype(np.float32)
        hsv = cv2.cvtColor(region.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        mean_sat = float(hsv[:, 1].mean())   # 0-255
        mean_val = float(hsv[:, 2].mean())   # 0-255

        # Bright & low-saturation (overcast) OR bright & blue-hued (clear)
        is_overcast = mean_val > 130 and mean_sat < 80
        mean_hue = float(hsv[:, 0].mean())  # 0-180 in OpenCV
        is_blue_sky = mean_val > 100 and mean_sat > 40 and 90 < mean_hue < 140
        if is_overcast or is_blue_sky:
            sky |= seg

    # Fallback: geometric top band if SAM found nothing
    if sky.sum() < H * W * 0.01:
        band = img[:top_cut].astype(np.float32)
        lum = band.mean(axis=2)
        sky[:top_cut] = lum > 160

    return sky


def _detect_floor(img: np.ndarray, masks: list, H: int, W: int, scale: float) -> np.ndarray:
    """
    Floor: centroid in bottom 30%, warm (high R/G, low B ratio) or neutral-dark.
    Covers wood, carpet, stone tiles.
    """
    bot_start = int(H * 0.68)
    floor = np.zeros((H, W), bool)

    for m in masks:
        seg = _up(m["segmentation"], H, W)
        if seg.sum() == 0:
            continue
        cy = np.argwhere(seg).mean(axis=0)[0]
        if cy < bot_start:
            continue
        bot_frac = seg[bot_start:].sum() / seg.sum()
        if bot_frac < 0.50:
            continue

        region = img[seg].astype(np.float32)
        r_mean = region[:, 0].mean()
        g_mean = region[:, 1].mean()
        b_mean = region[:, 2].mean()
        total  = (r_mean + g_mean + b_mean) / 3.0 + 1.0
        b_frac = b_mean / total

        # Warm (wood/carpet) or low-blue (stone/concrete)
        if r_mean >= g_mean * 0.88 and b_frac < 0.38:
            floor |= seg

    if floor.sum() < H * W * 0.01:
        floor[bot_start:] = True

    return floor


def _detect_windows(
    img: np.ndarray, masks: list, H: int, W: int, scale: float,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """
    Windows / light sources: very bright (mean luminance > 200), not sky.
    Typical in interior shots where windows blow highlights.
    """
    top_guard = int(H * 0.20)   # ignore top strip (captured by sky heuristic)
    wins = np.zeros((H, W), bool)

    for m in masks:
        seg = _up(m["segmentation"], H, W)
        seg[:top_guard] = False
        if exclude is not None:
            seg &= ~exclude
        if seg.sum() < 300:
            continue

        region = img[seg].astype(np.float32)
        if region.mean() > 200:     # very bright = light source / window
            wins |= seg

    return wins


# ── Download ──────────────────────────────────────────────────────────────────

def _download(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("sam.downloading", url=_SAM_URL, dest=str(path))
    def _progress(block, block_size, total):
        if total > 0:
            pct = block * block_size / total * 100
            print(f"\r  SAM ViT-B download: {min(pct, 100):.0f}%", end="", flush=True)
    urllib.request.urlretrieve(_SAM_URL, str(path), reporthook=_progress)
    print()
    log.info("sam.downloaded", size_mb=path.stat().st_size // 1_000_000)
