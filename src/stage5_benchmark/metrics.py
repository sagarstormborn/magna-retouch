"""
All metrics mapped to Matt's five criteria:

1. Clean/sharp at 100%      → local sharpness (Laplacian variance) + noise proxy
2. RAW conversion correct   → structural (MS-SSIM / PSNR) vs reference
3. Colour/tone integrity    → ΔE2000 vs reference
4. Matches Matt's look      → LPIPS + ΔE2000 vs Matt's retouched reference
5. Series consistency       → WB variance (R/B ratio std-dev across series)
HDR path                    → highlight protection (top-2-stops check)
"""
from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger(__name__)


# ── ΔE2000 ──────────────────────────────────────────────────────────────────

def delta_e2000(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Mean ΔE2000 between two uint16 RGB images.
    Both must be the same shape.
    """
    import colour

    f_a = img_a.astype(np.float32) / 65535.0
    f_b = img_b.astype(np.float32) / 65535.0

    # sRGB → XYZ → Lab (D65, 2°)
    lab_a = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(f_a))
    lab_b = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(f_b))

    de = colour.delta_E(lab_a, lab_b, method="CIE 2000")
    return float(np.mean(de))


# ── MS-SSIM / PSNR ──────────────────────────────────────────────────────────

def ms_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    a = img_a.astype(np.float32) / 65535.0
    b = img_b.astype(np.float32) / 65535.0
    score, _ = structural_similarity(a, b, channel_axis=2, full=True, data_range=1.0)
    return float(score)


def psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    from skimage.metrics import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(img_a, img_b, data_range=65535))


# ── LPIPS ────────────────────────────────────────────────────────────────────

def lpips_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    import lpips
    import torch

    _net = getattr(lpips_score, "_net", None)
    if _net is None:
        lpips_score._net = lpips.LPIPS(net="alex")
    net = lpips_score._net

    def _to_tensor(img: np.ndarray) -> "torch.Tensor":
        f = img.astype(np.float32) / 65535.0 * 2.0 - 1.0   # [-1, 1]
        return torch.from_numpy(f.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        score = net(_to_tensor(img_a), _to_tensor(img_b))
    return float(score.mean())


# ── Sharpness / noise ────────────────────────────────────────────────────────

def local_sharpness(img: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(lap.var())


# ── WB series consistency ────────────────────────────────────────────────────

def wb_series_variance(images: list[np.ndarray]) -> float:
    """Std-dev of the R/B ratio across images (proxy for CCT swing)."""
    ratios = []
    for img in images:
        f = img.astype(np.float32)
        r_mean = f[:, :, 0].mean()
        b_mean = f[:, :, 2].mean()
        ratios.append(r_mean / (b_mean + 1e-8))
    return float(np.std(ratios))


# ── Highlight protection ──────────────────────────────────────────────────────

def highlight_protection_ok(img: np.ndarray, stops: int = 2) -> tuple[bool, float]:
    """Returns (passes, blown_pct)."""
    threshold = int(65535 * (1.0 - 2.0 ** -stops))
    blown_pct = np.all(img >= threshold, axis=2).mean() * 100.0
    return blown_pct <= 0.1, float(blown_pct)
