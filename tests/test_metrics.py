"""
Unit tests for Stage 5 metrics.
All tests use synthetic numpy arrays — no real RAW files needed.
"""
import numpy as np
import pytest

from src.stage5_benchmark.metrics import (
    delta_e2000,
    ms_ssim,
    psnr,
    local_sharpness,
    wb_series_variance,
    highlight_protection_ok,
)


def _solid(r, g, b, h=64, w=64) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint16)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def test_delta_e_identical():
    img = _solid(32768, 32768, 32768)
    assert delta_e2000(img, img) < 0.01


def test_delta_e_differs():
    a = _solid(60000, 30000, 10000)
    b = _solid(10000, 30000, 60000)
    assert delta_e2000(a, b) > 10.0


def test_ms_ssim_identical():
    img = _solid(40000, 20000, 15000)
    assert ms_ssim(img, img) > 0.999


def test_psnr_identical():
    img = _solid(32768, 32768, 32768)
    assert psnr(img, img) > 80.0


def test_local_sharpness_sharp_vs_blurry():
    import cv2
    sharp = np.random.randint(0, 65535, (64, 64, 3), dtype=np.uint16)
    blurry = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert local_sharpness(sharp) > local_sharpness(blurry)


def test_wb_series_variance_consistent():
    # All same image — variance should be near 0
    img = _solid(40000, 32000, 28000)
    assert wb_series_variance([img, img, img]) < 0.01


def test_wb_series_variance_inconsistent():
    warm = _solid(50000, 32000, 20000)
    cool = _solid(20000, 32000, 50000)
    assert wb_series_variance([warm, cool]) > 0.5


def test_highlight_protection_ok_no_clipping():
    img = _solid(45000, 45000, 45000)   # below 2-stop threshold (~49151)
    ok, pct = highlight_protection_ok(img, stops=2)
    assert ok


def test_highlight_protection_fails_clipping():
    img = _solid(65535, 65535, 65535)   # fully clipped
    ok, pct = highlight_protection_ok(img, stops=2)
    assert not ok
    assert pct > 99.0
