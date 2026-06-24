import numpy as np
import pytest

from src.stage3_wb.estimator import shades_of_gray, apply_wb, series_illuminant, wb_cct_variance


def _img(r, g, b) -> np.ndarray:
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def test_shades_of_gray_neutral():
    img = _img(0.5, 0.5, 0.5)
    ill = shades_of_gray(img)
    np.testing.assert_allclose(ill, [1.0, 1.0, 1.0], atol=1e-5)


def test_shades_of_gray_warm():
    img = _img(0.8, 0.5, 0.3)
    ill = shades_of_gray(img)
    # Red channel illuminant should be largest for a warm-tinted scene
    assert ill[0] > ill[2]


def test_apply_wb_neutral():
    img_u16 = np.full((8, 8, 3), 32768, dtype=np.uint16)
    ill = np.array([1.0, 1.0, 1.0])
    result = apply_wb(img_u16, ill)
    np.testing.assert_array_equal(result, img_u16)


def test_series_illuminant_median():
    warm = np.array([1.5, 1.0, 0.7])
    cool = np.array([0.7, 1.0, 1.5])
    neutral = np.array([1.0, 1.0, 1.0])
    median = series_illuminant([warm, cool, neutral])
    np.testing.assert_allclose(median, neutral, atol=1e-5)


def test_wb_cct_variance_zero():
    ill = np.array([1.2, 1.0, 0.9])
    assert wb_cct_variance([ill, ill, ill]) < 1e-6
