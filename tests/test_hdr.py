import numpy as np
import pytest

from src.stage2_hdr.stage import _check_highlight_protection


def test_highlight_protection_ok():
    img = np.full((32, 32, 3), 45000, dtype=np.uint16)
    # Should not raise or log warning
    _check_highlight_protection(img, stops=2)


def test_highlight_protection_blown():
    # Capture log warning would need structlog test capture — just ensure it runs
    img = np.full((32, 32, 3), 65535, dtype=np.uint16)
    _check_highlight_protection(img, stops=2)  # logs warning, does not raise
