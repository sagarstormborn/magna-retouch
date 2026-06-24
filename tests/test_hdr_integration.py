"""
Integration test for Stage 2 HDR merge using synthetic brackets.

Creates three bracketed uint16 images (under / normal / over exposed)
and verifies Mertens fusion produces:
  - correct shape / dtype
  - better highlight headroom than the over-exposed bracket alone
  - better shadow detail than the under-exposed bracket alone
  - passes the top-2-stops highlight protection check
"""
import numpy as np
import pytest

from src.common.config import load_config
from src.stage2_hdr.stage import process


@pytest.fixture
def cfg():
    return load_config()


def _synthetic_brackets():
    """
    Simulate 3 brackets of the same scene:
      -2 EV  (under) : bright highlights preserved, dark shadows
       0 EV  (normal): balanced
      +2 EV  (over)  : bright shadows, clipped highlights

    Scene has a checkerboard texture so Mertens contrast weights are nonzero
    everywhere — a pure gradient gives near-zero contrast weight and artificially
    dark fused output because Mertens penalises flat regions.
    """
    h, w = 256, 256

    # Gradient base — capped at 0.60 so 0EV bracket never clips.
    # At +2EV (×4) right edge = 0.60×4=2.4 → clips: over-exposed bracket blows highlights.
    # At -2EV (×0.25) left edge = 0.05×0.25=0.0125: shadows are very dark.
    # This gives Mertens clean signal in every zone without ambiguous edge cases.
    base = np.linspace(0.05, 0.60, w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)

    # Checkerboard texture so every region has local contrast (Mertens penalises flat areas)
    checker = ((np.arange(h)[:, None] // 8 + np.arange(w)[None, :] // 8) % 2).astype(np.float32)
    base = np.clip(base + checker * 0.08, 0.0, 1.0)
    base = np.stack([base, base * 0.9, base * 0.8], axis=2)  # slight warm tint

    def to_uint16(f, ev):
        exposed = np.clip(f * (2.0 ** ev), 0, 1)
        return (exposed * 65535).astype(np.uint16)

    return [to_uint16(base, -2), to_uint16(base, 0), to_uint16(base, +2)]


def test_mertens_output_shape_dtype(cfg):
    brackets = _synthetic_brackets()
    result = process(brackets, cfg)
    assert result.shape == brackets[0].shape
    assert result.dtype == np.uint16


def test_mertens_better_shadows_than_underexposed(cfg):
    brackets = _synthetic_brackets()
    under_clean = brackets[0].copy()
    fused = process([b.copy() for b in brackets], cfg)
    # Left 10% = shadow region; fused should be brighter than -2EV bracket
    shadow_fused = fused[:, :25, :].mean()
    shadow_under = under_clean[:, :25, :].mean()
    assert shadow_fused > shadow_under, (
        f"Fused shadows ({shadow_fused:.1f}) should be > under-exposed ({shadow_under:.1f})"
    )


def test_mertens_better_highlights_than_overexposed(cfg):
    brackets = _synthetic_brackets()
    # Keep a clean copy of the over-exposed bracket before process() runs
    # (MTB aligner modifies inputs in-place; we must not share the reference)
    over_clean = brackets[2].copy()
    fused = process([b.copy() for b in brackets], cfg)
    # Right 10% = highlight region; fused should have less clipping than +2EV alone
    hi_fused_clipped = (fused[:, -25:, :] >= 65000).mean()
    hi_over_clipped = (over_clean[:, -25:, :] >= 65000).mean()
    assert hi_fused_clipped < hi_over_clipped, (
        f"Fused clipping ({hi_fused_clipped:.3f}) should be < over-exposed clipping ({hi_over_clipped:.3f})"
    )


def test_highlight_protection_passes(cfg):
    brackets = _synthetic_brackets()
    fused = process(brackets, cfg)
    stops = cfg["stage5_benchmark"]["highlight_protection_stops"]
    threshold = int(65535 * (1.0 - 2.0 ** -stops))
    blown_pct = np.all(fused >= threshold, axis=2).mean() * 100.0
    assert blown_pct <= 0.1, f"Blown highlights {blown_pct:.2f}% > 0.1% threshold"


def test_single_bracket_passthrough(cfg):
    brackets = _synthetic_brackets()
    result = process([brackets[1]], cfg)   # only normal exposure
    np.testing.assert_array_equal(result, brackets[1])


def test_mtb_alignment_doesnt_crash(cfg):
    """MTB alignment should run without error on synthetic brackets."""
    brackets = _synthetic_brackets()
    # Shift middle bracket by a few pixels to simulate micro-vibration
    shifted = np.roll(brackets[1], shift=3, axis=1)
    result = process([brackets[0], shifted, brackets[2]], cfg)
    assert result.shape == brackets[0].shape
