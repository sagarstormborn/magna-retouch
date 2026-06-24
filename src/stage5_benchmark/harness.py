"""
Benchmark harness — runs all metrics and maps them to Matt's five criteria.

Produces:
  - JSON results log per image
  - Pass/fail per criterion against configured thresholds
  - Side-by-side contact sheets

Ours vs Reference  →  criteria 2, 3, 4
Autoenhance vs Reference  →  same metrics for comparison baseline
Series consistency  →  criterion 5
HDR highlight protection  →  HDR path gate
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from .metrics import (
    delta_e2000,
    ms_ssim,
    psnr,
    lpips_score,
    local_sharpness,
    wb_series_variance,
    highlight_protection_ok,
)
from .contact_sheet import make_contact_sheet

log = structlog.get_logger(__name__)


def run_image(
    input_img: np.ndarray,
    our_img: np.ndarray,
    reference_img: np.ndarray,
    autoenhance_img: Optional[np.ndarray],
    cfg: dict,
    image_name: str = "unknown",
    contact_sheet_path: Optional[Path] = None,
) -> dict:
    """
    Evaluate a single image against reference.
    Returns a results dict keyed by metric name.
    """
    s5 = cfg["stage5_benchmark"]

    results: dict = {"image": image_name}

    # Criterion 1: sharpness
    results["sharpness_ours"] = local_sharpness(our_img)
    results["sharpness_input"] = local_sharpness(input_img)

    # Criterion 3: colour/tone integrity
    results["delta_e_ours"] = delta_e2000(our_img, reference_img)
    results["delta_e_pass"] = results["delta_e_ours"] <= s5["delta_e_threshold"]

    # Criterion 2: structural
    results["ms_ssim_ours"] = ms_ssim(our_img, reference_img)
    results["ms_ssim_pass"] = results["ms_ssim_ours"] >= s5["ms_ssim_threshold"]
    results["psnr_ours"] = psnr(our_img, reference_img)

    # Criterion 4: perceptual match
    results["lpips_ours"] = lpips_score(our_img, reference_img)
    results["lpips_pass"] = results["lpips_ours"] <= s5["lpips_threshold"]

    # Autoenhance baseline (same metrics)
    if autoenhance_img is not None:
        results["delta_e_autoenhance"] = delta_e2000(autoenhance_img, reference_img)
        results["ms_ssim_autoenhance"] = ms_ssim(autoenhance_img, reference_img)
        results["lpips_autoenhance"] = lpips_score(autoenhance_img, reference_img)

    # HDR highlight protection
    hp_ok, blown_pct = highlight_protection_ok(our_img, s5["highlight_protection_stops"])
    results["highlight_protection_ok"] = hp_ok
    results["highlight_blown_pct"] = blown_pct

    results["overall_pass"] = (
        results["delta_e_pass"]
        and results["ms_ssim_pass"]
        and results["lpips_pass"]
        and hp_ok
    )

    log.info("benchmark.image", **{k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()})

    if s5["contact_sheet"]["enabled"]:
        cs_path = contact_sheet_path or (
            Path(s5["contact_sheet"]["output_dir"]) / f"{image_name}_contact.tiff"
        )
        make_contact_sheet(
            [input_img, our_img, autoenhance_img, reference_img],
            labels=s5["contact_sheet"]["labels"],
            output_path=cs_path,
        )
        results["contact_sheet"] = str(cs_path)

    return results


def run_series(
    results_list: list[dict],
    our_images: list[np.ndarray],
    cfg: dict,
) -> dict:
    """Criterion 5: WB consistency across a property series."""
    s5 = cfg["stage5_benchmark"]
    variance = wb_series_variance(our_images)
    passes = variance <= s5["wb_variance_threshold"]
    series_result = {
        "wb_rb_std": round(variance, 4),
        "wb_variance_pass": passes,
        "wb_variance_threshold": s5["wb_variance_threshold"],
    }
    log.info("benchmark.series", **series_result)
    return series_result


def save_report(results: list[dict], series_result: dict, output_path: Path) -> None:
    report = {"images": results, "series": series_result}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("benchmark.report_saved", path=str(output_path))
