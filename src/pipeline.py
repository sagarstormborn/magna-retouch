"""
Main pipeline orchestrator.

Processes a directory of RAW brackets (grouped by bracket-set prefix),
runs all stages, and optionally benchmarks vs reference.

Usage:
    python -m src.pipeline process --input data/raw/property_001 --output data/processed/property_001
    python -m src.pipeline benchmark --input data/raw/property_001 --output data/processed/property_001 \
        --reference data/reference/property_001
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import click
import structlog

from src.common.config import load_config
from src.common.io import find_images, save_tiff_16
from src.common.logging import setup_logging

log = structlog.get_logger(__name__)


@click.group()
@click.option("--config", default="configs/pipeline.yaml", show_default=True)
@click.pass_context
def cli(ctx, config):
    cfg = load_config(config)
    setup_logging(cfg["logging"]["level"], cfg["logging"]["log_dir"])
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfg


@cli.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "output_dir", required=True)
@click.pass_context
def process(ctx, input_dir, output_dir):
    """Run Stages 1–4 on a directory of RAW brackets."""
    cfg = ctx.obj["cfg"]
    _ensure_spawn()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_files = find_images(input_path)
    if not raw_files:
        log.error("pipeline.no_raw_files", dir=str(input_path))
        raise click.Abort()

    log.info("pipeline.start", n_files=len(raw_files), input=str(input_path))

    # Group brackets by filename prefix (e.g., DSC_001_+0, DSC_001_+2, DSC_001_-2 → DSC_001)
    groups = _group_brackets(raw_files)
    log.info("pipeline.bracket_groups", n_groups=len(groups))

    results = []
    for group_name, paths in groups.items():
        result = _process_group(group_name, paths, output_path, cfg)
        results.append(result)

    log.info("pipeline.done", processed=len(results))


@cli.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path(exists=True))
@click.option("--reference", "ref_dir", required=True, type=click.Path(exists=True))
@click.option("--autoenhance", "ae_dir", default=None, type=click.Path())
@click.option("--report", "report_path", default="logs/benchmark_report.json")
@click.pass_context
def benchmark(ctx, input_dir, output_dir, ref_dir, ae_dir, report_path):
    """Run Stage 5 harness on processed vs reference images."""
    from src.stage5_benchmark.harness import run_image, run_series, save_report
    from src.common.io import load_tiff_16

    cfg = ctx.obj["cfg"]
    processed_files = sorted(Path(output_dir).glob("*.tiff")) + sorted(Path(output_dir).glob("*.tif"))
    ref_files = {p.stem: p for p in (sorted(Path(ref_dir).glob("*.tiff")) + sorted(Path(ref_dir).glob("*.tif")))}
    ae_files = {}
    if ae_dir:
        ae_files = {p.stem: p for p in (sorted(Path(ae_dir).glob("*.tiff")) + sorted(Path(ae_dir).glob("*.tif")))}

    input_files = {p.stem: p for p in (sorted(Path(input_dir).glob("*.tiff")) + sorted(Path(input_dir).glob("*.tif")))}

    image_results = []
    our_images = []
    for proc_path in processed_files:
        stem = proc_path.stem
        if stem not in ref_files:
            log.warning("benchmark.no_reference", stem=stem)
            continue

        our_img = load_tiff_16(proc_path)
        ref_img = load_tiff_16(ref_files[stem])
        inp_img = load_tiff_16(input_files[stem]) if stem in input_files else our_img
        ae_img = load_tiff_16(ae_files[stem]) if stem in ae_files else None

        result = run_image(inp_img, our_img, ref_img, ae_img, cfg, image_name=stem)
        image_results.append(result)
        our_images.append(our_img)

    series_result = run_series(image_results, our_images, cfg)
    save_report(image_results, series_result, Path(report_path))

    passed = sum(1 for r in image_results if r.get("overall_pass"))
    click.echo(f"\nBenchmark: {passed}/{len(image_results)} images passed all thresholds")
    click.echo(f"Series WB variance pass: {series_result['wb_variance_pass']}")
    click.echo(f"Report: {report_path}")


def _process_group(group_name: str, paths: list[Path], output_dir: Path, cfg: dict) -> Path:
    from src.stage1_raw.stage import process as s1
    from src.stage2_hdr.stage import process as s2
    from src.stage3_wb.stage import process_single as s3
    from src.stage4_look.stage import process as s4
    from src.stage6_zones.stage import process as s6

    log.info("pipeline.group_start", group=group_name, n_brackets=len(paths))

    # Stage 1: decode all brackets
    decoded = []
    for p in paths:
        img, exif = s1(p, cfg)
        decoded.append((img, exif))

    imgs = [d[0] for d in decoded]

    # Stage 2: HDR merge
    merged = s2(imgs, cfg)

    # Stage 3: WB
    wb_img, _ = s3(merged, cfg)

    # Stage 4: look (skipped if model not trained yet)
    try:
        lut_img = s4(wb_img, cfg)
    except FileNotFoundError as e:
        log.warning("pipeline.stage4_skipped", reason=str(e))
        lut_img = wb_img

    # Stage 6: SAM zone segmentation + per-zone corrections
    final = s6(lut_img, cfg)

    out_path = output_dir / f"{group_name}.tiff"
    save_tiff_16(out_path, final)
    log.info("pipeline.group_done", group=group_name, output=str(out_path))
    return out_path


def _group_brackets(paths: list[Path]) -> dict[str, list[Path]]:
    """
    Group by the common filename prefix before the last underscore-separated
    exposure indicator (e.g., DSC_001_-2EV, DSC_001_0EV, DSC_001_+2EV → DSC_001).
    Falls back to putting every file in its own group if no pattern is found.
    """
    from collections import defaultdict
    import re

    groups: dict[str, list[Path]] = defaultdict(list)
    ev_pattern = re.compile(r"[_\-+]\d+(?:ev|EV)?$")

    for p in paths:
        stem = p.stem
        match = ev_pattern.search(stem)
        key = stem[: match.start()] if match else stem
        groups[key].append(p)

    return dict(sorted(groups.items()))


def _ensure_spawn():
    """rawpy + OpenMP deadlocks under fork — set spawn before any Pool."""
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")


if __name__ == "__main__":
    cli()
