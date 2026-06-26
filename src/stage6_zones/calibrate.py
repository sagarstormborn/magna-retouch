"""
Calibrate zone corrections from training data.

For each training pair (our Stage 4 output vs Matt's retouched target):
  1. Run SAM to get zone masks
  2. Compute per-zone mean Lab(target) - Lab(stage4) → the correction Matt wants
  3. Average across all training pairs per zone

Saves calibrated coefficients to models/sam/zone_corrections.json.
zone_correct.py reads this file at runtime instead of using hardcoded values.

Usage:
    cd /path/to/magna-retouch
    python -m src.stage6_zones.calibrate
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from src.common.config import load_config
    from src.stage4_look.lut3d import load_model, apply_lut
    from src.stage6_zones.zone_seg import ZoneSegmenter
    import structlog
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))  # WARNING+

    cfg    = load_config("configs/pipeline.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(cfg, device)
    seg    = ZoneSegmenter(device=device,
                           checkpoint=cfg.get("stage6_zones", {})
                           .get("sam_checkpoint", "models/sam/sam_vit_b_01ec64.pth"))

    inp_dir = Path("data/train/our_input")
    tgt_dir = Path("data/train/our_target")

    # Use training pairs only (exclude DSCF test set)
    all_pairs = [(p, tgt_dir / p.name) for p in sorted(inp_dir.glob("*.jpg"))
                 if (tgt_dir / p.name).exists() and not p.stem.startswith("DSCF")]
    print(f"Calibrating on {len(all_pairs)} training pairs …")

    # Accumulator: zone_name → list of (da, db) corrections
    corrections: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    scene_counts: dict[str, int] = defaultdict(int)

    EVAL_LONG = 960
    tb = 0.585

    for i, (inp_path, tgt_path) in enumerate(all_pairs):
        print(f"  [{i+1:3d}/{len(all_pairs)}] {inp_path.stem}", end=" ", flush=True)

        inp_u8 = cv2.cvtColor(cv2.imread(str(inp_path)), cv2.COLOR_BGR2RGB)
        tgt_u8 = cv2.cvtColor(cv2.imread(str(tgt_path)), cv2.COLOR_BGR2RGB)

        # Resize to working resolution
        h, w = inp_u8.shape[:2]
        scale = EVAL_LONG / max(h, w)
        inp_u8 = cv2.resize(inp_u8, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        tgt_u8 = cv2.resize(tgt_u8, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        # Aspect-ratio crop (GFX 4:3 → 3:2)
        hi, wi = inp_u8.shape[:2]
        ht, wt = tgt_u8.shape[:2]
        ar_t = wt / ht
        new_wi = int(hi * ar_t)
        if new_wi <= wi:
            xo = (wi - new_wi) // 2
            inp_u8 = inp_u8[:, xo:xo+new_wi]
        inp_u8 = cv2.resize(inp_u8, (wt, ht), interpolation=cv2.INTER_AREA)

        # Stage 4 brightness norm + LUT
        inp_f = inp_u8.astype(np.float32) / 255.0
        mean  = inp_f.mean()
        if 1e-4 < mean < 0.999:
            g = max(0.3, min(3.0, math.log(tb) / math.log(mean)))
            inp_f = np.power(np.clip(inp_f, 1e-8, 1.0), g)
        lut_out = apply_lut((inp_f * 255).astype(np.uint8), model, device)

        # Zone segmentation
        try:
            zones, scene = seg.segment(lut_out)
        except Exception as e:
            print(f"SAM error: {e} — skipped")
            continue

        scene_counts[scene] += 1

        # Convert both to float32 Lab
        lut_lab = cv2.cvtColor(lut_out.astype(np.float32)/255.0, cv2.COLOR_RGB2Lab)
        tgt_lab = cv2.cvtColor(tgt_u8.astype(np.float32)/255.0,  cv2.COLOR_RGB2Lab)
        diff    = tgt_lab - lut_lab   # what Matt's grade adds per pixel

        for zone_name, mask in zones.items():
            if mask.sum() < 100:
                continue
            # Mean correction in this zone
            dL = float(diff[:, :, 0][mask].mean())
            da = float(diff[:, :, 1][mask].mean())
            db = float(diff[:, :, 2][mask].mean())
            corrections[f"{scene}/{zone_name}"].append((dL, da, db))

        print(f"scene={scene}  zones={list(zones)}", flush=True)

    # Aggregate
    print(f"\n=== Calibrated corrections (mean Lab shift Matt wants) ===")
    results = {}
    for key, vals in sorted(corrections.items()):
        arr = np.array(vals)
        dL_m, da_m, db_m = arr.mean(axis=0)
        results[key] = {"dL": round(dL_m, 3), "da": round(da_m, 3), "db": round(db_m, 3), "n": len(vals)}
        print(f"  {key:<30}  dL={dL_m:+.2f}  da={da_m:+.2f}  db={db_m:+.2f}  (n={len(vals)})")

    results["scene_counts"] = dict(scene_counts)

    out_path = Path("models/sam/zone_corrections.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
