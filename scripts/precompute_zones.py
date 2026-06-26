"""
Pre-compute SAM zone maps for all training input images.

Saves data/train/zone_maps/<stem>.npy — HxW uint8 zone IDs:
    0 = sky (exterior) / ceiling (interior)
    1 = floor
    2 = windows
    3 = walls (default / unclassified)

Also writes data/train/zone_maps/summary.json with per-image scene type
and zone fractions (used to verify calibration).

Idempotent: skips images that already have a zone map unless --force.

Usage (on GPU server):
    cd /path/to/magna-retouch
    python scripts/precompute_zones.py
    python scripts/precompute_zones.py --force   # recompute all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ZONE_DIR = Path("data/train/zone_maps")
INP_DIR  = Path("data/train/our_input")
SAVE_LONG = 512    # save zone maps at this long-side resolution (small, fast, sufficient)

ZONE_IDS = {"sky": 0, "ceiling": 0, "floor": 1, "windows": 2, "walls": 3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute existing maps")
    args = parser.parse_args()

    ZONE_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    from src.stage6_zones.zone_seg import ZoneSegmenter
    from src.common.config import load_config

    cfg    = load_config("configs/pipeline.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg    = ZoneSegmenter(device=device,
                           checkpoint=cfg["stage6_zones"]["sam_checkpoint"])

    imgs = sorted(INP_DIR.glob("*.jpg")) + sorted(INP_DIR.glob("*.tif"))
    print(f"Found {len(imgs)} input images  →  saving zone maps to {ZONE_DIR}/")

    summary = {}
    done = skipped = 0

    for i, inp_path in enumerate(imgs):
        stem = inp_path.stem.split("_")[0]
        out_path = ZONE_DIR / f"{stem}.npy"

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        print(f"  [{i+1:3d}/{len(imgs)}]  {stem}", end=" ", flush=True)

        img_bgr = cv2.imread(str(inp_path))
        if img_bgr is None:
            print("SKIP (unreadable)")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            zones, scene_type = seg.segment(img_rgb)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Build zone ID map at save resolution
        H, W = img_rgb.shape[:2]
        scale = SAVE_LONG / max(H, W)
        sh, sw = int(H * scale), int(W * scale)

        zone_map = np.full((sh, sw), ZONE_IDS["walls"], dtype=np.uint8)  # default = walls

        # Apply in reverse priority (lowest-priority first, so higher-priority overwrites)
        for zone_name in ("walls", "floor", "sky", "ceiling", "windows"):
            mask = zones.get(zone_name)
            if mask is None:
                continue
            # Resize mask to save resolution
            m_small = cv2.resize(mask.astype(np.uint8), (sw, sh),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            zone_map[m_small] = ZONE_IDS[zone_name]

        np.save(str(out_path), zone_map)

        fracs = {
            name: float((zone_map == zid).mean())
            for name, zid in {"sky_ceil": 0, "floor": 1, "windows": 2, "walls": 3}.items()
        }
        summary[stem] = {"scene_type": scene_type, **fracs}
        print(f"scene={scene_type}  "
              f"sky/ceil={fracs['sky_ceil']:.2f}  floor={fracs['floor']:.2f}  "
              f"win={fracs['windows']:.2f}  walls={fracs['walls']:.2f}")
        done += 1

    # Merge with existing summary if any
    summary_path = ZONE_DIR / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        existing.update(summary)
        summary = existing
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nDone. Computed={done}  Skipped={skipped}  Total={len(imgs)}")
    print(f"Zone maps: {ZONE_DIR}/")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
