import sys; sys.path.insert(0, '/Users/codesageml/Desktop/AiRetouch')
import re, cv2, math
from pathlib import Path
import numpy as np
from src.common.config import load_config
from src.common.logging import setup_logging
from src.common.io import uint16_to_float
import rawpy

BASE      = Path("/Users/codesageml/Library/Mobile Documents/com~apple~CloudDocs/MAGNALUX_Lux-Retouch_AI-samples/MAGNA LUX")
RAW_DIR   = BASE / "A MAGNA LUX RAW "
FINAL_DIR = BASE / "B MAGNA LUX FINAL "

OUT_INP = Path("/Users/codesageml/Desktop/AiRetouch/data/train/our_input")
OUT_TGT = Path("/Users/codesageml/Desktop/AiRetouch/data/train/our_target")
OUT_INP.mkdir(parents=True, exist_ok=True)
OUT_TGT.mkdir(parents=True, exist_ok=True)

cfg = load_config('/Users/codesageml/Desktop/AiRetouch/configs/pipeline.yaml')
setup_logging("WARNING", "/Users/codesageml/Desktop/AiRetouch/logs", "magna_lux_batch")

# Import pipeline stages
import os; os.chdir('/Users/codesageml/Desktop/AiRetouch')
from src.stage1_raw.lens_correction import correct_lens
from src.stage1_raw.decode import _extract_exif
from src.stage3_wb.stage import process_single as s3

rafs = sorted(RAW_DIR.glob("*.RAF"))[:40]
print(f"Processing {len(rafs)} MAGNA LUX RAFs (half_size=True for speed) ...")

TARGET_BRIGHTNESS = cfg["stage4_look"]["train"]["target_brightness"]

done = 0
for raf in rafs:
    num = re.search(r"_(\d{4})\.RAF", raf.name)
    if not num: continue
    num = num.group(1)
    final = FINAL_DIR / f"FINAL_MAGNALUX_Lux-retouch_AI-samples_{num}.jpg"
    if not final.exists():
        print(f"  skip {num} (no final)")
        continue

    try:
        # Decode at half size for speed (still 4140x3104 = plenty for 480 crops)
        with rawpy.imread(str(raf)) as raw:
            img = raw.postprocess(
                no_auto_bright=True, gamma=(1,1), output_bps=16,
                use_camera_wb=True,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                half_size=True,
            )
            exif = _extract_exif(raw, raf)

        # Apply lens correction (GF20-35mm not in Lensfun → will log miss + passthrough)
        from src.stage1_raw.lens_correction import correct_lens
        lc = cfg["stage1_raw"]["lens_correction"]
        img = correct_lens(img, exif,
                           interpolation=lc["interpolation"],
                           loose_search_fallback=lc["loose_search_fallback"])

        # Stage 3 WB
        wb16, _ = s3(img, cfg, exif=exif)

        # Convert to display uint8 with gamma
        f32 = uint16_to_float(wb16)
        p2, p998 = np.percentile(f32, 0.2), np.percentile(f32, 99.8)
        disp = np.power(np.clip((f32-p2)/(p998-p2+1e-8),0,1), 1/2.2)
        u8 = (disp * 255).astype(np.uint8)

        stem = f"ML{num}"
        out_path = OUT_INP / f"{stem}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 97])

        link = OUT_TGT / f"{stem}.jpg"
        if link.exists() or link.is_symlink(): link.unlink()
        link.symlink_to(final.resolve())

        done += 1
        print(f"  [{done:2d}/40] ML{num}  mean={u8.mean():.1f}  shape={u8.shape}")
    except Exception as e:
        print(f"  ERROR {num}: {e}")

total = len(list(OUT_INP.iterdir()))
print(f"\nDone. {done} new pairs. Total training set: {total} images")
