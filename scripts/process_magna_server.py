"""
Process MAGNA LUX RAFs on the server using multiprocessing.
Run on server after RAFs are transferred:
    python -u scripts/process_magna_server.py --workers 6
"""
import argparse
import multiprocessing as mp
import re
import sys
from pathlib import Path

import cv2
import numpy as np

def process_one(args):
    num, raf_path, final_path, out_inp, out_tgt = args
    try:
        import rawpy
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.common.config import load_config
        from src.common.io import uint16_to_float
        from src.stage1_raw.decode import _extract_exif
        from src.stage1_raw.lens_correction import correct_lens
        from src.stage3_wb.stage import process_single as s3

        cfg = load_config()
        lc = cfg["stage1_raw"]["lens_correction"]

        with rawpy.imread(str(raf_path)) as raw:
            img = raw.postprocess(
                no_auto_bright=True, gamma=(1,1), output_bps=16,
                use_camera_wb=True, half_size=True,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD)
            exif = _extract_exif(raw, raf_path)

        img = correct_lens(img, exif,
                           interpolation=lc["interpolation"],
                           loose_search_fallback=lc["loose_search_fallback"])
        wb16, _ = s3(img, cfg, exif=exif)

        f32 = uint16_to_float(wb16)
        p2, p998 = np.percentile(f32, 0.2), np.percentile(f32, 99.8)
        disp = np.power(np.clip((f32 - p2) / (p998 - p2 + 1e-8), 0, 1), 1/2.2)
        u8 = (disp * 255).astype(np.uint8)

        out_path = Path(out_inp) / f"ML{num}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 97])

        link = Path(out_tgt) / f"ML{num}.jpg"
        if link.exists() or link.is_symlink(): link.unlink()
        link.symlink_to(Path(final_path).resolve())

        return (num, True, round(float(u8.mean()), 1))
    except Exception as e:
        return (num, False, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir",  default="/home/sagar/Desktop/magna-retouch/data/raw_magna_lux")
    parser.add_argument("--final-dir", default="/Users/codesageml/Library/Mobile Documents/com~apple~CloudDocs/MAGNALUX_Lux-Retouch_AI-samples/MAGNA LUX/B MAGNA LUX FINAL ")
    parser.add_argument("--out-inp",  default="data/train/our_input")
    parser.add_argument("--out-tgt",  default="data/train/our_target")
    parser.add_argument("--workers",  type=int, default=6)
    args = parser.parse_args()

    raw_dir   = Path(args.raw_dir)
    final_dir = Path(args.final_dir)
    out_inp   = args.out_inp
    out_tgt   = args.out_tgt

    Path(out_inp).mkdir(parents=True, exist_ok=True)
    Path(out_tgt).mkdir(parents=True, exist_ok=True)

    already = {f.stem for f in Path(out_inp).glob("ML*.jpg")}

    tasks = []
    for raf in sorted(raw_dir.glob("*.RAF")):
        m = re.search(r"_(\d{4})\.RAF", raf.name)
        if not m: continue
        num = m.group(1)
        if f"ML{num}" in already:
            continue
        final = final_dir / f"FINAL_MAGNALUX_Lux-retouch_AI-samples_{num}.jpg"
        if not final.exists():
            continue
        tasks.append((num, str(raf), str(final), out_inp, out_tgt))

    if not tasks:
        print("Nothing to process — all done already.")
        return

    print(f"Processing {len(tasks)} RAFs with {args.workers} workers …", flush=True)
    mp.set_start_method("spawn", force=True)
    with mp.Pool(args.workers) as pool:
        for i, (num, ok, info) in enumerate(pool.imap_unordered(process_one, tasks), 1):
            if ok:
                print(f"  [{i:2d}/{len(tasks)}] ML{num}  mean={info}", flush=True)
            else:
                print(f"  ERR ML{num}: {info}", flush=True)

    total = len(list(Path(out_inp).glob("ML*.jpg")))
    print(f"\nDone. Total ML images: {total}")
    total_all = len(list(Path(out_inp).iterdir()))
    print(f"Total training set:    {total_all}")


if __name__ == "__main__":
    main()
