# Magna Lux AI Retouch — Full Project Overview

**Client:** Matt / Magna Multi ApS  
**Contractor:** CodeWaves / Sagar Parmar  
**Value:** €3,000 fixed price  
**Repo:** https://github.com/sagarstormborn/magna-retouch  
**Server:** sagar@49.206.252.63:2222 — Ubuntu 24.04, RTX 3060 (12 GB) + GTX 1660 SUPER (6 GB)

---

## 1. What We Are Trying To Do

Build an AI-powered retouching pipeline that takes raw real estate photos from a **Fujifilm camera** and produces finished JPEGs that **match Matt's professional retouching style** — specifically his "neutral to slightly warm daylight" look.

Matt retouches images manually in **Capture One** and **Photoshop**. The goal is to automate that process so each property's photos (20–40 rooms) can be processed overnight instead of manually.

### Matt's 5 Acceptance Criteria

| # | Criterion | Metric | Target |
|---|---|---|---|
| 1 | Clean and sharp at 100% | Laplacian variance | High |
| 2 | RAW conversion correct — noise, sharpening, lens profiles | MS-SSIM | ≥ 0.90 |
| 3 | Colour/tone integrity vs original RAW | **ΔE2000** | **≤ 2.0** |
| 4 | Comparable to Matt's results point-by-point | LPIPS + ΔE | Similar |
| 5 | Consistent across a property's image series — no WB drift | WB variance | No drift |
| + | HDR: protect highlights in top 2 stops | Blown pixel % | < 0.1% |

### Current Status vs Target

| Stage | ΔE2000 | MS-SSIM | Brightness | R/B ratio |
|---|---|---|---|---|
| RAW only (no LUT) | 45.6 | 0.32 | 80 | 1.13 |
| **SepLUT Run 7 (honest, held-out TEST)** | **12.73** | — | — | — |
| LUTwithBGrid Run 7 (honest, held-out TEST) | 14.69 | — | — | — |
| Matt's own C1 pre-retouch | 14.23 | 0.59 | 117 | 1.13 |
| **Matt's target** | **0.00** | **1.00** | 159 | 1.05 |

**SepLUT beats C1 (12.73 vs 14.23) on a genuinely held-out 21-image test set.**
Next run on 181 pairs (83 more MAGNA LUX in progress) targets ΔE ~9-11.

---

## 2. The Data

**4 properties, all Fujifilm RAFs, single-shot (no HDR brackets):**

| Property | Camera | Files | Role |
|---|---|---|---|
| N2500920001716 | X-S10 (26MP, APS-C) | 21 RAFs + C1 TIFs + Matt JPGs + AE JPGs | **Gold set** — training + benchmark |
| LL0000234 | X-S10 | 37 RAFs + 37 Matt JPGs | Additional training data |
| MAGNA LUX | GFX 50R (51MP, medium format) | 123 RAFs + 123 Matt JPGs | Large-scale training |
| N2500920001690 | X-S10 | 20 RAFs + 20 Autoenhance JPGs | Autoenhance comparison only |

**Training pairs assembled (98 total):**
- 21 from N2500920001716 (gold set)
- 37 from LL0000234
- 40 from MAGNA LUX (first 40 processed)

**Location on server:** `/home/sagar/Desktop/magna-retouch/data/train/`

---

## 3. The Pipeline — All Stages

```
Fujifilm RAF
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — RAW Decode + Lens Correction                        │
│  • rawpy with use_camera_wb=True (single-shot mode)            │
│  • lensfunpy: vignetting → TCA → geometry (in that order)      │
│  • Camera body via exiftool subprocess (rawpy gives lens name) │
│  • Fujifilm X-S10 + XF10-24mm WR: confirmed in Lensfun DB     │
│  • GFX 50R: body in DB, GF20-35mm lens NOT in DB (passthrough)│
└─────────────────┬───────────────────────────────────────────────┘
                  │  uint16 RGB, linear, 6246×4170 (X-S10)
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2 — HDR Merge (BYPASSED for this dataset)               │
│  All 4 properties are single-shot — no brackets exist.         │
│  Code: Mertens fusion + MTB alignment, ready for future use.   │
│  Key bug fixed: OpenCV 4.9 MergeMertens needs float32×255      │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3 — White Balance                                        │
│  Two-stage correction:                                          │
│  1. Daylight WB prior from EXIF (raw.daylight_whitebalance)    │
│     → corrects Bayer sensor spectral imbalance                 │
│  2. Shades-of-Gray (p=6) scene fine-tuning                     │
│  Result: R/B = 1.128 vs C1's 1.130 — within 0.002             │
│  Series lock: per-property median illuminant (criterion 5)     │
└─────────────────┬───────────────────────────────────────────────┘
                  │  uint16 WB-corrected
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4 — Look Matching (the AI part)                         │
│  Learns Matt's aesthetic from paired examples.                 │
│                                                                 │
│  Currently active: LUTwithBGrid (ECCV 2024, Apache-2.0)        │
│  Also available:  SepLUT (ECCV 2022), AdaptiveLUT3D            │
│  Select via: configs/pipeline.yaml → architecture              │
└─────────────────┬───────────────────────────────────────────────┘
                  │  uint8 RGB, display-ready
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5 — Benchmark Harness                                   │
│  Computes: ΔE2000, MS-SSIM, LPIPS, sharpness, WB variance     │
│  Generates 4-panel contact sheets for calibrated-monitor QA   │
│  22 unit tests, all passing                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Stage 4 Deep Dive — The AI Look-Matching Models

This is where the most research and iteration happened.

### Architecture 1: AdaptiveLUT3D (baseline)
- **Paper:** HuiZeng/Image-Adaptive-3DLUT (Apache-2.0)
- **How:** n basis 3D LUTs + CNN classifier predicts blending weights
- **Params:** 384K
- **Limitation:** global colour only — no spatial awareness

### Architecture 2: SepLUT (ECCV 2022, Apache-2.0) ← current best trained
- **Paper:** ImCharlesY/SepLUT
- **How:** 1D per-channel curves (brightness/contrast) cascade into 3D LUT (colour coupling)
- **Params:** 445K
- **Key improvement:** separable decomposition makes it more expressive than pure 3D LUT
- **Config:** `architecture: seplut`

### Architecture 3: LUTwithBGrid (ECCV 2024, Apache-2.0) ← training now
- **Paper:** WontaeaeKim/LUTwithBGrid
- **Our implementation:** Pure PyTorch (no custom CUDA extensions)
- **How:** CNN backbone → basis 3D LUTs (global grade) + basis bilateral grids (spatial)
- **Bilateral slice:** For each pixel at (x, y) with luminance l, query 5D grid at (x/W, y/H, l)
  → per-pixel local colour transform
- **Params:** 443K
- **Key improvement:** SPATIAL corrections — different colour treatment for windows, shadows, highlights
- **This is the answer to Matt's per-zone editing**
- **Config:** `architecture: lutwithbgrid`

### Research candidates (not implemented, verified licenses):
| Model | Venue | License | Status |
|---|---|---|---|
| AdaInt | CVPR 2022 | Apache-2.0 | Available, needs MMCV |
| NILUT | AAAI 2024 | MIT | Available, Hald-image paradigm |
| NeurOp | ECCV 2022 | MIT | Available, 28K params sequential operators |

### What WAS NOT used (license issues):
- **Afifi mixedillWB** — research-only, not for commercial use
- **CCMNet** — CC BY-NC
- **INRetouch** — CC BY-NC-SA
- **Zero-DCE** — CC BY-NC (not suitable for Stage 2 anyway)

---

## 5. Training History — All Runs

### Run 1: C1 TIFs as input → Matt JPGs (21 pairs)
- **Problem:** Domain mismatch — C1 TIFs are already tone-mapped/colour-graded by Capture One; our pipeline output looks different
- **Result:** ΔE = 44.2 (LUT couldn't learn useful correction)

### Run 2: Our S1+S3 output → Matt JPGs (21 pairs)
- **Problem:** Degenerate LUT initialization — all 3 basis LUTs initialised as identical identity → classifier collapses, learns one LUT only
- **Brightening factor:** 1.1× instead of needed 2×
- **Result:** ΔE = 44.4 (still broken)

### Run 3: BREAKTHROUGH (58 pairs, diverse init, gamma norm, LPIPS)
**Three fixes applied simultaneously:**
1. **Diverse LUT init** — basis LUTs: identity / brightness-boost (γ=0.75) / warm-grade
2. **Gamma brightness norm** — `γ = log(target_mean) / log(input_mean)` maps [0,1]→[0,1] without clipping highlights (vs linear scale which clipped 18%)
3. **LPIPS loss** (weight=0.1) — perceptual metric drives colour accuracy, not just pixel MSE
4. **More data** — 58 pairs (21 + 37 LL0000234)
- **Result (train-set):** ΔE = 13.15, MS-SSIM = 0.583 — but this was **train-set leakage** (DSCF gold images were in the training set). See Run 7 for honest numbers.

### Run 4: SepLUT, 71 pairs, GPU (Mac CPU, in progress)
- SepLUT architecture (1D+3D cascade)
- 71 pairs (58 + 13 MAGNA LUX processed before training started)
- Mac CPU, ~30s/epoch × 400 = ~3.5h

### Run 5: SepLUT, 71 pairs, Mac CPU (completed, current model)
- Series ΔE2000 mean=**15.64** std=4.99 (vs 15.56 for 58-pair run — essentially same)
- Brightness: **162.3** ← much closer to Matt's 159.3 (was 153.0 before)
- R/B ratio: **1.021** ← closer to target 1.049 (was 1.036)
- MS-SSIM: 0.563

### Run 7: Dual-GPU corrected run — ✅ COMPLETE
**Six correctness fixes applied:**
1. DSCF gold 21 images held out as TEST — never trained on, never used for model selection
2. 15% deterministic VAL split from remaining 77 pairs — model saved on best **val ΔE2000**
3. GFX aspect fix: `_center_crop_to_ar()` instead of stretch-resize → pixels aligned
4. In-memory downscaled cache → GPU ≥70% utilisation (was 0%)
5. Brightness norm pinned to `norm.json` sidecar → train ≡ inference
6. `colour-science` installed on server so val ΔE matches benchmark harness

**Final results (800 epochs, 65 train / 12 val / 21 test held-out):**

| Model | Val ΔE2000 | **TEST ΔE2000** |
|---|---|---|
| SepLUT (GTX 1660) | 12.83 | **12.73** ← winner, promoted |
| LUTwithBGrid (RTX 3060) | 15.46 | 14.69 |

**Honest TEST ΔE = 12.73** on 21 DSCF images never seen during training.  
This is the number to show Matt (not 13.15, which was train-set leakage).

**Next: retrain on 181 pairs** (83 more MAGNA LUX being processed now) → expect ΔE to drop to ~9-11.

### Run 6: LUTwithBGrid, 98 pairs, RTX 3060 (superseded by Run 7)
- New spatial architecture with bilateral grid slicing
- 98 pairs (21 + 37 LL + 40 MAGNA LUX)
- RTX 3060 GPU, ~30s/epoch × 400 = ~3.3h
- **Expected to significantly reduce ΔE by capturing local per-zone corrections**

---

## 6. Key Bugs Found & Fixed

All found by running on real data — not theoretical.

| # | Bug | Where | Fix |
|---|---|---|---|
| 1 | rawpy ≥0.20: `raw.metadata` removed | decode.py | Use `raw.lens`, `raw.other`, exiftool for body |
| 2 | lensfunpy `find_lenses()` arg order wrong | lens_correction.py | Use keyword arg |
| 3 | TCA coords shape (H,W,3,2) not (H,W,6) | lens_correction.py | Fix indexing `[:,c,:]` |
| 4 | lensfunpy rejects uint16 for vignetting | lens_correction.py | float32 round-trip |
| 5 | `user_wb=[1,1,1,1]` → R/B = 2.08 (broken WB) | decode.py | Added `wb_mode: camera` |
| 6 | OpenCV 4.9 MergeMertens float32 scale bug | stage2_hdr | Multiply inputs by 255 |
| 7 | MTB aligner modifies input arrays in-place | stage2_hdr | Deep-copy before aligner |
| 8 | lensfunpy==1.13.0 doesn't exist | requirements.txt | Changed to `>=1.14.0` |
| 9 | LPIPS on CPU, model on CUDA → crash | train.py | Move LPIPS net to training device |
| 10 | `import torch.cuda` shadowed global `torch` | train.py | Removed inner import |
| 11 | `mix` conv identity init on wrong channels | lut_bilateral.py | Fixed concat order [img, features] |
| 12 | rsync `--info=progress2` not on macOS rsync | transfer | Removed flag |

---

## 7. Infrastructure

### Local (macOS)
```
/Users/codesageml/Desktop/AiRetouch/
├── configs/pipeline.yaml     ← ALL hyperparams — change architecture, LUT sizes, loss weights here
├── setup.sh                  ← bash setup.sh / bash setup.sh --gpu
├── Makefile                  ← make install|test|process|train|benchmark|build-cpu|build-gpu
├── Dockerfile                ← cpu + gpu targets
├── docker-compose.yml
├── requirements.txt          ← full deps
├── requirements-train.txt    ← GPU server minimal (torch+cv2+lpips only)
├── src/
│   ├── pipeline.py           ← CLI: process / benchmark
│   ├── common/               ← config, TIFF I/O, structured JSON logging
│   ├── stage1_raw/           ← decode.py, lens_correction.py, stage.py
│   ├── stage2_hdr/           ← stage.py (Mertens, bypassed for single-shot)
│   ├── stage3_wb/            ← estimator.py, stage.py
│   ├── stage4_look/
│   │   ├── lut3d.py          ← AdaptiveLUT3D + SepLUT models + build_model factory
│   │   ├── lut_bilateral.py  ← LUTwithBGrid (pure PyTorch bilateral grid)
│   │   ├── stage.py          ← inference with brightness normalisation
│   │   └── train.py          ← training loop, PairDataset, CombinedLoss
│   └── stage5_benchmark/     ← metrics.py, contact_sheet.py, harness.py
├── tests/                    ← 22 unit tests (all green)
├── models/lut3d/
│   └── model_best.pth        ← current best trained weights (SepLUT, 58-pair run)
└── data/
    └── train/
        ├── our_input/        ← 98 × our S1+S3 pipeline output JPEGs
        └── our_target/       ← 98 × Matt's retouched JPEGs
```

### Server (sagar@49.206.252.63:2222)
```
/home/sagar/Desktop/magna-retouch/   ← cloned from GitHub
├── (same structure as local)
├── logs/
│   ├── train_bgrid.log       ← LUTwithBGrid GPU training (RUNNING NOW)
│   └── train_gpu.log         ← previous SepLUT GPU run
└── data/train/               ← 98 pairs transferred from Mac
```

**Connect:** `ssh -i ~/.ssh/id_ed25519 -p 2222 sagar@49.206.252.63`  
**Watch training:** `tmux attach -t train2`  
**Watch GPU:** `watch -n 0.5 nvidia-smi`  
**Training log:** `tail -f logs/train_bgrid.log`

---

## 8. What's Currently Running

| GPU | Architecture | Pairs | Batch | Crop | Epochs | Speed | ETA |
|---|---|---|---|---|---|---|---|
| **RTX 3060** | LUTwithBGrid | 98 | 16 | 800 | 800 | ~6s/ep | ~80 min |
| **GTX 1660 SUPER** | SepLUT | 98 | 8 | 640 | 800 | ~6s/ep | ~80 min |

**GPU utilisation:** RTX 3060 @ 72% (103W) · GTX 1660 @ 21% (42W)

**Key optimisations applied:**
- In-memory dataset: all 98 pairs preloaded into RAM → zero disk IO per batch
- LPIPS resized to 256px before perceptual loss (was 800px → 13× speedup)
- 2 dataloader workers share preloaded numpy arrays (no copy) via fork
- Gradient clipping at max_norm=1.0
- `python -u` (unbuffered) → every epoch logged

When done (monitor with `tmux attach -t gpu0` or `gpu1`):
1. `scp sagar@49.206.252.63:~/Desktop/magna-retouch/models/lut3d/bgrid_best.pth models/lut3d/`
2. `scp sagar@49.206.252.63:~/Desktop/magna-retouch/models/lut3d/seplut_best.pth models/lut3d/`
3. Run full 21-image series inference + metrics comparison
4. Push best model to GitHub

---

## 9. Research Done

### Stage 3 (White Balance) — Researched, no change needed
Our current WB (R/B = 1.128) already matches C1's (R/B = 1.130) within 0.002.

Alternatives researched:
- **FFCC** (google/ffcc, Apache-2.0) — MATLAB-based, not easily integrated
- **C5** (mahmoudnafifi/C5, Apache-2.0, ICCV 2021) — good, but WB is already accurate
- **mixedillWB** — perfect for mixed indoor lighting but **research-only license, cannot ship**

### Stage 4 (Look Matching) — Researched all ECCV/CVPR 2022–2024 models
All models verified with direct GitHub API lookups:

| Model | Venue | License | Decision |
|---|---|---|---|
| **SepLUT** | ECCV 2022 | ✅ Apache-2.0 | ✅ Implemented |
| **AdaInt** | CVPR 2022 | ✅ Apache-2.0 | Available, MMCV dependency |
| **LUTwithBGrid** | ECCV 2024 | ✅ Apache-2.0 | ✅ Implemented (pure PyTorch) |
| **NILUT** | AAAI 2024 | ✅ MIT | Available (Hald image paradigm) |
| **NeurOp** | ECCV 2022 | ✅ MIT | Available (28K params, sequential) |

CVPR 2024/2025: No paired retouching models with permissive licenses found — field moved to diffusion (too heavy).

### Stage 2 (HDR) — Researched, no change for current data
All data is single-shot. If Matt ever shoots brackets:
- **IAT** (Illumination-Adaptive-Transformer, Apache-2.0) — low-light, not blown windows
- **MAXIM** (google-research/maxim, Apache-2.0) — multi-task including enhancement
- True HDR: Mertens fusion (already implemented) or ECC alignment (already implemented)

---

## 10. Next Steps (In Priority Order)

### Immediate (when GPU training finishes ~3h)
1. Pull model from server → `scp` to local `models/lut3d/model_bgrid.pth`
2. Run inference + full 21-image metrics → compare to SepLUT (ΔE ~14-18 (honest val))
3. Commit + push to GitHub

### Near-term
4. Process remaining 83 MAGNA LUX GFX RAFs → 181 total training pairs
5. Retrain LUTwithBGrid with full 181-pair set → expect ΔE < 10
6. Implement LPIPS on GPU server (currently confirmed working)

### Before delivery
7. **Ratify thresholds with Matt on calibrated monitor** — the ΔE ≤ 2 target may be renegotiated once he sees our current ΔE ~14-18 (honest val) output (which already beats C1)
8. Run series consistency benchmark (criterion 5) across a full property set
9. Contact sheet generation for each property — deliverable for calibrated-monitor QA

### Escalation if needed
- **HDRNet** (Apache-2.0) — spatially-varying affine, similar to LUTwithBGrid but TF1 (complex port)
- **More MAGNA LUX data** — easy 83 pairs remaining, biggest data lever
- **LPIPS weight increase** to 0.2 — more perceptual pull

---

## 11. License Inventory (Shipable Path)

| Component | License | Status |
|---|---|---|
| rawpy / LibRaw | MIT / LGPL | ✅ Ship |
| lensfunpy / Lensfun | MIT / LGPL | ✅ Ship |
| OpenCV | Apache-2.0 | ✅ Ship |
| colour-science, scikit-image | BSD/Apache | ✅ Ship |
| SepLUT architecture | Apache-2.0 | ✅ Ship |
| LUTwithBGrid architecture (our impl) | Apache-2.0 | ✅ Ship |
| LPIPS / AlexNet weights | BSD | ✅ Ship |
| **Afifi mixedillWB** | research-only | ❌ Cannot ship |
| **CCMNet** | CC BY-NC | ❌ Cannot ship |
| **INRetouch** | CC BY-NC-SA | ❌ Cannot ship |
| **Zero-DCE** | CC BY-NC | ❌ Cannot ship |

---

*Last updated: 2026-06-26 · Training in progress · 22/22 tests passing*
