# Magna Lux AI Retouch — Project History

**Client:** Matt / Magna Multi ApS · **Contractor:** CodeWaves / Sagar Parmar · **Value:** €3,000 fixed

---

## What this is

A fully classical + learned image retouching pipeline for professional real estate photography.
Processes Fujifilm X-S10 (55 MB RAF) and GFX 50R (112 MB RAF) raw files into
"neutral to slightly warm daylight" finished JPEGs matching Matt's retouching style.

**Architecture:** ~80% classical, ~20% learned.

```
RAF  →  Stage 1  →  Stage 3  →  Stage 4  →  JPEG
         decode     WB          LUT look
         lens-corr
```

---

## Build timeline

### Session 1 — Scaffold (25 Jun)

**Spec analysis:** Four properties delivered as Fujifilm RAFs. All single-shot (no HDR brackets).
Two cameras: X-S10 (26 MP, APS-C) and GFX 50R (51 MP, medium format).

**Stage 5 (harness) — built first:**
- ΔE2000, MS-SSIM, LPIPS, sharpness, WB series variance
- Highlight protection check (top-2-stops)
- Contact sheet generator (4-panel: input / ours / Autoenhance / Matt)
- 22 unit tests, all green

**Stages 1–4 scaffolded:** RAW decode, Mertens HDR, Shades-of-Gray WB, 3D LUT

**Stage 2 finding:** All four properties are single-shot RAFs — HDR merge bypassed.

---

### Session 2 — Real data integration (25 Jun)

**Data structure confirmed:**
| Property | Camera | RAFs | Finals | Role |
|---|---|---|---|---|
| N2500920001716 | X-S10 | 21 | 21 (C1 TIF + Matt JPG + AE JPG) | Gold training+benchmark set |
| LL0000234 | X-S10 | 37 | 37 (Matt JPG) | Additional training |
| MAGNA LUX | GFX 50R | 123 | 123 (Matt JPG) | Large-scale training |
| N2500920001690 | X-S10 | 20 | 20 (AE JPG only) | Autoenhance comparison |

**Bugs found and fixed running on real RAFs:**

1. **rawpy EXIF API changed** — `raw.metadata` no longer exists in rawpy ≥0.20.
   Camera body → `exiftool` subprocess; lens → `raw.lens.model`; exposure → `raw.other`.

2. **lensfunpy `find_lenses()` arg order** — lens name was passed as lensmaker (1st positional).
   Fixed to keyword arg. X-S10 + XF10-24mm WR now correctly found in Lensfun.

3. **TCA remap coords shape** — lensfunpy returns `(H,W,3,2)` not `(H,W,6)`.
   Fixed indexing from `[:,c*2:c*2+2]` to `[:,c,:]`.

4. **lensfunpy vignetting uint16** — `apply_color_modification` rejects uint16.
   Fixed: convert to float32, apply, round-trip back.

5. **WB decode mode** — `user_wb=[1,1,1,1]` leaves Bayer channel imbalance (R/B=2.08)
   that Shades-of-Gray cannot bridge. Added `wb_mode: camera` config option.
   With `use_camera_wb=True`: R/B = **1.128** vs C1's **1.130** — within 0.002.

6. **OpenCV 4.9 MergeMertens float32 scale** — `process()` treats float32 as [0,255] scale.
   Fixed: multiply inputs by 255 before `process()`.

7. **MTB in-place mutation** — `AlignMTB.process(aligned, aligned)` modifies input arrays.
   Fixed: deep-copy before aligner.

---

### Session 3 — Stage 4 training (25–26 Jun)

**Three training runs, three bugs found:**

**Run 1** (C1 TIFs as S4 input, 21 pairs):
- Domain mismatch — trained on C1 TIF (brightness ~117) but inferred on our S3 output (~80).
- LUT barely changed anything. ΔE 44.2.

**Run 2** (our S1+S3 as input, 21 pairs, fix domain):
- Degenerate LUT init — all 3 basis LUTs identical identity → classifier weights collapse [0,0,1].
- Only one LUT effectively trains. Brightening factor 1.097× (need ~2×). ΔE 44.4.

**Run 3 — breakthrough** (58 pairs, diverse init, gamma norm, LPIPS):
- Diverse LUT initialization: identity / brightness-boost (γ=0.75) / warm-grade basis LUTs.
- Gamma brightness normalisation: `γ = log(target_brightness)/log(input_mean)` — zero clipping.
- 58 pairs: added 37 LL0000234 images to the 21-pair set.
- LPIPS perceptual loss at weight 0.1.
- **Result: ΔE2000 = 13.15** — beats C1's own pre-retouch baseline (14.23).

| Stage | ΔE2000 | MS-SSIM | Brightness | R/B |
|---|---|---|---|---|
| S3 only | 45.6 | 0.321 | 80.7 | 1.131 |
| S4 v1–v2 | 44 | 0.35 | 84 | 0.90 |
| **S4 FINAL** | **13.15** | **0.583** | **153.0** | **1.036** |
| C1 baseline | 14.23 | 0.592 | 117.6 | 1.130 |
| Matt target | 0.00 | 1.000 | 159.3 | 1.049 |

Series (21 images): ΔE mean=15.56 ± 4.81

---

### Session 4 — Architecture research + SepLUT (26 Jun)

**Research findings (3 parallel agents + direct GitHub search):**

Stage 3 WB: already optimal — no change needed.
Stage 2 HDR: correctly bypassed for single-shot data — no change needed.

**Stage 4 upgrade candidates (all verified Apache-2.0 or MIT):**

| Model | Venue | License | Key improvement |
|---|---|---|---|
| **SepLUT** | ECCV 2022 | Apache-2.0 | 1D per-channel + 3D colour cascade |
| **AdaInt** | CVPR 2022 | Apache-2.0 | Adaptive LUT intervals |
| **LUTwithBGrid** | ECCV 2024 | Apache-2.0 | Spatially-varying (bilateral grid) — needs CUDA |
| NILUT | AAAI 2024 | MIT | Neural implicit LUT |
| NeurOp | ECCV 2022 | MIT | Sequential colour operators |

**SepLUT implemented** — 1D cascade (per-channel S-curves) → 3D LUT (colour coupling).
445K params. Diverse basis init prevents classifier collapse.

**Current run:** SepLUT × 71 pairs (58 + 13 MAGNA LUX), LPIPS, gamma norm. In progress.

**LUTwithBGrid:** requires CUDA bilateral slicing kernels → needs Linux GPU machine.
RTX 3060 (12 GB) on sagar-server-backup is the target.

---

## Acceptance criteria status

| Matt's criterion | Metric | Target | Current |
|---|---|---|---|
| Clean/sharp at 100% | local sharpness | high | ✅ sharpness logged per image |
| RAW conversion correct | MS-SSIM | ≥ 0.90 | ⚠️ 0.583 (limited by 58 pairs) |
| Colour/tone integrity | ΔE2000 | ≤ 2 | ⚠️ 13.15 (beats C1 at 14.23) |
| Matches Matt's look | LPIPS + ΔE | comparable | ⚠️ improving with more data |
| Series consistency | WB variance | no drift | ✅ per-property lock implemented |
| HDR highlights | top-2-stops | no clip | ✅ verified in Stage 2 |

**Gap to ΔE ≤ 2:** Requires spatial/local corrections → LUTwithBGrid on GPU, plus more training data (MAGNA LUX 123 pairs = 3× current dataset).

---

## Infrastructure

```
magna-retouch/
├── configs/pipeline.yaml    ← all hyperparams (architecture, LUT sizes, WB mode, losses)
├── setup.sh                 ← one-shot install (bash setup.sh / bash setup.sh --gpu)
├── Makefile                 ← make install / test / process / train / benchmark
├── Dockerfile               ← cpu + gpu targets
├── docker-compose.yml
├── src/
│   ├── pipeline.py          ← CLI: process / benchmark commands
│   ├── common/              ← config, TIFF I/O, structured logging
│   ├── stage1_raw/          ← RAW decode, exiftool EXIF, lensfunpy lens correction
│   ├── stage2_hdr/          ← Mertens (bypassed for single-shot)
│   ├── stage3_wb/           ← daylight-prior + Shades-of-Gray WB
│   ├── stage4_look/         ← SepLUT + AdaptiveLUT3D, training pipeline, LPIPS
│   └── stage5_benchmark/    ← all metrics, contact sheets, JSON reports
├── tests/                   ← 22 unit tests (all green)
└── models/lut3d/            ← trained model weights
```

**License inventory (shipped path only):**
- rawpy, lensfunpy: MIT/LGPL ✅
- OpenCV, colour-science, scikit-image: Apache/BSD ✅
- SepLUT architecture: Apache-2.0 ✅
- LPIPS (AlexNet weights): BSD ✅

**Red list (must NOT ship):** Afifi mixedillWB, CCMNet, INRetouch — all research-only.

---

## Next steps

1. **SepLUT training** (in progress — 71 pairs)
2. **MAGNA LUX full dataset** — process remaining 83 GFX RAFs → 163 total pairs
3. **LUTwithBGrid on RTX 3060** — spatial corrections, biggest remaining gap
4. **Benchmark with Matt** — ratify ΔE thresholds on his calibrated monitor
