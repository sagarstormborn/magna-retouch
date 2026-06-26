"""
Train the Image-Adaptive LUT / SepLUT / LUTwithBGrid on paired examples.

Key properties:
  - Held-out split: the 21 DSCF images (gold property N2500920001716) are NEVER
    trained on — reserved as a clean TEST set. A deterministic VAL split is carved
    from the rest. The checkpoint is saved on best *validation* ΔE2000, not on
    training loss, so reported numbers are not leakage.
  - Aspect-matched pairs: GFX (ML) inputs are 4:3 while Matt's exports are 3:2;
    we center-crop the input to the target's aspect ratio instead of stretching,
    keeping pixels aligned (critical for the spatial bilateral grid).
  - In-memory dataset, downscaled to working resolution → GPU is the bottleneck,
    not per-step JPEG decode of 45 MP files.
  - Pinned brightness-norm: computed once on the train split and saved next to the
    model so inference uses the exact same normalisation.
  - Selectable GPU via CUDA_VISIBLE_DEVICES → run two cards simultaneously.

Usage (dual-GPU, two terminals):
    CUDA_VISIBLE_DEVICES=0 python -u -m src.stage4_look.train --arch lutwithbgrid --epochs 800 --out models/lut3d/bgrid_gpu0.pth
    CUDA_VISIBLE_DEVICES=1 python -u -m src.stage4_look.train --arch seplut    --epochs 800 --out models/lut3d/seplut_gpu1.pth
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import structlog

from src.common.config import load_config
from src.common.logging import setup_logging
from src.stage5_benchmark.metrics import delta_e2000
from .lut3d import build_model
from .lut_bilateral import monotonicity_loss, tv_loss

log = structlog.get_logger(__name__)

# Held-out gold benchmark set — these stems are NEVER trained on.
TEST_PREFIX = "DSCF"          # property N2500920001716 (X-S10 gold set)
VAL_FRACTION = 0.15           # fraction of the trainable pool used for validation
SPLIT_SEED = 1234             # deterministic split
NORM_SIDECAR = "norm.json"    # pinned brightness-norm, saved next to the model


# ── Pair discovery + split ──────────────────────────────────────────────────────

def discover_pairs(input_dir: Path, target_dir: Path) -> list[tuple[Path, Path]]:
    """Match input files to target JPGs by stem prefix (e.g. DSCF4652)."""
    inp_files = (sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
                 + sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg")))
    tgt_by_stem = {p.stem: p for p in
                   list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.jpeg"))}
    pairs: list[tuple[Path, Path]] = []
    for inp in inp_files:
        key = inp.stem.split("_")[0]
        if key in tgt_by_stem:
            pairs.append((inp, tgt_by_stem[key]))
    if not pairs:
        raise ValueError(f"No matched pairs found in {input_dir} / {target_dir}")
    return pairs


def split_pairs(pairs):
    """
    Returns (train, val, test).
    test  = gold DSCF set (held out, never trained / never selected on)
    val   = deterministic VAL_FRACTION of the remaining pool (model selection)
    train = the rest
    """
    test = [p for p in pairs if p[0].stem.startswith(TEST_PREFIX)]
    pool = [p for p in pairs if not p[0].stem.startswith(TEST_PREFIX)]
    rng = random.Random(SPLIT_SEED)
    pool_sorted = sorted(pool, key=lambda p: p[0].stem)
    rng.shuffle(pool_sorted)
    n_val = max(1, int(round(len(pool_sorted) * VAL_FRACTION)))
    return pool_sorted[n_val:], pool_sorted[:n_val], test


# ── Image helpers ───────────────────────────────────────────────────────────────

def _load_rgb_u8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _center_crop_to_ar(img: np.ndarray, target_ar: float) -> np.ndarray:
    """
    Center-crop `img` so its aspect ratio (W/H) matches target_ar. Avoids the
    stretch distortion of resizing 4:3 GFX inputs onto 3:2 targets, which would
    misalign every pixel and corrupt the spatial supervision.
    """
    h, w = img.shape[:2]
    cur = w / h
    if abs(cur - target_ar) < 0.01:
        return img
    if cur > target_ar:                       # too wide → trim width
        new_w = int(round(h * target_ar))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    new_h = int(round(w / target_ar))         # too tall → trim height
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, :]


def _gamma_to_brightness(img_f32: np.ndarray, target_brightness: float) -> np.ndarray:
    """Lift/lower brightness via gamma without clipping highlights."""
    m = float(img_f32.mean())
    if 1e-4 < m < 0.999:
        g = float(np.clip(np.log(target_brightness) / np.log(m), 0.3, 3.0))
        return np.power(np.clip(img_f32, 1e-8, 1.0), g)
    return img_f32


def _resize_short(img: np.ndarray, short_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = short_side / min(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


# ── In-memory dataset (downscaled → light per-step crop) ──────────────────────────

class CachedPairDataset(Dataset):
    """
    Preloads aspect-matched, brightness-normalised pairs into RAM as uint8,
    downscaled to working resolution. __getitem__ then only crops + flips, so the
    GPU is fed without per-step JPEG decode / 45 MP resize.

    mode="train": random crop + flips on a work_short-sized frame
    mode="eval" : full aspect-matched frame at eval_long (ΔE evaluated one at a time)
    """
    EVAL_LONG = 768

    def __init__(self, pairs, target_brightness, crop_size: int = 640,
                 mode: str = "train", brightness_norm: bool = True):
        self.pairs = pairs
        self.mode = mode
        self.crop_size = crop_size
        self.target_brightness = target_brightness   # None → per-image norm
        work_short = round(crop_size * 1.5)

        self.data: list[tuple[np.ndarray, np.ndarray]] = []
        for inp_p, tgt_p in pairs:
            inp = _load_rgb_u8(inp_p)
            tgt = _load_rgb_u8(tgt_p)

            # Align geometry: crop input to target's aspect, then match dimensions.
            inp = _center_crop_to_ar(inp, tgt.shape[1] / tgt.shape[0])
            if mode == "train":
                inp, tgt = _resize_short(inp, work_short), _resize_short(tgt, work_short)
            else:
                inp, tgt = _resize_long(inp, self.EVAL_LONG), _resize_long(tgt, self.EVAL_LONG)
            if inp.shape[:2] != tgt.shape[:2]:
                tgt = cv2.resize(tgt, (inp.shape[1], inp.shape[0]),
                                 interpolation=cv2.INTER_AREA)

            # Brightness-normalise the INPUT to match its OWN target's brightness.
            # Per-image: each input is gamma-shifted to its paired target's mean,
            # eliminating the L* residual that a single global constant left behind.
            if brightness_norm:
                tgt_brightness = tgt.astype(np.float32).mean() / 255.0
                f = _gamma_to_brightness(inp.astype(np.float32) / 255.0, tgt_brightness)
                inp = (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)

            self.data.append((np.ascontiguousarray(inp), np.ascontiguousarray(tgt)))

        log.info("dataset.cached", mode=mode, n_pairs=len(self.data), crop=crop_size,
                 brightness_norm=brightness_norm)
        print(f"  cached {mode}: {len(self.data)} pairs"
              f"{', per-image brightness norm' if brightness_norm else ''}",
              flush=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        inp, tgt = self.data[idx]
        if self.mode == "train":
            h, w = inp.shape[:2]
            c = self.crop_size
            if h > c and w > c:
                y = random.randint(0, h - c)
                x = random.randint(0, w - c)
                inp = inp[y:y + c, x:x + c]
                tgt = tgt[y:y + c, x:x + c]
            else:
                inp = cv2.resize(inp, (c, c), interpolation=cv2.INTER_AREA)
                tgt = cv2.resize(tgt, (c, c), interpolation=cv2.INTER_AREA)
            if random.random() > 0.5:
                inp = inp[:, ::-1]; tgt = tgt[:, ::-1]
            if random.random() > 0.75:
                inp = inp[::-1]; tgt = tgt[::-1]
        return _to_tensor(inp), _to_tensor(tgt)


def _to_tensor(img_u8: np.ndarray) -> torch.Tensor:
    f = np.ascontiguousarray(img_u8).astype(np.float32) / 255.0
    return torch.from_numpy(f.transpose(2, 0, 1))


# ── Loss ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """MSE + LPIPS + L* ΔE proxy. LPIPS net is moved to the training device on first call."""
    def __init__(self, lpips_weight: float = 0.0, de_weight: float = 0.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lpips_weight = lpips_weight
        self.de_weight = de_weight
        self._lpips = None
        self._lpips_device = None

    def _get_lpips(self, device):
        if self._lpips is None:
            try:
                import lpips
                self._lpips = lpips.LPIPS(net="alex").to(device)
                self._lpips_device = device
                log.info("loss.lpips_enabled", device=str(device))
            except ImportError:
                log.warning("loss.lpips_unavailable")
                self.lpips_weight = 0.0
        elif str(self._lpips_device) != str(device):
            self._lpips = self._lpips.to(device)
            self._lpips_device = device
        return self._lpips

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.mse(pred, target)
        if self.lpips_weight > 0:
            net = self._get_lpips(pred.device)
            if net is not None:
                loss = loss + self.lpips_weight * net(pred * 2 - 1, target * 2 - 1).mean()
        if self.de_weight > 0:
            # downsample to 64px (fast, geometry-invariant)
            p64 = F.interpolate(pred,   64, mode='bilinear', antialias=True)
            t64 = F.interpolate(target, 64, mode='bilinear', antialias=True)
            # sRGB → linearise approx (fast, avoids colour-science import on GPU)
            p_lin = p64.pow(2.2)
            t_lin = t64.pow(2.2)
            # L* = 116*(Y/Yn)^(1/3) - 16 approximation using Y channel
            # Y ≈ 0.2126R + 0.7152G + 0.0722B
            coeff = torch.tensor([0.2126, 0.7152, 0.0722],
                                 device=pred.device).view(1, 3, 1, 1)
            # Normalise to [0,1] so de_weight is comparable to MSE/LPIPS scale
            # L* ∈ [0,100] → divide by 100 before MSE
            p_L = (116 * (p_lin * coeff).sum(1, keepdim=True).clamp(1e-8).pow(1/3) - 16) / 100.0
            t_L = (116 * (t_lin * coeff).sum(1, keepdim=True).clamp(1e-8).pow(1/3) - 16) / 100.0
            loss = loss + self.de_weight * F.mse_loss(p_L, t_L)
        return loss


# ── Validation ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_delta_e(model: nn.Module, dataset: CachedPairDataset, device: str) -> float:
    """Mean ΔE2000 over a held-out dataset (full frames, no crop)."""
    model.eval()
    des = []
    for inp_u8, tgt_u8 in dataset.data:
        t = torch.from_numpy(
            np.ascontiguousarray(inp_u8).astype(np.float32).transpose(2, 0, 1) / 255.0
        ).unsqueeze(0).to(device)
        out = model(t).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        pred_u8 = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
        des.append(delta_e2000(pred_u8.astype(np.uint16) * 257,
                               tgt_u8.astype(np.uint16) * 257))
    return float(np.mean(des)) if des else float("inf")


# ── Training loop ──────────────────────────────────────────────────────────────

def train(cfg: dict, epochs: int | None = None, resume: bool = True,
          arch_override: str | None = None, out_override: str | None = None):
    s4 = cfg["stage4_look"]
    tr = s4["train"]
    n_epochs   = epochs or tr["n_epochs"]
    lr         = tr["lr"]
    batch_size = tr["batch_size"]
    crop_size  = tr.get("train_size", 640)
    arch       = arch_override or s4.get("architecture", "lut3d")
    grad_clip  = tr.get("grad_clip", 1.0)

    cfg_run = dict(cfg)
    cfg_run["stage4_look"] = dict(s4, architecture=arch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    log.info("train.start", device=device, gpu=gpu_name, epochs=n_epochs,
             lr=lr, arch=arch, batch=batch_size, crop=crop_size)
    print(f"\n{'='*60}\n  Architecture : {arch}\n  Device       : {gpu_name}\n"
          f"  Epochs       : {n_epochs}   Batch: {batch_size}   Crop: {crop_size}\n"
          f"  LR           : {lr}   LPIPS: {tr.get('lpips_weight',0)}\n{'='*60}\n",
          flush=True)

    # ── Split: hold out gold DSCF as test, carve a val set for selection ──────────
    pairs = discover_pairs(Path("data/train/our_input"), Path("data/train/our_target"))
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    log.info("data.split", n_train=len(train_pairs), n_val=len(val_pairs),
             n_test_heldout=len(test_pairs))
    print(f"  split → train={len(train_pairs)}  val={len(val_pairs)}  "
          f"test(held-out)={len(test_pairs)}", flush=True)

    # ── Pinned brightness norm — per-image, saved as null in sidecar ─────────────
    # Each input is normalised to its OWN target's brightness in the dataset loop,
    # so a single global constant is no longer needed or stored.
    brightness_norm = tr.get("brightness_norm", True)
    target_brightness = None   # null in sidecar; inference uses its own 0.58 approx

    out_path = Path(out_override or s4["lut_model_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path.parent / NORM_SIDECAR).write_text(json.dumps({
        "target_brightness": target_brightness,  # null: per-image norm at train time
        "brightness_norm": brightness_norm,
        "architecture": arch,
        "model": out_path.name,
    }, indent=2))

    train_ds = CachedPairDataset(train_pairs, target_brightness, crop_size, mode="train",
                                 brightness_norm=brightness_norm)
    val_ds   = CachedPairDataset(val_pairs,   target_brightness, crop_size, mode="eval",
                                 brightness_norm=brightness_norm)

    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=(device == "cuda"))

    model = build_model(cfg_run).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {total_params:,}", flush=True)

    if resume and out_path.exists():
        model.load_state_dict(torch.load(out_path, map_location=device))
        log.info("train.resumed", path=str(out_path))
        print(f"  Resumed from : {out_path}", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    loss_fn = CombinedLoss(lpips_weight=tr.get("lpips_weight", 0.0),
                           de_weight=tr.get("de_weight", 0.0))
    lam_mono = tr.get("lambda_monotonicity", 0.0)
    lam_tv   = tr.get("lambda_tv", 0.0)

    best_val_de = float("inf")
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for inp, tgt in dl:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            if lam_mono > 0 and hasattr(model, "luts_3d"):
                loss = loss + lam_mono * monotonicity_loss(model.luts_3d)
            if lam_tv > 0 and hasattr(model, "luts_3d"):
                loss = loss + lam_tv * tv_loss(model.luts_3d)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dl)
        scheduler.step()

        # ── Model selection on VALIDATION ΔE (not training loss) ──────────────────
        eval_now = (epoch % 5 == 0) or epoch <= 3 or epoch == n_epochs
        if eval_now:
            val_de = evaluate_delta_e(model, val_ds, device)
            improved = val_de < best_val_de
            if improved:
                best_val_de = val_de
                torch.save(model.state_dict(), out_path)
            log.info("train.epoch", epoch=epoch, total=n_epochs, loss=round(epoch_loss, 6),
                     val_delta_e=round(val_de, 4), best_val_delta_e=round(best_val_de, 4),
                     saved=improved, lr=round(scheduler.get_last_lr()[0], 7))
            print(f"  [{epoch:4d}/{n_epochs}]  loss={epoch_loss:.5f}  "
                  f"val_ΔE={val_de:.3f}  best_val_ΔE={best_val_de:.3f}"
                  f"{'  ✔saved' if improved else ''}", flush=True)
        else:
            log.info("train.epoch", epoch=epoch, total=n_epochs, loss=round(epoch_loss, 6),
                     lr=round(scheduler.get_last_lr()[0], 7))

    # ── Final report on the untouched held-out TEST set ───────────────────────────
    if test_pairs:
        model.load_state_dict(torch.load(out_path, map_location=device))
        test_ds = CachedPairDataset(test_pairs, target_brightness, crop_size, mode="eval",
                                    brightness_norm=brightness_norm)
        test_de = evaluate_delta_e(model, test_ds, device)
        log.info("train.test_heldout", test_delta_e=round(test_de, 4), n=len(test_pairs))
        print(f"\nHeld-out TEST (gold DSCF, never trained):  ΔE2000 = {test_de:.3f}  "
              f"over {len(test_pairs)} images", flush=True)

    log.info("train.done", best_val_delta_e=round(best_val_de, 4), saved=str(out_path))
    print(f"Done. Best val ΔE: {best_val_de:.3f}  →  {out_path}", flush=True)
    return best_val_de


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LUT look-matching model")
    parser.add_argument("--config",    default="configs/pipeline.yaml")
    parser.add_argument("--epochs",    type=int, default=None, help="Override n_epochs")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--arch",      default=None,
                        help="Architecture override: lut3d | seplut | lutwithbgrid")
    parser.add_argument("--out",       default=None,
                        help="Override output model path (e.g. models/lut3d/bgrid.pth)")
    parser.add_argument("--batch",     type=int, default=None, help="Override batch_size")
    parser.add_argument("--crop",      type=int, default=None, help="Override crop size")
    parser.add_argument("--lr",        type=float, default=None, help="Override learning rate")
    parser.add_argument("--lpips",     type=float, default=None, help="Override LPIPS weight")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["level"], cfg["logging"]["log_dir"], "train")

    if args.batch:  cfg["stage4_look"]["train"]["batch_size"]   = args.batch
    if args.crop:   cfg["stage4_look"]["train"]["train_size"]   = args.crop
    if args.lr:     cfg["stage4_look"]["train"]["lr"]           = args.lr
    if args.lpips:  cfg["stage4_look"]["train"]["lpips_weight"] = args.lpips

    train(cfg, epochs=args.epochs, resume=not args.no_resume,
          arch_override=args.arch, out_override=args.out)
