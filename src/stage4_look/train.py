"""
Train the Image-Adaptive LUT / SepLUT / LUTwithBGrid on paired examples.

Key optimisations vs naive version:
  - In-memory dataset: all images loaded into RAM at startup → zero disk IO per batch
  - Selectable GPU via --gpu 0|1 → run two processes simultaneously, one per card
  - Gradient clipping (max_norm=1.0) for training stability
  - Every-epoch JSON logging (not just every 25) → visible progress
  - Flush=True on all prints → no buffering surprises

Usage (dual-GPU, two terminals):
    CUDA_VISIBLE_DEVICES=0 python -u -m src.stage4_look.train --arch lutwithbgrid --epochs 800 --out models/lut3d/bgrid_gpu0.pth
    CUDA_VISIBLE_DEVICES=1 python -u -m src.stage4_look.train --arch seplut    --epochs 800 --out models/lut3d/seplut_gpu1.pth
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import structlog

from src.common.config import load_config
from src.common.logging import setup_logging
from .lut3d import build_model
from .lut_bilateral import monotonicity_loss, tv_loss

log = structlog.get_logger(__name__)


# ── In-memory dataset (zero disk IO after init) ───────────────────────────────

class CachedPairDataset(Dataset):
    """
    Preloads all images into RAM on first construction.
    After that, __getitem__ only does random crop + augmentation in memory.

    With 98 pairs × ~4 MB JPEG → ~400 MB RAM total.
    Eliminates NVMe reads during training → removes the #1 GPU starvation cause.
    """

    def __init__(self, input_dir: Path, target_dir: Path,
                 crop_size: int = 640, augment: bool = True,
                 brightness_norm: bool = True):
        self.crop_size = crop_size
        self.augment = augment

        # Discover pairs
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
            raise ValueError(f"No matched pairs in {input_dir} / {target_dir}")

        # Preload all images into RAM
        print(f"  Preloading {len(pairs)} image pairs into RAM …", flush=True)
        self.data: list[tuple[np.ndarray, np.ndarray]] = []
        tgt_means = []
        for inp_p, tgt_p in pairs:
            inp_img = _load_bgr(inp_p)
            tgt_img = _load_bgr(tgt_p)
            tgt_means.append(tgt_img.mean() / 255.0)
            # Resize target to input shape once (cheaper than doing it each __getitem__)
            if inp_img.shape[:2] != tgt_img.shape[:2]:
                tgt_img = cv2.resize(tgt_img, (inp_img.shape[1], inp_img.shape[0]),
                                     cv2.INTER_AREA)
            self.data.append((inp_img, tgt_img))

        self.target_brightness = float(np.median(tgt_means)) if brightness_norm else None
        print(f"  Done. target_brightness={self.target_brightness:.3f}", flush=True)
        log.info("dataset.cached", n_pairs=len(self.data), crop=crop_size,
                 target_brightness=self.target_brightness)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp_bgr, tgt_bgr = self.data[idx]

        # BGR → RGB float32 [0,1]
        inp = inp_bgr[:, :, ::-1].astype(np.float32) / 255.0
        tgt = tgt_bgr[:, :, ::-1].astype(np.float32) / 255.0

        # Gamma brightness normalisation (no highlight clipping)
        if self.target_brightness is not None:
            m = inp.mean()
            if 1e-4 < m < 0.999:
                g = float(np.clip(np.log(self.target_brightness) / np.log(m), 0.3, 3.0))
                inp = np.power(np.clip(inp, 1e-8, 1.0), g)

        # Random same-location crop
        h, w = inp.shape[:2]
        c = self.crop_size
        if h > c and w > c:
            y = random.randint(0, h - c)
            x = random.randint(0, w - c)
            inp = inp[y:y+c, x:x+c]
            tgt = tgt[y:y+c, x:x+c]
        else:
            inp = cv2.resize(inp, (c, c), cv2.INTER_AREA)
            tgt = cv2.resize(tgt, (c, c), cv2.INTER_AREA)

        # Augmentation: horizontal flip + vertical flip (25% each)
        if self.augment:
            if random.random() > 0.5:
                inp = inp[:, ::-1].copy(); tgt = tgt[:, ::-1].copy()
            if random.random() > 0.75:
                inp = inp[::-1].copy(); tgt = tgt[::-1].copy()

        return _to_tensor(inp), _to_tensor(tgt)


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1).copy())


# ── Loss ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """MSE + LPIPS. LPIPS net is moved to training device on first call."""
    def __init__(self, lpips_weight: float = 0.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lpips_weight = lpips_weight
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
        return loss


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

    # Allow arch override without mutating cfg permanently
    cfg_run = dict(cfg)
    cfg_run["stage4_look"] = dict(s4, architecture=arch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    log.info("train.start", device=device, gpu=gpu_name, epochs=n_epochs,
             lr=lr, arch=arch, batch=batch_size, crop=crop_size)
    print(f"\n{'='*60}", flush=True)
    print(f"  Architecture : {arch}", flush=True)
    print(f"  Device       : {gpu_name}", flush=True)
    print(f"  Epochs       : {n_epochs}   Batch: {batch_size}   Crop: {crop_size}", flush=True)
    print(f"  LR           : {lr}   LPIPS: {tr.get('lpips_weight',0)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Data — all images in RAM, no disk IO per batch
    brightness_norm = tr.get("brightness_norm", True)
    ds = CachedPairDataset(Path("data/train/our_input"), Path("data/train/our_target"),
                           crop_size=crop_size, augment=True,
                           brightness_norm=brightness_norm)
    # num_workers=0: data is already in RAM, workers add IPC overhead
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=(device == "cuda"))

    # Model
    model = build_model(cfg_run).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {total_params:,}", flush=True)

    out_path = Path(out_override or s4["lut_model_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    if resume and out_path.exists():
        model.load_state_dict(torch.load(out_path, map_location=device))
        log.info("train.resumed", path=str(out_path))
        print(f"  Resumed from : {out_path}", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    loss_fn = CombinedLoss(lpips_weight=tr.get("lpips_weight", 0.0))

    lam_mono = tr.get("lambda_monotonicity", 0.0)
    lam_tv   = tr.get("lambda_tv", 0.0)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for inp, tgt in dl:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp)
            loss = loss_fn(pred, tgt)

            # LUTwithBGrid regularisation
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

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), out_path)

        # Log every epoch (data is cheap, visibility is important)
        log.info("train.epoch", epoch=epoch, total=n_epochs,
                 loss=round(epoch_loss, 6), best=round(best_loss, 6),
                 lr=round(scheduler.get_last_lr()[0], 7))
        if epoch % 10 == 0 or epoch <= 5:
            print(f"  [{epoch:4d}/{n_epochs}]  loss={epoch_loss:.6f}  best={best_loss:.6f}"
                  f"  lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

    log.info("train.done", best_loss=round(best_loss, 6), saved=str(out_path))
    print(f"\nDone. Best loss: {best_loss:.6f}  →  {out_path}", flush=True)
    return best_loss


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LUT look-matching model")
    parser.add_argument("--config",    default="configs/pipeline.yaml")
    parser.add_argument("--epochs",   type=int, default=None, help="Override n_epochs")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--arch",     default=None,
                        help="Architecture override: lut3d | seplut | lutwithbgrid")
    parser.add_argument("--out",      default=None,
                        help="Override output model path (e.g. models/lut3d/bgrid.pth)")
    parser.add_argument("--batch",    type=int, default=None, help="Override batch_size")
    parser.add_argument("--crop",     type=int, default=None, help="Override crop size")
    parser.add_argument("--lr",       type=float, default=None, help="Override learning rate")
    parser.add_argument("--lpips",    type=float, default=None, help="Override LPIPS weight")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["level"], cfg["logging"]["log_dir"], "train")

    # Apply CLI overrides to config
    if args.batch:  cfg["stage4_look"]["train"]["batch_size"]  = args.batch
    if args.crop:   cfg["stage4_look"]["train"]["train_size"]  = args.crop
    if args.lr:     cfg["stage4_look"]["train"]["lr"]          = args.lr
    if args.lpips:  cfg["stage4_look"]["train"]["lpips_weight"]= args.lpips

    train(cfg, epochs=args.epochs, resume=not args.no_resume,
          arch_override=args.arch, out_override=args.out)
