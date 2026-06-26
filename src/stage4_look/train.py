"""
Train the Image-Adaptive 3D LUT on Matt's RAW→retouched pairs.

Input:  data/train/input/   — C1 TIFs (uint8, Matt's RAW conversion pre-retouch)
Target: data/train/target/  — Matt's retouched JPGs (uint8, matched by stem)

Usage:
    python -m src.stage4_look.train
    python -m src.stage4_look.train --config configs/pipeline.yaml --epochs 400
"""
from __future__ import annotations

import argparse
import random
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

log = structlog.get_logger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """
    Matches input TIFs to target JPGs by stem prefix (DSCF####).
    Both are uint8; loaded as RGB float32 in [0, 1].
    Random crops + flips for augmentation (21 pairs → no overfitting room without it).
    """
    def __init__(self, input_dir: Path, target_dir: Path,
                 crop_size: int = 480, augment: bool = True,
                 brightness_norm: bool = True):
        self.augment = augment
        self.crop_size = crop_size
        self.brightness_norm = brightness_norm

        inp_files = (sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
                     + sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg")))
        tgt_by_stem = {p.stem: p for p in
                       list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.jpeg"))}

        self.pairs: list[tuple[Path, Path]] = []
        for inp in inp_files:
            # input stem: "DSCF4652_MATTS..." or "DSCF4652" → key = DSCF4652
            key = inp.stem.split("_")[0]
            if key in tgt_by_stem:
                self.pairs.append((inp, tgt_by_stem[key]))

        if not self.pairs:
            raise ValueError(f"No matched pairs found in {input_dir} / {target_dir}")

        # Compute median target brightness for normalisation
        # Decouples per-image exposure from colour grade so LUT only learns colour
        if brightness_norm:
            tgt_means = []
            for _, tgt_p in self.pairs:
                b = cv2.imread(str(tgt_p))
                if b is not None:
                    tgt_means.append(b.mean() / 255.0)
            self.target_brightness = float(np.median(tgt_means)) if tgt_means else 0.625
        else:
            self.target_brightness = None

        log.info("dataset.loaded", n_pairs=len(self.pairs), crop=crop_size,
                 target_brightness=round(self.target_brightness or 0, 3) if brightness_norm else "off")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp_path, tgt_path = self.pairs[idx]

        inp = _load_rgb_f32(inp_path)
        tgt = _load_rgb_f32(tgt_path)

        # Brightness normalisation via gamma: maps current mean → target mean
        # without clipping highlights. gamma = log(target)/log(current) maps
        # the entire [0,1] range so dark images lift without blowing highlights.
        if self.target_brightness is not None:
            inp_mean = inp.mean()
            if inp_mean > 1e-4 and inp_mean < 0.999:
                gamma = np.log(self.target_brightness) / np.log(inp_mean)
                gamma = float(np.clip(gamma, 0.3, 3.0))   # safety bounds
                inp = np.power(np.clip(inp, 1e-8, 1.0), gamma)

        # Resize target to match input dimensions (C1 TIF > Matt JPG)
        if inp.shape[:2] != tgt.shape[:2]:
            tgt = cv2.resize(tgt, (inp.shape[1], inp.shape[0]),
                             interpolation=cv2.INTER_AREA)

        # Random crop
        h, w = inp.shape[:2]
        c = self.crop_size
        if h > c and w > c:
            y = random.randint(0, h - c)
            x = random.randint(0, w - c)
            inp = inp[y:y+c, x:x+c]
            tgt = tgt[y:y+c, x:x+c]
        else:
            inp = cv2.resize(inp, (c, c), interpolation=cv2.INTER_AREA)
            tgt = cv2.resize(tgt, (c, c), interpolation=cv2.INTER_AREA)

        # Augmentation
        if self.augment and random.random() > 0.5:
            inp = inp[:, ::-1].copy()
            tgt = tgt[:, ::-1].copy()

        return _to_tensor(inp), _to_tensor(tgt)


def _load_rgb_f32(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1))


# ── Loss ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """MSE + optional LPIPS perceptual loss (lpips import is deferred)."""
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
                log.warning("loss.lpips_not_installed_falling_back_to_mse")
                self.lpips_weight = 0.0
        elif str(self._lpips_device) != str(device):
            # Move to new device if needed (e.g. GPU training after CPU init)
            self._lpips = self._lpips.to(device)
            self._lpips_device = device
        return self._lpips

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.mse(pred, target)
        if self.lpips_weight > 0:
            net = self._get_lpips(pred.device)
            if net is not None:
                # lpips expects [-1, 1]
                p = pred  * 2 - 1
                t = target * 2 - 1
                loss = loss + self.lpips_weight * net(p, t).mean()
        return loss


# ── Training loop ──────────────────────────────────────────────────────────────

def train(cfg: dict, epochs: int | None = None, resume: bool = True):
    s4 = cfg["stage4_look"]
    tr = s4["train"]
    n_epochs   = epochs or tr["n_epochs"]
    lr         = tr["lr"]
    batch_size = tr["batch_size"]
    crop_size  = tr.get("train_size", 480)
    arch       = s4.get("architecture", "lut3d")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("train.start", device=device, epochs=n_epochs, lr=lr, arch=arch)

    # Data
    brightness_norm = tr.get("brightness_norm", True)
    ds = PairDataset(Path("data/train/our_input"), Path("data/train/our_target"),
                     crop_size=crop_size, augment=True, brightness_norm=brightness_norm)
    import torch.cuda
    n_workers = 4 if torch.cuda.is_available() else 0   # parallel IO on GPU, sync on CPU
    pin = torch.cuda.is_available()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=n_workers, pin_memory=pin, persistent_workers=n_workers > 0)

    # Model — selected by config
    model = build_model(cfg).to(device)
    out_path = Path(s4["lut_model_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_loss = float("inf")
    if resume and out_path.exists():
        model.load_state_dict(torch.load(out_path, map_location=device))
        log.info("train.resumed", path=str(out_path))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Cosine annealing: gradually reduce lr to 0 over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    loss_fn = CombinedLoss(lpips_weight=tr.get("lpips_weight", 0.0))

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for inp, tgt in dl:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dl)
        scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), out_path)

        if epoch % 25 == 0 or epoch <= 5:
            log.info("train.epoch", epoch=epoch, loss=round(epoch_loss, 6),
                     lr=round(scheduler.get_last_lr()[0], 7), best=round(best_loss, 6))
            print(f"  [{epoch:4d}/{n_epochs}]  loss={epoch_loss:.6f}  best={best_loss:.6f}"
                  f"  lr={scheduler.get_last_lr()[0]:.2e}")

    log.info("train.done", best_loss=round(best_loss, 6), saved=str(out_path))
    print(f"\nTraining complete. Best loss: {best_loss:.6f}  →  {out_path}")
    return best_loss


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["level"], cfg["logging"]["log_dir"], "train")
    train(cfg, epochs=args.epochs, resume=not args.no_resume)
