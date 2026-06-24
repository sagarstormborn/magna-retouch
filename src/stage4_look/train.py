"""
Train the Image-Adaptive 3D LUT on Matt's RAW→retouched pairs.

Usage:
    python -m src.stage4_look.train --config configs/pipeline.yaml \
        --train_dir data/train --val_dir data/val

data/train layout expected:
    data/train/input/   ← Stage 1-3 processed images (TIFF 16-bit)
    data/train/target/  ← Matt's retouched output (TIFF 16-bit, matched filenames)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import structlog

from src.common.config import load_config
from src.common.io import load_tiff_16, uint16_to_float
from .lut3d import AdaptiveLUT3DModel

log = structlog.get_logger(__name__)


class PairDataset(Dataset):
    def __init__(self, input_dir: Path, target_dir: Path, size: int = 480):
        self.inputs = sorted(input_dir.glob("*.tiff")) + sorted(input_dir.glob("*.tif"))
        self.targets = sorted(target_dir.glob("*.tiff")) + sorted(target_dir.glob("*.tif"))
        assert len(self.inputs) == len(self.targets), "Input/target count mismatch"
        self.size = size

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = _resize(uint16_to_float(load_tiff_16(self.inputs[idx])), self.size)
        tgt = _resize(uint16_to_float(load_tiff_16(self.targets[idx])), self.size)
        return (
            torch.from_numpy(inp.transpose(2, 0, 1)),
            torch.from_numpy(tgt.transpose(2, 0, 1)),
        )


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    scale = size / min(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def train(cfg: dict, train_dir: Path, val_dir: Path):
    s4 = cfg["stage4_look"]["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("stage4.train_start", device=device, epochs=s4["n_epochs"])

    model = AdaptiveLUT3DModel(lut_size=cfg["stage4_look"]["lut_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=s4["lr"])
    loss_fn = nn.MSELoss()

    train_ds = PairDataset(train_dir / "input", train_dir / "target", s4["train_size"])
    train_dl = DataLoader(train_ds, batch_size=s4["batch_size"], shuffle=True, num_workers=2)

    out_path = Path(cfg["stage4_look"]["lut_model_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, s4["n_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for inp, tgt in train_dl:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_dl)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), out_path)

        if epoch % 50 == 0:
            log.info("stage4.train_epoch", epoch=epoch, loss=round(epoch_loss, 6))

    log.info("stage4.train_done", best_loss=round(best_loss, 6), saved=str(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--train_dir", default="data/train")
    parser.add_argument("--val_dir", default="data/val")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, Path(args.train_dir), Path(args.val_dir))
