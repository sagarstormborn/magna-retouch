"""
Image-Adaptive 3D LUT — inference wrapper.
Training uses HuiZeng/Image-Adaptive-3DLUT (Apache-2.0).

The model learns basis LUTs + a small CNN that blends them per-image.
We run inference only here; training is a separate CLI entry-point.

If model weights are not found, raises FileNotFoundError with an install hint.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import structlog

log = structlog.get_logger(__name__)


class TrilinearInterpolation(nn.Module):
    """Pure-PyTorch trilinear LUT interpolation (fallback when CUDA op unavailable)."""

    def forward(self, lut: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        # lut: (batch, 3, D, D, D)   img: (batch, 3, H, W) in [0,1]
        b, c, h, w = img.shape
        d = lut.shape[2]

        # Normalise img to grid_sample coordinates [-1, 1]
        coords = img.permute(0, 2, 3, 1).unsqueeze(1)  # (B,1,H,W,3)
        coords = coords * 2.0 - 1.0

        out_channels = []
        for i in range(3):
            ch = nn.functional.grid_sample(
                lut[:, i:i+1, :, :, :],
                coords,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            out_channels.append(ch.squeeze(1))
        return torch.stack(out_channels, dim=1)


class AdaptiveLUT3DModel(nn.Module):
    """
    Minimal re-implementation of the inference path of Image-Adaptive-3DLUT.
    Compatible with weights trained from the original repo.
    """

    def __init__(self, n_luts: int = 3, lut_size: int = 33):
        super().__init__()
        self.lut_size = lut_size

        # Basis LUTs (learnable)
        d = lut_size
        self.luts = nn.Parameter(torch.zeros(n_luts, 3, d, d, d))
        nn.init.normal_(self.luts, std=0.01)

        # Tiny CNN: produces per-image weights over the n_luts basis LUTs
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((256, 256)),
            nn.Flatten(),
            nn.Linear(3 * 256 * 256, 64),
            nn.ReLU(),
            nn.Linear(64, n_luts),
            nn.Softmax(dim=1),
        )

        self.interp = TrilinearInterpolation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.classifier(x)                        # (B, n_luts)
        lut = (weights[:, :, None, None, None, None] * self.luts.unsqueeze(0)).sum(dim=1)
        return self.interp(lut, x)


def load_model(cfg: dict, device: str = "cpu") -> AdaptiveLUT3DModel:
    model_path = Path(cfg["stage4_look"]["lut_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"LUT model not found at {model_path}. "
            "Train first: python -m src.stage4_look.train --config configs/pipeline.yaml"
        )
    model = AdaptiveLUT3DModel(lut_size=cfg["stage4_look"]["lut_size"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    log.info("stage4.model_loaded", path=str(model_path), device=device)
    return model


def apply_lut(img_uint16: np.ndarray, model: AdaptiveLUT3DModel, device: str = "cpu") -> np.ndarray:
    """Run LUT inference on a uint16 HxWx3 image. Returns uint16."""
    f32 = img_uint16.astype(np.float32) / 65535.0
    t = torch.from_numpy(f32.transpose(2, 0, 1)).unsqueeze(0).to(device)  # (1,3,H,W)
    with torch.no_grad():
        out = model(t)
    out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # HxWx3
    return (np.clip(out_np, 0, 1) * 65535 + 0.5).astype(np.uint16)
