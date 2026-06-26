"""
LUTwithBGrid — spatially-varying colour correction via bilateral grid + 3D LUT.

Adapted from: WontaeaeKim/LUTwithBGrid (ECCV 2024, Apache-2.0).
This implementation is pure PyTorch — no custom CUDA extensions needed.
Uses F.grid_sample for differentiable bilateral slicing on CPU and GPU.

Architecture:
  img → CNN backbone → features
             ├─► n_lut_basis blending weights  → blended 3D LUT   (global grade)
             └─► n_grid_basis blending weights → blended bilateral grid (spatial)

  Bilateral slice: for each pixel at (x, y) with luminance l,
    sample the 5D grid at coord (x/W, y/H, l) → per-pixel local features.

  Final: concat(local_features, img) → 1×1 conv → clamp → 3D LUT → output.

Key advantage over plain 3D LUT: local corrections — each spatial zone / luminance
zone gets its own colour transform. Directly addresses Matt's per-zone editing.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

from .lut3d import LUTClassifier, TrilinearInterp, _diverse_3d_luts

log = structlog.get_logger(__name__)


# ── Bilateral grid initialisation ─────────────────────────────────────────────

def _identity_grid(n: int, d: int, n_out: int) -> torch.Tensor:
    """
    n basis bilateral grids, each (d, d, d, n_out).
    Identity: no local correction (passthrough when weights are 0).
    Small random perturbations for n > 1 to break symmetry.
    """
    grids = []
    for i in range(n):
        g = torch.zeros(d, d, d, n_out)
        if i == 0:
            pass  # zero → no local offset
        else:
            g = g + torch.randn_like(g) * 0.01
        grids.append(g)
    return torch.stack(grids)    # (n, d, d, d, n_out)


# ── Bilateral slicing via F.grid_sample ───────────────────────────────────────

def bilateral_slice(grid: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """
    Slice the bilateral grid at each pixel's (x_pos, y_pos, luminance) location.

    grid : (B, C, D, D, D) — 5D bilateral grid: [B, channels, luma, y, x]
    img  : (B, 3, H, W)    — pixel values in [0, 1] used as guide

    Returns (B, C, H, W) — per-pixel sliced features.

    F.grid_sample with 5D input expects:
      input: (B, C, D_in, H_in, W_in)
      grid:  (B, D_out, H_out, W_out, 3)  — (x, y, z) coords in [-1, 1]
    Here D_out=H_out=1 and W_out=H*W for a flat output, then reshape.
    """
    B, C, D, _, _ = grid.shape
    _, _, H, W = img.shape

    # Luminance as intensity guide
    luma = img.mean(dim=1, keepdim=True)          # (B, 1, H, W)

    # Pixel grid coordinates in [-1, 1]
    # xs: varies along W dimension, ys: varies along H dimension
    xs = torch.linspace(-1, 1, W, device=img.device)
    ys = torch.linspace(-1, 1, H, device=img.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W) each

    # Luminance coordinate: pixel value mapped to [-1, 1]
    grid_z = luma.squeeze(1) * 2.0 - 1.0          # (B, H, W)

    # Stack into (B, H, W, 3) — (x, y, z) = (x_pos, y_pos, luma)
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
    sample_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)   # (B, H, W, 3)
    # Add D_out=1 dim: (B, 1, H, W, 3)
    sample_coords = sample_coords.unsqueeze(1)

    # Sample: (B, C, 1, H, W)
    sliced = F.grid_sample(grid, sample_coords, mode="bilinear",
                           padding_mode="border", align_corners=True)
    return sliced.squeeze(2)    # (B, C, H, W)


# ── Main model ────────────────────────────────────────────────────────────────

class LUTwithBGridModel(nn.Module):
    """
    Basis 3D LUTs + Basis bilateral grids, blended per-image by a shared CNN.

    n_lut_basis    : number of basis 3D LUTs (global colour grade)
    n_grid_basis   : number of basis bilateral grids (local spatial correction)
    lut_size       : 3D LUT grid resolution (33 = 33³ nodes)
    grid_size      : bilateral grid resolution (17 recommended)
    n_grid_channels: channels per grid node (9 = 3×3 local affine or just 3 additive)
    """
    def __init__(self, n_lut_basis: int = 3, n_grid_basis: int = 4,
                 lut_size: int = 33, grid_size: int = 17,
                 n_grid_channels: int = 3):
        super().__init__()
        self.n_lut_basis = n_lut_basis
        self.n_grid_basis = n_grid_basis
        self.n_grid_channels = n_grid_channels

        # Basis 3D LUTs (global grade) — same diverse init as SepLUT
        self.luts_3d = nn.Parameter(_diverse_3d_luts(n_lut_basis, lut_size))

        # Basis bilateral grids (local correction) — zero init (no-op start)
        self.grids = nn.Parameter(
            _identity_grid(n_grid_basis, grid_size, n_grid_channels))

        # Shared CNN backbone — produces per-image features for both classifiers
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Separate heads for LUT vs grid weights
        self.lut_head  = nn.Sequential(nn.Linear(64, n_lut_basis),  nn.Softmax(dim=1))
        self.grid_head = nn.Sequential(nn.Linear(64, n_grid_basis),  nn.Softmax(dim=1))

        # 1×1 conv: cat([img(3), local_grid(C)]) → 3
        # Input layout: channels 0-2 = image, channels 3+ = grid features
        self.mix = nn.Conv2d(3 + n_grid_channels, 3, 1)
        with torch.no_grad():
            self.mix.weight.zero_()
            # Identity on image channels (0-2) → passthrough at init
            self.mix.weight[:, :3] = torch.eye(3).unsqueeze(-1).unsqueeze(-1)
            self.mix.bias.zero_()

        self.interp = TrilinearInterp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        features = self.backbone(x)                          # (B, 64)

        # ── Global 3D LUT (image-adaptive colour grade) ───────────────────────
        lut_w = self.lut_head(features)                      # (B, n_lut_basis)
        lut = (lut_w[:, :, None, None, None, None] *
               self.luts_3d.unsqueeze(0)).sum(1)             # (B, D, D, D, 3)

        # ── Spatially-varying bilateral grid ─────────────────────────────────
        grid_w = self.grid_head(features)                    # (B, n_grid_basis)
        # (n_grid_basis, D, D, D, C) → (B, C, D, D, D) for bilateral_slice
        g = self.grids.permute(0, 4, 1, 2, 3)               # (n, C, D, D, D)
        blended_grid = (grid_w[:, :, None, None, None, None] *
                        g.unsqueeze(0)).sum(1)               # (B, C, D, D, D)

        # Slice bilateral grid at each pixel's (x, y, luma) position
        local_features = bilateral_slice(blended_grid, x)   # (B, C, H, W)

        # Image first (channels 0-2), grid features after — matches weight init
        x_mixed = torch.clamp(self.mix(torch.cat([x, local_features], dim=1)), 0, 1)

        # Apply 3D LUT image-adaptively
        out = []
        for i in range(B):
            out.append(self.interp(lut[i:i+1], x_mixed[i:i+1]))
        return torch.cat(out, dim=0)


# ── Losses specific to bilateral grid training ────────────────────────────────

def monotonicity_loss(luts_3d: torch.Tensor) -> torch.Tensor:
    """
    Penalise non-monotone LUT entries along each axis.
    A monotone LUT cannot produce hue inversions.
    """
    # Differences along each of the 3 LUT axes
    d_r = luts_3d[:, 1:, :, :, :] - luts_3d[:, :-1, :, :, :]
    d_g = luts_3d[:, :, 1:, :, :] - luts_3d[:, :, :-1, :, :]
    d_b = luts_3d[:, :, :, 1:, :] - luts_3d[:, :, :, :-1, :]
    return (F.relu(-d_r).mean() + F.relu(-d_g).mean() + F.relu(-d_b).mean())


def tv_loss(luts_3d: torch.Tensor) -> torch.Tensor:
    """Total-variation smoothness on the 3D LUT."""
    d_r = (luts_3d[:, 1:] - luts_3d[:, :-1]).pow(2).mean()
    d_g = (luts_3d[:, :, 1:] - luts_3d[:, :, :-1]).pow(2).mean()
    d_b = (luts_3d[:, :, :, 1:] - luts_3d[:, :, :, :-1]).pow(2).mean()
    return d_r + d_g + d_b


# ── Load / apply ──────────────────────────────────────────────────────────────

def build_bilateral_model(cfg: dict) -> LUTwithBGridModel:
    s4 = cfg["stage4_look"]
    return LUTwithBGridModel(
        n_lut_basis=s4.get("n_lut_basis", 3),
        n_grid_basis=s4.get("n_grid_basis", 4),
        lut_size=s4.get("lut_size", 33),
        grid_size=s4.get("grid_size", 17),
        n_grid_channels=s4.get("n_grid_channels", 3),
    )


def apply_bilateral(img_uint8: np.ndarray, model: LUTwithBGridModel,
                    device: str = "cpu") -> np.ndarray:
    f32 = img_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(f32.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return (np.clip(out_np, 0, 1) * 255 + 0.5).astype(np.uint8)
