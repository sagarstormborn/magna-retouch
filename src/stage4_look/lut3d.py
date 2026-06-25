"""
Image-Adaptive 3D LUT — model definition and inference.

Architecture (after HuiZeng/Image-Adaptive-3DLUT, Apache-2.0):
  - n_luts basis LUTs (learnable 3-D colour lookup tables)
  - Lightweight CNN classifier: predicts per-image blending weights
  - Output = weighted sum of basis LUTs applied via trilinear interpolation

Classifier uses GlobalAveragePool so it runs at any input resolution.
Training uses random 480p crops; inference applies the LUT at full resolution.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

log = structlog.get_logger(__name__)


# ── Trilinear LUT interpolation (pure PyTorch) ────────────────────────────────

class TrilinearInterp(nn.Module):
    """
    Apply a 3D LUT to an image via trilinear interpolation using grid_sample.
    lut  : (1, D, D, D, 3) — one LUT in grid_sample "flow" layout
    img  : (B, 3, H, W)    — pixel values in [0, 1]
    """
    def forward(self, lut: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        # grid_sample expects coords in [-1, 1]
        # img pixels (R, G, B) → (x, y, z) grid coords
        coords = img.permute(0, 2, 3, 1) * 2.0 - 1.0   # (B, H, W, 3)
        coords = coords.unsqueeze(1)                      # (B, 1, H, W, 3)
        # lut: (1, D, D, D, 3) → (1, 3, D, D, D) for grid_sample
        lut_5d = lut.permute(0, 4, 1, 2, 3)
        lut_5d = lut_5d.expand(b, -1, -1, -1, -1)
        out = F.grid_sample(lut_5d, coords,
                            mode="bilinear", padding_mode="border",
                            align_corners=True)              # (B, 3, 1, H, W)
        return out.squeeze(2)                                # (B, 3, H, W)


# ── Lightweight classifier CNN ────────────────────────────────────────────────

class LUTClassifier(nn.Module):
    """
    Predicts per-image blending weights over n_luts basis LUTs.
    Works at any input resolution via GlobalAveragePool.
    ~35 K parameters.
    """
    def __init__(self, n_luts: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_luts),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full model ────────────────────────────────────────────────────────────────

class AdaptiveLUT3DModel(nn.Module):
    def __init__(self, n_luts: int = 3, lut_size: int = 33):
        super().__init__()
        d = lut_size
        self.lut_size = lut_size
        self.n_luts = n_luts

        # Basis LUTs — identity-initialised so untrained model = passthrough
        self.luts = nn.Parameter(self._identity_lut(n_luts, d))
        self.classifier = LUTClassifier(n_luts)
        self.interp = TrilinearInterp()

    @staticmethod
    def _identity_lut(n: int, d: int) -> torch.Tensor:
        """
        Initialise basis LUTs with diversity so the classifier has a meaningful
        gradient from step 1. All-identical identity init causes classifier collapse
        (weights like [0,0,1]) and the model degenerates to a single LUT.

        Basis LUT roles:
          0 — neutral identity (passthrough)
          1 — brightness boost (~+0.3 stop, for underexposed rooms)
          2 — warm grade (slight R+, B−, to approximate Matt's neutral-warm target)
          3+ — small random perturbations for remaining slots
        """
        lin = torch.linspace(0, 1, d)
        r, g, b = torch.meshgrid(lin, lin, lin, indexing="ij")
        identity = torch.stack([r, g, b], dim=-1)      # (D, D, D, 3)

        luts = []
        for i in range(n):
            if i == 0:
                lut = identity.clone()
            elif i == 1:
                # Brightness boost: gamma < 1 brightens (x^0.75)
                lut = identity.clone().pow(0.75)
            elif i == 2:
                # Warm colour grade: lift R, suppress B slightly
                lut = identity.clone()
                lut[..., 0] = (identity[..., 0] * 1.05).clamp(0, 1)  # R +5%
                lut[..., 2] = (identity[..., 2] * 0.95).clamp(0, 1)  # B −5%
            else:
                # Random perturbation for additional slots
                lut = identity.clone() + torch.randn_like(identity) * 0.02
            luts.append(lut)

        return torch.stack(luts, dim=0)   # (n, D, D, D, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.classifier(x)                     # (B, n_luts)
        # Blend basis LUTs: (B, 1, 1, 1, 1) × (n, D, D, D, 3) summed over n
        lut = (weights[:, :, None, None, None, None] *
               self.luts.unsqueeze(0)).sum(dim=1)        # (B, D, D, D, 3)
        lut = lut.unsqueeze(0) if lut.dim() == 4 else lut
        # Apply per-image
        out = []
        for i in range(x.shape[0]):
            out.append(self.interp(lut[i:i+1], x[i:i+1]))
        return torch.cat(out, dim=0)


# ── Load / apply ──────────────────────────────────────────────────────────────

def load_model(cfg: dict, device: str = "cpu") -> AdaptiveLUT3DModel:
    model_path = Path(cfg["stage4_look"]["lut_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"LUT model not found at {model_path}.\n"
            "Train first:  make train"
        )
    model = AdaptiveLUT3DModel(
        n_luts=cfg["stage4_look"].get("n_luts", 3),
        lut_size=cfg["stage4_look"]["lut_size"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    log.info("stage4.model_loaded", path=str(model_path))
    return model


def apply_lut(img_uint8: np.ndarray, model: AdaptiveLUT3DModel, device: str = "cpu") -> np.ndarray:
    """Run LUT inference on a uint8 HxWx3 RGB image. Returns uint8."""
    f32 = img_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(f32.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return (np.clip(out_np, 0, 1) * 255 + 0.5).astype(np.uint8)
