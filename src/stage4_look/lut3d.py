"""
Image-Adaptive LUT models — three architectures, all Apache-2.0 / MIT:

  AdaptiveLUT3DModel  — original 3D-LUT with diverse basis init (our current model)
  SepLUTModel         — Separable 1D+3D cascade (ECCV 2022, ImCharlesY/SepLUT)
                        1D per-channel curves handle brightness/contrast;
                        3D handles colour coupling. More expressive, same params.

Both share the same training pipeline and inference path.
Select via configs/pipeline.yaml → stage4_look.architecture: lut3d | seplut
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

log = structlog.get_logger(__name__)


# ── Shared components ─────────────────────────────────────────────────────────

class LUTClassifier(nn.Module):
    """
    Image backbone that predicts per-image LUT blending weights.

    backbone="resnet18" (default): pretrained ResNet-18 truncated at pool layer
        → 512-dim features → Linear → softmax. 11 M params.  ImageNet pretrained
        features understand image content (exposure zones, colour casts, room type)
        far better than a 4-conv net trained from scratch on 70 images.

    backbone="lightweight": original 4-conv, ~35 K params. Kept for CPU/small-GPU.
    """
    def __init__(self, n_out: int, backbone: str = "resnet18"):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            import torchvision.models as M
            r = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
            # Drop the final FC; keep everything through avgpool
            self.feat = nn.Sequential(*list(r.children())[:-1], nn.Flatten())
            feat_dim = 512
        else:
            self.feat = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            feat_dim = 64
        self.head = nn.Sequential(nn.Linear(feat_dim, n_out), nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet expects at least 224×224; downsample large crops for the backbone
        if self.backbone_name == "resnet18" and min(x.shape[-2:]) > 256:
            x_small = F.interpolate(x, size=224, mode="bilinear", antialias=True)
        else:
            x_small = x
        return self.head(self.feat(x_small))


class TrilinearInterp(nn.Module):
    """Apply one 3D LUT (B, D, D, D, 3) to (B, 3, H, W) via grid_sample."""
    def forward(self, lut: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        b, _, h, w = img.shape
        coords = (img.permute(0, 2, 3, 1) * 2.0 - 1.0).unsqueeze(1)  # (B,1,H,W,3)
        lut_5d = lut.permute(0, 4, 1, 2, 3).expand(b, -1, -1, -1, -1)
        out = F.grid_sample(lut_5d, coords, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.squeeze(2)                                            # (B,3,H,W)


def _apply_1d_lut(img: torch.Tensor, lut_1d: torch.Tensor) -> torch.Tensor:
    """
    Apply per-channel 1D LUTs via linear interpolation (index-based, no grid_sample).
    img    : (B, 3, H, W) in [0, 1]
    lut_1d : (B, 3, D)    — one curve per channel per image
    Returns (B, 3, H, W)
    """
    B, C, H, W = img.shape
    D = lut_1d.shape[-1]

    pixels = img.reshape(B, C, -1)          # (B, 3, H*W)
    idx = pixels * (D - 1)                   # fractional LUT index in [0, D-1]
    idx_lo = idx.long().clamp(0, D - 2)     # floor index
    idx_hi = (idx_lo + 1).clamp(0, D - 1)  # ceil index
    frac = (idx - idx_lo.float())           # interpolation weight

    lo = lut_1d.gather(2, idx_lo)           # (B, 3, H*W)
    hi = lut_1d.gather(2, idx_hi)
    out = lo + frac * (hi - lo)             # linear interpolation
    return out.reshape(B, C, H, W)


# ── Diverse initialisation helpers ────────────────────────────────────────────

def _diverse_3d_luts(n: int, d: int) -> torch.Tensor:
    """Identity + brightness-boost + warm-grade basis LUTs. Prevents classifier collapse."""
    lin = torch.linspace(0, 1, d)
    r, g, b = torch.meshgrid(lin, lin, lin, indexing="ij")
    identity = torch.stack([r, g, b], dim=-1)
    luts = []
    for i in range(n):
        if i == 0:
            luts.append(identity.clone())
        elif i == 1:
            luts.append(identity.clone().pow(0.75))        # brighten (γ < 1)
        elif i == 2:
            w = identity.clone()
            w[..., 0] = (w[..., 0] * 1.05).clamp(0, 1)   # R+5%
            w[..., 2] = (w[..., 2] * 0.95).clamp(0, 1)   # B−5%
            luts.append(w)
        else:
            luts.append(identity.clone() + torch.randn_like(identity) * 0.02)
    return torch.stack(luts)                                # (n, D, D, D, 3)


def _diverse_1d_luts(n: int, d: int) -> torch.Tensor:
    """Identity + s-curve + linear-shift basis 1D LUTs per channel."""
    lin = torch.linspace(0, 1, d)
    identity = lin.unsqueeze(0).expand(3, -1)               # (3, D)
    luts = []
    for i in range(n):
        if i == 0:
            luts.append(identity.clone())
        elif i == 1:
            # Soft S-curve: lifts midtones
            t = lin
            s = 3 * t**2 - 2 * t**3                        # smoothstep
            luts.append(s.unsqueeze(0).expand(3, -1).clone())
        elif i == 2:
            # Linear lift: shifts entire curve up (overall brightening)
            luts.append((identity + 0.05).clamp(0, 1).clone())
        else:
            luts.append((identity + torch.randn_like(identity) * 0.01).clamp(0, 1))
    return torch.stack(luts)                                # (n, 3, D)


# ═══════════════════════════════════════════════════════════════════════════════
# Model 1: AdaptiveLUT3DModel  (current baseline, kept for comparison)
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveLUT3DModel(nn.Module):
    def __init__(self, n_luts: int = 3, lut_size: int = 33, backbone: str = "resnet18"):
        super().__init__()
        self.lut_size = lut_size
        self.n_luts = n_luts
        self.luts = nn.Parameter(_diverse_3d_luts(n_luts, lut_size))
        self.classifier = LUTClassifier(n_luts, backbone=backbone)
        self.interp = TrilinearInterp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.classifier(x)                         # (B, n_luts)
        lut = (weights[:, :, None, None, None, None] *
               self.luts.unsqueeze(0)).sum(dim=1)            # (B, D, D, D, 3)
        out = []
        for i in range(x.shape[0]):
            out.append(self.interp(lut[i:i+1], x[i:i+1]))
        return torch.cat(out, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Model 2: SepLUTModel  (ECCV 2022 — separable 1D→3D cascade)
#
# Key idea: component-independent (1D per-channel) followed by
# component-correlated (3D colour coupling). More expressive than pure 3D.
# ═══════════════════════════════════════════════════════════════════════════════

class SepLUTModel(nn.Module):
    def __init__(self, n_1d: int = 3, n_3d: int = 3,
                 lut_1d_size: int = 64, lut_3d_size: int = 33,
                 backbone: str = "resnet18"):
        super().__init__()
        self.n_1d = n_1d
        self.n_3d = n_3d

        self.luts_1d = nn.Parameter(_diverse_1d_luts(n_1d, lut_1d_size))
        self.luts_3d = nn.Parameter(_diverse_3d_luts(n_3d, lut_3d_size))

        # Shared ResNet-18 backbone — both 1D and 3D classifiers reuse features
        # without duplicating the 11M-param network
        if backbone == "resnet18":
            import torchvision.models as M
            r = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(r.children())[:-1], nn.Flatten())
            feat_dim = 512
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            feat_dim = 64
        self.backbone_name = backbone
        self.head_1d = nn.Sequential(nn.Linear(feat_dim, n_1d), nn.Softmax(dim=1))
        self.head_3d = nn.Sequential(nn.Linear(feat_dim, n_3d), nn.Softmax(dim=1))
        self.interp = TrilinearInterp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Shared backbone (single forward pass — shared features for both heads)
        x_small = F.interpolate(x, 224, mode="bilinear", antialias=True) \
                  if self.backbone_name == "resnet18" and min(x.shape[-2:]) > 256 else x
        feats = self.backbone(x_small)                       # (B, feat_dim)

        # ── Stage 1: 1D LUT (per-channel tone curves) ────────────────────────
        w1d  = self.head_1d(feats)                           # (B, n_1d)
        lut_1d = (w1d[:, :, None, None] *
                  self.luts_1d.unsqueeze(0)).sum(dim=1)      # (B, 3, D_1d)
        x_1d = _apply_1d_lut(x, lut_1d)                     # (B, 3, H, W)

        # ── Stage 2: 3D LUT (colour coupling) ───────────────────────────────
        w3d  = self.head_3d(feats)                           # (B, n_3d)
        lut_3d = (w3d[:, :, None, None, None, None] *
                  self.luts_3d.unsqueeze(0)).sum(dim=1)      # (B, D, D, D, 3)
        out = []
        for i in range(B):
            out.append(self.interp(lut_3d[i:i+1], x_1d[i:i+1]))
        return torch.cat(out, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory, load, apply
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(cfg: dict) -> nn.Module:
    s4 = cfg["stage4_look"]
    arch = s4.get("architecture", "lut3d")
    backbone = s4.get("backbone", "resnet18")
    if arch == "seplut":
        return SepLUTModel(
            n_1d=s4.get("n_1d_luts", 3),
            n_3d=s4.get("n_3d_luts", 3),
            lut_1d_size=s4.get("lut_1d_size", 64),
            lut_3d_size=s4.get("lut_size", 33),
            backbone=backbone,
        )
    if arch == "lutwithbgrid":
        from .lut_bilateral import build_bilateral_model
        return build_bilateral_model(cfg)
    return AdaptiveLUT3DModel(
        n_luts=s4.get("n_luts", 3),
        lut_size=s4.get("lut_size", 33),
        backbone=backbone,
    )


def load_model(cfg: dict, device: str = "cpu") -> nn.Module:
    model_path = Path(cfg["stage4_look"]["lut_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"LUT model not found at {model_path}.\nTrain first: make train"
        )
    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    log.info("stage4.model_loaded", path=str(model_path),
             arch=cfg["stage4_look"].get("architecture", "lut3d"))
    return model


def apply_lut(img_uint8: np.ndarray, model: nn.Module, device: str = "cpu") -> np.ndarray:
    """Run LUT inference on a uint8 HxWx3 RGB image. Returns uint8."""
    f32 = img_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(f32.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return (np.clip(out_np, 0, 1) * 255 + 0.5).astype(np.uint8)
