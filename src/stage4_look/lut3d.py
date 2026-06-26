"""
Image-Adaptive LUT models — three architectures, all Apache-2.0 / MIT:

  AdaptiveLUT3DModel  — original 3D-LUT with diverse basis init
  SepLUTModel         — Separable 1D+3D cascade (ECCV 2022) + optional metadata conditioning
  LUTwithBGridModel   — Bilateral grid + 3D LUT (ECCV 2024, in lut_bilateral.py)

Metadata conditioning (Tier-1 recommendation from research):
  5-dim vector: [log_brightness, sin(hour), cos(hour), sin(month), cos(month)]
  - log_brightness: computed from the image itself (interior/exterior proxy)
  - sin/cos(hour): time of day cyclical encoding
  - sin/cos(month): season cyclical encoding
  Encoded → 32-dim MLP → concat with ResNet 512-dim → 544-dim → LUT heads
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

log = structlog.get_logger(__name__)


# ── Metadata encoder ─────────────────────────────────────────────────────────

def encode_metadata(images: torch.Tensor,
                    hour: torch.Tensor | None = None,
                    month: torch.Tensor | None = None,
                    zone_fracs: torch.Tensor | None = None) -> torch.Tensor:
    """
    Build metadata vector from image + optional datetime + optional zone fractions.

    Base (5-dim):
      0: log_brightness  — log10(mean pixel), encodes indoor/outdoor exposure
      1-2: sin/cos(hour) — time of day, cyclical [0,24]
      3-4: sin/cos(month)— season, cyclical [1,12]

    With zone conditioning (+4-dim = 9-dim total):
      5-8: [sky_ceil_frac, floor_frac, window_frac, walls_frac]
           — fraction of image covered by each SAM zone (sums to ~1)
           — teaches model: "this image is 30% sky, warm it differently"

    At inference without zone maps, zone_fracs=None → uniform [0.25×4].
    """
    B = images.shape[0]
    dev = images.device
    dtype = images.dtype

    brightness = images.mean(dim=(1, 2, 3)).clamp(min=1e-4)
    log_b = torch.log10(brightness).unsqueeze(1)   # (B,1)

    if hour is None:
        hour = torch.full((B,), 12.0, device=dev, dtype=dtype)
    h_norm = hour * (2 * 3.14159265 / 24.0)
    h_sin = torch.sin(h_norm).unsqueeze(1)
    h_cos = torch.cos(h_norm).unsqueeze(1)

    if month is None:
        month = torch.full((B,), 6.0, device=dev, dtype=dtype)
    m_norm = month * (2 * 3.14159265 / 12.0)
    m_sin = torch.sin(m_norm).unsqueeze(1)
    m_cos = torch.cos(m_norm).unsqueeze(1)

    meta_5 = torch.cat([log_b, h_sin, h_cos, m_sin, m_cos], dim=1)  # (B, 5)

    if zone_fracs is not None:
        return torch.cat([meta_5, zone_fracs.to(dev, dtype)], dim=1)  # (B, 9)
    return meta_5  # (B, 5)


class MetaEncoder(nn.Module):
    """Metadata → 32-dim embedding. meta_dim=5 (base) or 9 (zone-aware)."""
    def __init__(self, meta_dim: int = 5, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, out_dim),
        )

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.net(meta)


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
            self.feat = nn.Sequential(*list(r.children())[:-1], nn.Flatten())
            # Freeze backbone — use as fixed feature extractor.
            # 11M params on 65 pairs would overfit; only the tiny head trains.
            for p in self.feat.parameters():
                p.requires_grad = False
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
        # Dropout before head prevents overfitting on the 512-dim bottleneck
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, n_out),
            nn.Softmax(dim=1),
        )

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
                 backbone: str = "resnet18", zone_aware: bool = False):
        super().__init__()
        self.n_1d = n_1d
        self.n_3d = n_3d
        self.zone_aware = zone_aware

        self.luts_1d = nn.Parameter(_diverse_1d_luts(n_1d, lut_1d_size))
        self.luts_3d = nn.Parameter(_diverse_3d_luts(n_3d, lut_3d_size))

        if backbone == "resnet18":
            import torchvision.models as M
            r = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(r.children())[:-1], nn.Flatten())
            for p in self.backbone.parameters():
                p.requires_grad = False
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
        # 5-dim base metadata; +4 zone fractions when zone_aware=True
        meta_dim = 9 if zone_aware else 5
        self.meta_encoder = MetaEncoder(meta_dim=meta_dim, out_dim=32)
        combined_dim = feat_dim + 32
        self.head_1d = nn.Sequential(nn.Dropout(0.3), nn.Linear(combined_dim, n_1d), nn.Softmax(dim=1))
        self.head_3d = nn.Sequential(nn.Dropout(0.3), nn.Linear(combined_dim, n_3d), nn.Softmax(dim=1))
        self.interp = TrilinearInterp()

    def forward(self, x: torch.Tensor,
                hour: torch.Tensor | None = None,
                month: torch.Tensor | None = None,
                zone_fracs: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]

        x_small = F.interpolate(x, 224, mode="bilinear", antialias=True) \
                  if self.backbone_name == "resnet18" and min(x.shape[-2:]) > 256 else x
        feats = self.backbone(x_small)                        # (B, 512)

        # Zone-aware metadata: 5-dim base + optional 4-dim zone fracs → 9-dim
        zf = zone_fracs if self.zone_aware else None
        meta     = encode_metadata(x, hour, month, zf)        # (B, 5 or 9)
        meta_emb = self.meta_encoder(meta)                    # (B, 32)
        feats_cond = torch.cat([feats, meta_emb], dim=1)     # (B, 544)

        w1d  = self.head_1d(feats_cond)
        lut_1d = (w1d[:, :, None, None] *
                  self.luts_1d.unsqueeze(0)).sum(dim=1)       # (B, 3, D_1d)
        x_1d = _apply_1d_lut(x, lut_1d)

        w3d  = self.head_3d(feats_cond)
        lut_3d = (w3d[:, :, None, None, None, None] *
                  self.luts_3d.unsqueeze(0)).sum(dim=1)
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
            zone_aware=s4.get("zone_aware", False),
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
