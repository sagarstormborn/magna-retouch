from pathlib import Path

import numpy as np
import structlog

from .lut3d import load_model, apply_lut

log = structlog.get_logger(__name__)

_model_cache: dict = {}


def process(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply look-matching LUT. Caches model across calls in the same process."""
    s4 = cfg["stage4_look"]
    if s4["method"] != "lut3d":
        raise NotImplementedError(f"method={s4['method']} not yet implemented (hdrnet is stage-gated)")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    key = s4["lut_model_path"]
    if key not in _model_cache:
        _model_cache[key] = load_model(cfg, device)

    result = apply_lut(img, _model_cache[key], device)
    log.info("stage4.done", shape=result.shape)
    return result
