from pathlib import Path

import numpy as np
import structlog

from .lut3d import load_model, apply_lut

log = structlog.get_logger(__name__)

_model_cache: dict = {}


def process(img: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Apply look-matching LUT to a uint8 RGB image.

    If brightness_norm is enabled in config, the input is scaled to the
    training target brightness before the LUT and the scale is reported.
    This ensures inference matches the training domain.
    """
    s4 = cfg["stage4_look"]
    if s4["method"] != "lut3d":
        raise NotImplementedError(f"method={s4['method']} not yet implemented")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    key = s4["lut_model_path"]
    if key not in _model_cache:
        _model_cache[key] = load_model(cfg, device)

    inp = img
    scale_applied = 1.0

    if s4["train"].get("brightness_norm", False):
        import math
        target_brightness = s4["train"].get("target_brightness", 0.609)
        inp_f32 = img.astype(np.float32) / 255.0
        inp_mean = inp_f32.mean()
        if inp_mean > 1e-4 and inp_mean < 0.999:
            gamma = math.log(target_brightness) / math.log(inp_mean)
            gamma = float(max(0.3, min(3.0, gamma)))
            scale_applied = gamma
            inp = np.power(np.clip(inp_f32, 1e-8, 1.0), gamma)
            inp = (inp * 255).astype(np.uint8)

    result = apply_lut(inp, _model_cache[key], device)
    log.info("stage4.done", shape=result.shape, brightness_scale=round(scale_applied, 3))
    return result
