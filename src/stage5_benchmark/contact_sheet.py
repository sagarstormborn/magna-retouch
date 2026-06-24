"""
Generate side-by-side contact sheets:
    Input | Ours | Autoenhance | Matt's Reference

Output is a 16-bit TIFF per image — the deliverable for calibrated-monitor inspection.
If an Autoenhance path is absent, that column shows a grey placeholder.
"""
from pathlib import Path

import cv2
import numpy as np


_LABELS = ["Input", "Ours", "Autoenhance", "Matt's Reference"]
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_THUMB_WIDTH = 800          # px per column at output res
_LABEL_H = 40               # px for text strip


def make_contact_sheet(
    panels: list[np.ndarray | None],
    labels: list[str] | None = None,
    output_path: Path | None = None,
) -> np.ndarray:
    """
    panels: [input, ours, autoenhance, reference] — None = grey placeholder.
    Returns a uint16 HxWx3 montage.
    """
    labels = labels or _LABELS
    assert len(panels) == len(labels), "panel/label count must match"

    thumbs = [_thumb(p) for p in panels]
    h = max(t.shape[0] for t in thumbs)

    # Pad all to same height
    padded = [_pad_to(t, h) for t in thumbs]

    # Draw label strip
    strips = [_label_strip(l) for l in labels]
    columns = [np.concatenate([s, p], axis=0) for s, p in zip(strips, padded)]

    sheet = np.concatenate(columns, axis=1)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    return sheet


def _thumb(img: np.ndarray | None) -> np.ndarray:
    if img is None:
        placeholder = np.full((_THUMB_WIDTH, _THUMB_WIDTH, 3), 32768, dtype=np.uint16)
        return placeholder
    h, w = img.shape[:2]
    scale = _THUMB_WIDTH / w
    new_w, new_h = _THUMB_WIDTH, int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _pad_to(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h >= target_h:
        return img
    pad = np.zeros((target_h - h, w, 3), dtype=img.dtype)
    return np.concatenate([img, pad], axis=0)


def _label_strip(text: str) -> np.ndarray:
    strip = np.zeros((_LABEL_H, _THUMB_WIDTH, 3), dtype=np.uint16)
    # Draw white text scaled to 16-bit (8-bit value 200 → 16-bit ~51400)
    cv2.putText(
        strip,
        text,
        (10, _LABEL_H - 10),
        _FONT,
        0.9,
        (51400, 51400, 51400),
        1,
        cv2.LINE_AA,
    )
    return strip
