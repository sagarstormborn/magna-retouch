from pathlib import Path
import numpy as np
import cv2


RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2"}


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def find_images(directory: Path, extensions: set[str] | None = None) -> list[Path]:
    exts = extensions or RAW_EXTENSIONS
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in exts)


def save_tiff_16(path: Path, img: np.ndarray) -> None:
    """Save a uint16 HxWx3 image as 16-bit TIFF (lossless, preserves full range)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV expects BGR; flip RGB→BGR here and back on load
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_tiff_16(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def float_to_uint16(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16)


def uint16_to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 65535.0
