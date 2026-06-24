from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_images(folder: Path, exclude_dirs: tuple[Path, ...] = ()) -> list[Path]:
    excluded = set(exclude_dirs)
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(parent in excluded for parent in path.parents)
    )


def load_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, data = cv2.imencode(path.suffix or ".png", bgr)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    data.tofile(str(path))


def load_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    if not path.exists():
        if shape is None:
            raise FileNotFoundError(path)
        return np.zeros(shape, dtype=np.uint8)
    data = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    return mask


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    ok, data = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError(f"Cannot encode mask: {path}")
    data.tofile(str(path))


def mask_path_for(image_path: Path, mask_dir: Path, variant: str | None = None) -> Path:
    suffix = f".mask.{variant}.png" if variant else ".mask.png"
    return mask_dir / f"{image_path.stem}{suffix}"
