from __future__ import annotations

import numpy as np


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 32, 32),
    opacity: float = 0.35,
) -> np.ndarray:
    if opacity <= 0.0 or not np.any(mask):
        return image.copy()
    alpha = ((mask > 127).astype(np.float32) * max(0.0, min(1.0, opacity)))[:, :, None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    result = image.astype(np.float32) * (1.0 - alpha) + color_arr * alpha
    return np.clip(result, 0, 255).astype(np.uint8)
