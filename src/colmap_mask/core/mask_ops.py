from __future__ import annotations

import cv2
import numpy as np


def normalize_mask(mask: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if shape is not None and mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def postprocess_mask(mask: np.ndarray, dilate_px: int = 8, feather_px: int = 5) -> np.ndarray:
    mask = normalize_mask(mask)
    if dilate_px > 0:
        size = dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)
    if feather_px > 0:
        size = feather_px * 2 + 1
        if size % 2 == 0:
            size += 1
        mask = cv2.GaussianBlur(mask, (size, size), 0)
    return mask.astype(np.uint8)


def draw_mask_circle(
    mask: np.ndarray,
    center: tuple[int, int],
    radius: int,
    value: int,
) -> np.ndarray:
    edited = mask.copy()
    cv2.circle(edited, center, max(1, radius), int(value), thickness=-1, lineType=cv2.LINE_AA)
    return edited


def resize_keep_aspect(image: np.ndarray, width: int = 0, height: int = 0, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    h, w = image.shape[:2]
    if width <= 0 and height <= 0:
        return image.copy()
    if width > 0:
        scale = width / w
        out_w = width
        out_h = max(1, round(h * scale))
    else:
        scale = height / h
        out_w = max(1, round(w * scale))
        out_h = height
    return cv2.resize(image, (out_w, out_h), interpolation=interpolation)


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask > 127))
