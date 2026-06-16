from __future__ import annotations

import numpy as np

from colmap_mask.core.mask_ops import normalize_mask, resize_keep_aspect
from colmap_mask.core.overlay import overlay_mask


def test_normalize_mask_binary() -> None:
    mask = np.array([[0, 128], [127, 255]], dtype=np.uint8)
    actual = normalize_mask(mask)
    assert actual.tolist() == [[0, 255], [0, 255]]


def test_resize_keep_aspect_by_width() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = resize_keep_aspect(image, width=50)
    assert resized.shape[:2] == (25, 50)


def test_overlay_mask_only_changes_masked_pixels() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 255
    result = overlay_mask(image, mask, color=(100, 0, 0), opacity=0.5)
    assert result[0, 0].tolist() == [0, 0, 0]
    assert result[1, 1].tolist() == [50, 0, 0]
