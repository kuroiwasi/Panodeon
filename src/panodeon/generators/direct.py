from __future__ import annotations

from time import perf_counter

import numpy as np

from panodeon.core.mask_ops import postprocess_mask

from .base import MaskOptions, MaskResult, Segmenter


class DirectEquirectangularGenerator:
    def __init__(self, segmenter: Segmenter | None = None) -> None:
        self.segmenter = segmenter

    def generate(self, image: np.ndarray, options: MaskOptions) -> MaskResult:
        start = perf_counter()
        if self.segmenter is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            metadata: dict[str, object] = {"stub": True}
        else:
            mask = self.segmenter.predict_mask(image, options)
            mask = postprocess_mask(mask, options.dilate_px, options.feather_px)
            metadata = {"stub": False}
        return MaskResult(
            mask=mask,
            strategy="direct",
            elapsed_sec=perf_counter() - start,
            metadata=metadata,
        )
