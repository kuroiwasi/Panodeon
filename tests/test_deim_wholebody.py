from __future__ import annotations

import numpy as np

from panodeon.generators.base import MaskOptions
from panodeon.inference.deim_wholebody import (
    DeimWholebodySegmenter,
    class_ids_from_names,
    clip_mask_to_box,
    denormalize_box,
    resize_probability_mask,
)


def test_class_ids_from_names() -> None:
    assert class_ids_from_names(("body", "head", "hand", "unknown")) == {0, 7, 32}


def test_denormalize_box_accepts_normalized_coords() -> None:
    assert denormalize_box(np.array([0.25, 0.25, 0.75, 0.75]), 200, 100) == (50, 25, 150, 75)


def test_resize_probability_mask_shape() -> None:
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 0.9
    resized = resize_probability_mask(mask, (16, 16))
    assert resized.shape == (16, 16)
    assert resized.max() > 0.5


def test_clip_mask_to_box() -> None:
    mask = np.ones((10, 10), dtype=bool)
    clipped = clip_mask_to_box(mask, np.array([2, 2, 5, 5], dtype=np.float32), (10, 10), padding=0)
    assert clipped.sum() == 9


def test_predict_mask_raises_runtime_provider_error() -> None:
    segmenter = DeimWholebodySegmenter.__new__(DeimWholebodySegmenter)
    segmenter.session = FailingSession()
    segmenter.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    segmenter.requested_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    segmenter.image_input_name = "images"
    segmenter.input_names = {"images"}
    segmenter.output_names = {"label_xyxy_score"}
    segmenter.image_size = None
    segmenter.normalize = False

    try:
        segmenter.predict_mask(np.zeros((4, 4, 3), dtype=np.uint8), MaskOptions(dilate_px=0, feather_px=0))
    except RuntimeError as exc:
        assert "CUDA Conv failed" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


class FailingSession:
    def run(self, outputs, feed):
        raise RuntimeError("CUDA Conv failed")


def test_predict_masks_fallback_to_predict_mask() -> None:
    segmenter = DeimWholebodySegmenter.__new__(DeimWholebodySegmenter)
    segmenter.inputs = [FakeInput([1, 3, 640, 640])]
    
    called = []
    def fake_predict_mask(image, options):
        called.append(image)
        return np.ones(image.shape[:2], dtype=np.uint8)
    segmenter.predict_mask = fake_predict_mask
    
    images = [np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)]
    results = segmenter.predict_masks(images, MaskOptions())
    
    assert len(results) == 2
    assert len(called) == 2
    assert results[0].max() == 1


class FakeInput:
    def __init__(self, shape):
        self.shape = shape

