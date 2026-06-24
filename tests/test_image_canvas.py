from __future__ import annotations

from panodeon.ui.image_canvas import build_stroke_mask


def test_build_stroke_mask_draws_line() -> None:
    mask = build_stroke_mask((32, 32), [(4, 4), (28, 28)], radius=2)
    assert mask[4, 4] > 0
    assert mask[16, 16] > 0
    assert mask[28, 28] > 0


def test_build_stroke_mask_empty_points() -> None:
    mask = build_stroke_mask((16, 16), [], radius=4)
    assert mask.sum() == 0
