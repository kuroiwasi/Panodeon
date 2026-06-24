from __future__ import annotations

import numpy as np

from panodeon.projection.cubemap import equirect_to_face, face_to_equirect_mask
from panodeon.projection.perspective import equirect_to_perspective


def test_equirect_to_face_shape() -> None:
    image = np.zeros((64, 128, 3), dtype=np.uint8)
    image[:, :64, 0] = 255
    face = equirect_to_face(image, face="front", face_size=32, fov_deg=90)
    assert face.shape == (32, 32, 3)


def test_face_to_equirect_mask_shape_and_values() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    eq = face_to_equirect_mask(mask, output_shape=(64, 128), face="front", fov_deg=90)
    assert eq.shape == (64, 128)
    assert eq.max() == 255
    assert eq.min() == 0


def test_equirect_to_perspective_shape() -> None:
    image = np.zeros((64, 128, 3), dtype=np.uint8)
    tile = equirect_to_perspective(image, yaw_deg=60, pitch_deg=-60, size=32, fov_deg=90)
    assert tile.shape == (32, 32, 3)
