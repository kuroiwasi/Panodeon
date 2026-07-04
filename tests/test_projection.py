from __future__ import annotations

import math

import cv2
import numpy as np

from panodeon.projection.cubemap import (
    CUBE_FACES,
    _face_rotation,
    _sphere_to_equirect_uv,
    accumulate_face_mask,
    equirect_to_face,
    face_to_equirect_mask,
)
from panodeon.projection.perspective import equirect_to_perspective


def _float_map_face(image: np.ndarray, face: str, face_size: int, fov_deg: float) -> np.ndarray:
    """Uncached float32-map reference for equirect_to_face."""
    height, width = image.shape[:2]
    rot = _face_rotation(face)
    half = math.tan(math.radians(fov_deg) * 0.5)
    coords = (np.arange(face_size, dtype=np.float32) + 0.5) / face_size
    px = (coords * 2.0 - 1.0) * half
    py = (1.0 - coords * 2.0) * half
    x_grid, y_grid = np.meshgrid(px, py)
    camera_dirs = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)
    camera_dirs /= np.linalg.norm(camera_dirs, axis=-1, keepdims=True)
    world_dirs = camera_dirs @ rot.T
    map_x, map_y = _sphere_to_equirect_uv(world_dirs, width, height)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


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


def test_equirect_to_face_matches_float_map_reference() -> None:
    # Cached maps are CV_16SC2 fixed point (1/32 px); on a smooth gradient the
    # interpolation error stays within a couple of grey levels.
    xs = np.linspace(0, 255, 128, dtype=np.float32)
    ys = np.linspace(0, 255, 64, dtype=np.float32)
    image = np.dstack([np.tile(xs, (64, 1)), np.tile(ys[:, None], (1, 128)), np.full((64, 128), 128.0)])
    image = image.astype(np.uint8)
    for face in CUBE_FACES:
        got = equirect_to_face(image, face=face, face_size=32, fov_deg=105.0)
        ref = _float_map_face(image, face, 32, 105.0)
        assert int(np.abs(got.astype(np.int16) - ref.astype(np.int16)).max()) <= 4


def test_equirect_to_face_cache_does_not_leak_image_content() -> None:
    rng = np.random.default_rng(7)
    first = rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)
    second = rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)
    equirect_to_face(first, face="front", face_size=32, fov_deg=90)
    got = equirect_to_face(second, face="front", face_size=32, fov_deg=90)
    ref = _float_map_face(second, "front", 32, 90.0)
    diff = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    # random-noise image: fixed-point quantisation may shift bilinear weights,
    # but the result must track the second image, not the first
    assert float(np.mean(diff > 16)) < 0.01


def test_accumulate_face_mask_max_merges_in_place() -> None:
    mask = np.full((32, 32), 200, dtype=np.uint8)
    output = np.full((64, 128), 210, dtype=np.uint8)
    accumulate_face_mask(output, mask, face="front", fov_deg=90)
    assert output.min() == 210  # existing higher values survive the merge

    output = np.zeros((64, 128), dtype=np.uint8)
    accumulate_face_mask(output, mask, face="front", fov_deg=90)
    assert output.max() == 200


def test_face_to_equirect_mask_covers_all_faces() -> None:
    # 6 full faces at 90 deg tile the sphere; nearest-neighbour rounding ties
    # may drop isolated seam pixels only
    full = np.full((32, 32), 255, dtype=np.uint8)
    output = np.zeros((64, 128), dtype=np.uint8)
    for face in CUBE_FACES:
        accumulate_face_mask(output, full, face=face, fov_deg=90)
    assert float(np.mean(output != 255)) < 0.005


def test_equirect_to_perspective_shape() -> None:
    image = np.zeros((64, 128, 3), dtype=np.uint8)
    tile = equirect_to_perspective(image, yaw_deg=60, pitch_deg=-60, size=32, fov_deg=90)
    assert tile.shape == (32, 32, 3)
