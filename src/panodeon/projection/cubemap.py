from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

CUBE_FACES = ("front", "right", "back", "left", "up", "down")


def _face_rotation(face: str) -> np.ndarray:
    # Columns are world-space camera axes: right, up, forward.
    if face == "front":
        right, up, forward = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    elif face == "back":
        right, up, forward = (-1, 0, 0), (0, 1, 0), (0, 0, -1)
    elif face == "right":
        right, up, forward = (0, 0, -1), (0, 1, 0), (1, 0, 0)
    elif face == "left":
        right, up, forward = (0, 0, 1), (0, 1, 0), (-1, 0, 0)
    elif face == "up":
        right, up, forward = (1, 0, 0), (0, 0, -1), (0, 1, 0)
    elif face == "down":
        right, up, forward = (1, 0, 0), (0, 0, 1), (0, -1, 0)
    else:
        raise ValueError(f"Unknown cube face: {face}")
    return np.array([right, up, forward], dtype=np.float32).T


def _sphere_to_equirect_uv(direction: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x = direction[..., 0]
    y = direction[..., 1]
    z = direction[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y, -1.0, 1.0))
    map_x = ((lon / (2.0 * math.pi)) + 0.5) * width
    map_y = (0.5 - lat / math.pi) * height
    return map_x.astype(np.float32), map_y.astype(np.float32)


def _equirect_dirs_block(
    width: int,
    height: int,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
) -> np.ndarray:
    xs = (np.arange(x_start, x_end, dtype=np.float32) + 0.5) / width
    ys = (np.arange(y_start, y_end, dtype=np.float32) + 0.5) / height
    lon = (xs - 0.5) * (2.0 * math.pi)
    lat = (0.5 - ys) * math.pi
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat_grid)
    return np.stack(
        [
            np.sin(lon_grid) * cos_lat,
            np.sin(lat_grid),
            np.cos(lon_grid) * cos_lat,
        ],
        axis=-1,
    ).astype(np.float32)


def _wrap_column_ranges(x_start: int, x_end: int, width: int) -> list[tuple[int, int]]:
    if x_end - x_start >= width:
        return [(0, width)]
    start = x_start % width
    end = start + (x_end - x_start)
    if end <= width:
        return [(start, end)]
    return [(start, width), (0, end - width)]


def _face_equirect_bounds(
    face: str,
    width: int,
    height: int,
    fov_deg: float,
    margin: int = 2,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Conservative equirect bounding region (rows + wrapped column ranges)
    outside of which the face cannot project. Pixels inside are still tested
    exactly, so a superset is sufficient."""
    if fov_deg >= 179.0:
        return 0, height, [(0, width)]
    half = math.tan(math.radians(fov_deg) * 0.5)
    delta = math.atan(half)
    if face in ("up", "down"):
        cap = math.atan(1.0 / (math.sqrt(2.0) * half))
        if face == "up":
            lat_min, lat_max = cap, math.pi * 0.5
        else:
            lat_min, lat_max = -math.pi * 0.5, -cap
        x_ranges = [(0, width)]
    else:
        lat_min, lat_max = -delta, delta
        centers = {"front": 0.0, "right": math.pi * 0.5, "back": math.pi, "left": -math.pi * 0.5}
        lon_min = centers[face] - delta
        lon_max = centers[face] + delta
        x0 = math.floor(((lon_min / (2.0 * math.pi)) + 0.5) * width) - margin
        x1 = math.ceil(((lon_max / (2.0 * math.pi)) + 0.5) * width) + margin
        x_ranges = _wrap_column_ranges(x0, x1, width)
    y0 = max(0, math.floor((0.5 - lat_max / math.pi) * height) - margin)
    y1 = min(height, math.ceil((0.5 - lat_min / math.pi) * height) + margin)
    return y0, y1, x_ranges


@lru_cache(maxsize=16)
def _face_sample_maps(
    face: str,
    face_size: int,
    fov_deg: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample maps depend only on geometry, never image content; stored
    pre-converted to CV_16SC2 fixed point (half the float32 footprint)."""
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
    map1, map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
    map1.flags.writeable = False
    map2.flags.writeable = False
    return map1, map2


def equirect_to_face(
    image: np.ndarray,
    face: str,
    face_size: int,
    fov_deg: float = 90.0,
) -> np.ndarray:
    height, width = image.shape[:2]
    map1, map2 = _face_sample_maps(face, face_size, float(fov_deg), width, height)
    return cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


@lru_cache(maxsize=8)
def _face_mask_maps(
    face: str,
    width: int,
    height: int,
    fov_deg: float,
    face_h: int,
    face_w: int,
) -> tuple[tuple[int, int, int, int, np.ndarray], ...]:
    """Bounded-region back-projection maps per wrapped column range, as
    (y_lo, y_hi, x_start, x_end, map1). CV_16SC2 nearest-neighbour form is
    4 bytes/px (~150-250 MB across 6 faces at 8192x4096), hence the small
    LRU; map_y fits int16 for equirect heights up to 32k."""
    rot = _face_rotation(face)
    half = math.tan(math.radians(fov_deg) * 0.5)
    y_lo, y_hi, x_ranges = _face_equirect_bounds(face, width, height, fov_deg)
    entries: list[tuple[int, int, int, int, np.ndarray]] = []
    for x_start, x_end in x_ranges:
        dirs = _equirect_dirs_block(width, height, x_start, x_end, y_lo, y_hi)
        cam = dirs @ rot
        z = cam[..., 2]
        valid = z > 1e-6
        x_norm = cam[..., 0] / np.maximum(z, 1e-6)
        y_norm = cam[..., 1] / np.maximum(z, 1e-6)
        valid &= np.abs(x_norm) <= half
        valid &= np.abs(y_norm) <= half
        map_x = ((x_norm / half) + 1.0) * 0.5 * face_w - 0.5
        map_y = (1.0 - ((y_norm / half) + 1.0) * 0.5) * face_h - 0.5
        map_x = np.where(valid, map_x, -1).astype(np.float32)
        map_y = np.where(valid, map_y, -1).astype(np.float32)
        map1, _ = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2, nninterpolation=True)
        map1.flags.writeable = False
        entries.append((y_lo, y_hi, x_start, x_end, map1))
    return tuple(entries)


def accumulate_face_mask(
    output: np.ndarray,
    face_mask: np.ndarray,
    face: str,
    fov_deg: float = 90.0,
) -> None:
    """Back-project a cube-face mask into `output` (max-merge, in place)."""
    height, width = output.shape[:2]
    face_h, face_w = face_mask.shape[:2]
    for y_lo, y_hi, x_start, x_end, map1 in _face_mask_maps(
        face, width, height, float(fov_deg), face_h, face_w
    ):
        block = cv2.remap(
            face_mask,
            map1,
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        target = output[y_lo:y_hi, x_start:x_end]
        np.maximum(target, block, out=target)


def face_to_equirect_mask(
    face_mask: np.ndarray,
    output_shape: tuple[int, int],
    face: str,
    fov_deg: float = 90.0,
) -> np.ndarray:
    height, width = output_shape
    output = np.zeros((height, width), dtype=np.uint8)
    accumulate_face_mask(output, face_mask, face, fov_deg)
    return output
