from __future__ import annotations

import math

import cv2
import numpy as np


def camera_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    forward = np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
            math.cos(yaw) * math.cos(pitch),
        ],
        dtype=np.float32,
    )
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float32)
    up = np.cross(forward, right).astype(np.float32)
    right /= np.linalg.norm(right)
    up /= np.linalg.norm(up)
    forward /= np.linalg.norm(forward)
    return np.stack([right, up, forward], axis=1)


def equirect_to_perspective(
    image: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    size: int,
    fov_deg: float = 90.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be positive")
    height, width = image.shape[:2]
    world_dirs = perspective_world_directions(yaw_deg, pitch_deg, size, fov_deg)
    map_x, map_y = sphere_to_equirect_uv(world_dirs, width, height)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_WRAP,
    )


def perspective_world_directions(yaw_deg: float, pitch_deg: float, size: int, fov_deg: float = 90.0) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be positive")
    rot = camera_rotation(yaw_deg, pitch_deg)
    half = math.tan(math.radians(fov_deg) * 0.5)
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    px = (coords * 2.0 - 1.0) * half
    py = (1.0 - coords * 2.0) * half
    x_grid, y_grid = np.meshgrid(px, py)
    camera_dirs = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)
    camera_dirs /= np.linalg.norm(camera_dirs, axis=-1, keepdims=True)
    return camera_dirs @ rot.T


def sphere_to_equirect_uv(direction: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x = direction[..., 0]
    y = direction[..., 1]
    z = direction[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y, -1.0, 1.0))
    map_x = ((lon / (2.0 * math.pi)) + 0.5) * width
    map_y = (0.5 - lat / math.pi) * height
    return map_x.astype(np.float32), map_y.astype(np.float32)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return tuple(float(value) for value in quat)
