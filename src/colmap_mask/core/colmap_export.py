from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from colmap_mask.core.image_io import load_mask, load_rgb, save_mask, save_rgb
from colmap_mask.core.project_state import ImageItem
from colmap_mask.projection.perspective import (
    camera_rotation,
    equirect_to_perspective,
    rotation_matrix_to_quaternion,
)

ICOSAHEDRON_X_ROTATION_DEG = 25.0
ICOSAHEDRON_Z_ROTATION_DEG = 18.0


@dataclass(frozen=True)
class ColmapExportSettings:
    tile_size: int = 3072
    fov_deg: float = 90.0


@dataclass(frozen=True)
class VirtualCamera:
    camera_id: int
    yaw_deg: float
    pitch_deg: float
    direction: tuple[float, float, float]

    @property
    def name(self) -> str:
        return f"cam{self.camera_id:02d}_y{round_angle_for_name(self.yaw_deg):03d}_p{round_angle_for_name(self.pitch_deg):+03d}"


def virtual_cameras() -> list[VirtualCamera]:
    directions = rotated_icosahedron_directions()
    angles = [direction_to_yaw_pitch(direction) for direction in directions]
    order = sorted(range(len(directions)), key=lambda index: (-angles[index][1], angles[index][0]))
    return [
        VirtualCamera(
            camera_id=camera_id,
            yaw_deg=angles[index][0],
            pitch_deg=angles[index][1],
            direction=tuple(float(value) for value in directions[index]),
        )
        for camera_id, index in enumerate(order, start=1)
    ]


def rotated_icosahedron_directions() -> list[np.ndarray]:
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    vertices = [
        (-1.0, phi, 0.0),
        (1.0, phi, 0.0),
        (-1.0, -phi, 0.0),
        (1.0, -phi, 0.0),
        (0.0, -1.0, phi),
        (0.0, 1.0, phi),
        (0.0, -1.0, -phi),
        (0.0, 1.0, -phi),
        (phi, 0.0, -1.0),
        (phi, 0.0, 1.0),
        (-phi, 0.0, -1.0),
        (-phi, 0.0, 1.0),
    ]
    rotation = rotation_z(ICOSAHEDRON_Z_ROTATION_DEG) @ rotation_x(ICOSAHEDRON_X_ROTATION_DEG)
    directions: list[np.ndarray] = []
    for vertex in vertices:
        direction = rotation @ np.asarray(vertex, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        directions.append(direction)
    return directions


def rotation_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ],
        dtype=np.float64,
    )


def rotation_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.asarray(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def direction_to_yaw_pitch(direction: np.ndarray) -> tuple[float, float]:
    x, y, z = (float(value) for value in direction)
    yaw = math.degrees(math.atan2(x, z)) % 360.0
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, y))))
    return yaw, pitch


def round_angle_for_name(angle_deg: float) -> int:
    return int(round(angle_deg)) % 360 if angle_deg >= 0.0 else int(round(angle_deg))


def export_item_for_colmap(
    item: ImageItem,
    export_dir: Path,
    settings: ColmapExportSettings,
) -> None:
    image = load_rgb(item.path)
    mask = load_mask(item.mask_path, image.shape[:2])
    for camera in virtual_cameras():
        rel_name = item.relative_dir / item.path.name
        image_path = export_dir / "images" / camera.name / rel_name
        mask_path = export_dir / "masks" / camera.name / rel_name.with_name(f"{rel_name.name}.png")
        tile = equirect_to_perspective(
            image,
            camera.yaw_deg,
            camera.pitch_deg,
            settings.tile_size,
            settings.fov_deg,
            interpolation=cv2.INTER_LINEAR,
        )
        tile_mask = equirect_to_perspective(
            mask,
            camera.yaw_deg,
            camera.pitch_deg,
            settings.tile_size,
            settings.fov_deg,
            interpolation=cv2.INTER_NEAREST,
        )
        colmap_mask = np.where(tile_mask > 127, 0, 255).astype(np.uint8)
        save_rgb(image_path, tile)
        save_mask(mask_path, colmap_mask)


def write_colmap_metadata(export_dir: Path, settings: ColmapExportSettings) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    cameras = virtual_cameras()
    ref_camera = cameras[0]
    payload = [
        {
            "cameras": [camera_config_json(camera, ref_camera) for camera in cameras],
        }
    ]
    (export_dir / "rig_config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_readme(export_dir, settings)


def camera_config_json(camera: VirtualCamera, ref_camera: VirtualCamera) -> dict[str, object]:
    config: dict[str, object] = {
        "image_prefix": f"{camera.name}/",
    }
    if camera.camera_id == ref_camera.camera_id:
        config["ref_sensor"] = True
        return config
    config.update(camera_pose_json(camera, ref_camera))
    return config


def camera_pose_json(camera: VirtualCamera, ref_camera: VirtualCamera) -> dict[str, list[float]]:
    cam_from_rig = world_from_colmap_camera(camera).T @ world_from_colmap_camera(ref_camera)
    qw, qx, qy, qz = rotation_matrix_to_quaternion(cam_from_rig)
    return {
        "cam_from_rig_rotation": [qw, qx, qy, qz],
        "cam_from_rig_translation": [0.0, 0.0, 0.0],
    }


def world_from_colmap_camera(camera: VirtualCamera) -> np.ndarray:
    rotation = camera_rotation(camera.yaw_deg, camera.pitch_deg)
    return rotation @ np.diag([1.0, -1.0, 1.0])


def write_readme(export_dir: Path, settings: ColmapExportSettings) -> None:
    text = f"""COLMAP export

images/: projected perspective images.
masks/: COLMAP feature masks. Black pixels are ignored by feature extraction.
rig_config.json: input for colmap rig_configurator.

Example:
colmap feature_extractor --database_path database.db --image_path images --ImageReader.mask_path masks --ImageReader.single_camera_per_folder 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params {settings.tile_size * 0.5 / np.tan(np.radians(settings.fov_deg) * 0.5):.6f},{settings.tile_size * 0.5 / np.tan(np.radians(settings.fov_deg) * 0.5):.6f},{settings.tile_size / 2:.6f},{settings.tile_size / 2:.6f}
colmap rig_configurator --database_path database.db --rig_config_path rig_config.json
"""
    (export_dir / "README_colmap.txt").write_text(text, encoding="utf-8")
