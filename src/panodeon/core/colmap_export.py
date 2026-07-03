from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from panodeon.core.image_io import load_mask, load_rgb, save_mask, save_rgb
from panodeon.core.project_state import ImageItem
from panodeon.projection.perspective import (
    camera_rotation,
    equirect_to_perspective,
    perspective_world_directions,
    rotation_matrix_to_quaternion,
)

PANORAMA_YAW_STEPS = 4
PANORAMA_PITCHES_DEG = (-35.0, 0.0, 35.0)


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
    cameras: list[VirtualCamera] = []
    yaw_step_deg = 360.0 / PANORAMA_YAW_STEPS
    for pitch_deg in PANORAMA_PITCHES_DEG:
        yaw_offset_deg = yaw_step_deg * 0.5 if pitch_deg > 0.0 else 0.0
        for yaw_index in range(PANORAMA_YAW_STEPS):
            yaw_deg = (yaw_index * yaw_step_deg + yaw_offset_deg) % 360.0
            cameras.append(
                VirtualCamera(
                    camera_id=len(cameras) + 1,
                    yaw_deg=yaw_deg,
                    pitch_deg=pitch_deg,
                    direction=direction_from_yaw_pitch(yaw_deg, pitch_deg),
                )
            )
    return cameras


def direction_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    return (
        math.sin(yaw) * math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw) * math.cos(pitch),
    )


def round_angle_for_name(angle_deg: float) -> int:
    return int(round(angle_deg)) % 360 if angle_deg >= 0.0 else int(round(angle_deg))


def export_item_for_colmap(
    item: ImageItem,
    export_dir: Path,
    settings: ColmapExportSettings,
) -> None:
    image = load_rgb(item.path)
    mask = load_mask(item.mask_path, image.shape[:2])
    cameras = virtual_cameras()
    for camera in cameras:
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
        owner_mask = virtual_camera_owner_mask(camera, cameras, settings.tile_size, settings.fov_deg)
        valid_mask = owner_mask & (tile_mask <= 127)
        colmap_mask = np.where(valid_mask, 255, 0).astype(np.uint8)
        save_rgb(image_path, tile)
        save_mask(mask_path, colmap_mask)


def write_colmap_metadata(export_dir: Path, settings: ColmapExportSettings) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    cameras = virtual_cameras()
    ref_camera = next(camera for camera in cameras if camera.pitch_deg == 0.0)
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


def virtual_camera_owner_mask(
    camera: VirtualCamera,
    cameras: list[VirtualCamera],
    tile_size: int,
    fov_deg: float,
) -> np.ndarray:
    world_dirs = perspective_world_directions(camera.yaw_deg, camera.pitch_deg, tile_size, fov_deg)
    camera_dirs = np.asarray([other.direction for other in cameras], dtype=np.float32)
    closest_camera_ids = np.argmax(world_dirs @ camera_dirs.T, axis=-1) + 1
    return closest_camera_ids == camera.camera_id


def write_readme(export_dir: Path, settings: ColmapExportSettings) -> None:
    text = f"""COLMAP export

images/: projected perspective images.
masks/: COLMAP feature masks. Black pixels are ignored by feature extraction. Pixels are valid only for their nearest virtual camera and non-person regions.
rig_config.json: input for colmap rig_configurator.

Example:
colmap feature_extractor --database_path database.db --image_path images --ImageReader.mask_path masks --ImageReader.single_camera_per_folder 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params {settings.tile_size * 0.5 / np.tan(np.radians(settings.fov_deg) * 0.5):.6f},{settings.tile_size * 0.5 / np.tan(np.radians(settings.fov_deg) * 0.5):.6f},{settings.tile_size / 2:.6f},{settings.tile_size / 2:.6f}
colmap rig_configurator --database_path database.db --rig_config_path rig_config.json
"""
    (export_dir / "README_colmap.txt").write_text(text, encoding="utf-8")
