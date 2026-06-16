from __future__ import annotations

import json
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

PITCHES = (0, 60, -60)
YAWS = (0, 60, 120, 180, 240, 300)


@dataclass(frozen=True)
class ColmapExportSettings:
    tile_size: int = 3072
    fov_deg: float = 90.0


@dataclass(frozen=True)
class VirtualCamera:
    camera_id: int
    yaw_deg: int
    pitch_deg: int

    @property
    def name(self) -> str:
        return f"cam{self.camera_id:02d}_y{self.yaw_deg:03d}_p{self.pitch_deg:+03d}"


def virtual_cameras() -> list[VirtualCamera]:
    cameras: list[VirtualCamera] = []
    camera_id = 1
    for pitch in PITCHES:
        for yaw in YAWS:
            cameras.append(VirtualCamera(camera_id, yaw, pitch))
            camera_id += 1
    return cameras


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
            "ref_camera_id": ref_camera.camera_id,
            "cameras": [camera_config_json(camera, ref_camera) for camera in cameras],
        }
    ]
    (export_dir / "rig_config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_readme(export_dir, settings)


def camera_config_json(camera: VirtualCamera, ref_camera: VirtualCamera) -> dict[str, object]:
    config: dict[str, object] = {
        "camera_id": camera.camera_id,
        "image_prefix": f"{camera.name}/",
    }
    if camera.camera_id == ref_camera.camera_id:
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
