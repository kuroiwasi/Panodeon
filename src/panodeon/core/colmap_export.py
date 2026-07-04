from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
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
    sphere_to_equirect_uv,
)

PANORAMA_YAW_STEPS = 4
PANORAMA_PITCHES_DEG = (0.0, -35.0, 35.0)
EXPORT_MAX_WORKERS = min(8, os.cpu_count() or 4)
# Cached remap maps cost ~10 bytes/pixel/camera; above this budget they are
# recomputed per image instead of held for the whole export run.
MAX_CACHED_MAP_BYTES = 2 * 1024**3


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
    return list(_virtual_cameras())


@lru_cache(maxsize=1)
def _virtual_cameras() -> tuple[VirtualCamera, ...]:
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
    return tuple(cameras)


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


@dataclass(frozen=True)
class VirtualCameraMaps:
    tile_map1: np.ndarray
    tile_map2: np.ndarray
    mask_map1: np.ndarray
    mask_map2: np.ndarray | None


@dataclass(frozen=True)
class ColmapExportContext:
    settings: ColmapExportSettings
    width: int
    height: int
    cameras: tuple[VirtualCamera, ...]
    owner_masks: tuple[np.ndarray, ...]
    camera_maps: tuple[VirtualCameraMaps, ...] | None


class ColmapExportContextCache:
    """Per-export-run cache of contexts keyed by settings and source resolution."""

    def __init__(self) -> None:
        self._contexts: dict[tuple[ColmapExportSettings, int, int], ColmapExportContext] = {}

    def get(self, settings: ColmapExportSettings, width: int, height: int) -> ColmapExportContext:
        key = (settings, width, height)
        context = self._contexts.get(key)
        if context is None:
            context = build_export_context(settings, width, height)
            self._contexts[key] = context
        return context

    def clear(self) -> None:
        self._contexts.clear()


def build_export_context(settings: ColmapExportSettings, width: int, height: int) -> ColmapExportContext:
    cameras = _virtual_cameras()
    cache_maps = (
        estimated_map_cache_bytes(len(cameras), settings.tile_size) <= MAX_CACHED_MAP_BYTES
        and max(width, height) < 2**15  # CV_16SC2 stores int16 source coordinates
    )
    with ThreadPoolExecutor(max_workers=EXPORT_MAX_WORKERS) as executor:
        entries = list(
            executor.map(
                lambda camera: _build_camera_entry(camera, cameras, settings, width, height, cache_maps),
                cameras,
            )
        )
    return ColmapExportContext(
        settings=settings,
        width=width,
        height=height,
        cameras=cameras,
        owner_masks=tuple(owner_mask for owner_mask, _ in entries),
        camera_maps=tuple(maps for _, maps in entries if maps is not None) if cache_maps else None,
    )


def estimated_map_cache_bytes(camera_count: int, tile_size: int) -> int:
    # CV_16SC2 tile map (4 B/px) + CV_16UC1 coefficients (2 B/px) + CV_16SC2 nearest map (4 B/px).
    return camera_count * tile_size * tile_size * 10


def _build_camera_entry(
    camera: VirtualCamera,
    cameras: Sequence[VirtualCamera],
    settings: ColmapExportSettings,
    width: int,
    height: int,
    cache_maps: bool,
) -> tuple[np.ndarray, VirtualCameraMaps | None]:
    world_dirs = perspective_world_directions(camera.yaw_deg, camera.pitch_deg, settings.tile_size, settings.fov_deg)
    owner_mask = _owner_mask_from_world_dirs(camera, cameras, world_dirs)
    if not cache_maps:
        return owner_mask, None
    map_x, map_y = sphere_to_equirect_uv(world_dirs, width, height)
    tile_map1, tile_map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
    mask_map1, mask_map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2, nninterpolation=True)
    return owner_mask, VirtualCameraMaps(tile_map1, tile_map2, mask_map1, mask_map2)


def export_item_for_colmap(
    item: ImageItem,
    export_dir: Path,
    settings: ColmapExportSettings,
    context_cache: ColmapExportContextCache | None = None,
) -> None:
    image = load_rgb(item.path)
    mask = load_mask(item.mask_path, image.shape[:2])
    height, width = image.shape[:2]
    if context_cache is not None:
        context = context_cache.get(settings, width, height)
    else:
        context = build_export_context(settings, width, height)
    rel_name = item.relative_dir / item.path.name
    with ThreadPoolExecutor(max_workers=EXPORT_MAX_WORKERS) as executor:
        futures = [
            executor.submit(_export_camera_tile, context, index, image, mask, export_dir, rel_name)
            for index in range(len(context.cameras))
        ]
        for future in futures:
            future.result()


def _export_camera_tile(
    context: ColmapExportContext,
    camera_index: int,
    image: np.ndarray,
    mask: np.ndarray,
    export_dir: Path,
    rel_name: Path,
) -> None:
    camera = context.cameras[camera_index]
    image_path = export_dir / "images" / camera.name / rel_name
    mask_path = export_dir / "masks" / camera.name / rel_name.with_name(f"{rel_name.name}.png")
    maps = context.camera_maps[camera_index] if context.camera_maps is not None else None
    if maps is not None:
        tile = cv2.remap(image, maps.tile_map1, maps.tile_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        tile_mask = cv2.remap(mask, maps.mask_map1, maps.mask_map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    else:
        tile = equirect_to_perspective(
            image,
            camera.yaw_deg,
            camera.pitch_deg,
            context.settings.tile_size,
            context.settings.fov_deg,
            interpolation=cv2.INTER_LINEAR,
        )
        tile_mask = equirect_to_perspective(
            mask,
            camera.yaw_deg,
            camera.pitch_deg,
            context.settings.tile_size,
            context.settings.fov_deg,
            interpolation=cv2.INTER_NEAREST,
        )
    valid_mask = context.owner_masks[camera_index] & (tile_mask <= 127)
    colmap_mask = np.where(valid_mask, 255, 0).astype(np.uint8)
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


def virtual_camera_owner_mask(
    camera: VirtualCamera,
    cameras: list[VirtualCamera],
    tile_size: int,
    fov_deg: float,
) -> np.ndarray:
    world_dirs = perspective_world_directions(camera.yaw_deg, camera.pitch_deg, tile_size, fov_deg)
    return _owner_mask_from_world_dirs(camera, cameras, world_dirs)


def _owner_mask_from_world_dirs(
    camera: VirtualCamera,
    cameras: Sequence[VirtualCamera],
    world_dirs: np.ndarray,
) -> np.ndarray:
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
