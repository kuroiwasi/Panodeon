from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pytest

from panodeon.core import colmap_export
from panodeon.core.colmap_export import (
    ColmapExportContextCache,
    ColmapExportSettings,
    build_export_context,
    export_item_for_colmap,
    virtual_cameras,
    write_colmap_metadata,
)
from panodeon.core.image_io import load_mask, save_mask, save_rgb
from panodeon.core.project_state import ImageItem


def make_item(tmp_path: Path, image: np.ndarray, mask: np.ndarray) -> ImageItem:
    image_path = tmp_path / "pano.jpg"
    mask_path = tmp_path / "masks" / "pano.mask.png"
    save_rgb(image_path, image)
    save_mask(mask_path, mask)
    return ImageItem(
        path=image_path,
        relative_dir=Path("."),
        mask_path=mask_path,
        direct_mask_path=tmp_path / "unused.direct.png",
        cubemap_mask_path=tmp_path / "unused.cubemap.png",
    )


def test_export_item_for_colmap_writes_12_images_and_masks(tmp_path: Path) -> None:
    image_path = tmp_path / "pano.jpg"
    mask_path = tmp_path / "masks" / "pano.mask.png"
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[..., 0] = 128
    mask = np.full((32, 64), 255, dtype=np.uint8)
    save_rgb(image_path, image)
    save_mask(mask_path, mask)
    item = ImageItem(
        path=image_path,
        relative_dir=Path("."),
        mask_path=mask_path,
        direct_mask_path=tmp_path / "unused.direct.png",
        cubemap_mask_path=tmp_path / "unused.cubemap.png",
    )

    export_dir = tmp_path / "exports"
    export_item_for_colmap(item, export_dir, ColmapExportSettings(tile_size=16, fov_deg=90))

    images = sorted((export_dir / "images").rglob("*.jpg"))
    masks = sorted((export_dir / "masks").rglob("*.png"))
    assert len(images) == 12
    assert len(masks) == 12
    assert masks[0].name == "pano.jpg.png"
    assert load_mask(masks[0]).max() == 0


def test_export_item_for_colmap_combines_owner_and_person_masks(tmp_path: Path) -> None:
    image_path = tmp_path / "pano.jpg"
    mask_path = tmp_path / "masks" / "pano.mask.png"
    image = np.zeros((64, 128, 3), dtype=np.uint8)
    person_mask = np.zeros((64, 128), dtype=np.uint8)
    save_rgb(image_path, image)
    save_mask(mask_path, person_mask)
    item = ImageItem(
        path=image_path,
        relative_dir=Path("."),
        mask_path=mask_path,
        direct_mask_path=tmp_path / "unused.direct.png",
        cubemap_mask_path=tmp_path / "unused.cubemap.png",
    )

    export_dir = tmp_path / "exports"
    export_item_for_colmap(item, export_dir, ColmapExportSettings(tile_size=32, fov_deg=90))

    exported_masks = [load_mask(path) for path in sorted((export_dir / "masks").rglob("*.png"))]
    assert any(mask.min() == 0 and mask.max() == 255 for mask in exported_masks)


def test_export_item_for_colmap_with_context_cache_matches_default(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)
    mask = np.zeros((64, 128), dtype=np.uint8)
    mask[:, :32] = 255
    item = make_item(tmp_path, image, mask)
    settings = ColmapExportSettings(tile_size=32, fov_deg=90)

    default_dir = tmp_path / "exports_default"
    cached_dir = tmp_path / "exports_cached"
    export_item_for_colmap(item, default_dir, settings)
    context_cache = ColmapExportContextCache()
    export_item_for_colmap(item, cached_dir, settings, context_cache)
    context_cache.clear()

    default_files = sorted(path for path in default_dir.rglob("*") if path.is_file())
    cached_files = sorted(path for path in cached_dir.rglob("*") if path.is_file())
    assert [path.relative_to(default_dir) for path in default_files] == [
        path.relative_to(cached_dir) for path in cached_files
    ]
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(default_files, cached_files))


def test_export_context_skips_map_cache_above_memory_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(colmap_export, "MAX_CACHED_MAP_BYTES", 0)
    settings = ColmapExportSettings(tile_size=16, fov_deg=90)

    context = build_export_context(settings, 64, 32)
    assert context.camera_maps is None
    assert len(context.owner_masks) == len(virtual_cameras())

    image = np.full((32, 64, 3), 128, dtype=np.uint8)
    mask = np.zeros((32, 64), dtype=np.uint8)
    item = make_item(tmp_path, image, mask)
    export_dir = tmp_path / "exports"
    export_item_for_colmap(item, export_dir, settings, ColmapExportContextCache())

    assert len(sorted((export_dir / "images").rglob("*.jpg"))) == 12
    assert len(sorted((export_dir / "masks").rglob("*.png"))) == 12


def test_write_colmap_metadata_writes_rig_config(tmp_path: Path) -> None:
    write_colmap_metadata(tmp_path, ColmapExportSettings(tile_size=1024, fov_deg=90))

    payload = json.loads((tmp_path / "rig_config.json").read_text(encoding="utf-8"))
    assert len(payload[0]["cameras"]) == len(virtual_cameras())
    assert payload[0]["cameras"][0]["image_prefix"] == f"{virtual_cameras()[0].name}/"
    assert payload[0]["cameras"][0]["ref_sensor"] is True
    assert "cam_from_rig_rotation" not in payload[0]["cameras"][0]
    assert "ref_sensor" not in payload[0]["cameras"][1]
    assert "cam_from_rig_rotation" in payload[0]["cameras"][1]
    assert (tmp_path / "README_colmap.txt").exists()


def test_virtual_cameras_match_official_overlapping_panorama_layout() -> None:
    cameras = virtual_cameras()

    assert len(cameras) == 12
    assert {camera.pitch_deg for camera in cameras} == {-35.0, 0.0, 35.0}
    assert cameras[0].pitch_deg == 0.0
    assert [camera.pitch_deg for camera in cameras[:4]] == [0.0, 0.0, 0.0, 0.0]
    assert {camera.yaw_deg for camera in cameras if camera.pitch_deg == -35.0} == {0.0, 90.0, 180.0, 270.0}
    assert {camera.yaw_deg for camera in cameras if camera.pitch_deg == 35.0} == {45.0, 135.0, 225.0, 315.0}
    assert len({camera.name for camera in cameras}) == 12
