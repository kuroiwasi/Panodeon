from __future__ import annotations

import os
import sqlite3
import struct
from pathlib import Path

from PySide6.QtWidgets import QApplication

from colmap_mask.ui.main_window import (
    MainWindow,
    colmap_image_mask_counts,
    colmap_pipeline_status,
    colmap_registered_image_count,
    colmap_run_start_text,
    default_colmap_executable,
    dropped_path_from_mime,
    folder_from_mime,
    short_error,
)


def test_short_error_limits_multiline_message() -> None:
    message = short_error(RuntimeError("line1\n" + "x" * 300), limit=20)
    assert "\n" not in message
    assert len(message) <= 23


def test_folder_from_mime_accepts_directory(tmp_path: Path) -> None:
    assert folder_from_mime(FakeMime([FakeUrl(tmp_path)])) == tmp_path


def test_folder_from_mime_accepts_image_parent(tmp_path: Path) -> None:
    image_path = tmp_path / "pano.jpg"
    image_path.write_bytes(b"")
    assert folder_from_mime(FakeMime([FakeUrl(image_path)])) == tmp_path


def test_folder_from_mime_ignores_non_images(tmp_path: Path) -> None:
    text_path = tmp_path / "note.txt"
    text_path.write_text("")
    assert folder_from_mime(FakeMime([FakeUrl(text_path)])) is None


def test_dropped_path_from_mime_accepts_video(tmp_path: Path) -> None:
    video_path = tmp_path / "walk.mp4"
    video_path.write_bytes(b"")
    assert dropped_path_from_mime(FakeMime([FakeUrl(video_path)])) == video_path
    assert folder_from_mime(FakeMime([FakeUrl(video_path)])) is None


def test_colmap_image_mask_counts_matching_masks(tmp_path: Path) -> None:
    images_dir = tmp_path / "exports" / "images"
    masks_dir = tmp_path / "exports" / "masks"
    image_a = images_dir / "cam01" / "frame_000001.jpg"
    image_b = images_dir / "cam01" / "frame_000002.jpg"
    image_a.parent.mkdir(parents=True)
    image_a.write_bytes(b"")
    image_b.write_bytes(b"")
    mask_a = masks_dir / "cam01" / "frame_000001.jpg.png"
    mask_a.parent.mkdir(parents=True)
    mask_a.write_bytes(b"")

    assert colmap_image_mask_counts(images_dir, masks_dir) == (2, 1)


def test_colmap_registered_image_count_uses_latest_snapshot(tmp_path: Path) -> None:
    sparse_model = tmp_path / "exports" / "sparse" / "0"
    snapshot_model = tmp_path / "exports" / "snapshots" / "50"
    sparse_model.mkdir(parents=True)
    snapshot_model.mkdir(parents=True)
    (sparse_model / "images.bin").write_bytes(struct.pack("<Q", 12))
    (snapshot_model / "images.bin").write_bytes(struct.pack("<Q", 50))
    os.utime(sparse_model / "images.bin", (100, 100))
    os.utime(snapshot_model / "images.bin", (200, 200))

    assert colmap_registered_image_count(tmp_path / "exports") == 50


def test_colmap_pipeline_status_detects_resume_step(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    images_dir = export_dir / "images" / "cam01"
    masks_dir = export_dir / "masks" / "cam01"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    (images_dir / "frame_000001.jpg").write_bytes(b"")
    (masks_dir / "frame_000001.jpg.png").write_bytes(b"")
    (export_dir / "rig_config.json").write_text("{}", encoding="utf-8")
    with sqlite3.connect(export_dir / "database.db") as connection:
        connection.execute("CREATE TABLE images(image_id INTEGER)")
        connection.execute("INSERT INTO images VALUES (1)")
        connection.execute("CREATE TABLE frames(frame_id INTEGER)")
        connection.execute("INSERT INTO frames VALUES (1)")

    status = colmap_pipeline_status(export_dir)

    assert status.export_ready
    assert status.feature_done
    assert status.rig_done
    assert not status.matching_done
    assert status.next_step == "Feature matching"
    assert colmap_run_start_text(status, overwrite=False, skip_completed=True) == "Feature matching"
    assert colmap_run_start_text(status, overwrite=True, skip_completed=False) == "Feature extraction (overwrite)"


def test_colmap_pipeline_status_detects_rig_ba_step(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    images_dir = export_dir / "images" / "cam01"
    masks_dir = export_dir / "masks" / "cam01"
    sparse_model = export_dir / "sparse" / "0"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    sparse_model.mkdir(parents=True)
    (images_dir / "frame_000001.jpg").write_bytes(b"")
    (masks_dir / "frame_000001.jpg.png").write_bytes(b"")
    (export_dir / "rig_config.json").write_text("{}", encoding="utf-8")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse_model / name).write_bytes(struct.pack("<Q", 1) if name == "images.bin" else b"")
    with sqlite3.connect(export_dir / "database.db") as connection:
        connection.execute("CREATE TABLE images(image_id INTEGER)")
        connection.execute("INSERT INTO images VALUES (1)")
        connection.execute("CREATE TABLE frames(frame_id INTEGER)")
        connection.execute("INSERT INTO frames VALUES (1)")
        connection.execute("CREATE TABLE matches(pair_id INTEGER)")
        connection.execute("INSERT INTO matches VALUES (1)")

    status = colmap_pipeline_status(export_dir, rig_ba_enabled=True)

    assert status.sparse_done
    assert not status.rig_ba_done
    assert status.next_step == "Rig bundle adjustment"


def test_colmap_overwrite_is_checked_by_default() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.colmap_overwrite_check.isChecked()
        assert window.tile_size_spin.value() == 3072
        assert window.colmap_matcher_combo.currentText() == "sequential"
        assert window.colmap_sparse_mapper_combo.currentText() == "mapper"
        assert window.colmap_use_gpu_check.isChecked()
        assert window.colmap_gpu_index_edit.text() == "-1"
        assert window.colmap_snapshot_check.isChecked()
        assert window.colmap_snapshot_freq_spin.value() == 50
        assert not window.colmap_skip_completed_check.isChecked()
        assert not window.colmap_rig_ba_check.isChecked()
        assert not window.colmap_dense_check.isChecked()
        assert window.colmap_path_edit.text() == default_colmap_executable()
    finally:
        window.close()


def test_default_colmap_executable_returns_string() -> None:
    assert default_colmap_executable()


class FakeMime:
    def __init__(self, urls: list["FakeUrl"]) -> None:
        self._urls = urls

    def hasUrls(self) -> bool:  # noqa: N802
        return bool(self._urls)

    def urls(self) -> list["FakeUrl"]:
        return self._urls


class FakeUrl:
    def __init__(self, path: Path, local: bool = True) -> None:
        self._path = path
        self._local = local

    def isLocalFile(self) -> bool:  # noqa: N802
        return self._local

    def toLocalFile(self) -> str:  # noqa: N802
        return str(self._path)
