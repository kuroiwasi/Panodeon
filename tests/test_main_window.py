from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from colmap_mask.ui.main_window import (
    MainWindow,
    colmap_image_mask_counts,
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


def test_colmap_overwrite_is_checked_by_default() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.colmap_overwrite_check.isChecked()
        assert window.tile_size_spin.value() == 3072
        assert window.colmap_matcher_combo.currentText() == "pairs"
        assert window.colmap_use_gpu_check.isChecked()
        assert window.colmap_gpu_index_edit.text() == "-1"
        assert not window.colmap_skip_completed_check.isChecked()
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
