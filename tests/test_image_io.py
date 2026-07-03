from __future__ import annotations

from pathlib import Path

from panodeon.core.image_io import find_images, is_mask_image_path


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_find_images_excludes_mask_files_and_mask_dirs(tmp_path: Path) -> None:
    touch(tmp_path / "pano.jpg")
    touch(tmp_path / "pano.mask.png")
    touch(tmp_path / "pano.mask.direct.png")
    touch(tmp_path / "pano.mask.cubemap.png")
    touch(tmp_path / "nested" / "frame.png")
    touch(tmp_path / "nested" / "frame.mask.png")
    touch(tmp_path / "masks" / "pano.png")
    touch(tmp_path / "mask" / "pano.jpg")

    actual = {path.relative_to(tmp_path).as_posix() for path in find_images(tmp_path)}

    assert actual == {"nested/frame.png", "pano.jpg"}


def test_is_mask_image_path_detects_saved_mask_variants(tmp_path: Path) -> None:
    assert is_mask_image_path(tmp_path / "pano.mask.png")
    assert is_mask_image_path(tmp_path / "pano.mask.direct.png")
    assert is_mask_image_path(tmp_path / "pano.mask.cubemap.png")
    assert not is_mask_image_path(tmp_path / "pano.png")
