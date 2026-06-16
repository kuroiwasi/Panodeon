from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .image_io import find_images, mask_path_for


@dataclass
class ImageItem:
    path: Path
    relative_dir: Path
    mask_path: Path
    direct_mask_path: Path
    cubemap_mask_path: Path


@dataclass
class ProjectState:
    folder: Path | None = None
    mask_dir: Path | None = None
    export_dir: Path | None = None
    images: list[ImageItem] = field(default_factory=list)

    def load_folder(self, folder: Path) -> None:
        self.folder = folder
        self.mask_dir = folder / "masks"
        self.export_dir = folder / "exports"
        images: list[ImageItem] = []
        for path in find_images(folder, exclude_dirs=(self.mask_dir, self.export_dir)):
            relative_dir = path.parent.relative_to(folder)
            mask_dir = self.mask_dir / relative_dir
            images.append(
                ImageItem(
                    path=path,
                    relative_dir=relative_dir,
                    mask_path=mask_path_for(path, mask_dir),
                    direct_mask_path=mask_path_for(path, mask_dir, "direct"),
                    cubemap_mask_path=mask_path_for(path, mask_dir, "cubemap"),
                )
            )
        self.images = images
