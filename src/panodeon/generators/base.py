from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class JobCancelled(RuntimeError):
    """Raised when a mask generation job is cancelled by the user."""


@dataclass
class MaskOptions:
    strategy: str = "direct"
    selected_classes: tuple[str, ...] = ("body", "head", "face", "hand", "foot")
    score_threshold: float = 0.35
    dilate_px: int = 8
    feather_px: int = 5
    cube_face_size: int = 1024
    cube_fov_deg: float = 105.0
    mask_threshold: float = 0.4


@dataclass
class MaskResult:
    mask: np.ndarray
    strategy: str
    elapsed_sec: float
    metadata: dict[str, object] = field(default_factory=dict)


class Segmenter(Protocol):
    def predict_mask(self, image: np.ndarray, options: MaskOptions) -> np.ndarray:
        ...

    def predict_masks(self, images: list[np.ndarray], options: MaskOptions) -> list[np.ndarray]:
        ...


class MaskGenerator(Protocol):
    def generate(self, image: np.ndarray, options: MaskOptions) -> MaskResult:
        ...
