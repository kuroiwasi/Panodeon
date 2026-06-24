from __future__ import annotations

import threading
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from panodeon.core.mask_ops import postprocess_mask
from panodeon.projection.cubemap import CUBE_FACES, accumulate_face_mask, equirect_to_face

from .base import JobCancelled, MaskOptions, MaskResult, Segmenter

CubemapProgress = Callable[[int, int, str], None]


class CubemapGenerator:
    def __init__(self, segmenter: Segmenter | None = None) -> None:
        self.segmenter = segmenter

    def generate(
        self,
        image: np.ndarray,
        options: MaskOptions,
        progress: CubemapProgress | None = None,
        cancel_event: threading.Event | None = None,
    ) -> MaskResult:
        def check_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled("cubemap job cancelled")

        check_cancelled()
        start = perf_counter()
        out_shape = image.shape[:2]
        merged = np.zeros(out_shape, dtype=np.uint8)
        face_times: dict[str, float] = {}

        total_faces = len(CUBE_FACES)
        face_images = []

        # 1. 投影および推論のフェーズ
        if self.segmenter is None:
            # セグメンターなし（Stub）
            for index, face in enumerate(CUBE_FACES, start=1):
                if progress is not None:
                    progress(index, total_faces, face)
                face_start = perf_counter()
                face_image = equirect_to_face(
                    image,
                    face=face,
                    face_size=options.cube_face_size,
                    fov_deg=options.cube_fov_deg,
                )
                face_images.append(face_image)
                face_times[face] = perf_counter() - face_start
            face_masks = [np.zeros(img.shape[:2], dtype=np.uint8) for img in face_images]
        else:
            predict_masks_fn = getattr(self.segmenter, "predict_masks", None)
            # モデルのバッチサポート判定
            first_dim = self.segmenter.inputs[0].shape[0]
            is_batch_supported = not isinstance(first_dim, int) or first_dim < 0 or first_dim > 1

            if predict_masks_fn is not None and is_batch_supported:
                # バッチ処理
                for face in CUBE_FACES:
                    face_start = perf_counter()
                    face_image = equirect_to_face(
                        image,
                        face=face,
                        face_size=options.cube_face_size,
                        fov_deg=options.cube_fov_deg,
                    )
                    face_images.append(face_image)
                    face_times[face] = perf_counter() - face_start

                if progress is not None:
                    progress(0, total_faces, "inferring batch")
                face_masks = predict_masks_fn(face_images, options)
            else:
                # 逐次処理（フォールバック）
                face_masks = []
                for index, face in enumerate(CUBE_FACES, start=1):
                    check_cancelled()
                    if progress is not None:
                        progress(index, total_faces, face)
                    face_start = perf_counter()
                    face_image = equirect_to_face(
                        image,
                        face=face,
                        face_size=options.cube_face_size,
                        fov_deg=options.cube_fov_deg,
                    )
                    face_times[face] = perf_counter() - face_start
                    
                    face_mask = self.segmenter.predict_mask(face_image, options)
                    face_masks.append(face_mask)

        # 2. 得られたマスクを逆投影してマージする
        check_cancelled()
        for face, face_mask in zip(CUBE_FACES, face_masks):
            face_start = perf_counter()
            accumulate_face_mask(
                merged,
                face_mask,
                face=face,
                fov_deg=options.cube_fov_deg,
            )
            face_times[face] += perf_counter() - face_start

        if self.segmenter is not None:
            merged = postprocess_mask(merged, options.dilate_px, options.feather_px)
        return MaskResult(
            mask=merged.astype(np.uint8),
            strategy="cubemap",
            elapsed_sec=perf_counter() - start,
            metadata={"stub": self.segmenter is None, "face_times": face_times},
        )


def merge_compare_masks(direct: np.ndarray, cubemap: np.ndarray) -> np.ndarray:
    h = max(direct.shape[0], cubemap.shape[0])
    w = max(direct.shape[1], cubemap.shape[1])
    direct = cv2.resize(direct, (w, h), interpolation=cv2.INTER_NEAREST)
    cubemap = cv2.resize(cubemap, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.maximum(direct, cubemap)
