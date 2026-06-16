from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".insv"}


def is_video_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def extract_video_frames(video_path: Path, fps: float) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not is_video_path(video_path):
        raise ValueError(f"Unsupported video: {video_path}")
    output_folder = video_path.parent / video_path.stem
    inputs_dir = output_folder / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in inputs_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if source_fps <= 0:
            source_fps = fps
        written = 0
        index = 0
        next_time = 0.0
        interval = 1.0 / fps
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_time = index / source_fps
            if frame_time + 1e-9 >= next_time:
                written += 1
                out_path = inputs_dir / f"frame_{written:06d}.jpg"
                save_bgr_jpeg(out_path, frame)
                next_time += interval
            index += 1
        if written == 0 and frame_count != 0:
            raise ValueError(f"No frames extracted: {video_path}")
    finally:
        capture.release()
    return output_folder


def save_bgr_jpeg(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError(f"Cannot encode frame: {path}")
    data.tofile(str(path))
