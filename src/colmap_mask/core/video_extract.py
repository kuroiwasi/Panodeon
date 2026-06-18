from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".insv"}


def is_video_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def extract_video_frames(video_path: Path, fps: float, quality_select: bool = False) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not is_video_path(video_path):
        raise ValueError(f"Unsupported video: {video_path}")
    output_folder = video_path.parent / video_path.stem
    inputs_dir = output_folder / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in inputs_dir.glob("frame_*.jpg"):
        old_frame.unlink()
    quality_csv = inputs_dir / "frame_quality.csv"
    if quality_csv.exists():
        quality_csv.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if source_fps <= 0:
            source_fps = fps
        if quality_select:
            written = extract_best_frames_by_quality(capture, inputs_dir, source_fps, fps)
        else:
            written = extract_frames_by_interval(capture, inputs_dir, source_fps, fps)
        if written == 0 and frame_count != 0:
            raise ValueError(f"No frames extracted: {video_path}")
    finally:
        capture.release()
    return output_folder


def extract_frames_by_interval(capture: cv2.VideoCapture, inputs_dir: Path, source_fps: float, fps: float) -> int:
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
            save_bgr_jpeg(inputs_dir / f"frame_{written:06d}.jpg", frame)
            next_time += interval
        index += 1
    return written


def extract_best_frames_by_quality(capture: cv2.VideoCapture, inputs_dir: Path, source_fps: float, fps: float) -> int:
    interval = 1.0 / fps
    written = 0
    index = 0
    current_window = -1
    best_frame: np.ndarray | None = None
    best_score = -math.inf
    best_time = 0.0
    rows: list[tuple[int, float, float]] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_time = index / source_fps
        window = int(frame_time / interval)
        if current_window >= 0 and window != current_window and best_frame is not None:
            written += 1
            save_bgr_jpeg(inputs_dir / f"frame_{written:06d}.jpg", best_frame)
            rows.append((written, best_time, best_score))
            best_frame = None
            best_score = -math.inf
        current_window = window
        score = frame_quality_score(frame)
        if score > best_score:
            best_frame = frame.copy()
            best_score = score
            best_time = frame_time
        index += 1
    if best_frame is not None:
        written += 1
        save_bgr_jpeg(inputs_dir / f"frame_{written:06d}.jpg", best_frame)
        rows.append((written, best_time, best_score))
    write_quality_csv(inputs_dir / "frame_quality.csv", rows)
    return written


def frame_quality_score(frame: np.ndarray) -> float:
    band = middle_latitude_band(frame)
    if max(band.shape[:2]) > 1024:
        scale = 1024.0 / max(band.shape[:2])
        band = cv2.resize(band, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    sharpness = math.log1p(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    contrast = float(gray.std()) / 64.0
    dark_clip = float(np.mean(gray <= 5))
    bright_clip = float(np.mean(gray >= 250))
    exposure = max(0.0, 1.0 - dark_clip - bright_clip)
    feature_score = min(1.0, count_orb_features(gray) / 1000.0)
    return sharpness + 2.0 * feature_score + contrast + exposure


def middle_latitude_band(frame: np.ndarray) -> np.ndarray:
    height = frame.shape[0]
    top = max(0, height // 6)
    bottom = min(height, height - top)
    return frame[top:bottom]


def count_orb_features(gray: np.ndarray) -> int:
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints = orb.detect(gray, None)
    return len(keypoints)


def write_quality_csv(path: Path, rows: list[tuple[int, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "source_time_sec", "quality_score"])
        writer.writerows((frame, f"{time_sec:.6f}", f"{score:.6f}") for frame, time_sec, score in rows)


def save_bgr_jpeg(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError(f"Cannot encode frame: {path}")
    data.tofile(str(path))
