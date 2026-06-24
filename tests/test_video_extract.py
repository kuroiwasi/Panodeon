from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from panodeon.core.video_extract import extract_video_frames, is_video_path


def test_is_video_path_accepts_video_extensions(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"")
    assert is_video_path(path)


def test_extract_video_frames_writes_inputs_folder(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        4.0,
        (16, 12),
    )
    assert writer.isOpened()
    try:
        for index in range(8):
            frame = np.full((12, 16, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    output_folder = extract_video_frames(video_path, fps=2.0)

    frames = sorted((output_folder / "inputs").glob("frame_*.jpg"))
    assert output_folder == tmp_path / "clip"
    assert len(frames) == 4
    assert frames[0].name == "frame_000001.jpg"


def test_extract_video_frames_quality_selects_sharpest_frame_per_interval(tmp_path: Path) -> None:
    video_path = tmp_path / "quality.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        4.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for index in range(8):
            frame = np.full((48, 64, 3), 120, dtype=np.uint8)
            if index in (2, 5):
                frame = checkerboard_frame(64, 48)
            writer.write(frame)
    finally:
        writer.release()

    output_folder = extract_video_frames(video_path, fps=1.0, quality_select=True)

    frames = sorted((output_folder / "inputs").glob("frame_*.jpg"))
    assert len(frames) == 2
    with (output_folder / "inputs" / "frame_quality.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [round(float(row["source_time_sec"]), 2) for row in rows] == [0.5, 1.25]


def checkerboard_frame(width: int, height: int) -> np.ndarray:
    y, x = np.indices((height, width))
    board = ((x // 4 + y // 4) % 2 * 255).astype(np.uint8)
    return np.repeat(board[:, :, None], 3, axis=2)
