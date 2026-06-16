from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from colmap_mask.core.video_extract import extract_video_frames, is_video_path


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
