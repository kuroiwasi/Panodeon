from __future__ import annotations

import bisect
import csv
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from ..io import atomic_text_writer
from ..models import TrajectoryRecord
from ..video import VideoFrameRecord


@dataclass(frozen=True)
class TumPose:
    timestamp: float
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass(frozen=True)
class StellaRunConfig:
    executable: Path
    vocabulary: Path
    camera_config: Path
    frame_skip: int = 1
    log_level: str = "info"


def write_default_equirectangular_config(path: Path, fps: float) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    with atomic_text_writer(path) as handle:
        handle.write(
            f'''# Generated speed-oriented default.
Camera:
  name: "360 Camera 2K"
  setup: "monocular"
  model: "equirectangular"
  fps: {fps:.9g}
  cols: 1920
  rows: 960
  color_order: "BGR"

Preprocessing:
  min_size: 800

Feature:
  name: "default ORB feature extraction setting"
  scale_factor: 1.2
  num_levels: 8
  ini_fast_threshold: 20
  min_fast_threshold: 7

Mapping:
  backend: "g2o"
  baseline_dist_thr_ratio: 0.02
  redundant_obs_ratio_thr: 0.95
  num_covisibilities_for_landmark_generation: 20
  num_covisibilities_for_landmark_fusion: 20
  residual_deg_thr: 0.4

Tracking:
  backend: "g2o"

LoopDetector:
  backend: "g2o"
  enabled: true
  reject_by_graph_distance: true
  min_distance_on_graph: 50

GraphOptimizer:
  min_num_shared_lms: 200

GlobalOptimizer:
  thr_neighbor_keyframes: 100

System:
  map_format: "msgpack"
  num_grid_cols: 96
  num_grid_rows: 48
'''
        )


def run_stella_video(video_path: Path, work_dir: Path, config: StellaRunConfig) -> Path:
    for path, label in (
        (video_path, "video"),
        (config.executable, "stella executable"),
        (config.vocabulary, "ORB vocabulary"),
        (config.camera_config, "camera config"),
    ):
        if not path.is_file():
            raise ValueError(f"Missing {label}: {path}")
    if config.frame_skip <= 0:
        raise ValueError("frame_skip must be positive")
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.executable.resolve()),
        "--vocab",
        str(config.vocabulary.resolve()),
        "--video",
        str(video_path.resolve()),
        "--config",
        str(config.camera_config.resolve()),
        "--frame-skip",
        str(config.frame_skip),
        "--no-sleep",
        "--wait-loop-ba",
        "--eval-log-dir",
        str(work_dir.resolve()),
        "--start-timestamp",
        "0.0",
        "--viewer",
        "none",
        "--log-level",
        config.log_level,
    ]
    process = subprocess.run(command, cwd=work_dir, text=True, capture_output=True, check=False)
    with atomic_text_writer(work_dir / "stella_stdout.log") as handle:
        handle.write(process.stdout)
    with atomic_text_writer(work_dir / "stella_stderr.log") as handle:
        handle.write(process.stderr)
    if process.returncode != 0:
        tail = "\n".join(process.stderr.splitlines()[-20:])
        raise RuntimeError(f"stella_vslam failed with exit code {process.returncode}\n{tail}")
    trajectory_path = work_dir / "frame_trajectory.txt"
    if not trajectory_path.is_file():
        raise RuntimeError("stella_vslam did not produce frame_trajectory.txt")
    return trajectory_path


def load_tum_trajectory(path: Path) -> list[TumPose]:
    poses: list[TumPose] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 8:
            raise ValueError(f"Invalid TUM trajectory line {line_number}: expected 8 fields")
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"Invalid TUM trajectory line {line_number}: {exc}") from exc
        poses.append(TumPose(*values))
    if not poses:
        raise ValueError("TUM trajectory is empty")
    poses.sort(key=lambda pose: pose.timestamp)
    return poses


def align_stella_trajectory(
    poses: list[TumPose],
    frames: list[VideoFrameRecord],
    *,
    split_gap_seconds: float = 0.5,
    max_timestamp_error_seconds: float | None = None,
) -> list[TrajectoryRecord]:
    if split_gap_seconds <= 0:
        raise ValueError("split_gap_seconds must be positive")
    if not frames:
        raise ValueError("Video frame index is empty")
    first_timestamp = frames[0].timestamp
    relative_times = [float(frame.timestamp - first_timestamp) for frame in frames]
    if max_timestamp_error_seconds is None:
        deltas = [right - left for left, right in zip(relative_times, relative_times[1:]) if right > left]
        median_delta = sorted(deltas)[len(deltas) // 2] if deltas else 0.0
        max_timestamp_error_seconds = max(0.05, median_delta * 0.55)
    if max_timestamp_error_seconds <= 0:
        raise ValueError("max_timestamp_error_seconds must be positive")

    matches: dict[int, tuple[float, TumPose]] = {}
    for pose in poses:
        index = _nearest_index(relative_times, pose.timestamp)
        error = abs(relative_times[index] - pose.timestamp)
        if error > max_timestamp_error_seconds:
            continue
        previous = matches.get(index)
        if previous is None or error < previous[0]:
            matches[index] = (error, pose)
    if not matches:
        raise ValueError("No stella_vslam poses matched video PTS")

    records: list[TrajectoryRecord] = []
    segment_id = 0
    previous_timestamp: float | None = None
    for frame_list_index in sorted(matches):
        pose = matches[frame_list_index][1]
        if previous_timestamp is not None and pose.timestamp - previous_timestamp > split_gap_seconds:
            segment_id += 1
        previous_timestamp = pose.timestamp
        frame = frames[frame_list_index]
        records.append(
            TrajectoryRecord(
                frame_index=frame.frame_index,
                pts=frame.pts,
                time_base_num=frame.time_base_num,
                time_base_den=frame.time_base_den,
                segment_id=segment_id,
                coordinate_group_id=0,
                cx=pose.tx,
                cy=pose.ty,
                cz=pose.tz,
                qw=pose.qw,
                qx=pose.qx,
                qy=pose.qy,
                qz=pose.qz,
            )
        )
    return records


def remap_trajectory_frame_timing(
    records: list[TrajectoryRecord],
    source_frames: list[VideoFrameRecord],
) -> list[TrajectoryRecord]:
    by_index = {frame.frame_index: frame for frame in source_frames}
    remapped: list[TrajectoryRecord] = []
    for record in records:
        try:
            frame = by_index[record.frame_index]
        except KeyError as exc:
            raise ValueError(
                f"Trajectory frame is absent from source frame index: {record.frame_index}"
            ) from exc
        remapped.append(
            replace(
                record,
                pts=frame.pts,
                time_base_num=frame.time_base_num,
                time_base_den=frame.time_base_den,
            )
        )
    return remapped


def write_trajectory_csv(path: Path, records: list[TrajectoryRecord]) -> None:
    fields = [
        "frame_index",
        "pts",
        "time_base_num",
        "time_base_den",
        "segment_id",
        "coordinate_group_id",
        "cx",
        "cy",
        "cz",
        "qw",
        "qx",
        "qy",
        "qz",
        "pose_valid",
        "tracking_state",
        "candidate_valid",
        "quality_score",
    ]
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in fields})


def _nearest_index(values: list[float], target: float) -> int:
    right = bisect.bisect_left(values, target)
    if right == 0:
        return 0
    if right == len(values):
        return len(values) - 1
    left = right - 1
    return left if target - values[left] <= values[right] - target else right

