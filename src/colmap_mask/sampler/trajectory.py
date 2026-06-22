from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import TrajectoryRecord


@dataclass(frozen=True)
class PreparedTrajectory:
    records: tuple[TrajectoryRecord, ...]
    centers: np.ndarray
    path_s: np.ndarray
    candidate_mask: np.ndarray


def prepare_trajectory(
    records: list[TrajectoryRecord],
    *,
    jitter_threshold: float,
    allow_weak_tracking: bool,
) -> PreparedTrajectory:
    ordered = tuple(
        sorted(
            records,
            key=lambda record: (
                record.coordinate_group_id,
                record.segment_id,
                record.pts_seconds,
                record.frame_index,
            ),
        )
    )
    centers = np.asarray([(record.cx, record.cy, record.cz) for record in ordered], dtype=np.float64)
    finite = np.all(np.isfinite(centers), axis=1)
    pose_valid = np.asarray([record.pose_valid for record in ordered], dtype=bool) & finite
    accepted_tracking = {"good"}
    if allow_weak_tracking:
        accepted_tracking.add("weak")
    candidate_mask = np.asarray(
        [record.candidate_valid and record.tracking_state in accepted_tracking for record in ordered],
        dtype=bool,
    )
    candidate_mask &= pose_valid

    path_s = np.zeros(len(ordered), dtype=np.float64)
    by_segment: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(ordered):
        by_segment.setdefault((record.coordinate_group_id, record.segment_id), []).append(index)

    for indices in by_segment.values():
        cumulative = 0.0
        previous_valid: int | None = None
        for index in indices:
            if pose_valid[index]:
                if previous_valid is not None:
                    delta = float(np.linalg.norm(centers[index] - centers[previous_valid]))
                    if delta > jitter_threshold:
                        cumulative += delta
                previous_valid = index
            path_s[index] = cumulative

    return PreparedTrajectory(ordered, centers, path_s, candidate_mask)


def robust_extent(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return 2.0 * float(np.quantile(distances, 0.95))


def grouped_indices(trajectory: PreparedTrajectory) -> dict[int, np.ndarray]:
    result: dict[int, list[int]] = {}
    for index, record in enumerate(trajectory.records):
        result.setdefault(record.coordinate_group_id, []).append(index)
    return {group: np.asarray(indices, dtype=np.int64) for group, indices in result.items()}


def segmented_indices(trajectory: PreparedTrajectory) -> dict[tuple[int, int], np.ndarray]:
    result: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(trajectory.records):
        key = (record.coordinate_group_id, record.segment_id)
        result.setdefault(key, []).append(index)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in result.items()}

