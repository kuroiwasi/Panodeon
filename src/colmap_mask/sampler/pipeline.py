from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .models import SamplerConfig, SelectionRecord, TrajectoryRecord
from .orb import OverlapEvaluator
from .sampling import FpsSelection, radius_fps, target_count_fps
from .trajectory import PreparedTrajectory, grouped_indices, prepare_trajectory, robust_extent, segmented_indices


@dataclass(frozen=True)
class SamplingResult:
    selections: list[SelectionRecord]
    report: dict[str, Any]


@dataclass(frozen=True)
class _PendingSelection:
    index: int
    selection_type: str
    nearest_distance: float | None
    radius: float


def sample_trajectory(
    records: list[TrajectoryRecord],
    config: SamplerConfig,
    overlap_evaluator: OverlapEvaluator | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> SamplingResult:
    config.validate()
    _notify_progress(progress, "候補準備", 0.45)
    trajectory = prepare_trajectory(
        records,
        jitter_threshold=config.trajectory.jitter_threshold,
        allow_weak_tracking=config.sampling.allow_weak_tracking,
    )
    if not np.any(trajectory.candidate_mask):
        raise ValueError("Trajectory has no valid sampling candidates")

    groups = grouped_indices(trajectory)
    if config.sampling.scale_mode == "target_count" and len(groups) != 1:
        raise ValueError("target_count mode currently requires one coordinate_group_id")

    quality = np.asarray([record.quality_score for record in trajectory.records], dtype=np.float64)
    frame_indices = np.asarray([record.frame_index for record in trajectory.records], dtype=np.int64)
    pending: list[_PendingSelection] = []
    spatial_selected: set[int] = set()
    group_reports: dict[str, Any] = {}
    radii: dict[int, float] = {}

    _notify_progress(progress, "空間被覆選定", 0.55)
    for group_id in sorted(groups):
        group_indices = groups[group_id]
        valid_indices = group_indices[trajectory.candidate_mask[group_indices]]
        if len(valid_indices) == 0:
            continue
        seeds = _endpoint_seeds(trajectory, group_id, config)
        for seed in seeds:
            if seed not in spatial_selected:
                pending.append(_PendingSelection(seed, "endpoint", None, 0.0))
                spatial_selected.add(seed)

        if config.sampling.scale_mode == "metric":
            radius = float(config.sampling.radius)
            fps = radius_fps(
                trajectory.centers,
                valid_indices,
                radius=radius,
                seed_indices=seeds,
                quality=quality,
                frame_indices=frame_indices,
                distance_tolerance=config.sampling.distance_tolerance,
                tie_distance_ratio=config.sampling.tie_distance_ratio,
            )
        elif config.sampling.scale_mode == "robust_extent":
            scale = robust_extent(trajectory.centers[valid_indices])
            radius = scale * config.sampling.radius_ratio
            fps = radius_fps(
                trajectory.centers,
                valid_indices,
                radius=radius,
                seed_indices=seeds,
                quality=quality,
                frame_indices=frame_indices,
                distance_tolerance=config.sampling.distance_tolerance,
                tie_distance_ratio=config.sampling.tie_distance_ratio,
            )
        else:
            fps, radius = target_count_fps(
                trajectory.centers,
                valid_indices,
                target_count=int(config.sampling.target_count),
                seed_indices=seeds,
                quality=quality,
                frame_indices=frame_indices,
                tie_distance_ratio=config.sampling.tie_distance_ratio,
            )

        radii[group_id] = radius
        for addition in fps.additions:
            if addition.index not in spatial_selected:
                pending.append(_from_fps(addition, radius))
                spatial_selected.add(addition.index)
        group_reports[str(group_id)] = {
            "candidate_count": int(len(valid_indices)),
            "radius": radius,
            "max_cover_distance": fps.max_cover_distance,
        }

    if config.continuity.enabled:
        continuity_stage = (
            "ORB接続判定・bridge追加"
            if overlap_evaluator is not None
            else "軌跡間隔bridge追加"
        )
        _notify_progress(progress, continuity_stage, 0.70)
        bridges, unresolved, unresolved_orb = _path_bridges(
            trajectory,
            spatial_selected,
            radii,
            config,
            overlap_evaluator,
            progress,
        )
        for bridge in bridges:
            pending.append(bridge)
            spatial_selected.add(bridge.index)
    else:
        unresolved = 0
        unresolved_orb = 0

    _notify_progress(progress, "レポート生成", 0.85)
    selections = [_to_record(order, item, trajectory) for order, item in enumerate(pending)]
    report = _build_report(
        trajectory,
        pending,
        radii,
        group_reports,
        unresolved,
        unresolved_orb,
        overlap_evaluator,
    )
    return SamplingResult(selections, report)


def _notify_progress(
    callback: Callable[[str, float], None] | None,
    stage_name: str,
    progress: float,
) -> None:
    if callback is not None:
        callback(stage_name, progress)


def _notify_bridge_progress(
    callback: Callable[[str, float], None] | None,
    pair_number: int,
    total_pairs: int,
) -> None:
    if callback is None or total_pairs <= 0:
        return
    fraction = pair_number / total_pairs
    progress = 0.70 + (0.85 - 0.70) * fraction
    callback(f"bridgeマッチング {pair_number}/{total_pairs} 被覆点間", progress)


def _endpoint_seeds(trajectory: PreparedTrajectory, group_id: int, config: SamplerConfig) -> list[int]:
    seeds: list[int] = []
    for (candidate_group, _segment_id), indices in sorted(segmented_indices(trajectory).items()):
        if candidate_group != group_id:
            continue
        valid = [int(index) for index in indices if trajectory.candidate_mask[index]]
        if not valid:
            continue
        if config.sampling.force_first_frame:
            seeds.append(valid[0])
        if config.sampling.force_segment_endpoints and valid[-1] != valid[0]:
            seeds.append(valid[-1])
    return list(dict.fromkeys(seeds))


def _from_fps(selection: FpsSelection, radius: float) -> _PendingSelection:
    return _PendingSelection(selection.index, "coverage", selection.nearest_distance, radius)


def _path_bridges(
    trajectory: PreparedTrajectory,
    selected: set[int],
    radii: dict[int, float],
    config: SamplerConfig,
    overlap_evaluator: OverlapEvaluator | None,
    progress: Callable[[str, float], None] | None = None,
) -> tuple[list[_PendingSelection], int, int]:
    additions: list[_PendingSelection] = []
    unresolved = 0
    unresolved_orb = 0
    selected_with_bridges = set(selected)
    segments = sorted(segmented_indices(trajectory).items())
    anchors_by_segment = {
        key: sorted(
            (int(index) for index in indices if int(index) in selected),
            key=lambda index: trajectory.path_s[index],
        )
        for key, indices in segments
    }
    total_pairs = sum(max(len(anchors) - 1, 0) for anchors in anchors_by_segment.values())
    pair_number = 0
    for (group_id, _segment_id), indices in segments:
        max_gap = radii.get(group_id, 0.0) * config.continuity.max_path_gap_ratio
        if max_gap <= 0:
            continue
        anchors = anchors_by_segment[(group_id, _segment_id)]
        for left, right in zip(anchors, anchors[1:]):
            pair_number += 1
            _notify_bridge_progress(progress, pair_number, total_pairs)
            candidates = [
                int(index)
                for index in indices
                if trajectory.candidate_mask[index]
                and int(index) not in selected_with_bridges
                and trajectory.path_s[left] < trajectory.path_s[index] < trajectory.path_s[right]
            ]
            path = None
            if overlap_evaluator is not None:
                path = _orb_connection_path(
                    trajectory,
                    left,
                    right,
                    candidates,
                    radii[group_id],
                    selected_with_bridges,
                    overlap_evaluator,
                    config,
                )
            if path is None:
                if overlap_evaluator is not None:
                    unresolved_orb += 1
                bridge_indices, gap_unresolved = _path_gap_bridge_indices(
                    trajectory,
                    left,
                    right,
                    candidates,
                    max_gap,
                    radii[group_id]
                    * min(config.continuity.bridge_min_separation_ratios),
                    selected_with_bridges,
                )
                unresolved += gap_unresolved
            else:
                bridge_indices = path[1:-1]

            for index in bridge_indices:
                if index in selected_with_bridges:
                    continue
                distance = _distance_to_selected(
                    trajectory.centers, index, selected_with_bridges
                )
                additions.append(
                    _PendingSelection(index, "bridge_path", distance, radii[group_id])
                )
                selected_with_bridges.add(index)
    return additions, unresolved, unresolved_orb


def _orb_connection_path(
    trajectory: PreparedTrajectory,
    left: int,
    right: int,
    candidates: list[int],
    radius: float,
    selected: set[int],
    evaluator: OverlapEvaluator,
    config: SamplerConfig,
) -> list[int] | None:
    direct = _orb_connected(trajectory, left, right, evaluator)
    if direct:
        return [left, right]

    for separation_ratio in config.continuity.bridge_min_separation_ratios:
        search_budget = [_MAX_RECURSIVE_ORB_STATES]
        path = _recursive_orb_path(
            trajectory,
            left,
            right,
            candidates,
            radius * separation_ratio,
            selected,
            set(),
            evaluator,
            config,
            0,
            search_budget,
        )
        if path is not None:
            return path
    return None


def _recursive_orb_path(
    trajectory: PreparedTrajectory,
    left: int,
    right: int,
    candidates: list[int],
    minimum_distance: float,
    selected: set[int],
    reserved: set[int],
    evaluator: OverlapEvaluator,
    config: SamplerConfig,
    depth: int,
    search_budget: list[int],
) -> list[int] | None:
    search_budget[0] -= 1
    if search_budget[0] < 0:
        return None
    if _orb_connected(trajectory, left, right, evaluator):
        return [left, right]
    if depth >= config.continuity.bridge_max_recursion_depth:
        return None

    middle_candidates = _midpoint_candidates(
        trajectory,
        left,
        right,
        candidates,
        selected | reserved,
        minimum_distance,
    )
    for middle in middle_candidates:
        middle_path_s = trajectory.path_s[middle]
        left_candidates = [
            index for index in candidates if trajectory.path_s[index] < middle_path_s
        ]
        right_candidates = [
            index for index in candidates if middle_path_s < trajectory.path_s[index]
        ]
        next_reserved = reserved | {middle}
        left_path = _recursive_orb_path(
            trajectory,
            left,
            middle,
            left_candidates,
            minimum_distance,
            selected,
            next_reserved,
            evaluator,
            config,
            depth + 1,
            search_budget,
        )
        if left_path is None:
            continue
        left_bridges = set(left_path[1:-1])
        right_path = _recursive_orb_path(
            trajectory,
            middle,
            right,
            right_candidates,
            minimum_distance,
            selected,
            next_reserved | left_bridges,
            evaluator,
            config,
            depth + 1,
            search_budget,
        )
        if right_path is not None:
            return [*left_path[:-1], *right_path]
    return None


def _midpoint_candidates(
    trajectory: PreparedTrajectory,
    left: int,
    right: int,
    candidates: list[int],
    selected: set[int],
    minimum_distance: float,
) -> list[int]:
    midpoint = (trajectory.path_s[left] + trajectory.path_s[right]) * 0.5
    center_outward = sorted(
        candidates,
        key=lambda index: (
            abs(trajectory.path_s[index] - midpoint),
            trajectory.records[index].frame_index,
        ),
    )
    accepted: list[int] = []
    for index in center_outward:
        references = selected | set(accepted)
        distance = _distance_to_selected(trajectory.centers, index, references)
        if distance is not None and distance + 1.0e-12 < minimum_distance:
            continue
        accepted.append(index)
    return accepted


def _orb_connected(
    trajectory: PreparedTrajectory,
    left: int,
    right: int,
    evaluator: OverlapEvaluator,
) -> bool:
    match = evaluator.match(
        trajectory.records[left].frame_index,
        trajectory.records[right].frame_index,
    )
    return match.is_connected(evaluator.config)


_MAX_RECURSIVE_ORB_STATES = 4096


def _path_gap_bridge_indices(
    trajectory: PreparedTrajectory,
    left: int,
    right: int,
    candidates: list[int],
    max_gap: float,
    minimum_distance: float,
    selected: set[int],
) -> tuple[list[int], int]:
    available = set(candidates)
    additions: list[int] = []
    unresolved = 0
    pending = [(left, right)]
    while pending:
        interval_left, interval_right = pending.pop(0)
        gap = trajectory.path_s[interval_right] - trajectory.path_s[interval_left]
        if gap <= max_gap * (1.0 + 1.0e-9):
            continue
        references = selected | set(additions)
        interval_candidates = [
            index
            for index in available
            if trajectory.path_s[interval_left]
            < trajectory.path_s[index]
            < trajectory.path_s[interval_right]
            and (
                (_distance_to_selected(trajectory.centers, index, references) or 0.0)
                + 1.0e-12
                >= minimum_distance
            )
        ]
        if not interval_candidates:
            unresolved += 1
            continue
        midpoint = (
            trajectory.path_s[interval_left] + trajectory.path_s[interval_right]
        ) * 0.5
        best = min(
            interval_candidates,
            key=lambda index: (
                abs(trajectory.path_s[index] - midpoint),
                -(_distance_to_selected(
                    trajectory.centers, index, selected | set(additions)
                ) or 0.0),
                -trajectory.records[index].quality_score,
                trajectory.records[index].frame_index,
            ),
        )
        available.remove(best)
        additions.append(best)
        pending.extend(((interval_left, best), (best, interval_right)))
    additions.sort(key=lambda index: trajectory.path_s[index])
    return additions, unresolved


def _distance_to_selected(points: np.ndarray, index: int, selected: set[int]) -> float | None:
    if not selected:
        return None
    selected_indices = np.asarray(sorted(selected), dtype=np.int64)
    return float(np.min(np.linalg.norm(points[selected_indices] - points[index], axis=1)))


def _to_record(order: int, item: _PendingSelection, trajectory: PreparedTrajectory) -> SelectionRecord:
    source = trajectory.records[item.index]
    normalized = None
    if item.nearest_distance is not None and item.radius > 0:
        normalized = item.nearest_distance / item.radius
    return SelectionRecord(
        selection_order=order,
        frame_index=source.frame_index,
        pts=source.pts,
        time_base_num=source.time_base_num,
        time_base_den=source.time_base_den,
        pts_seconds=source.pts_seconds,
        segment_id=source.segment_id,
        coordinate_group_id=source.coordinate_group_id,
        selection_type=item.selection_type,
        nearest_selected_distance=item.nearest_distance,
        normalized_distance=normalized,
        path_s=float(trajectory.path_s[item.index]),
        quality_score=source.quality_score,
    )


def _build_report(
    trajectory: PreparedTrajectory,
    pending: list[_PendingSelection],
    radii: dict[int, float],
    group_reports: dict[str, Any],
    unresolved: int,
    unresolved_orb: int,
    overlap_evaluator: OverlapEvaluator | None,
) -> dict[str, Any]:
    selected = {item.index for item in pending}
    types: dict[str, int] = {}
    for item in pending:
        types[item.selection_type] = types.get(item.selection_type, 0) + 1
    max_path_gap = 0.0
    for indices in segmented_indices(trajectory).values():
        ordered = [int(index) for index in indices if int(index) in selected]
        ordered.sort(key=lambda index: trajectory.path_s[index])
        for left, right in zip(ordered, ordered[1:]):
            max_path_gap = max(max_path_gap, float(trajectory.path_s[right] - trajectory.path_s[left]))

    coverage = [item.index for item in pending if item.selection_type == "coverage"]
    min_separation = _minimum_separation(trajectory.centers, coverage)
    return {
        "schema_version": 1,
        "input_count": len(trajectory.records),
        "candidate_count": int(np.count_nonzero(trajectory.candidate_mask)),
        "selected_count": len(pending),
        "selection_counts": types,
        "coordinate_groups": group_reports,
        "minimum_coverage_separation": min_separation,
        "maximum_selected_path_gap": max_path_gap,
        "unresolved_path_gaps": unresolved,
        "unresolved_orb_connections": unresolved_orb,
        "radii": {str(key): value for key, value in sorted(radii.items())},
        "continuity": {
            "mode": "orb" if overlap_evaluator is not None else "path",
            **(overlap_evaluator.report() if overlap_evaluator is not None else {}),
        },
    }


def _minimum_separation(points: np.ndarray, indices: list[int]) -> float | None:
    if len(indices) < 2:
        return None
    minimum = math.inf
    for offset, left in enumerate(indices[:-1]):
        distances = np.linalg.norm(points[np.asarray(indices[offset + 1 :])] - points[left], axis=1)
        minimum = min(minimum, float(np.min(distances)))
    return minimum

