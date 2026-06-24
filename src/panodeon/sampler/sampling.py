from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FpsSelection:
    index: int
    nearest_distance: float | None


@dataclass(frozen=True)
class FpsResult:
    additions: tuple[FpsSelection, ...]
    max_cover_distance: float


def radius_fps(
    points: np.ndarray,
    valid_indices: np.ndarray,
    *,
    radius: float,
    seed_indices: list[int],
    quality: np.ndarray,
    frame_indices: np.ndarray,
    distance_tolerance: float,
    tie_distance_ratio: float,
) -> FpsResult:
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    if len(valid_indices) == 0:
        return FpsResult((), 0.0)
    valid_set = set(int(index) for index in valid_indices)
    selected = list(dict.fromkeys(index for index in seed_indices if index in valid_set))
    selected_set = set(selected)
    min_d2 = np.full(len(points), np.inf, dtype=np.float64)
    for index in selected:
        min_d2 = np.minimum(min_d2, _squared_distances(points, index))
    additions: list[FpsSelection] = []

    if not selected:
        center = np.median(points[valid_indices], axis=0)
        distances = np.sum((points[valid_indices] - center) ** 2, axis=1)
        first = int(valid_indices[int(np.argmax(distances))])
        selected.append(first)
        selected_set.add(first)
        additions.append(FpsSelection(first, None))
        min_d2 = np.minimum(min_d2, _squared_distances(points, first))

    threshold = max(0.0, radius * (1.0 - distance_tolerance))
    while radius > 0:
        candidates = np.asarray([index for index in valid_indices if int(index) not in selected_set], dtype=np.int64)
        if len(candidates) == 0:
            break
        max_d2 = float(np.max(min_d2[candidates]))
        if math.sqrt(max_d2) < threshold:
            break
        index = _tie_break(candidates, min_d2, max_d2, quality, frame_indices, tie_distance_ratio)
        distance = math.sqrt(float(min_d2[index]))
        additions.append(FpsSelection(index, distance))
        selected.append(index)
        selected_set.add(index)
        min_d2 = np.minimum(min_d2, _squared_distances(points, index))

    remaining = np.asarray([index for index in valid_indices if int(index) not in selected_set], dtype=np.int64)
    max_cover = 0.0 if len(remaining) == 0 else math.sqrt(float(np.max(min_d2[remaining])))
    return FpsResult(tuple(additions), max_cover)


def target_count_fps(
    points: np.ndarray,
    valid_indices: np.ndarray,
    *,
    target_count: int,
    seed_indices: list[int],
    quality: np.ndarray,
    frame_indices: np.ndarray,
    tie_distance_ratio: float,
) -> tuple[FpsResult, float]:
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    valid_set = set(int(index) for index in valid_indices)
    seeds = list(dict.fromkeys(index for index in seed_indices if index in valid_set))
    additions_needed = max(0, min(len(valid_indices), target_count) - len(seeds))
    selected = list(seeds)
    selected_set = set(seeds)
    min_d2 = np.full(len(points), np.inf, dtype=np.float64)
    for index in selected:
        min_d2 = np.minimum(min_d2, _squared_distances(points, index))
    additions: list[FpsSelection] = []

    while len(additions) < additions_needed:
        candidates = np.asarray([index for index in valid_indices if int(index) not in selected_set], dtype=np.int64)
        if len(candidates) == 0:
            break
        if not selected:
            center = np.median(points[valid_indices], axis=0)
            distances = np.sum((points[candidates] - center) ** 2, axis=1)
            index = int(candidates[int(np.argmax(distances))])
            nearest = None
        else:
            max_d2 = float(np.max(min_d2[candidates]))
            index = _tie_break(candidates, min_d2, max_d2, quality, frame_indices, tie_distance_ratio)
            nearest = math.sqrt(float(min_d2[index]))
        additions.append(FpsSelection(index, nearest))
        selected.append(index)
        selected_set.add(index)
        min_d2 = np.minimum(min_d2, _squared_distances(points, index))

    remaining = np.asarray([index for index in valid_indices if int(index) not in selected_set], dtype=np.int64)
    max_cover = 0.0 if len(remaining) == 0 else math.sqrt(float(np.max(min_d2[remaining])))
    radius = max_cover * (1.0 + 1.0e-9) if max_cover > 0 else 0.0
    return FpsResult(tuple(additions), max_cover), radius


def _squared_distances(points: np.ndarray, index: int) -> np.ndarray:
    difference = points - points[index]
    return np.einsum("ij,ij->i", difference, difference)


def _tie_break(
    candidates: np.ndarray,
    min_d2: np.ndarray,
    max_d2: float,
    quality: np.ndarray,
    frame_indices: np.ndarray,
    tie_distance_ratio: float,
) -> int:
    minimum_tied_d2 = max_d2 * (1.0 - tie_distance_ratio) ** 2
    tied = candidates[min_d2[candidates] >= minimum_tied_d2]
    ordered = sorted((int(index) for index in tied), key=lambda index: (-quality[index], frame_indices[index]))
    return ordered[0]

