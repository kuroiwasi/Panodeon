from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import av
import cv2
import numpy as np

from .models import OrbConfig, TrajectoryRecord


@dataclass(frozen=True)
class OrbMatch:
    good_matches: int
    inliers: int
    inlier_ratio: float
    overlap_score: float

    def is_connected(self, config: OrbConfig) -> bool:
        return self.inliers >= config.min_inliers and self.inlier_ratio >= config.min_inlier_ratio


class OverlapEvaluator(Protocol):
    config: OrbConfig

    def match(self, left: int, right: int) -> OrbMatch:
        ...

    def report(self) -> dict[str, int | float]:
        ...


@dataclass(frozen=True)
class _Features:
    points: np.ndarray
    descriptors: np.ndarray | None
    # Unit-sphere direction for each keypoint, populated in spherical mode.
    bearings: np.ndarray | None = None


class OrbOverlapEvaluator:
    def __init__(
        self,
        video_path: Path,
        records: list[TrajectoryRecord],
        config: OrbConfig,
        *,
        match_by_frame_index: bool = False,
    ) -> None:
        if not video_path.is_file():
            raise ValueError(f"Video does not exist: {video_path}")
        self.config = config
        self._features = _load_features(video_path, records, config, match_by_frame_index)
        self._cache: dict[tuple[int, int], OrbMatch] = {}
        self._cache_hits = 0

    def match(self, left: int, right: int) -> OrbMatch:
        if left == right:
            count = len(self._features[left].points)
            return OrbMatch(count, count, 1.0 if count else 0.0, 1.0 if count else 0.0)
        key = (left, right) if left < right else (right, left)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        result = _match_features(self._features[left], self._features[right], self.config)
        self._cache[key] = result
        return result

    def report(self) -> dict[str, int | float]:
        counts = [len(item.points) for item in self._features.values()]
        return {
            "feature_frames": len(counts),
            "mean_features": float(np.mean(counts)) if counts else 0.0,
            "evaluated_pairs": len(self._cache),
            "cache_hits": self._cache_hits,
        }


def _load_features(
    video_path: Path,
    records: list[TrajectoryRecord],
    config: OrbConfig,
    match_by_frame_index: bool,
) -> dict[int, _Features]:
    timestamp_targets: dict[Fraction, list[int]] = {}
    frame_targets: dict[int, list[int]] = {}
    for record in records:
        if match_by_frame_index:
            frame_targets.setdefault(record.frame_index, []).append(record.frame_index)
        else:
            timestamp = Fraction(record.pts * record.time_base_num, record.time_base_den)
            timestamp_targets.setdefault(timestamp, []).append(record.frame_index)
    features: dict[int, _Features] = {}
    detector = cv2.ORB_create(nfeatures=config.feature_count)
    tangent_views = _tangent_views(config) if config.spherical else None
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError("Input has no video stream")
        stream = container.streams.video[0]
        for decoded_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                continue
            if match_by_frame_index:
                matching = frame_targets.pop(decoded_index, [])
            else:
                timestamp = Fraction(int(frame.pts)) * Fraction(frame.time_base)
                matching = timestamp_targets.pop(timestamp, [])
            if not matching:
                continue
            gray = frame.to_ndarray(format="gray")
            if gray.shape[1] > config.max_width:
                scale = config.max_width / gray.shape[1]
                gray = cv2.resize(
                    gray,
                    (config.max_width, max(1, round(gray.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            if tangent_views is not None:
                item = _detect_spherical(gray, detector, tangent_views, config)
            else:
                keypoints, descriptors = detector.detectAndCompute(gray, None)
                points = np.asarray(
                    [item.pt for item in keypoints], dtype=np.float32
                ).reshape(-1, 2)
                item = _Features(points, descriptors)
            for frame_index in matching:
                features[frame_index] = item
            if not timestamp_targets and not frame_targets:
                break
    missing = sum(len(indices) for indices in timestamp_targets.values()) + sum(
        len(indices) for indices in frame_targets.values()
    )
    if missing:
        identity = "frame index" if match_by_frame_index else "PTS"
        raise ValueError(f"Trajectory {identity} not found in video for ORB: {missing} frame(s)")
    return features


def _match_features(left: _Features, right: _Features, config: OrbConfig) -> OrbMatch:
    if left.descriptors is None or right.descriptors is None:
        return OrbMatch(0, 0, 0.0, 0.0)
    if len(left.descriptors) < 2 or len(right.descriptors) < 2:
        return OrbMatch(0, 0, 0.0, 0.0)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = _ratio_matches(
        matcher.knnMatch(left.descriptors, right.descriptors, k=2), config.ratio_test
    )
    reverse = _ratio_matches(
        matcher.knnMatch(right.descriptors, left.descriptors, k=2), config.ratio_test
    )
    mutual = [
        match
        for query, match in forward.items()
        if match.trainIdx in reverse and reverse[match.trainIdx].trainIdx == query
    ]
    if len(mutual) < 4:
        return OrbMatch(len(mutual), 0, 0.0, 0.0)
    if left.bearings is not None and right.bearings is not None:
        left_bearings = np.asarray(
            [left.bearings[item.queryIdx] for item in mutual], dtype=np.float64
        )
        right_bearings = np.asarray(
            [right.bearings[item.trainIdx] for item in mutual], dtype=np.float64
        )
        inliers = _spherical_inliers(
            left_bearings, right_bearings, config.spherical_ransac_threshold
        )
    else:
        left_points = np.asarray(
            [left.points[item.queryIdx] for item in mutual], dtype=np.float32
        )
        right_points = np.asarray(
            [right.points[item.trainIdx] for item in mutual], dtype=np.float32
        )
        inliers = _geometric_inliers(left_points, right_points, config.ransac_threshold)
    denominator = min(len(left.points), len(right.points))
    return OrbMatch(
        len(mutual),
        inliers,
        inliers / len(mutual),
        inliers / denominator if denominator else 0.0,
    )


def _ratio_matches(rows: tuple | list, ratio: float) -> dict[int, cv2.DMatch]:
    return {
        row[0].queryIdx: row[0]
        for row in rows
        if len(row) >= 2 and row[0].distance < ratio * row[1].distance
    }


def _geometric_inliers(left: np.ndarray, right: np.ndarray, threshold: float) -> int:
    left = np.ascontiguousarray(left, dtype=np.float32)
    right = np.ascontiguousarray(right, dtype=np.float32)
    finite = np.all(np.isfinite(left), axis=1) & np.all(np.isfinite(right), axis=1)
    left = left[finite]
    right = right[finite]
    best = 0
    if len(left) >= 8:
        try:
            _, mask = cv2.findFundamentalMat(
                left, right, cv2.FM_RANSAC, threshold, 0.99
            )
        except cv2.error:
            mask = None
        if mask is not None:
            best = max(best, int(np.count_nonzero(mask)))
    if len(left) >= 4:
        try:
            _, mask = cv2.findHomography(left, right, cv2.RANSAC, threshold)
        except cv2.error:
            mask = None
        if mask is not None:
            best = max(best, int(np.count_nonzero(mask)))
    return best


@dataclass(frozen=True)
class _TangentView:
    """Precomputed equirectangular sample map and back-projection basis."""

    map_x: np.ndarray
    map_y: np.ndarray
    rotation: np.ndarray  # camera-to-world basis, columns [right, up, forward]
    focal: float
    center: float


def _tangent_views(config: OrbConfig) -> list[_TangentView]:
    # A cubemap tiles the sphere exactly with six orthogonal 90-degree faces, so
    # the per-face field of view is fixed regardless of tangent_fov_deg.
    fov = np.radians(90.0 if config.tangent_layout == "cubemap" else config.tangent_fov_deg)
    size = config.tangent_size
    focal = (size / 2) / np.tan(fov / 2)
    center = (size - 1) / 2
    # Pixel grid shared by every view; bearings are rotated per viewpoint.
    cols, rows = np.meshgrid(np.arange(size), np.arange(size))
    cam = np.stack(
        [(cols - center) / focal, (center - rows) / focal, np.ones_like(cols, dtype=float)],
        axis=-1,
    )
    cam /= np.linalg.norm(cam, axis=-1, keepdims=True)
    rotations = (
        _cubemap_rotations()
        if config.tangent_layout == "cubemap"
        else _equatorial_rotations(config)
    )
    views: list[_TangentView] = []
    for rotation in rotations:
        world = cam @ rotation.T
        map_x, map_y = _bearings_to_equirect_pixels(world)
        views.append(
            _TangentView(map_x.astype(np.float32), map_y.astype(np.float32), rotation, focal, center)
        )
    return views


def _equatorial_rotations(config: OrbConfig) -> list[np.ndarray]:
    yaws = np.linspace(0.0, 2 * np.pi, config.tangent_yaw_count, endpoint=False)
    return [
        _basis(
            [
                np.cos(np.radians(pitch)) * np.sin(yaw),
                np.sin(np.radians(pitch)),
                np.cos(np.radians(pitch)) * np.cos(yaw),
            ],
            [0.0, 1.0, 0.0],
        )
        for pitch in config.tangent_pitch_deg
        for yaw in yaws
    ]


def _cubemap_rotations() -> list[np.ndarray]:
    """Six orthogonal faces: +Z, +X, -Z, -X (sides), +Y (up), -Y (down)."""
    return [
        _basis([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        _basis([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        _basis([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        _basis([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        _basis([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        _basis([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
    ]


def _basis(forward: list[float], up_hint: list[float]) -> np.ndarray:
    """Camera-to-world basis, columns [right, up, forward], for a given heading.

    ``up_hint`` only needs to be non-parallel to ``forward``; it picks the roll.
    """
    f = np.asarray(forward, dtype=float)
    f /= np.linalg.norm(f)
    right = np.cross(np.asarray(up_hint, dtype=float), f)
    right /= np.linalg.norm(right)
    up = np.cross(f, right)
    return np.stack([right, up, f], axis=1)


def _bearings_to_equirect_pixels(
    bearings: np.ndarray, *, width: int | None = None, height: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Map unit bearings to fractional equirectangular pixel coordinates."""
    x, y, z = bearings[..., 0], bearings[..., 1], bearings[..., 2]
    lon = np.arctan2(x, z)  # 0 at +Z, increasing toward +X
    lat = np.arcsin(np.clip(y, -1.0, 1.0))
    u = (lon / (2 * np.pi) + 0.5)
    v = (0.5 - lat / np.pi)
    if width is not None and height is not None:
        return u * (width - 1), v * (height - 1)
    return u, v


def _detect_spherical(
    gray: np.ndarray,
    detector: cv2.ORB,
    views: list[_TangentView],
    config: OrbConfig,
) -> _Features:
    height, width = gray.shape[:2]
    points: list[np.ndarray] = []
    bearings: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    for view in views:
        patch = cv2.remap(
            gray,
            view.map_x * (width - 1),
            view.map_y * (height - 1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        keypoints, patch_desc = detector.detectAndCompute(patch, None)
        if patch_desc is None or not keypoints:
            continue
        pixels = np.asarray([kp.pt for kp in keypoints], dtype=np.float64).reshape(-1, 2)
        cam = np.stack(
            [
                (pixels[:, 0] - view.center) / view.focal,
                (view.center - pixels[:, 1]) / view.focal,
                np.ones(len(pixels)),
            ],
            axis=-1,
        )
        cam /= np.linalg.norm(cam, axis=-1, keepdims=True)
        world = cam @ view.rotation.T
        points.append(pixels.astype(np.float32))
        bearings.append(world)
        descriptors.append(patch_desc)
    if not descriptors:
        return _Features(np.empty((0, 2), np.float32), None, np.empty((0, 3)))
    return _Features(
        np.concatenate(points),
        np.concatenate(descriptors),
        np.concatenate(bearings),
    )


def _spherical_inliers(left: np.ndarray, right: np.ndarray, threshold: float) -> int:
    """Count matches consistent with a single essential matrix on the sphere.

    ``threshold`` is the maximum symmetric epipolar distance (Sampson) measured
    on unit bearings, so it is roughly an angular tolerance in radians.
    """
    if len(left) < 8:
        return 0
    finite = np.all(np.isfinite(left), axis=1) & np.all(np.isfinite(right), axis=1)
    left = left[finite]
    right = right[finite]
    best = 0
    for _ in range(_RANSAC_ITERATIONS):
        sample = _RNG.choice(len(left), size=8, replace=False)
        essential = _eight_point(left[sample], right[sample])
        if essential is None:
            continue
        residual = _sampson_distance(essential, left, right)
        inliers = int(np.count_nonzero(residual < threshold))
        best = max(best, inliers)
    return best


def _eight_point(left: np.ndarray, right: np.ndarray) -> np.ndarray | None:
    """Linear essential-matrix estimate from corresponding unit bearings."""
    a = np.stack(
        [
            right[:, 0] * left[:, 0],
            right[:, 0] * left[:, 1],
            right[:, 0] * left[:, 2],
            right[:, 1] * left[:, 0],
            right[:, 1] * left[:, 1],
            right[:, 1] * left[:, 2],
            right[:, 2] * left[:, 0],
            right[:, 2] * left[:, 1],
            right[:, 2] * left[:, 2],
        ],
        axis=-1,
    )
    try:
        _, _, vh = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None
    essential = vh[-1].reshape(3, 3)
    # Enforce the rank-2, equal-singular-value structure of an essential matrix.
    u, _, vt = np.linalg.svd(essential)
    return u @ np.diag([1.0, 1.0, 0.0]) @ vt


def _sampson_distance(essential: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    el = left @ essential.T  # E x_left, per row
    er = right @ essential
    numerator = np.einsum("ij,ij->i", right, el) ** 2
    denominator = (
        el[:, 0] ** 2 + el[:, 1] ** 2 + er[:, 0] ** 2 + er[:, 1] ** 2
    )
    return numerator / np.where(denominator > 0, denominator, np.inf)


_RANSAC_ITERATIONS = 256
_RNG = np.random.default_rng(0)
