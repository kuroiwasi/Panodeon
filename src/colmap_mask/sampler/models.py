from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrajectoryRecord:
    frame_index: int
    pts: int
    time_base_num: int
    time_base_den: int
    segment_id: int
    coordinate_group_id: int
    cx: float
    cy: float
    cz: float
    pose_valid: bool = True
    tracking_state: str = "good"
    candidate_valid: bool = True
    quality_score: float = 0.0
    qw: float | None = None
    qx: float | None = None
    qy: float | None = None
    qz: float | None = None

    @property
    def pts_seconds(self) -> float:
        return self.pts * self.time_base_num / self.time_base_den


@dataclass(frozen=True)
class SamplingConfig:
    scale_mode: str = "robust_extent"
    radius: float | None = None
    radius_ratio: float = 0.01
    target_count: int | None = None
    force_first_frame: bool = True
    force_segment_endpoints: bool = False
    allow_weak_tracking: bool = False
    distance_tolerance: float = 1.0e-6
    tie_distance_ratio: float = 1.0e-3

    def validate(self) -> None:
        if self.scale_mode not in {"metric", "robust_extent", "target_count"}:
            raise ValueError(f"Unsupported scale_mode: {self.scale_mode}")
        if self.scale_mode == "metric" and (self.radius is None or self.radius <= 0):
            raise ValueError("sampling.radius must be positive in metric mode")
        if self.radius_ratio <= 0:
            raise ValueError("sampling.radius_ratio must be positive")
        if self.scale_mode == "target_count" and (self.target_count is None or self.target_count <= 0):
            raise ValueError("sampling.target_count must be positive in target_count mode")
        if not 0 <= self.distance_tolerance < 1:
            raise ValueError("sampling.distance_tolerance must be in [0, 1)")
        if not 0 <= self.tie_distance_ratio < 1:
            raise ValueError("sampling.tie_distance_ratio must be in [0, 1)")


@dataclass(frozen=True)
class OrbConfig:
    enabled: bool = True
    max_width: int = 720
    feature_count: int = 1000
    ratio_test: float = 0.75
    min_inliers: int = 40
    min_inlier_ratio: float = 0.2
    ransac_threshold: float = 2.0
    # Spherical (tangent-plane) mode: project the equirectangular frame onto
    # several perspective patches before running ORB, avoiding pole distortion.
    spherical: bool = False
    # "cubemap": six orthogonal 90-degree faces covering the whole sphere with
    # no overlap (fewest patches, fastest). "equatorial": a yaw ring at several
    # pitches with overlap (configurable coverage, more patches).
    tangent_layout: str = "cubemap"
    tangent_fov_deg: float = 90.0
    tangent_size: int = 512
    tangent_yaw_count: int = 6
    tangent_pitch_deg: tuple[float, ...] = (-30.0, 0.0, 30.0)
    spherical_ransac_threshold: float = 0.01

    def validate(self) -> None:
        if self.max_width <= 0:
            raise ValueError("continuity.orb.max_width must be positive")
        if self.feature_count < 8:
            raise ValueError("continuity.orb.feature_count must be at least 8")
        if not 0 < self.ratio_test < 1:
            raise ValueError("continuity.orb.ratio_test must be in (0, 1)")
        if self.min_inliers < 4:
            raise ValueError("continuity.orb.min_inliers must be at least 4")
        if not 0 <= self.min_inlier_ratio <= 1:
            raise ValueError("continuity.orb.min_inlier_ratio must be in [0, 1]")
        if self.ransac_threshold <= 0:
            raise ValueError("continuity.orb.ransac_threshold must be positive")
        if self.tangent_layout not in {"cubemap", "equatorial"}:
            raise ValueError("continuity.orb.tangent_layout must be 'cubemap' or 'equatorial'")
        if not 0 < self.tangent_fov_deg < 180:
            raise ValueError("continuity.orb.tangent_fov_deg must be in (0, 180)")
        if self.tangent_size <= 0:
            raise ValueError("continuity.orb.tangent_size must be positive")
        if self.tangent_yaw_count <= 0:
            raise ValueError("continuity.orb.tangent_yaw_count must be positive")
        if not self.tangent_pitch_deg:
            raise ValueError("continuity.orb.tangent_pitch_deg must not be empty")
        if any(not -90 < pitch < 90 for pitch in self.tangent_pitch_deg):
            raise ValueError("continuity.orb.tangent_pitch_deg entries must be in (-90, 90)")
        if self.spherical_ransac_threshold <= 0:
            raise ValueError("continuity.orb.spherical_ransac_threshold must be positive")


@dataclass(frozen=True)
class ContinuityConfig:
    enabled: bool = True
    max_path_gap_ratio: float = 2.0
    bridge_min_separation_ratios: tuple[float, ...] = (0.25, 0.1)
    bridge_max_recursion_depth: int = 12
    orb: OrbConfig = field(default_factory=OrbConfig)

    def validate(self) -> None:
        if self.max_path_gap_ratio <= 0:
            raise ValueError("continuity.max_path_gap_ratio must be positive")
        if not self.bridge_min_separation_ratios:
            raise ValueError("continuity.bridge_min_separation_ratios must not be empty")
        if any(ratio < 0 for ratio in self.bridge_min_separation_ratios):
            raise ValueError("continuity.bridge_min_separation_ratios must be non-negative")
        if self.bridge_max_recursion_depth <= 0:
            raise ValueError("continuity.bridge_max_recursion_depth must be positive")
        self.orb.validate()


@dataclass(frozen=True)
class TrajectoryConfig:
    jitter_threshold: float = 0.0

    def validate(self) -> None:
        if self.jitter_threshold < 0:
            raise ValueError("trajectory.jitter_threshold must be non-negative")


@dataclass(frozen=True)
class SamplerConfig:
    schema_version: int = SCHEMA_VERSION
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    continuity: ContinuityConfig = field(default_factory=ContinuityConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplerConfig:
        schema_version = int(data.get("schema_version", SCHEMA_VERSION))
        continuity_data = dict(data.get("continuity", {}))
        orb_data = dict(continuity_data.pop("orb", {}))
        orb_data.pop("max_frame_gap", None)
        if "bridge_min_separation_ratios" in continuity_data:
            continuity_data["bridge_min_separation_ratios"] = tuple(
                continuity_data["bridge_min_separation_ratios"]
            )
        if "tangent_pitch_deg" in orb_data:
            orb_data["tangent_pitch_deg"] = tuple(orb_data["tangent_pitch_deg"])
        config = cls(
            schema_version=schema_version,
            sampling=SamplingConfig(**data.get("sampling", {})),
            continuity=ContinuityConfig(orb=OrbConfig(**orb_data), **continuity_data),
            trajectory=TrajectoryConfig(**data.get("trajectory", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported config schema_version: {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        self.sampling.validate()
        self.continuity.validate()
        self.trajectory.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionRecord:
    selection_order: int
    frame_index: int
    pts: int
    time_base_num: int
    time_base_den: int
    pts_seconds: float
    segment_id: int
    coordinate_group_id: int
    selection_type: str
    nearest_selected_distance: float | None
    normalized_distance: float | None
    path_s: float
    quality_score: float
