from __future__ import annotations

import argparse
import math
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from panodeon.core.colmap_export import virtual_cameras
from panodeon.core.image_io import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ColmapGpuOptions:
    extraction_use_option: str | None = "--SiftExtraction.use_gpu"
    extraction_index_option: str | None = "--SiftExtraction.gpu_index"
    matching_use_option: str | None = "--SiftMatching.use_gpu"
    matching_index_option: str | None = "--SiftMatching.gpu_index"

    @property
    def extraction_supported(self) -> bool:
        return self.extraction_use_option is not None and self.extraction_index_option is not None

    @property
    def matching_supported(self) -> bool:
        return self.matching_use_option is not None and self.matching_index_option is not None


@dataclass(frozen=True)
class ColmapMapperOptions:
    ignore_two_view_tracks_option: str | None = "--Mapper.tri_ignore_two_view_tracks"
    snapshot_path_option: str | None = "--Mapper.snapshot_path"
    snapshot_images_freq_option: str | None = "--Mapper.snapshot_images_freq"
    refine_sensor_from_rig_option: str | None = "--Mapper.ba_refine_sensor_from_rig"

    @property
    def snapshot_supported(self) -> bool:
        return self.snapshot_path_option is not None and self.snapshot_images_freq_option is not None


@dataclass(frozen=True)
class ColmapRunSettings:
    colmap: str = "colmap"
    tile_size: int = 3072
    fov_deg: float = 90.0
    matcher: str = "sequential"
    sparse_mapper: str = "mapper"
    camera_model: str = "PINHOLE"
    skip_mapping: bool = False
    rig_bundle_adjustment: bool = False
    dense_reconstruction: bool = False
    pair_temporal_window: int = 3
    camera_neighbor_count: int = 4
    use_gpu: bool = True
    gpu_index: str = "-1"
    covariant_sift: bool = False
    mapper_snapshot_path: Path | None = None
    mapper_snapshot_images_freq: int = 50
    gpu_options: ColmapGpuOptions = field(default_factory=ColmapGpuOptions)
    mapper_options: ColmapMapperOptions = field(default_factory=ColmapMapperOptions)
    skip_completed: bool = False


@dataclass(frozen=True)
class ColmapStep:
    name: str
    command: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run COLMAP on a Panodeon export folder.")
    parser.add_argument("export_dir", type=Path, help="Folder containing images/, masks/, and rig_config.json.")
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable path.")
    parser.add_argument("--tile-size", type=int, default=3072, help="Perspective tile size used during export.")
    parser.add_argument("--fov", type=float, default=90.0, help="Perspective FOV used during export.")
    parser.add_argument("--matcher", choices=("sequential", "pairs", "exhaustive", "vocab_tree"), default="sequential")
    parser.add_argument("--sparse-mapper", choices=("mapper", "hierarchical_mapper"), default="mapper")
    parser.add_argument("--camera-model", default="PINHOLE")
    parser.add_argument("--camera-neighbors", type=int, default=4, help="Nearest virtual cameras to match across neighboring frames.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing database.db and sparse/ before running.")
    parser.add_argument("--skip-mapping", action="store_true", help="Run only feature extraction, rig setup, and matching.")
    parser.add_argument("--rig-ba", action="store_true", help="Run bundle_adjuster after sparse mapping.")
    parser.add_argument("--dense", action="store_true", help="Run COLMAP dense reconstruction after sparse mapping.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip COLMAP steps that already have output.")
    parser.add_argument("--no-gpu", action="store_true", help="Disable COLMAP SIFT GPU extraction and matching.")
    parser.add_argument("--gpu-index", default="-1", help="COLMAP GPU index. -1 lets COLMAP choose/use all available GPUs.")
    parser.add_argument(
        "--covariant-sift",
        action="store_true",
        help="Enable affine shape and domain size pooling. This uses COLMAP's Covariant SIFT CPU extractor.",
    )
    parser.add_argument("--mapper-snapshot-path", type=Path, help="Folder for mapper snapshot models.")
    parser.add_argument(
        "--mapper-snapshot-images-freq",
        type=int,
        default=50,
        help="Registered-image interval for mapper snapshots.",
    )
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    validate_export_dir(export_dir)
    database_path = export_dir / "database.db"
    sparse_dir = export_dir / "sparse"
    dense_dir = export_dir / "dense"
    rig_ba_dir = export_dir / "sparse_rig_ba"
    mapper_snapshot_path = args.mapper_snapshot_path
    if mapper_snapshot_path is not None:
        if not mapper_snapshot_path.is_absolute():
            mapper_snapshot_path = export_dir / mapper_snapshot_path
    if should_overwrite_outputs(args.overwrite, args.skip_completed):
        if database_path.exists():
            database_path.unlink()
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        if args.rig_ba and rig_ba_dir.exists():
            shutil.rmtree(rig_ba_dir)
        if args.dense and dense_dir.exists():
            shutil.rmtree(dense_dir)
        if mapper_snapshot_path is not None and mapper_snapshot_path.exists():
            shutil.rmtree(mapper_snapshot_path)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if mapper_snapshot_path is not None:
        mapper_snapshot_path.mkdir(parents=True, exist_ok=True)

    gpu_options = detect_colmap_gpu_options(args.colmap)
    mapper_options = detect_colmap_mapper_options(args.colmap, args.sparse_mapper)
    if not args.no_gpu and not (gpu_options.extraction_supported and gpu_options.matching_supported):
        print("GPU options unsupported by this COLMAP binary. Running without GPU flags.", flush=True)
    settings = ColmapRunSettings(
        colmap=args.colmap,
        tile_size=args.tile_size,
        fov_deg=args.fov,
        matcher=args.matcher,
        sparse_mapper=args.sparse_mapper,
        camera_model=args.camera_model,
        camera_neighbor_count=args.camera_neighbors,
        skip_mapping=args.skip_mapping,
        rig_bundle_adjustment=args.rig_ba,
        dense_reconstruction=args.dense,
        use_gpu=not args.no_gpu,
        gpu_index=args.gpu_index,
        covariant_sift=args.covariant_sift,
        mapper_snapshot_path=mapper_snapshot_path,
        mapper_snapshot_images_freq=args.mapper_snapshot_images_freq,
        gpu_options=gpu_options,
        mapper_options=mapper_options,
        skip_completed=args.skip_completed,
    )
    for step in build_colmap_steps(export_dir, settings):
        run(step.command, export_dir)
    return 0


def validate_export_dir(export_dir: Path) -> None:
    missing = [name for name in ("images", "masks", "rig_config.json") if not (export_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing COLMAP export files in {export_dir}: {', '.join(missing)}")


def should_overwrite_outputs(overwrite: bool, skip_completed: bool) -> bool:
    return overwrite and not skip_completed


def camera_params_for(tile_size: int, fov_deg: float) -> str:
    focal = tile_size * 0.5 / math.tan(math.radians(fov_deg) * 0.5)
    center = tile_size * 0.5
    return f"{focal:.6f},{focal:.6f},{center:.6f},{center:.6f}"


def build_colmap_steps(export_dir: Path, settings: ColmapRunSettings) -> list[ColmapStep]:
    database_path = export_dir / "database.db"
    sparse_dir = export_dir / "sparse"
    rig_ba_dir = export_dir / "sparse_rig_ba"
    dense_dir = export_dir / "dense"
    camera_params = camera_params_for(settings.tile_size, settings.fov_deg)
    feature_command = [
        settings.colmap,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(export_dir / "images"),
        "--ImageReader.mask_path",
        str(export_dir / "masks"),
        "--ImageReader.single_camera_per_folder",
        "1",
        "--ImageReader.camera_model",
        settings.camera_model,
        "--ImageReader.camera_params",
        camera_params,
        "--FeatureExtraction.max_image_size",
        str(settings.tile_size),
        "--SiftExtraction.max_num_features",
        "16384",
    ]
    if settings.covariant_sift:
        feature_command.extend(
            [
                "--SiftExtraction.estimate_affine_shape",
                "1",
                "--SiftExtraction.domain_size_pooling",
                "1",
            ]
        )
    if settings.use_gpu and settings.gpu_options.extraction_supported:
        feature_command.extend(
            [
                settings.gpu_options.extraction_use_option,
                "1",
                settings.gpu_options.extraction_index_option,
                settings.gpu_index,
            ]
        )
    steps: list[ColmapStep] = []
    image_count = colmap_image_count(export_dir / "images")
    if not (settings.skip_completed and feature_extraction_done(database_path, image_count)):
        steps.append(ColmapStep("Feature extraction", feature_command))
    if not (settings.skip_completed and database_has_rows(database_path, "frames")):
        steps.append(
            ColmapStep(
                "Rig configuration",
                [
                    settings.colmap,
                    "rig_configurator",
                    "--database_path",
                    str(database_path),
                    "--rig_config_path",
                    str(export_dir / "rig_config.json"),
                ],
            )
        )
    if settings.matcher == "pairs":
        match_list_path = write_match_pair_list(export_dir, settings.pair_temporal_window, settings.camera_neighbor_count)
        match_command = [
            settings.colmap,
            "matches_importer",
            "--database_path",
            str(database_path),
            "--match_list_path",
            str(match_list_path),
            "--match_type",
            "pairs",
            "--FeatureMatching.guided_matching",
            "1",
            "--FeatureMatching.rig_verification",
            "1",
            "--FeatureMatching.skip_image_pairs_in_same_frame",
            "1",
        ]
        if settings.use_gpu and settings.gpu_options.matching_supported:
            match_command.extend(
                [
                    settings.gpu_options.matching_use_option,
                    "1",
                    settings.gpu_options.matching_index_option,
                    settings.gpu_index,
                ]
            )
        if not (settings.skip_completed and database_has_rows(database_path, "matches")):
            steps.append(ColmapStep("Pair-list matching", match_command))
    else:
        match_command = [
            settings.colmap,
            f"{settings.matcher}_matcher",
            "--database_path",
            str(database_path),
            "--FeatureMatching.guided_matching",
            "1",
            "--FeatureMatching.rig_verification",
            "1",
            "--FeatureMatching.skip_image_pairs_in_same_frame",
            "1",
        ]
        if settings.use_gpu and settings.gpu_options.matching_supported:
            match_command.extend(
                [
                    settings.gpu_options.matching_use_option,
                    "1",
                    settings.gpu_options.matching_index_option,
                    settings.gpu_index,
                ]
            )
        if not (settings.skip_completed and database_has_rows(database_path, "matches")):
            steps.append(ColmapStep(f"{settings.matcher.title()} matching", match_command))
    if not settings.skip_mapping and not (settings.skip_completed and sparse_model_exists(sparse_dir)):
        mapper_command = [
            settings.colmap,
            settings.sparse_mapper,
            "--database_path",
            str(database_path),
            "--image_path",
            str(export_dir / "images"),
            "--output_path",
            str(sparse_dir),
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.multiple_models",
            "0",
        ]
        if settings.mapper_options.refine_sensor_from_rig_option is not None:
            mapper_command.extend([settings.mapper_options.refine_sensor_from_rig_option, "0"])
        if settings.mapper_options.ignore_two_view_tracks_option is not None:
            mapper_command.extend([settings.mapper_options.ignore_two_view_tracks_option, "0"])
        if (
            settings.mapper_snapshot_path is not None
            and settings.mapper_snapshot_images_freq > 0
            and settings.mapper_options.snapshot_supported
        ):
            mapper_command.extend(
                [
                    settings.mapper_options.snapshot_path_option,
                    str(settings.mapper_snapshot_path),
                    settings.mapper_options.snapshot_images_freq_option,
                    str(settings.mapper_snapshot_images_freq),
                ]
            )
        mapper_command.extend(
            [
                "--Mapper.tri_min_angle",
                "0.75",
            ]
        )
        step_name = "Hierarchical sparse mapping" if settings.sparse_mapper == "hierarchical_mapper" else "Sparse mapping"
        steps.append(ColmapStep(step_name, mapper_command))
    if (
        settings.rig_bundle_adjustment
        and not settings.skip_mapping
        and not (settings.skip_completed and sparse_model_exists(rig_ba_dir))
    ):
        steps.append(
            ColmapStep(
                "Rig bundle adjustment",
                [
                    settings.colmap,
                    "bundle_adjuster",
                    "--input_path",
                    str(best_sparse_model_dir(sparse_dir)),
                    "--output_path",
                    str(rig_ba_dir),
                    "--BundleAdjustment.refine_focal_length",
                    "0",
                    "--BundleAdjustment.refine_principal_point",
                    "0",
                    "--BundleAdjustment.refine_extra_params",
                    "0",
                    "--BundleAdjustment.refine_sensor_from_rig",
                    "0",
                ],
            )
        )
    if settings.dense_reconstruction and not (settings.skip_completed and dense_model_exists(dense_dir)):
        sparse_model_dir = best_reconstruction_model_dir(export_dir, prefer_rig_ba=settings.rig_bundle_adjustment)
        steps.extend(
            [
                ColmapStep(
                    "Image undistortion",
                    [
                        settings.colmap,
                        "image_undistorter",
                        "--image_path",
                        str(export_dir / "images"),
                        "--input_path",
                        str(sparse_model_dir),
                        "--output_path",
                        str(dense_dir),
                        "--output_type",
                        "COLMAP",
                        "--max_image_size",
                        str(settings.tile_size),
                    ],
                ),
                ColmapStep(
                    "Patch-match stereo",
                    [
                        settings.colmap,
                        "patch_match_stereo",
                        "--workspace_path",
                        str(dense_dir),
                        "--workspace_format",
                        "COLMAP",
                        "--PatchMatchStereo.geom_consistency",
                        "1",
                    ],
                ),
                ColmapStep(
                    "Stereo fusion",
                    [
                        settings.colmap,
                        "stereo_fusion",
                        "--workspace_path",
                        str(dense_dir),
                        "--workspace_format",
                        "COLMAP",
                        "--input_type",
                        "geometric",
                        "--output_path",
                        str(dense_dir / "fused.ply"),
                    ],
                ),
            ]
        )
    return steps


def database_has_rows(database_path: Path, table_name: str) -> bool:
    return database_table_count(database_path, table_name) > 0


def database_table_count(database_path: Path, table_name: str) -> int:
    if not database_path.exists():
        return 0
    try:
        with sqlite3.connect(str(database_path)) as connection:
            row = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            if row is None:
                return 0
            count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    except sqlite3.Error:
        return 0
    return int(count)


def colmap_image_count(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    return sum(1 for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def feature_extraction_done(database_path: Path, expected_images: int) -> bool:
    if expected_images <= 0:
        return False
    return (
        database_table_count(database_path, "images") >= expected_images
        and database_table_count(database_path, "keypoints") >= expected_images
    )


def sparse_model_exists(sparse_dir: Path) -> bool:
    if not sparse_dir.exists():
        return False
    for model_dir in [sparse_dir, *sorted(path for path in sparse_dir.iterdir() if path.is_dir())]:
        if (
            (model_dir / "cameras.bin").exists()
            and (model_dir / "images.bin").exists()
            and (model_dir / "points3D.bin").exists()
        ):
            return True
    return False


def best_sparse_model_dir(sparse_dir: Path) -> Path:
    if not sparse_dir.exists():
        return sparse_dir / "0"
    if model_files_exist(sparse_dir):
        return sparse_dir
    model_dirs = [path for path in sparse_dir.iterdir() if path.is_dir()]
    complete_model_dirs = [path for path in model_dirs if sparse_model_exists(path)]
    if complete_model_dirs:
        return sorted(complete_model_dirs)[0]
    if model_dirs:
        return sorted(model_dirs)[0]
    return sparse_dir / "0"


def model_files_exist(model_dir: Path) -> bool:
    return (
        (model_dir / "cameras.bin").exists()
        and (model_dir / "images.bin").exists()
        and (model_dir / "points3D.bin").exists()
    )


def best_reconstruction_model_dir(export_dir: Path, prefer_rig_ba: bool = True) -> Path:
    rig_ba_dir = export_dir / "sparse_rig_ba"
    if prefer_rig_ba and sparse_model_exists(rig_ba_dir):
        return best_sparse_model_dir(rig_ba_dir)
    return best_sparse_model_dir(export_dir / "sparse")


def dense_model_exists(dense_dir: Path) -> bool:
    return (dense_dir / "fused.ply").exists()


def write_match_pair_list(export_dir: Path, temporal_window: int = 3, camera_neighbor_count: int = 4) -> Path:
    pairs = build_match_pairs(export_dir / "images", temporal_window=temporal_window, camera_neighbor_count=camera_neighbor_count)
    path = export_dir / "match_pairs.txt"
    path.write_text("\n".join(f"{left} {right}" for left, right in pairs) + ("\n" if pairs else ""), encoding="utf-8")
    return path


def detect_colmap_gpu_options(colmap: str) -> ColmapGpuOptions:
    extraction_use, extraction_index = select_supported_options(
        colmap,
        "feature_extractor",
        (
            ("--FeatureExtraction.use_gpu", "--FeatureExtraction.gpu_index"),
            ("--SiftExtraction.use_gpu", "--SiftExtraction.gpu_index"),
        ),
    )
    matching_use, matching_index = select_supported_options(
        colmap,
        "matches_importer",
        (
            ("--FeatureMatching.use_gpu", "--FeatureMatching.gpu_index"),
            ("--SiftMatching.use_gpu", "--SiftMatching.gpu_index"),
        ),
    )
    return ColmapGpuOptions(
        extraction_use_option=extraction_use,
        extraction_index_option=extraction_index,
        matching_use_option=matching_use,
        matching_index_option=matching_index,
    )


def detect_colmap_mapper_options(colmap: str, command: str = "mapper") -> ColmapMapperOptions:
    help_text = colmap_command_help(colmap, command)
    if not help_text:
        if command != "mapper":
            return ColmapMapperOptions(None, None, None, None)
        return ColmapMapperOptions()
    snapshot_path = "--Mapper.snapshot_path" if "--Mapper.snapshot_path" in help_text else None
    snapshot_images_freq = "--Mapper.snapshot_images_freq" if "--Mapper.snapshot_images_freq" in help_text else None
    if snapshot_images_freq is None and "--Mapper.snapshot_frames_freq" in help_text:
        snapshot_images_freq = "--Mapper.snapshot_frames_freq"
    refine_sensor_from_rig = "--Mapper.ba_refine_sensor_from_rig" if "--Mapper.ba_refine_sensor_from_rig" in help_text else None
    if "--Mapper.tri_ignore_two_view_tracks" in help_text:
        return ColmapMapperOptions(
            "--Mapper.tri_ignore_two_view_tracks",
            snapshot_path,
            snapshot_images_freq,
            refine_sensor_from_rig,
        )
    if "--Mapper.ignore_two_view_tracks" in help_text:
        return ColmapMapperOptions(
            "--Mapper.ignore_two_view_tracks",
            snapshot_path,
            snapshot_images_freq,
            refine_sensor_from_rig,
        )
    return ColmapMapperOptions(None, snapshot_path, snapshot_images_freq, refine_sensor_from_rig)


def select_supported_options(
    colmap: str,
    command: str,
    candidates: tuple[tuple[str, str], ...],
) -> tuple[str | None, str | None]:
    help_text = colmap_command_help(colmap, command)
    if not help_text:
        return None, None
    for use_option, index_option in candidates:
        if use_option in help_text and index_option in help_text:
            return use_option, index_option
    return None, None


def colmap_command_supports_option(colmap: str, command: str, option: str) -> bool:
    return option in colmap_command_help(colmap, command)


def colmap_command_help(colmap: str, command: str) -> str:
    try:
        result = subprocess.run(
            [colmap, command, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return f"{result.stdout}\n{result.stderr}"


def build_match_pairs(images_dir: Path, temporal_window: int = 3, camera_neighbor_count: int = 4) -> list[tuple[str, str]]:
    sources = collect_images_by_source(images_dir)
    source_keys = sorted(sources)
    camera_neighbors = adjacent_camera_names(camera_neighbor_count)
    pair_set: set[tuple[str, str]] = set()
    temporal_window = max(1, int(temporal_window))
    for index, source_key in enumerate(source_keys):
        current = sources[source_key]
        for offset in range(1, temporal_window + 1):
            next_index = index + offset
            if next_index >= len(source_keys):
                break
            following = sources[source_keys[next_index]]
            for camera_name, image_name in current.items():
                add_pair(pair_set, image_name, following.get(camera_name))
                for neighbor_name in camera_neighbors.get(camera_name, ()):
                    add_pair(pair_set, image_name, following.get(neighbor_name))
    return sorted(pair_set)


def collect_images_by_source(images_dir: Path) -> dict[str, dict[str, str]]:
    images: dict[str, dict[str, str]] = {}
    if not images_dir.exists():
        return images
    for image_path in sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
        rel = image_path.relative_to(images_dir)
        if len(rel.parts) < 2:
            continue
        camera_name = rel.parts[0]
        source_key = Path(*rel.parts[1:]).as_posix()
        images.setdefault(source_key, {})[camera_name] = rel.as_posix()
    return images


def adjacent_camera_names(neighbor_count: int = 4) -> dict[str, list[str]]:
    cameras = virtual_cameras()
    neighbor_count = max(1, int(neighbor_count))
    neighbors: dict[str, list[str]] = {}
    for camera in cameras:
        direction = camera_direction(camera)
        candidates = []
        for other in cameras:
            if other.name == camera.name:
                continue
            angle = angle_between(direction, camera_direction(other))
            candidates.append((angle, other.name))
        neighbors[camera.name] = [name for _, name in sorted(candidates)[:neighbor_count]]
    return neighbors


def camera_direction(camera) -> tuple[float, float, float]:
    return tuple(float(value) for value in camera.direction)


def angle_between(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    dot = sum(left[index] * right[index] for index in range(3))
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def add_pair(pair_set: set[tuple[str, str]], left: str, right: str | None) -> None:
    if right is None or left == right:
        return
    pair_set.add(tuple(sorted((left, right))))


def run(command: list[str], cwd: Path) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


if __name__ == "__main__":
    raise SystemExit(main())
