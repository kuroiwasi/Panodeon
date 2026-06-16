from __future__ import annotations

import argparse
import math
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from colmap_mask.core.colmap_export import virtual_cameras
from colmap_mask.core.image_io import IMAGE_EXTENSIONS


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
class ColmapRunSettings:
    colmap: str = "colmap"
    tile_size: int = 3072
    fov_deg: float = 90.0
    matcher: str = "pairs"
    camera_model: str = "PINHOLE"
    skip_mapping: bool = False
    pair_temporal_window: int = 3
    use_gpu: bool = True
    gpu_index: str = "-1"
    gpu_options: ColmapGpuOptions = field(default_factory=ColmapGpuOptions)
    skip_completed: bool = False


@dataclass(frozen=True)
class ColmapStep:
    name: str
    command: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run COLMAP on a 360 Colmap Mask export folder.")
    parser.add_argument("export_dir", type=Path, help="Folder containing images/, masks/, and rig_config.json.")
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable path.")
    parser.add_argument("--tile-size", type=int, default=3072, help="Perspective tile size used during export.")
    parser.add_argument("--fov", type=float, default=90.0, help="Perspective FOV used during export.")
    parser.add_argument("--matcher", choices=("pairs", "exhaustive", "sequential", "vocab_tree"), default="pairs")
    parser.add_argument("--camera-model", default="PINHOLE")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing database.db and sparse/ before running.")
    parser.add_argument("--skip-mapping", action="store_true", help="Run only feature extraction, rig setup, and matching.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip COLMAP steps that already have output.")
    parser.add_argument("--no-gpu", action="store_true", help="Disable COLMAP SIFT GPU extraction and matching.")
    parser.add_argument("--gpu-index", default="-1", help="COLMAP GPU index. -1 lets COLMAP choose/use all available GPUs.")
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    validate_export_dir(export_dir)
    database_path = export_dir / "database.db"
    sparse_dir = export_dir / "sparse"
    if args.overwrite:
        if database_path.exists():
            database_path.unlink()
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    gpu_options = detect_colmap_gpu_options(args.colmap)
    if not args.no_gpu and not (gpu_options.extraction_supported and gpu_options.matching_supported):
        print("GPU options unsupported by this COLMAP binary. Running without GPU flags.", flush=True)
    settings = ColmapRunSettings(
        colmap=args.colmap,
        tile_size=args.tile_size,
        fov_deg=args.fov,
        matcher=args.matcher,
        camera_model=args.camera_model,
        skip_mapping=args.skip_mapping,
        use_gpu=not args.no_gpu,
        gpu_index=args.gpu_index,
        gpu_options=gpu_options,
        skip_completed=args.skip_completed,
    )
    for step in build_colmap_steps(export_dir, settings):
        run(step.command, export_dir)
    return 0


def validate_export_dir(export_dir: Path) -> None:
    missing = [name for name in ("images", "masks", "rig_config.json") if not (export_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing COLMAP export files in {export_dir}: {', '.join(missing)}")


def camera_params_for(tile_size: int, fov_deg: float) -> str:
    focal = tile_size * 0.5 / math.tan(math.radians(fov_deg) * 0.5)
    center = tile_size * 0.5
    return f"{focal:.6f},{focal:.6f},{center:.6f},{center:.6f}"


def build_colmap_steps(export_dir: Path, settings: ColmapRunSettings) -> list[ColmapStep]:
    database_path = export_dir / "database.db"
    sparse_dir = export_dir / "sparse"
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
        "--SiftExtraction.estimate_affine_shape",
        "1",
        "--SiftExtraction.domain_size_pooling",
        "1",
    ]
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
    if not (settings.skip_completed and database_has_rows(database_path, "images")):
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
        match_list_path = write_match_pair_list(export_dir, settings.pair_temporal_window)
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
        steps.append(
            ColmapStep(
                "Sparse mapping",
                [
                    settings.colmap,
                    "mapper",
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
                    "--Mapper.ignore_two_view_tracks",
                    "0",
                    "--Mapper.tri_min_angle",
                    "0.75",
                ],
            )
        )
    return steps


def database_has_rows(database_path: Path, table_name: str) -> bool:
    if not database_path.exists():
        return False
    try:
        with sqlite3.connect(str(database_path)) as connection:
            row = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            if row is None:
                return False
            count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    except sqlite3.Error:
        return False
    return int(count) > 0


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


def write_match_pair_list(export_dir: Path, temporal_window: int = 3) -> Path:
    pairs = build_match_pairs(export_dir / "images", temporal_window=temporal_window)
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


def build_match_pairs(images_dir: Path, temporal_window: int = 3) -> list[tuple[str, str]]:
    sources = collect_images_by_source(images_dir)
    source_keys = sorted(sources)
    camera_neighbors = adjacent_camera_names()
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


def adjacent_camera_names() -> dict[str, list[str]]:
    cameras = virtual_cameras()
    by_pose = {(camera.pitch_deg, camera.yaw_deg): camera.name for camera in cameras}
    yaw_values = sorted({camera.yaw_deg for camera in cameras})
    pitch_values = sorted({camera.pitch_deg for camera in cameras})
    neighbors: dict[str, list[str]] = {}
    for camera in cameras:
        names: list[str] = []
        yaw_index = yaw_values.index(camera.yaw_deg)
        for yaw in (yaw_values[(yaw_index - 1) % len(yaw_values)], yaw_values[(yaw_index + 1) % len(yaw_values)]):
            neighbor = by_pose.get((camera.pitch_deg, yaw))
            if neighbor is not None:
                names.append(neighbor)
        for pitch in pitch_values:
            if pitch == camera.pitch_deg:
                continue
            neighbor = by_pose.get((pitch, camera.yaw_deg))
            if neighbor is not None:
                names.append(neighbor)
        neighbors[camera.name] = sorted(set(names))
    return neighbors


def add_pair(pair_set: set[tuple[str, str]], left: str, right: str | None) -> None:
    if right is None or left == right:
        return
    pair_set.add(tuple(sorted((left, right))))


def run(command: list[str], cwd: Path) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


if __name__ == "__main__":
    raise SystemExit(main())
