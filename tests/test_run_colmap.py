from __future__ import annotations

import math
import sqlite3

from colmap_mask.tools.run_colmap import (
    ColmapGpuOptions,
    ColmapRunSettings,
    build_colmap_steps,
    build_match_pairs,
    camera_params_for,
    colmap_command_supports_option,
    database_has_rows,
    select_supported_options,
    sparse_model_exists,
)


def test_camera_params_for_pinhole() -> None:
    params = [float(value) for value in camera_params_for(2048, 90).split(",")]
    assert params == [1024.0, 1024.0, 1024.0, 1024.0]


def test_camera_params_for_non_90_fov() -> None:
    params = [float(value) for value in camera_params_for(1000, 60).split(",")]
    expected_focal = 500 / math.tan(math.radians(30))
    assert params[0] == round(expected_focal, 6)
    assert params[1] == round(expected_focal, 6)
    assert params[2:] == [500.0, 500.0]


def test_build_colmap_steps_can_skip_mapping(tmp_path) -> None:
    steps = build_colmap_steps(tmp_path, ColmapRunSettings(colmap="colmap.exe", skip_mapping=True))
    assert [step.name for step in steps] == [
        "Feature extraction",
        "Rig configuration",
        "Pair-list matching",
    ]
    assert steps[0].command[0] == "colmap.exe"
    assert "--SiftExtraction.max_num_features" in steps[0].command
    assert "16384" in steps[0].command
    assert steps[0].command[steps[0].command.index("--SiftExtraction.use_gpu") + 1] == "1"
    assert "--FeatureMatching.guided_matching" in steps[2].command
    assert steps[2].command[steps[2].command.index("--SiftMatching.use_gpu") + 1] == "1"
    assert steps[2].command[1] == "matches_importer"
    assert (tmp_path / "match_pairs.txt").exists()


def test_build_colmap_steps_can_disable_gpu(tmp_path) -> None:
    steps = build_colmap_steps(tmp_path, ColmapRunSettings(use_gpu=False, skip_mapping=True))
    assert "--SiftExtraction.use_gpu" not in steps[0].command
    assert "--SiftMatching.use_gpu" not in steps[2].command


def test_build_colmap_steps_skips_unsupported_gpu_options(tmp_path) -> None:
    steps = build_colmap_steps(
        tmp_path,
        ColmapRunSettings(
            use_gpu=True,
            skip_mapping=True,
            gpu_options=ColmapGpuOptions(None, None, None, None),
        ),
    )
    assert "--SiftExtraction.use_gpu" not in steps[0].command
    assert "--SiftMatching.use_gpu" not in steps[2].command


def test_colmap_command_supports_option_returns_false_for_missing_binary() -> None:
    assert not colmap_command_supports_option("missing-colmap-binary", "feature_extractor", "--SiftExtraction.use_gpu")


def test_build_colmap_steps_supports_colmap_313_gpu_option_names(tmp_path) -> None:
    options = ColmapGpuOptions(
        "--FeatureExtraction.use_gpu",
        "--FeatureExtraction.gpu_index",
        "--FeatureMatching.use_gpu",
        "--FeatureMatching.gpu_index",
    )
    steps = build_colmap_steps(tmp_path, ColmapRunSettings(gpu_options=options, skip_mapping=True))
    assert steps[0].command[steps[0].command.index("--FeatureExtraction.use_gpu") + 1] == "1"
    assert steps[2].command[steps[2].command.index("--FeatureMatching.use_gpu") + 1] == "1"


def test_select_supported_options_prefers_new_colmap_option_names(monkeypatch) -> None:
    def fake_help(colmap: str, command: str) -> str:
        return "--FeatureExtraction.use_gpu\n--FeatureExtraction.gpu_index\n--SiftExtraction.use_gpu\n--SiftExtraction.gpu_index"

    monkeypatch.setattr("colmap_mask.tools.run_colmap.colmap_command_help", fake_help)
    assert select_supported_options(
        "colmap",
        "feature_extractor",
        (
            ("--FeatureExtraction.use_gpu", "--FeatureExtraction.gpu_index"),
            ("--SiftExtraction.use_gpu", "--SiftExtraction.gpu_index"),
        ),
    ) == ("--FeatureExtraction.use_gpu", "--FeatureExtraction.gpu_index")


def test_build_colmap_steps_uses_optimized_mapper_options(tmp_path) -> None:
    steps = build_colmap_steps(tmp_path, ColmapRunSettings())
    mapper_command = steps[-1].command
    assert "--Mapper.ba_refine_focal_length" in mapper_command
    assert "--Mapper.ba_refine_principal_point" in mapper_command
    assert "--Mapper.ba_refine_extra_params" in mapper_command
    assert "--Mapper.ignore_two_view_tracks" in mapper_command
    assert "--Mapper.tri_min_angle" in mapper_command


def test_build_match_pairs_uses_temporal_and_adjacent_cameras(tmp_path) -> None:
    images_dir = tmp_path / "images"
    for frame in ("frame_000001.jpg", "frame_000002.jpg"):
        for camera in ("cam01_y000_p+00", "cam02_y060_p+00", "cam07_y000_p+60"):
            path = images_dir / camera / frame
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    pairs = build_match_pairs(images_dir, temporal_window=1)

    assert ("cam01_y000_p+00/frame_000001.jpg", "cam01_y000_p+00/frame_000002.jpg") in pairs
    assert ("cam01_y000_p+00/frame_000001.jpg", "cam02_y060_p+00/frame_000002.jpg") in pairs
    assert ("cam01_y000_p+00/frame_000001.jpg", "cam07_y000_p+60/frame_000002.jpg") in pairs


def test_database_has_rows_detects_existing_table(tmp_path) -> None:
    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE images(image_id INTEGER)")
        connection.execute("INSERT INTO images VALUES (1)")
    assert database_has_rows(database_path, "images")
    assert not database_has_rows(database_path, "matches")


def test_sparse_model_exists_detects_mapper_output(tmp_path) -> None:
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model_dir / name).write_bytes(b"")
    assert sparse_model_exists(tmp_path / "sparse")


def test_build_colmap_steps_skips_completed_outputs(tmp_path) -> None:
    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE images(image_id INTEGER)")
        connection.execute("INSERT INTO images VALUES (1)")
        connection.execute("CREATE TABLE frames(frame_id INTEGER)")
        connection.execute("INSERT INTO frames VALUES (1)")
        connection.execute("CREATE TABLE matches(pair_id INTEGER)")
        connection.execute("INSERT INTO matches VALUES (1)")
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model_dir / name).write_bytes(b"")

    steps = build_colmap_steps(tmp_path, ColmapRunSettings(skip_completed=True))

    assert steps == []
