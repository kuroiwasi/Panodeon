from __future__ import annotations

import math
import sqlite3

from colmap_mask.core.colmap_export import virtual_cameras
from colmap_mask.tools.run_colmap import (
    ColmapGpuOptions,
    ColmapMapperOptions,
    ColmapRunSettings,
    build_colmap_steps,
    build_match_pairs,
    camera_params_for,
    adjacent_camera_names,
    colmap_command_supports_option,
    database_has_rows,
    dense_model_exists,
    detect_colmap_mapper_options,
    select_supported_options,
    best_reconstruction_model_dir,
    should_overwrite_outputs,
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
        "Sequential matching",
    ]
    assert steps[0].command[0] == "colmap.exe"
    assert "--SiftExtraction.max_num_features" in steps[0].command
    assert "16384" in steps[0].command
    assert "--SiftExtraction.estimate_affine_shape" not in steps[0].command
    assert "--SiftExtraction.domain_size_pooling" not in steps[0].command
    assert steps[0].command[steps[0].command.index("--SiftExtraction.use_gpu") + 1] == "1"
    assert "--FeatureMatching.guided_matching" in steps[2].command
    assert "--FeatureMatching.rig_verification" in steps[2].command
    assert "--FeatureMatching.skip_image_pairs_in_same_frame" in steps[2].command
    assert steps[2].command[steps[2].command.index("--SiftMatching.use_gpu") + 1] == "1"
    assert steps[2].command[1] == "sequential_matcher"


def test_build_colmap_steps_can_use_pair_list_matching(tmp_path) -> None:
    steps = build_colmap_steps(tmp_path, ColmapRunSettings(matcher="pairs", skip_mapping=True))
    assert steps[2].name == "Pair-list matching"
    assert steps[2].command[1] == "matches_importer"
    assert (tmp_path / "match_pairs.txt").exists()


def test_build_colmap_steps_can_disable_gpu(tmp_path) -> None:
    steps = build_colmap_steps(tmp_path, ColmapRunSettings(use_gpu=False, skip_mapping=True))
    assert "--SiftExtraction.use_gpu" not in steps[0].command
    assert "--SiftMatching.use_gpu" not in steps[2].command


def test_build_colmap_steps_can_enable_covariant_sift(tmp_path) -> None:
    steps = build_colmap_steps(tmp_path, ColmapRunSettings(covariant_sift=True, skip_mapping=True))
    assert steps[0].command[steps[0].command.index("--SiftExtraction.estimate_affine_shape") + 1] == "1"
    assert steps[0].command[steps[0].command.index("--SiftExtraction.domain_size_pooling") + 1] == "1"


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
    assert mapper_command[1] == "mapper"
    assert "--Mapper.ba_refine_focal_length" in mapper_command
    assert "--Mapper.ba_refine_principal_point" in mapper_command
    assert "--Mapper.ba_refine_extra_params" in mapper_command
    assert mapper_command[mapper_command.index("--Mapper.ba_refine_sensor_from_rig") + 1] == "0"
    assert "--Mapper.tri_ignore_two_view_tracks" in mapper_command
    assert "--Mapper.tri_min_angle" in mapper_command


def test_build_colmap_steps_can_use_hierarchical_mapper(tmp_path) -> None:
    steps = build_colmap_steps(tmp_path, ColmapRunSettings(sparse_mapper="hierarchical_mapper"))
    mapper_step = steps[-1]
    assert mapper_step.name == "Hierarchical sparse mapping"
    assert mapper_step.command[1] == "hierarchical_mapper"
    assert mapper_step.command[mapper_step.command.index("--output_path") + 1] == str(tmp_path / "sparse")


def test_build_colmap_steps_can_enable_mapper_snapshots(tmp_path) -> None:
    snapshot_path = tmp_path / "snapshots"
    steps = build_colmap_steps(
        tmp_path,
        ColmapRunSettings(mapper_snapshot_path=snapshot_path, mapper_snapshot_images_freq=25),
    )
    mapper_command = steps[-1].command
    assert mapper_command[mapper_command.index("--Mapper.snapshot_path") + 1] == str(snapshot_path)
    assert mapper_command[mapper_command.index("--Mapper.snapshot_images_freq") + 1] == "25"


def test_build_colmap_steps_skips_unsupported_mapper_snapshots(tmp_path) -> None:
    steps = build_colmap_steps(
        tmp_path,
        ColmapRunSettings(
            mapper_snapshot_path=tmp_path / "snapshots",
            mapper_options=ColmapMapperOptions(
                ignore_two_view_tracks_option="--Mapper.tri_ignore_two_view_tracks",
                snapshot_path_option=None,
                snapshot_images_freq_option=None,
            ),
        ),
    )
    assert "--Mapper.snapshot_path" not in steps[-1].command
    assert "--Mapper.snapshot_images_freq" not in steps[-1].command


def test_build_colmap_steps_can_run_dense_reconstruction(tmp_path) -> None:
    steps = build_colmap_steps(tmp_path, ColmapRunSettings(dense_reconstruction=True))
    step_names = [step.name for step in steps]
    assert step_names[-3:] == [
        "Image undistortion",
        "Patch-match stereo",
        "Stereo fusion",
    ]
    assert steps[-3].command[1] == "image_undistorter"
    assert steps[-3].command[steps[-3].command.index("--input_path") + 1] == str(tmp_path / "sparse" / "0")
    assert steps[-2].command[1] == "patch_match_stereo"
    assert "--PatchMatchStereo.geom_consistency" in steps[-2].command
    assert steps[-1].command[1] == "stereo_fusion"
    assert steps[-1].command[steps[-1].command.index("--output_path") + 1] == str(tmp_path / "dense" / "fused.ply")


def test_build_colmap_steps_can_run_rig_bundle_adjustment(tmp_path) -> None:
    steps = build_colmap_steps(tmp_path, ColmapRunSettings(rig_bundle_adjustment=True))
    rig_ba_step = steps[-1]
    assert rig_ba_step.name == "Rig bundle adjustment"
    assert rig_ba_step.command[1] == "bundle_adjuster"
    assert rig_ba_step.command[rig_ba_step.command.index("--input_path") + 1] == str(tmp_path / "sparse" / "0")
    assert rig_ba_step.command[rig_ba_step.command.index("--output_path") + 1] == str(tmp_path / "sparse_rig_ba")
    assert "--BundleAdjustment.refine_focal_length" in rig_ba_step.command
    assert rig_ba_step.command[rig_ba_step.command.index("--BundleAdjustment.refine_sensor_from_rig") + 1] == "0"


def test_dense_reconstruction_prefers_rig_bundle_adjusted_model(tmp_path) -> None:
    rig_ba_dir = tmp_path / "sparse_rig_ba"
    rig_ba_dir.mkdir()
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (rig_ba_dir / name).write_bytes(b"")

    assert best_reconstruction_model_dir(tmp_path, prefer_rig_ba=True) == rig_ba_dir
    steps = build_colmap_steps(tmp_path, ColmapRunSettings(rig_bundle_adjustment=True, dense_reconstruction=True))
    dense_step = next(step for step in steps if step.name == "Image undistortion")
    assert dense_step.command[dense_step.command.index("--input_path") + 1] == str(rig_ba_dir)


def test_build_colmap_steps_skips_completed_dense_output(tmp_path) -> None:
    dense_dir = tmp_path / "dense"
    dense_dir.mkdir()
    (dense_dir / "fused.ply").write_bytes(b"")

    steps = build_colmap_steps(tmp_path, ColmapRunSettings(dense_reconstruction=True, skip_completed=True))

    assert not any(step.name == "Stereo fusion" for step in steps)
    assert dense_model_exists(dense_dir)


def test_build_colmap_steps_supports_old_mapper_two_view_option(tmp_path) -> None:
    steps = build_colmap_steps(
        tmp_path,
        ColmapRunSettings(mapper_options=ColmapMapperOptions("--Mapper.ignore_two_view_tracks")),
    )
    mapper_command = steps[-1].command
    assert "--Mapper.ignore_two_view_tracks" in mapper_command
    assert "--Mapper.tri_ignore_two_view_tracks" not in mapper_command


def test_detect_colmap_mapper_options_prefers_current_name(monkeypatch) -> None:
    def fake_help(colmap: str, command: str) -> str:
        return (
            "--Mapper.tri_ignore_two_view_tracks\n"
            "--Mapper.ignore_two_view_tracks\n"
            "--Mapper.snapshot_path\n"
            "--Mapper.snapshot_images_freq\n"
            "--Mapper.ba_refine_sensor_from_rig"
        )

    monkeypatch.setattr("colmap_mask.tools.run_colmap.colmap_command_help", fake_help)
    options = detect_colmap_mapper_options("colmap")
    assert options.ignore_two_view_tracks_option == "--Mapper.tri_ignore_two_view_tracks"
    assert options.snapshot_supported
    assert options.refine_sensor_from_rig_option == "--Mapper.ba_refine_sensor_from_rig"


def test_detect_colmap_mapper_options_falls_back_to_old_name(monkeypatch) -> None:
    def fake_help(colmap: str, command: str) -> str:
        return "--Mapper.ignore_two_view_tracks"

    monkeypatch.setattr("colmap_mask.tools.run_colmap.colmap_command_help", fake_help)
    assert detect_colmap_mapper_options("colmap").ignore_two_view_tracks_option == "--Mapper.ignore_two_view_tracks"


def test_detect_colmap_mapper_options_disables_unknown_hierarchical_options(monkeypatch) -> None:
    monkeypatch.setattr("colmap_mask.tools.run_colmap.colmap_command_help", lambda colmap, command: "")
    options = detect_colmap_mapper_options("colmap", "hierarchical_mapper")
    assert options.ignore_two_view_tracks_option is None
    assert not options.snapshot_supported


def test_build_match_pairs_uses_temporal_and_adjacent_cameras(tmp_path) -> None:
    images_dir = tmp_path / "images"
    cameras = virtual_cameras()
    anchor = cameras[0].name
    neighbor = adjacent_camera_names(neighbor_count=1)[anchor][0]
    for frame in ("frame_000001.jpg", "frame_000002.jpg"):
        for camera in (anchor, neighbor):
            path = images_dir / camera / frame
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    pairs = build_match_pairs(images_dir, temporal_window=1, camera_neighbor_count=1)

    assert (f"{anchor}/frame_000001.jpg", f"{anchor}/frame_000002.jpg") in pairs
    assert tuple(sorted((f"{anchor}/frame_000001.jpg", f"{neighbor}/frame_000002.jpg"))) in pairs


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


def test_skip_completed_takes_priority_over_overwrite() -> None:
    assert should_overwrite_outputs(overwrite=True, skip_completed=False)
    assert not should_overwrite_outputs(overwrite=True, skip_completed=True)
    assert not should_overwrite_outputs(overwrite=False, skip_completed=True)
