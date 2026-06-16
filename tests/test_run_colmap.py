from __future__ import annotations

import math

from colmap_mask.tools.run_colmap import ColmapRunSettings, build_colmap_steps, build_match_pairs, camera_params_for


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
    assert "--FeatureMatching.guided_matching" in steps[2].command
    assert steps[2].command[1] == "matches_importer"
    assert (tmp_path / "match_pairs.txt").exists()


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
