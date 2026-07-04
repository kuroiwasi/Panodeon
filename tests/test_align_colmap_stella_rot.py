from __future__ import annotations

import math

import numpy as np

from panodeon.core.colmap_export import virtual_cameras, world_from_colmap_camera
from panodeon.sampler.io import load_trajectory_csv
from panodeon.sampler.models import TrajectoryRecord
import pytest

from panodeon.tools.align_colmap_stella_rot import (
    STELLA_ALIGNMENT_MARKER,
    ColmapImage,
    ColmapPoint3D,
    align_export_sparse_to_stella_up,
    axis_angle_to_rotmat,
    estimate_up_alignment_rotation,
    find_stella_trajectory,
    qvec_to_rotmat,
    read_images_binary,
    read_points3d_binary,
    rotmat_to_qvec,
    rotation_angle_deg,
    rotation_between,
    sparse_is_stella_aligned,
    write_images_binary,
    write_points3d_binary,
)


def test_load_trajectory_csv_reads_quaternion_columns(tmp_path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text(
        "frame_index,pts,time_base_num,time_base_den,cx,cy,cz,qw,qx,qy,qz\n"
        "1,1,1,30,0,0,0,0.5,0.5,0.5,0.5\n",
        encoding="utf-8",
    )

    record = load_trajectory_csv(path)[0]

    assert record.qw == 0.5
    assert record.qx == 0.5
    assert record.qy == 0.5
    assert record.qz == 0.5


def test_find_stella_trajectory_walks_up_to_sampler_output(tmp_path) -> None:
    export_dir = tmp_path / "frames" / "exports"
    export_dir.mkdir(parents=True)
    trajectory = tmp_path / "stella" / "trajectory.csv"
    trajectory.parent.mkdir()
    trajectory.write_text("", encoding="utf-8")

    assert find_stella_trajectory(export_dir) == trajectory


def test_find_stella_trajectory_prefers_extra_bases(tmp_path) -> None:
    export_dir = tmp_path / "frames" / "exports"
    export_dir.mkdir(parents=True)
    project_trajectory = tmp_path / "frames" / "stella" / "trajectory.csv"
    project_trajectory.parent.mkdir()
    project_trajectory.write_text("", encoding="utf-8")
    sampler_output = tmp_path / "video_frames"
    sampler_trajectory = sampler_output / "stella" / "trajectory.csv"
    sampler_trajectory.parent.mkdir(parents=True)
    sampler_trajectory.write_text("", encoding="utf-8")

    assert find_stella_trajectory(export_dir, (sampler_output,)) == sampler_trajectory


def test_find_stella_trajectory_defaults_to_first_candidate(tmp_path) -> None:
    export_dir = tmp_path / "frames" / "exports"
    export_dir.mkdir(parents=True)

    assert find_stella_trajectory(export_dir) == tmp_path / "frames" / "stella" / "trajectory.csv"


def test_estimate_up_alignment_rotation_uses_stella_up() -> None:
    target = rotation_between(np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    qvec = rotmat_to_qvec(target)
    images = [
        ColmapImage(1, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 1, "cam01_y000_p+00/frame_000001.jpg", b""),
        ColmapImage(2, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 1, "cam01_y000_p+00/frame_000002.jpg", b""),
    ]
    trajectory = {
        1: TrajectoryRecord(1, 1, 1, 30, 0, 0, 0, 0, 0, qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3]),
        2: TrajectoryRecord(2, 2, 1, 30, 0, 0, 0, 0, 0, qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3]),
    }

    rotation, matched = estimate_up_alignment_rotation(images, trajectory)

    assert matched == 2
    np.testing.assert_allclose(rotation @ np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), atol=1e-7)
    assert abs(rotation_angle_deg(rotation) - 90.0) < 1e-6


def test_estimate_up_alignment_rotation_uses_level_rig_reference() -> None:
    # The pano frame is left-handed, so a reflection relates it to the proper COLMAP world.
    reflect = np.diag([1.0, 1.0, -1.0])
    offset = axis_angle_to_rotmat(np.array([1.0, 0.0, 0.0]), math.radians(25.0))
    images = []
    for camera in virtual_cameras():
        world_from_cam = offset @ reflect @ world_from_colmap_camera(camera)
        images.append(
            ColmapImage(
                camera.camera_id,
                rotmat_to_qvec(world_from_cam.T),
                np.zeros(3),
                camera.camera_id,
                f"{camera.name}/frame_000001.jpg",
                b"",
            )
        )
    stella_qvec = rotmat_to_qvec(reflect @ np.diag([1.0, -1.0, 1.0]))
    trajectory = {
        1: TrajectoryRecord(1, 1, 1, 30, 0, 0, 0, 0, 0, qw=stella_qvec[0], qx=stella_qvec[1], qy=stella_qvec[2], qz=stella_qvec[3]),
    }

    rotation, matched = estimate_up_alignment_rotation(images, trajectory)

    assert matched == len(images)
    np.testing.assert_allclose(rotation @ offset @ np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-5)
    assert abs(rotation_angle_deg(rotation) - 25.0) < 1e-3


def _write_test_export(tmp_path) -> tuple:
    export_dir = tmp_path / "exports"
    model_dir = export_dir / "sparse" / "0"
    model_dir.mkdir(parents=True)
    target = rotation_between(np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    qvec = rotmat_to_qvec(target)
    images = [
        ColmapImage(1, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 1, "cam01_y000_p+00/frame_000001.jpg", b""),
        ColmapImage(2, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 1, "cam01_y000_p+00/frame_000002.jpg", b""),
    ]
    (model_dir / "cameras.bin").write_bytes(b"test-cameras")
    write_images_binary(model_dir / "images.bin", images)
    write_points3d_binary(model_dir / "points3D.bin", [ColmapPoint3D(9, np.array([1.0, 2.0, 3.0]), (1, 2, 3), 0.25, b"")])
    trajectory_path = tmp_path / "trajectory.csv"
    rows = "".join(f"{index},{index},1,30,0,0,0,{qvec[0]},{qvec[1]},{qvec[2]},{qvec[3]}\n" for index in (1, 2))
    trajectory_path.write_text(
        "frame_index,pts,time_base_num,time_base_den,cx,cy,cz,qw,qx,qy,qz\n" + rows,
        encoding="utf-8",
    )
    return export_dir, trajectory_path


def test_align_export_sparse_backs_up_original_and_replaces_sparse(tmp_path) -> None:
    export_dir, trajectory_path = _write_test_export(tmp_path)

    report = align_export_sparse_to_stella_up(export_dir, trajectory_path)

    assert report.input_model == export_dir / "sparse_orig" / "0"
    assert report.output_model == export_dir / "sparse" / "0"
    assert (export_dir / "sparse_orig" / "0" / "cameras.bin").read_bytes() == b"test-cameras"
    assert sparse_is_stella_aligned(export_dir / "sparse")
    original = read_images_binary(export_dir / "sparse_orig" / "0" / "images.bin")
    aligned = read_images_binary(export_dir / "sparse" / "0" / "images.bin")
    np.testing.assert_allclose(qvec_to_rotmat(original[0].qvec), np.eye(3))
    assert abs(rotation_angle_deg(qvec_to_rotmat(aligned[0].qvec)) - 90.0) < 1e-6


def test_align_export_sparse_rerun_requires_overwrite_and_keeps_backup(tmp_path) -> None:
    export_dir, trajectory_path = _write_test_export(tmp_path)
    align_export_sparse_to_stella_up(export_dir, trajectory_path)

    with pytest.raises(ValueError, match="already exists"):
        align_export_sparse_to_stella_up(export_dir, trajectory_path)

    report = align_export_sparse_to_stella_up(export_dir, trajectory_path, overwrite=True)
    assert report.input_model == export_dir / "sparse_orig" / "0"
    assert (export_dir / "sparse_orig" / "0" / "cameras.bin").read_bytes() == b"test-cameras"


def test_align_export_sparse_rejects_stale_backup(tmp_path) -> None:
    export_dir, trajectory_path = _write_test_export(tmp_path)
    (export_dir / "sparse_orig").mkdir()

    with pytest.raises(ValueError, match="sparse_orig"):
        align_export_sparse_to_stella_up(export_dir, trajectory_path)


def test_align_export_sparse_restores_original_on_failure(tmp_path) -> None:
    export_dir, trajectory_path = _write_test_export(tmp_path)
    trajectory_path.write_text(
        "frame_index,pts,time_base_num,time_base_den,cx,cy,cz,qw,qx,qy,qz\n1,1,1,30,0,0,0,,,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        align_export_sparse_to_stella_up(export_dir, trajectory_path)

    assert (export_dir / "sparse" / "0" / "cameras.bin").read_bytes() == b"test-cameras"
    assert not (export_dir / "sparse_orig").exists()


def test_colmap_binary_roundtrip(tmp_path) -> None:
    images = [
        ColmapImage(
            7,
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 3.0]),
            3,
            "cam01_y000_p+00/frame_000001.jpg",
            b"",
        )
    ]
    points = [ColmapPoint3D(9, np.array([1.0, 2.0, 3.0]), (1, 2, 3), 0.25, b"")]

    write_images_binary(tmp_path / "images.bin", images)
    write_points3d_binary(tmp_path / "points3D.bin", points)

    loaded_images = read_images_binary(tmp_path / "images.bin")
    loaded_points = read_points3d_binary(tmp_path / "points3D.bin")
    assert loaded_images[0].name == images[0].name
    np.testing.assert_allclose(qvec_to_rotmat(loaded_images[0].qvec), np.eye(3))
    np.testing.assert_allclose(loaded_images[0].tvec, images[0].tvec)
    assert loaded_points[0].point3d_id == 9
    np.testing.assert_allclose(loaded_points[0].xyz, points[0].xyz)
