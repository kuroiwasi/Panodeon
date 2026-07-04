from __future__ import annotations

import numpy as np

from panodeon.sampler.io import load_trajectory_csv
from panodeon.sampler.models import TrajectoryRecord
from panodeon.tools.align_colmap_stella_rot import (
    ColmapImage,
    ColmapPoint3D,
    estimate_up_alignment_rotation,
    qvec_to_rotmat,
    read_images_binary,
    read_points3d_binary,
    rotmat_to_qvec,
    rotation_angle_deg,
    rotation_between,
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
