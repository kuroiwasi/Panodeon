from __future__ import annotations

import argparse
import math
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from panodeon.core.colmap_export import (
    PANORAMA_YAW_STEPS,
    VirtualCamera,
    direction_from_yaw_pitch,
    virtual_cameras,
    world_from_colmap_camera,
)
from panodeon.sampler.io import load_trajectory_csv
from panodeon.sampler.models import TrajectoryRecord
from panodeon.tools.run_colmap import best_reconstruction_model_dir, model_files_exist


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    points2d: bytes


@dataclass(frozen=True)
class ColmapPoint3D:
    point3d_id: int
    xyz: np.ndarray
    rgb: tuple[int, int, int]
    error: float
    track: bytes


@dataclass(frozen=True)
class AlignmentReport:
    image_pairs: int
    angle_deg: float
    input_model: Path
    output_model: Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Rotate a COLMAP sparse model so its up direction matches stella.")
    parser.add_argument("path", type=Path, help="Project folder, exports folder, or sparse model folder.")
    parser.add_argument("--trajectory", type=Path, help="stella trajectory.csv. Defaults to the nearest stella/trajectory.csv above the exports folder.")
    parser.add_argument("--input-model", type=Path, help="Input COLMAP sparse model folder.")
    parser.add_argument("--output-model", type=Path, help="Output model folder. Defaults to exports/sparse_stella_rot/0.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output model folder.")
    args = parser.parse_args()

    export_dir, input_model = resolve_export_and_model(args.path, args.input_model)
    trajectory_path = args.trajectory or find_stella_trajectory(export_dir)
    output_model = args.output_model or export_dir / "sparse_stella_rot" / "0"
    report = align_colmap_model_to_stella_up(input_model, trajectory_path, output_model, overwrite=args.overwrite)
    print(f"input: {report.input_model}")
    print(f"output: {report.output_model}")
    print(f"matched images: {report.image_pairs}")
    print(f"rotation angle deg: {report.angle_deg:.6f}")
    return 0


def resolve_export_and_model(path: Path, input_model: Path | None) -> tuple[Path, Path]:
    path = path.resolve()
    if input_model is not None:
        return resolve_export_dir(path), input_model.resolve()
    if model_files_exist(path):
        export_dir = path.parent.parent if path.parent.name in {"sparse", "sparse_rig_ba", "snapshots"} else path.parent
        return export_dir, path
    export_dir = resolve_export_dir(path)
    return export_dir, best_reconstruction_model_dir(export_dir, prefer_rig_ba=True)


def find_stella_trajectory(export_dir: Path, extra_bases: tuple[Path, ...] = ()) -> Path:
    # The sampler writes <output>/stella/trajectory.csv while the UI loads
    # <output>/frames as the project, so exports sits two levels below <output>.
    bases = [*extra_bases, export_dir.parent, export_dir.parent.parent]
    candidates = [Path(base) / "stella" / "trajectory.csv" for base in bases]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def resolve_export_dir(path: Path) -> Path:
    if (path / "images").is_dir() and (path / "masks").is_dir():
        return path
    export_dir = path / "exports"
    if (export_dir / "images").is_dir() and (export_dir / "masks").is_dir():
        return export_dir
    raise SystemExit(f"Not a Panodeon project, export folder, or COLMAP model folder: {path}")


def align_colmap_model_to_stella_up(
    input_model: Path,
    trajectory_path: Path,
    output_model: Path,
    *,
    overwrite: bool = False,
) -> AlignmentReport:
    input_model = input_model.resolve()
    output_model = output_model.resolve()
    if not model_files_exist(input_model):
        raise ValueError(f"Missing COLMAP model files: {input_model}")
    if not trajectory_path.is_file():
        raise ValueError(f"Missing stella trajectory: {trajectory_path}")
    if output_model.exists():
        if not overwrite:
            raise ValueError(f"Output model already exists: {output_model}")
        shutil.rmtree(output_model)

    images = read_images_binary(input_model / "images.bin")
    points3d = read_points3d_binary(input_model / "points3D.bin")
    trajectory = {record.frame_index: record for record in load_trajectory_csv(trajectory_path)}
    rotation, matched_count = estimate_up_alignment_rotation(images, trajectory)
    pivot = camera_center_pivot(images)
    translation = pivot - rotation @ pivot

    output_model.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_model / "cameras.bin", output_model / "cameras.bin")
    write_images_binary(output_model / "images.bin", rotate_images(images, rotation, translation))
    write_points3d_binary(output_model / "points3D.bin", rotate_points3d(points3d, rotation, translation))
    return AlignmentReport(
        image_pairs=matched_count,
        angle_deg=rotation_angle_deg(rotation),
        input_model=input_model,
        output_model=output_model,
    )


def estimate_up_alignment_rotation(
    images: list[ColmapImage],
    trajectory_by_frame: dict[int, TrajectoryRecord],
) -> tuple[np.ndarray, int]:
    camera_rig_rotations = camera_from_rig_by_name()
    colmap_up = np.zeros(3, dtype=np.float64)
    stella_up = np.zeros(3, dtype=np.float64)
    matched_count = 0
    for image in images:
        camera_name = image.name.replace("\\", "/").split("/", 1)[0]
        cam_from_rig = camera_rig_rotations.get(camera_name)
        frame_index = frame_index_from_image_name(image.name)
        if cam_from_rig is None or frame_index is None:
            continue
        record = trajectory_by_frame.get(frame_index)
        if record is None or not record.pose_valid or None in (record.qw, record.qx, record.qy, record.qz):
            continue
        world_from_colmap_cam = qvec_to_rotmat(image.qvec).T
        world_from_colmap_rig = world_from_colmap_cam @ cam_from_rig
        colmap_up += normalize(-world_from_colmap_rig[:, 1])
        world_from_stella_cam = qvec_to_rotmat(np.array([record.qw, record.qx, record.qy, record.qz], dtype=np.float64))
        stella_up += normalize(-world_from_stella_cam[:, 1])
        matched_count += 1
    if matched_count < 2:
        raise ValueError("Not enough COLMAP/stella pose pairs to estimate rotation")
    return rotation_between(normalize(colmap_up), normalize(stella_up)), matched_count


def camera_from_rig_by_name() -> dict[str, np.ndarray]:
    layouts = (virtual_cameras(), virtual_cameras_for_pitches((-35.0, 0.0, 35.0)))
    result = {}
    for cameras in layouts:
        rig_world_from_camera = world_from_colmap_camera(cameras[0])
        for camera in cameras:
            result[camera.name] = world_from_colmap_camera(camera).T @ rig_world_from_camera
    return result


def virtual_cameras_for_pitches(pitches: tuple[float, ...]) -> list[VirtualCamera]:
    cameras: list[VirtualCamera] = []
    yaw_step_deg = 360.0 / PANORAMA_YAW_STEPS
    for pitch_deg in pitches:
        yaw_offset_deg = yaw_step_deg * 0.5 if pitch_deg > 0.0 else 0.0
        for yaw_index in range(PANORAMA_YAW_STEPS):
            yaw_deg = (yaw_index * yaw_step_deg + yaw_offset_deg) % 360.0
            cameras.append(
                VirtualCamera(
                    camera_id=len(cameras) + 1,
                    yaw_deg=yaw_deg,
                    pitch_deg=pitch_deg,
                    direction=direction_from_yaw_pitch(yaw_deg, pitch_deg),
                )
            )
    return cameras


def frame_index_from_image_name(name: str) -> int | None:
    stem = Path(name.replace("\\", "/")).stem
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else None


def camera_center_pivot(images: list[ColmapImage]) -> np.ndarray:
    centers = []
    for image in images:
        rotation = qvec_to_rotmat(image.qvec)
        centers.append(-rotation.T @ image.tvec)
    return np.mean(np.asarray(centers), axis=0) if centers else np.zeros(3, dtype=np.float64)


def rotate_images(images: list[ColmapImage], rotation: np.ndarray, translation: np.ndarray) -> list[ColmapImage]:
    rotated = []
    for image in images:
        new_rotation = qvec_to_rotmat(image.qvec) @ rotation.T
        new_tvec = image.tvec - new_rotation @ translation
        rotated.append(ColmapImage(image.image_id, rotmat_to_qvec(new_rotation), new_tvec, image.camera_id, image.name, image.points2d))
    return rotated


def rotate_points3d(points3d: list[ColmapPoint3D], rotation: np.ndarray, translation: np.ndarray) -> list[ColmapPoint3D]:
    return [
        ColmapPoint3D(point.point3d_id, rotation @ point.xyz + translation, point.rgb, point.error, point.track)
        for point in points3d
    ]


def read_images_binary(path: Path) -> list[ColmapImage]:
    data = path.read_bytes()
    offset = 0
    (image_count,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    images = []
    for _ in range(image_count):
        image_id = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        qvec = np.array(struct.unpack_from("<4d", data, offset), dtype=np.float64)
        offset += 32
        tvec = np.array(struct.unpack_from("<3d", data, offset), dtype=np.float64)
        offset += 24
        camera_id = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        name_end = data.index(0, offset)
        name = data[offset:name_end].decode("utf-8")
        offset = name_end + 1
        points2d_count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        points2d_size = points2d_count * 24
        points2d = data[offset : offset + points2d_size]
        offset += points2d_size
        images.append(ColmapImage(image_id, qvec, tvec, camera_id, name, points2d))
    return images


def write_images_binary(path: Path, images: list[ColmapImage]) -> None:
    chunks = [struct.pack("<Q", len(images))]
    for image in images:
        chunks.extend(
            [
                struct.pack("<I", image.image_id),
                struct.pack("<4d", *image.qvec),
                struct.pack("<3d", *image.tvec),
                struct.pack("<I", image.camera_id),
                image.name.encode("utf-8") + b"\0",
                struct.pack("<Q", len(image.points2d) // 24),
                image.points2d,
            ]
        )
    path.write_bytes(b"".join(chunks))


def read_points3d_binary(path: Path) -> list[ColmapPoint3D]:
    data = path.read_bytes()
    offset = 0
    (point_count,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    points = []
    for _ in range(point_count):
        point3d_id = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        xyz = np.array(struct.unpack_from("<3d", data, offset), dtype=np.float64)
        offset += 24
        rgb = struct.unpack_from("<3B", data, offset)
        offset += 3
        error = struct.unpack_from("<d", data, offset)[0]
        offset += 8
        track_length = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        track_size = track_length * 8
        track = data[offset : offset + track_size]
        offset += track_size
        points.append(ColmapPoint3D(point3d_id, xyz, rgb, error, track))
    return points


def write_points3d_binary(path: Path, points: list[ColmapPoint3D]) -> None:
    chunks = [struct.pack("<Q", len(points))]
    for point in points:
        chunks.extend(
            [
                struct.pack("<Q", point.point3d_id),
                struct.pack("<3d", *point.xyz),
                struct.pack("<3B", *point.rgb),
                struct.pack("<d", point.error),
                struct.pack("<Q", len(point.track) // 8),
                point.track,
            ]
        )
    path.write_bytes(b"".join(chunks))


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(qvec)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    m = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qvec = np.array([0.25 * scale, (m[2, 1] - m[1, 2]) / scale, (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale])
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qvec = np.array([(m[2, 1] - m[1, 2]) / scale, 0.25 * scale, (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale])
        elif axis == 1:
            scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qvec = np.array([(m[0, 2] - m[2, 0]) / scale, (m[0, 1] + m[1, 0]) / scale, 0.25 * scale, (m[1, 2] + m[2, 1]) / scale])
        else:
            scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qvec = np.array([(m[1, 0] - m[0, 1]) / scale, (m[0, 2] + m[2, 0]) / scale, (m[1, 2] + m[2, 1]) / scale, 0.25 * scale])
    qvec = normalize(qvec)
    return qvec if qvec[0] >= 0 else -qvec


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = normalize(source)
    target = normalize(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1.0e-12:
        return np.eye(3)
    if dot < -1.0 + 1.0e-12:
        axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1.0e-12:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0]))
        return axis_angle_to_rotmat(axis, math.pi)
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / float(np.dot(cross, cross)))


def axis_angle_to_rotmat(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    v = 1.0 - c
    return np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ],
        dtype=np.float64,
    )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (np.trace(rotation) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(value, -1.0, 1.0))))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return np.asarray(vector, dtype=np.float64) / norm


if __name__ == "__main__":
    raise SystemExit(main())
