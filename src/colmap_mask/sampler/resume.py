from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .io import write_json

STATE_FILENAME = ".workflow_state.json"
STATE_SCHEMA_VERSION = 1
STEP_KEYS = {
    "中間動画生成": "proxy",
    "軌跡抽出": "stella",
    "フレーム選定": "sample",
    "画像抽出": "extract",
}


def step_key(step_name: str) -> str:
    return STEP_KEYS[step_name]


def load_workflow_state(output_dir: Path) -> dict[str, Any]:
    path = output_dir / STATE_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"schema_version": STATE_SCHEMA_VERSION, "steps": {}}
    if data.get("schema_version") != STATE_SCHEMA_VERSION or not isinstance(data.get("steps"), dict):
        return {"schema_version": STATE_SCHEMA_VERSION, "steps": {}}
    return data


def step_signature(settings: Any, key: str) -> str:
    if key == "proxy":
        width, height = settings.proxy_dimensions()
        payload = {
            "video": _file_identity(settings.video_path),
            "width": width,
            "height": height,
            "codec": settings.proxy_codec,
            "crf": settings.proxy_crf,
            "preset": settings.proxy_preset,
        }
    elif key == "stella":
        payload = {
            "video": _file_identity(settings.proxy_video_path()),
            "source_frame_index": _file_identity(
                settings.output_dir / "proxy" / "video_frames.csv"
            ),
            "executable": _file_identity(settings.executable),
            "vocabulary": _file_identity(settings.vocabulary),
            "camera_config": _optional_file_identity(settings.camera_config),
            "frame_skip": settings.frame_skip,
        }
    elif key == "sample":
        payload = {
            "trajectory": _file_identity(settings.output_dir / "stella" / "trajectory.csv"),
            "video": _file_identity(settings.proxy_video_path()),
            "video_frame_aligned": True,
            "sampler_config": _optional_file_identity(settings.sampler_config),
        }
    elif key == "extract":
        payload = {
            "video": _file_identity(settings.video_path),
            "selected": _file_identity(settings.output_dir / "sampled" / "selected_frames.csv"),
            "image_format": settings.image_format,
            "jpeg_quality": settings.jpeg_quality,
        }
    else:
        raise ValueError(f"Unknown workflow step: {key}")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def legacy_stella_signature(settings: Any) -> str:
    payload = {
        "video": _file_identity(settings.video_path),
        "executable": _file_identity(settings.executable),
        "vocabulary": _file_identity(settings.vocabulary),
        "camera_config": _optional_file_identity(settings.camera_config),
        "frame_skip": settings.frame_skip,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def step_is_complete(settings: Any, key: str, state: dict[str, Any], signature: str) -> bool:
    if not _artifacts_complete(settings, key):
        return False
    stored_signature = state.get("steps", {}).get(key)
    if stored_signature is None or stored_signature == signature:
        return True
    return key == "stella" and stored_signature == legacy_stella_signature(settings)


def mark_step_complete(output_dir: Path, state: dict[str, Any], key: str, signature: str) -> None:
    state.setdefault("steps", {})[key] = signature
    write_json(output_dir / STATE_FILENAME, state)


def _artifacts_complete(settings: Any, key: str) -> bool:
    if key == "proxy":
        return _proxy_complete(settings)
    if key == "stella":
        return _stella_complete(settings)
    if key == "sample":
        return _sample_complete(settings.output_dir)
    if key == "extract":
        return _extract_complete(settings.output_dir)
    return False


def _proxy_complete(settings: Any) -> bool:
    directory = settings.output_dir / "proxy"
    video = settings.proxy_video_path()
    metadata_path = directory / "proxy_metadata.json"
    frame_index = directory / "video_frames.csv"
    if not all(_nonempty(path) for path in (video, metadata_path, frame_index)):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = Path(metadata["source_video_path"])
        width, height = settings.proxy_dimensions()
        return (
            os.path.normcase(str(source.resolve()))
            == os.path.normcase(str(settings.video_path.resolve()))
            and Path(metadata["proxy_video_path"]).resolve() == video.resolve()
            and int(metadata["source_frame_count"]) == _csv_row_count(frame_index)
            and int(metadata["width"]) == width
            and int(metadata["height"]) == height
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _stella_complete(settings: Any) -> bool:
    directory = settings.output_dir / "stella"
    trajectory = directory / "trajectory.csv"
    metadata_path = directory / "video_metadata.json"
    if not _nonempty(trajectory) or not _nonempty(metadata_path):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        recorded_video = Path(metadata["video_path"])
    except (OSError, ValueError, TypeError, KeyError):
        return False
    recorded = os.path.normcase(str(recorded_video.resolve()))
    accepted = {
        os.path.normcase(str(settings.video_path.resolve())),
        os.path.normcase(str(settings.proxy_video_path().resolve())),
    }
    return recorded in accepted


def _sample_complete(output_dir: Path) -> bool:
    directory = output_dir / "sampled"
    selected = directory / "selected_frames.csv"
    report_path = directory / "sampler_report.json"
    if not _nonempty(selected) or not _nonempty(report_path):
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected_count = _csv_row_count(selected)
        return selected_count > 0 and int(report["selected_count"]) == selected_count
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _extract_complete(output_dir: Path) -> bool:
    directory = output_dir / "frames"
    report_path = directory / "extraction_report.json"
    manifest_path = directory / "extracted_frames.csv"
    selected_path = output_dir / "sampled" / "selected_frames.csv"
    if not all(_nonempty(path) for path in (report_path, manifest_path, selected_path)):
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected_count = _csv_row_count(selected_path)
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if int(report["requested_count"]) != selected_count:
            return False
        if int(report["extracted_count"]) != selected_count or len(rows) != selected_count:
            return False
        return all(_nonempty(directory / row["export_path"]) for row in rows)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return {"path": str(resolved), "missing": True}
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _optional_file_identity(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else _file_identity(path)


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))
