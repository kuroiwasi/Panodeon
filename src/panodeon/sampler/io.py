from __future__ import annotations

import csv
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

from .models import SCHEMA_VERSION, SamplerConfig, SelectionRecord, TrajectoryRecord

REQUIRED_TRAJECTORY_COLUMNS = {
    "frame_index",
    "pts",
    "time_base_num",
    "time_base_den",
    "cx",
    "cy",
    "cz",
}


def load_config(path: Path | None) -> SamplerConfig:
    if path is None:
        config = SamplerConfig()
    else:
        with path.open(encoding="utf-8") as handle:
            config = SamplerConfig.from_dict(json.load(handle))
    config.validate()
    return config


def load_trajectory_csv(path: Path) -> list[TrajectoryRecord]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_TRAJECTORY_COLUMNS - columns
        if missing:
            raise ValueError(f"Missing trajectory columns: {', '.join(sorted(missing))}")
        records = [_trajectory_record(row, line_number) for line_number, row in enumerate(reader, start=2)]

    if not records:
        raise ValueError("Trajectory is empty")
    frame_indices = [record.frame_index for record in records]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError("frame_index must be unique")
    return records


def write_selected_frames(path: Path, records: list[SelectionRecord]) -> None:
    fieldnames = [
        "selection_order",
        "frame_index",
        "pts",
        "time_base_num",
        "time_base_den",
        "pts_seconds",
        "segment_id",
        "coordinate_group_id",
        "selection_type",
        "nearest_selected_distance",
        "normalized_distance",
        "path_s",
        "quality_score",
    ]
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {name: getattr(record, name) for name in fieldnames}
            row["pts_seconds"] = f"{record.pts_seconds:.9f}"
            row["path_s"] = f"{record.path_s:.9f}"
            row["quality_score"] = f"{record.quality_score:.9f}"
            for name in ("nearest_selected_distance", "normalized_distance"):
                value = row[name]
                row[name] = "" if value is None else f"{value:.9f}"
            writer.writerow(row)


def load_selected_frames_csv(path: Path) -> list[SelectionRecord]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        records: list[SelectionRecord] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                records.append(
                    SelectionRecord(
                        selection_order=int(row["selection_order"]),
                        frame_index=int(row["frame_index"]),
                        pts=int(row["pts"]),
                        time_base_num=int(row["time_base_num"]),
                        time_base_den=int(row["time_base_den"]),
                        pts_seconds=float(row["pts_seconds"]),
                        segment_id=int(row["segment_id"]),
                        coordinate_group_id=int(row["coordinate_group_id"]),
                        selection_type=row["selection_type"],
                        nearest_selected_distance=_optional_float(
                            row.get("nearest_selected_distance")
                        ),
                        normalized_distance=_optional_float(row.get("normalized_distance")),
                        path_s=float(row["path_s"]),
                        quality_score=float(row["quality_score"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid selected frame row {line_number}: {exc}") from exc
    if not records:
        raise ValueError("Selected frames are empty")
    return records


def _optional_float(value: str | None) -> float | None:
    return float(value) if value else None


def write_json(path: Path, data: dict[str, Any]) -> None:
    with atomic_text_writer(path) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


@contextmanager
def atomic_text_writer(path: Path, newline: str | None = None) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline=newline) as handle:
            yield handle
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except PermissionError:
                pass


def _replace_with_retry(source: Path, destination: Path, attempts: int = 8) -> None:
    delay = 0.02
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.5)


def _trajectory_record(row: dict[str, str], line_number: int) -> TrajectoryRecord:
    try:
        record = TrajectoryRecord(
            frame_index=int(row["frame_index"]),
            pts=int(row["pts"]),
            time_base_num=int(row["time_base_num"]),
            time_base_den=int(row["time_base_den"]),
            segment_id=int(row.get("segment_id") or 0),
            coordinate_group_id=int(row.get("coordinate_group_id") or 0),
            cx=float(row["cx"]),
            cy=float(row["cy"]),
            cz=float(row["cz"]),
            pose_valid=_parse_bool(row.get("pose_valid"), default=True),
            tracking_state=(row.get("tracking_state") or "good").strip().lower(),
            candidate_valid=_parse_bool(row.get("candidate_valid"), default=True),
            quality_score=float(row.get("quality_score") or 0.0),
            qw=_optional_float(row.get("qw")),
            qx=_optional_float(row.get("qx")),
            qy=_optional_float(row.get("qy")),
            qz=_optional_float(row.get("qz")),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid trajectory row {line_number}: {exc}") from exc
    if record.time_base_num <= 0 or record.time_base_den <= 0:
        raise ValueError(f"Invalid trajectory row {line_number}: time base must be positive")
    return record


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Invalid boolean: {value}")


def base_report() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION}

