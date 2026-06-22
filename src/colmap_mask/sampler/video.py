from __future__ import annotations

import csv
import json
import uuid
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import av
import cv2

from .io import _replace_with_retry, atomic_text_writer, write_json


@dataclass(frozen=True)
class VideoFrameRecord:
    frame_index: int
    pts: int
    time_base_num: int
    time_base_den: int
    key_frame: bool

    @property
    def timestamp(self) -> Fraction:
        return Fraction(self.pts * self.time_base_num, self.time_base_den)

    @property
    def pts_seconds(self) -> float:
        return float(self.timestamp)


@dataclass(frozen=True)
class VideoProbeResult:
    metadata: dict[str, Any]
    frames: tuple[VideoFrameRecord, ...]


@dataclass(frozen=True)
class ExtractionRecord:
    selection_order: int
    frame_index: int
    pts: int
    time_base_num: int
    time_base_den: int
    export_path: str


@dataclass(frozen=True)
class ProxyVideoResult:
    source_probe: VideoProbeResult
    output_path: Path
    width: int
    height: int


def probe_video(video_path: Path) -> VideoProbeResult:
    if not video_path.is_file():
        raise ValueError(f"Video does not exist: {video_path}")
    frames: list[VideoFrameRecord] = []
    with av.open(str(video_path)) as container:
        stream = _first_video_stream(container)
        stream_metadata = _stream_metadata(container, stream)
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Decoded frame has no PTS: frame_index={frame_index}")
            time_base = Fraction(frame.time_base)
            frames.append(
                VideoFrameRecord(
                    frame_index=frame_index,
                    pts=int(frame.pts),
                    time_base_num=time_base.numerator,
                    time_base_den=time_base.denominator,
                    key_frame=bool(frame.key_frame),
                )
            )
    return _build_probe_result(video_path, stream_metadata, frames)


def create_proxy_video(
    video_path: Path,
    output_path: Path,
    *,
    width: int = 1920,
    height: int = 960,
    codec: str = "libx264",
    crf: int = 15,
    preset: str = "veryfast",
    progress: Callable[[int, int | None], None] | None = None,
) -> ProxyVideoResult:
    if not video_path.is_file():
        raise ValueError(f"Video does not exist: {video_path}")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Proxy dimensions must be positive even numbers")
    if not 0 <= crf <= 51:
        raise ValueError("Proxy CRF must be in [0, 51]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    frames: list[VideoFrameRecord] = []
    try:
        with av.open(str(video_path)) as source:
            input_stream = _first_video_stream(source)
            stream_metadata = _stream_metadata(source, input_stream)
            expected_frames = int(input_stream.frames) if input_stream.frames else None
            rate = input_stream.average_rate or 30
            input_time_base = Fraction(input_stream.time_base)
            first_timestamp: Fraction | None = None
            with av.open(str(temporary), mode="w", format="matroska") as destination:
                output_stream = destination.add_stream(
                    codec,
                    rate=rate,
                    options={"crf": str(crf), "preset": preset, "tune": "fastdecode"},
                )
                output_stream.width = width
                output_stream.height = height
                output_stream.pix_fmt = "yuv420p"
                output_stream.time_base = input_time_base
                output_stream.codec_context.time_base = input_time_base

                for frame_index, frame in enumerate(source.decode(input_stream)):
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError(f"Decoded frame has no PTS: frame_index={frame_index}")
                    frame_time_base = Fraction(frame.time_base)
                    timestamp = Fraction(int(frame.pts)) * frame_time_base
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    frames.append(
                        VideoFrameRecord(
                            frame_index=frame_index,
                            pts=int(frame.pts),
                            time_base_num=frame_time_base.numerator,
                            time_base_den=frame_time_base.denominator,
                            key_frame=bool(frame.key_frame),
                        )
                    )
                    proxy_frame = frame.reformat(
                        width=width,
                        height=height,
                        format="yuv420p",
                        interpolation="AREA",
                    )
                    relative_timestamp = timestamp - first_timestamp
                    proxy_frame.pts = round(relative_timestamp / input_time_base)
                    proxy_frame.time_base = input_time_base
                    for packet in output_stream.encode(proxy_frame):
                        destination.mux(packet)
                    if progress is not None and (frame_index + 1) % 30 == 0:
                        progress(frame_index + 1, expected_frames)
                for packet in output_stream.encode(None):
                    destination.mux(packet)

        if not frames:
            raise ValueError(f"Video contains no decoded frames: {video_path}")
        _replace_with_retry(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)

    if progress is not None:
        progress(len(frames), len(frames))
    source_probe = _build_probe_result(video_path, stream_metadata, frames)
    return ProxyVideoResult(source_probe, output_path, width, height)


def _build_probe_result(
    video_path: Path,
    stream_metadata: dict[str, Any],
    frames: list[VideoFrameRecord],
) -> VideoProbeResult:
    if not frames:
        raise ValueError(f"Video contains no decoded frames: {video_path}")
    timestamps = [frame.timestamp for frame in frames]
    deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
    duplicate_pts = sum(delta == 0 for delta in deltas)
    backward_pts = sum(delta < 0 for delta in deltas)
    positive_deltas = [float(delta) for delta in deltas if delta > 0]
    delta_min = min(positive_deltas) if positive_deltas else None
    delta_max = max(positive_deltas) if positive_deltas else None
    suspected_vfr = bool(delta_min and delta_max and delta_max / delta_min > 1.1)
    duration = float(timestamps[-1] - timestamps[0])
    if positive_deltas:
        duration += sorted(positive_deltas)[len(positive_deltas) // 2]

    metadata = {
        "schema_version": 1,
        "video_path": str(video_path.resolve()),
        **stream_metadata,
        "decoded_frame_count": len(frames),
        "first_pts": frames[0].pts,
        "first_time_base_num": frames[0].time_base_num,
        "first_time_base_den": frames[0].time_base_den,
        "duration_seconds": duration,
        "duplicate_pts_count": duplicate_pts,
        "backward_pts_count": backward_pts,
        "timestamp_delta_min_seconds": delta_min,
        "timestamp_delta_max_seconds": delta_max,
        "variable_frame_rate_suspected": suspected_vfr,
    }
    return VideoProbeResult(metadata, tuple(frames))


def write_video_probe(output_dir: Path, result: VideoProbeResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "video_metadata.json", result.metadata)
    write_video_frame_index(output_dir / "video_frames.csv", result.frames)


def write_video_frame_index(path: Path, frames: tuple[VideoFrameRecord, ...] | list[VideoFrameRecord]) -> None:
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frame_index", "pts", "time_base_num", "time_base_den", "pts_seconds", "key_frame"]
        )
        for frame in frames:
            writer.writerow(
                [
                    frame.frame_index,
                    frame.pts,
                    frame.time_base_num,
                    frame.time_base_den,
                    f"{frame.pts_seconds:.9f}",
                    int(frame.key_frame),
                ]
            )


def load_video_frame_index(path: Path) -> list[VideoFrameRecord]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_index", "pts", "time_base_num", "time_base_den"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing video frame columns: {', '.join(sorted(missing))}")
        frames = [
            VideoFrameRecord(
                frame_index=int(row["frame_index"]),
                pts=int(row["pts"]),
                time_base_num=int(row["time_base_num"]),
                time_base_den=int(row["time_base_den"]),
                key_frame=(row.get("key_frame", "0").strip().lower() in {"1", "true", "yes"}),
            )
            for row in reader
        ]
    if not frames:
        raise ValueError("Video frame index is empty")
    return frames


def extract_selected_frames(
    video_path: Path,
    selected_frames_path: Path,
    output_dir: Path,
    *,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    skip_existing: bool = True,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> list[ExtractionRecord]:
    image_format = image_format.lower().lstrip(".")
    if image_format not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"Unsupported image format: {image_format}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    extension = "jpg" if image_format == "jpeg" else image_format
    targets = _load_extraction_targets(selected_frames_path)
    for target in targets:
        target["output_path"] = output_dir / _frame_filename(target, extension)

    total_targets = len(targets)
    if on_progress is not None:
        on_progress(0, total_targets, 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[ExtractionRecord] = []
    created_paths: list[Path] = []
    skipped_count = 0
    decoded_count = 0

    # When every selected frame already exists, there is nothing to decode: avoid
    # opening the container at all (full decode is by far the dominant cost).
    if skip_existing and all(target["output_path"].is_file() for target in targets):
        for target in targets:
            extracted.append(_extraction_record(target))
        skipped_count = total_targets
        if on_progress is not None:
            on_progress(total_targets, total_targets, 0)
        return _finalize_extraction(output_dir, targets, extracted, skipped_count)

    targets_by_time: dict[Fraction, list[dict[str, Any]]] = {}
    for target in targets:
        timestamp = Fraction(target["pts"] * target["time_base_num"], target["time_base_den"])
        targets_by_time.setdefault(timestamp, []).append(target)

    with av.open(str(video_path)) as container:
        stream = _first_video_stream(container)
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            decoded_count += 1
            timestamp = Fraction(int(frame.pts)) * Fraction(frame.time_base)
            matching = targets_by_time.pop(timestamp, [])
            if not matching:
                # Heartbeat while scanning toward the next selected frame so the
                # caller can show the decode is alive before the first match.
                if on_progress is not None and decoded_count % 30 == 0:
                    on_progress(len(extracted), total_targets, decoded_count)
                continue
            # An identical filename means an identical frame (the name encodes
            # frame index, selection order and PTS), so reuse what is on disk and
            # only decode + encode the targets that still need to be written.
            pending = [
                target
                for target in matching
                if not (skip_existing and target["output_path"].is_file())
            ]
            skipped_count += len(matching) - len(pending)
            if pending:
                image = frame.to_ndarray(format="bgr24")
                encode_extension = ".jpg" if extension == "jpg" else ".png"
                parameters = (
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if extension == "jpg" else []
                )
                ok, encoded = cv2.imencode(encode_extension, image, parameters)
                if not ok:
                    raise ValueError(f"Cannot encode frame at PTS {frame.pts}")
                encoded_bytes = encoded.tobytes()
                for target in pending:
                    _atomic_write_bytes(target["output_path"], encoded_bytes)
                    created_paths.append(target["output_path"])
            for target in matching:
                extracted.append(_extraction_record(target))
            if on_progress is not None:
                on_progress(len(extracted), total_targets, decoded_count)
            if not targets_by_time:
                break

    if targets_by_time:
        for path in created_paths:
            path.unlink(missing_ok=True)
        missing_count = sum(len(items) for items in targets_by_time.values())
        raise ValueError(f"Selected PTS not found in video: {missing_count} frame(s)")
    return _finalize_extraction(output_dir, targets, extracted, skipped_count)


def _stream_metadata(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> dict[str, Any]:
    average_rate = Fraction(stream.average_rate) if stream.average_rate is not None else None
    time_base = Fraction(stream.time_base)
    codec = stream.codec_context
    return {
        "container_format": container.format.name,
        "stream_index": stream.index,
        "codec": codec.name,
        "width": codec.width,
        "height": codec.height,
        "pixel_format": codec.format.name if codec.format is not None else None,
        "stream_time_base_num": time_base.numerator,
        "stream_time_base_den": time_base.denominator,
        "average_rate_num": average_rate.numerator if average_rate else None,
        "average_rate_den": average_rate.denominator if average_rate else None,
        "container_metadata": dict(container.metadata),
        "stream_metadata": dict(stream.metadata),
    }


def _first_video_stream(container: av.container.InputContainer) -> av.video.stream.VideoStream:
    if not container.streams.video:
        raise ValueError("Input has no video stream")
    return container.streams.video[0]


def _load_extraction_targets(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"selection_order", "frame_index", "pts", "time_base_num", "time_base_den"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing selected frame columns: {', '.join(sorted(missing))}")
        targets = [{name: int(row[name]) for name in required} for row in reader]
    if not targets:
        raise ValueError("Selected frame list is empty")
    return targets


def _write_extraction_manifest(path: Path, records: list[ExtractionRecord]) -> None:
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _frame_filename(target: dict[str, Any], extension: str) -> str:
    return (
        f"{target['frame_index']:09d}_{target['selection_order']:06d}_"
        f"{_safe_pts(target['pts'])}.{extension}"
    )


def _extraction_record(target: dict[str, Any]) -> ExtractionRecord:
    return ExtractionRecord(
        selection_order=target["selection_order"],
        frame_index=target["frame_index"],
        pts=target["pts"],
        time_base_num=target["time_base_num"],
        time_base_den=target["time_base_den"],
        export_path=target["output_path"].name,
    )


def _finalize_extraction(
    output_dir: Path,
    targets: list[dict[str, Any]],
    extracted: list[ExtractionRecord],
    skipped_count: int,
) -> list[ExtractionRecord]:
    extracted.sort(key=lambda record: record.selection_order)
    _write_extraction_manifest(output_dir / "extracted_frames.csv", extracted)
    write_json(
        output_dir / "extraction_report.json",
        {
            "schema_version": 1,
            "requested_count": len(targets),
            "extracted_count": len(extracted),
            "written_count": len(extracted) - skipped_count,
            "skipped_existing_count": skipped_count,
        },
    )
    return extracted


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except PermissionError:
                pass


def _safe_pts(pts: int) -> str:
    return str(pts) if pts >= 0 else f"m{abs(pts)}"

