from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backends.stella import (
    StellaRunConfig,
    align_stella_trajectory,
    load_tum_trajectory,
    remap_trajectory_frame_timing,
    run_stella_video,
    write_default_equirectangular_config,
    write_trajectory_csv,
)
from .io import write_json
from .video import (
    create_proxy_video,
    extract_selected_frames,
    load_video_frame_index,
    probe_video,
    write_video_probe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="360-frame-video")
    commands = parser.add_subparsers(dest="command", required=True)

    proxy = commands.add_parser("create-proxy", help="Create a spatially resized shared video")
    proxy.add_argument("--video", type=Path, required=True)
    proxy.add_argument("--output", type=Path, required=True)
    proxy.add_argument("--width", type=int, default=1920)
    proxy.add_argument("--height", type=int, default=960)
    proxy.add_argument("--codec", default="libx264")
    proxy.add_argument("--crf", type=int, default=15)
    proxy.add_argument("--preset", default="veryfast")

    probe = commands.add_parser("probe", help="Decode video PTS and write a frame index")
    probe.add_argument("--video", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)

    extract = commands.add_parser("extract", help="Extract selected source PTS")
    extract.add_argument("--video", type=Path, required=True)
    extract.add_argument("--selected", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--format", choices=("jpg", "png"), default="jpg")
    extract.add_argument("--jpeg-quality", type=int, default=95)
    extract.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Rewrite images even when an identical file already exists",
    )
    extract.set_defaults(skip_existing=True)

    import_stella = commands.add_parser("import-stella", help="Align a TUM trajectory to video PTS")
    import_stella.add_argument("--tum", type=Path, required=True)
    import_stella.add_argument("--frame-index", type=Path, required=True)
    import_stella.add_argument("--output", type=Path, required=True)
    import_stella.add_argument("--split-gap-sec", type=float, default=0.5)
    import_stella.add_argument("--max-timestamp-error-sec", type=float)

    run_stella = commands.add_parser("run-stella", help="Run stella_vslam and create trajectory.csv")
    run_stella.add_argument("--video", type=Path, required=True)
    run_stella.add_argument("--source-frame-index", type=Path)
    run_stella.add_argument("--executable", type=Path, required=True)
    run_stella.add_argument("--vocab", type=Path, required=True)
    run_stella.add_argument("--camera-config", type=Path)
    run_stella.add_argument("--output", type=Path, required=True)
    run_stella.add_argument("--frame-skip", type=int, default=1)
    run_stella.add_argument("--split-gap-sec", type=float, default=0.5)
    run_stella.add_argument("--max-timestamp-error-sec", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create-proxy":
            return _create_proxy(args)
        if args.command == "probe":
            return _probe(args.video, args.output)
        if args.command == "extract":
            return _extract(
                args.video,
                args.selected,
                args.output,
                args.format,
                args.jpeg_quality,
                args.skip_existing,
            )
        if args.command == "import-stella":
            return _import_stella(
                args.tum,
                args.frame_index,
                args.output,
                args.split_gap_sec,
                args.max_timestamp_error_sec,
            )
        if args.command == "run-stella":
            return _run_stella(args)
    except (OSError, RuntimeError, ValueError) as exc:
        _event("error", message=str(exc), stream=sys.stderr)
        return 2
    return 2


def _create_proxy(args: argparse.Namespace) -> int:
    output_path = args.output / "video.mkv"
    _event("progress", stage="proxy", progress=0.0)

    last_progress = -1.0

    def report(frame_count: int, total: int | None) -> None:
        nonlocal last_progress
        if total is None or total <= 0:
            return
        value = min(0.99, frame_count / total)
        if value - last_progress >= 0.01:
            last_progress = value
            _event("progress", stage="proxy", progress=value, frame_count=frame_count)

    result = create_proxy_video(
        args.video,
        output_path,
        width=args.width,
        height=args.height,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        progress=report,
    )
    write_video_probe(args.output, result.source_probe)
    write_json(
        args.output / "proxy_metadata.json",
        {
            "schema_version": 1,
            "source_video_path": str(args.video.resolve()),
            "proxy_video_path": str(output_path.resolve()),
            "source_frame_count": len(result.source_probe.frames),
            "width": result.width,
            "height": result.height,
            "codec": args.codec,
            "crf": args.crf,
            "preset": args.preset,
        },
    )
    _event(
        "complete",
        progress=1.0,
        frame_count=len(result.source_probe.frames),
        output=str(output_path),
    )
    return 0


def _probe(video_path: Path, output_dir: Path) -> int:
    _event("progress", stage="probe", progress=0.0)
    result = probe_video(video_path)
    write_video_probe(output_dir, result)
    _event("complete", progress=1.0, frame_count=len(result.frames), output=str(output_dir))
    return 0


def _extract(
    video_path: Path,
    selected_path: Path,
    output_dir: Path,
    image_format: str,
    jpeg_quality: int,
    skip_existing: bool = True,
) -> int:
    _event("progress", stage="extract", progress=0.0)

    def _report_progress(done: int, total: int, decoded: int) -> None:
        progress = done / total if total else 1.0
        _event(
            "progress",
            stage="extract",
            progress=progress,
            done=done,
            total=total,
            decoded=decoded,
        )

    records = extract_selected_frames(
        video_path,
        selected_path,
        output_dir,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        skip_existing=skip_existing,
        on_progress=_report_progress,
    )
    report = json.loads((output_dir / "extraction_report.json").read_text(encoding="utf-8"))
    _event(
        "complete",
        progress=1.0,
        extracted_count=len(records),
        written_count=report.get("written_count"),
        skipped_existing_count=report.get("skipped_existing_count"),
        output=str(output_dir),
    )
    return 0


def _import_stella(
    tum_path: Path,
    frame_index_path: Path,
    output_path: Path,
    split_gap_seconds: float,
    max_timestamp_error_seconds: float | None,
) -> int:
    poses = load_tum_trajectory(tum_path)
    frames = load_video_frame_index(frame_index_path)
    records = align_stella_trajectory(
        poses,
        frames,
        split_gap_seconds=split_gap_seconds,
        max_timestamp_error_seconds=max_timestamp_error_seconds,
    )
    write_trajectory_csv(output_path, records)
    _event("complete", progress=1.0, pose_count=len(records), output=str(output_path))
    return 0


def _run_stella(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    _event("progress", stage="probe", progress=0.0)
    probe = probe_video(args.video)
    write_video_probe(args.output, probe)
    source_frames = None
    if args.source_frame_index is not None:
        source_frames = load_video_frame_index(args.source_frame_index)
        if len(source_frames) != len(probe.frames):
            raise ValueError(
                "Proxy and source frame counts differ: "
                f"{len(probe.frames)} != {len(source_frames)}"
            )
    camera_config = args.camera_config
    if camera_config is None:
        camera_config = args.output / "stella_camera.resolved.yaml"
        rate_den = int(probe.metadata.get("average_rate_den") or 1)
        rate_num = int(probe.metadata.get("average_rate_num") or 30)
        write_default_equirectangular_config(camera_config, rate_num / rate_den)
    _event("progress", stage="stella_vslam", progress=0.2)
    tum_path = run_stella_video(
        args.video,
        args.output,
        StellaRunConfig(
            executable=args.executable,
            vocabulary=args.vocab,
            camera_config=camera_config,
            frame_skip=args.frame_skip,
        ),
    )
    _event("progress", stage="align_pts", progress=0.9)
    records = align_stella_trajectory(
        load_tum_trajectory(tum_path),
        list(probe.frames),
        split_gap_seconds=args.split_gap_sec,
        max_timestamp_error_seconds=args.max_timestamp_error_sec,
    )
    if source_frames is not None:
        records = remap_trajectory_frame_timing(records, source_frames)
    output_path = args.output / "trajectory.csv"
    write_trajectory_csv(output_path, records)
    _event("complete", progress=1.0, pose_count=len(records), output=str(output_path))
    return 0


def _event(event: str, *, stream=sys.stdout, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), file=stream, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

