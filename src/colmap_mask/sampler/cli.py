from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import load_config, load_trajectory_csv, write_json, write_selected_frames
from .orb import OrbOverlapEvaluator
from .pipeline import sample_trajectory
from .visualization import write_trajectory_visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="360-frame-sampler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Select frames from an external camera trajectory")
    sample.add_argument("--trajectory", type=Path, required=True, help="Input trajectory CSV")
    sample.add_argument("--video", type=Path, help="Source video for ORB continuity")
    sample.add_argument(
        "--video-frame-aligned",
        action="store_true",
        help="Match trajectory records to video by decoded frame index",
    )
    sample.add_argument("--output", type=Path, required=True, help="Output work directory")
    sample.add_argument("--config", type=Path, help="JSON configuration")

    visualize = subparsers.add_parser("visualize", help="Visualize a trajectory and selected points")
    visualize.add_argument("--trajectory", type=Path, required=True, help="Input trajectory CSV")
    visualize.add_argument("--selected", type=Path, required=True, help="Selected frames CSV")
    visualize.add_argument("--output", type=Path, required=True, help="Output HTML")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "sample":
            return _sample(
                args.trajectory,
                args.video,
                args.output,
                args.config,
                args.video_frame_aligned,
            )
        if args.command == "visualize":
            return _visualize(args.trajectory, args.selected, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


def _sample(
    trajectory_path: Path,
    video_path: Path | None,
    output_dir: Path,
    config_path: Path | None,
    video_frame_aligned: bool = False,
) -> int:
    config = load_config(config_path)
    _substage("軌跡読込", 0.05)
    trajectory = load_trajectory_csv(trajectory_path)
    overlap_evaluator = None
    if video_path is not None and config.continuity.enabled and config.continuity.orb.enabled:
        _substage("ORB特徴抽出", 0.15)
        overlap_evaluator = OrbOverlapEvaluator(
            video_path,
            trajectory,
            config.continuity.orb,
            match_by_frame_index=video_frame_aligned,
        )
    result = sample_trajectory(
        trajectory,
        config,
        overlap_evaluator,
        progress=_substage,
    )
    _substage("結果出力", 0.95)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_selected_frames(output_dir / "selected_frames.csv", result.selections)
    write_json(output_dir / "sampler_report.json", result.report)
    write_json(output_dir / "run_config.resolved.json", config.to_dict())
    write_trajectory_visualization(
        output_dir / "trajectory_visualization.html", trajectory, result.selections
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "progress": 1.0,
                "selected_count": len(result.selections),
                "output": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _substage(stage_name: str, progress: float) -> None:
    print(
        json.dumps(
            {
                "event": "substage",
                "stage": "sample",
                "stage_name": stage_name,
                "progress": progress,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _visualize(trajectory_path: Path, selected_path: Path, output_path: Path) -> int:
    from .io import load_selected_frames_csv

    trajectory = load_trajectory_csv(trajectory_path)
    selections = load_selected_frames_csv(selected_path)
    write_trajectory_visualization(output_path, trajectory, selections)
    print(
        json.dumps(
            {"event": "complete", "progress": 1.0, "output": str(output_path)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

