from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .resume import (
    load_workflow_state,
    mark_step_complete,
    step_is_complete,
    step_key,
    step_signature,
)


@dataclass(frozen=True)
class PipelineSettings:
    video_path: Path
    output_dir: Path
    executable: Path
    vocabulary: Path
    frame_skip: int = 2
    image_format: str = "jpg"
    jpeg_quality: int = 95
    camera_config: Path | None = None
    sampler_config: Path | None = None
    proxy_width: int = 1920
    proxy_height: int = 960
    proxy_codec: str = "libx264"
    proxy_crf: int = 15
    proxy_preset: str = "veryfast"

    def proxy_dimensions(self) -> tuple[int, int]:
        if self.camera_config is not None:
            text = self.camera_config.read_text(encoding="utf-8")
            cols = re.search(r"(?m)^\s*cols:\s*(\d+)\s*$", text)
            rows = re.search(r"(?m)^\s*rows:\s*(\d+)\s*$", text)
            if cols is None or rows is None:
                raise ValueError("カメラ設定にcolsとrowsが必要")
            return int(cols.group(1)), int(rows.group(1))
        return self.proxy_width, self.proxy_height

    def proxy_video_path(self) -> Path:
        return self.output_dir / "proxy" / "video.mkv"

    def validate(self) -> None:
        for path, label in (
            (self.video_path, "動画"),
            (self.executable, "run_video_slam.exe"),
            (self.vocabulary, "ORB vocabulary"),
        ):
            if not path.is_file():
                raise ValueError(f"{label}が見つからない: {path}")
        for path, label in (
            (self.camera_config, "カメラ設定"),
            (self.sampler_config, "選定設定"),
        ):
            if path is not None and not path.is_file():
                raise ValueError(f"{label}が見つからない: {path}")
        width, height = self.proxy_dimensions()
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("中間動画の解像度は正の偶数")
        if not 0 <= self.proxy_crf <= 51:
            raise ValueError("中間動画CRFは0から51")
        if not self.proxy_codec or not self.proxy_preset:
            raise ValueError("中間動画のcodecとpresetが必要")
        if self.frame_skip <= 0:
            raise ValueError("frame skipは1以上")
        if self.image_format not in {"jpg", "png"}:
            raise ValueError("画像形式はjpgまたはpng")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG品質は1から100")


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]
    progress_start: float
    progress_end: float


def project_root() -> Path:
    configured = os.environ.get("FRAME_SAMPLER_ROOT")
    if configured:
        return Path(configured).resolve()
    # Vendored layout: this module lives at <root>/src/panodeon/sampler/workflow.py,
    # so the panodeon project root (holding pyproject.toml and third_party/) is parents[4].
    source_root = Path(__file__).resolve().parents[4]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.cwd().resolve()


def console_python() -> Path:
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        candidate = executable.with_name("python.exe")
        if candidate.is_file():
            return candidate
    return executable


def build_pipeline_steps(
    settings: PipelineSettings,
    python_executable: Path | None = None,
) -> tuple[PipelineStep, ...]:
    python = str((python_executable or console_python()).resolve())
    output = settings.output_dir.resolve()
    proxy_output = output / "proxy"
    proxy_video = proxy_output / "video.mkv"
    stella_output = output / "stella"
    sampled_output = output / "sampled"
    proxy_width, proxy_height = settings.proxy_dimensions()

    proxy_command = [
        python,
        "-m",
        "panodeon.sampler.video_cli",
        "create-proxy",
        "--video",
        str(settings.video_path.resolve()),
        "--output",
        str(proxy_output),
        "--width",
        str(proxy_width),
        "--height",
        str(proxy_height),
        "--codec",
        settings.proxy_codec,
        "--crf",
        str(settings.proxy_crf),
        "--preset",
        settings.proxy_preset,
    ]

    stella_command = [
        python,
        "-m",
        "panodeon.sampler.video_cli",
        "run-stella",
        "--video",
        str(proxy_video),
        "--source-frame-index",
        str(proxy_output / "video_frames.csv"),
        "--executable",
        str(settings.executable.resolve()),
        "--vocab",
        str(settings.vocabulary.resolve()),
        "--output",
        str(stella_output),
        "--frame-skip",
        str(settings.frame_skip),
    ]
    if settings.camera_config is not None:
        stella_command.extend(("--camera-config", str(settings.camera_config.resolve())))

    sample_command = [
        python,
        "-m",
        "panodeon.sampler.cli",
        "sample",
        "--trajectory",
        str(stella_output / "trajectory.csv"),
        "--video",
        str(proxy_video),
        "--video-frame-aligned",
        "--output",
        str(sampled_output),
    ]
    if settings.sampler_config is not None:
        sample_command.extend(("--config", str(settings.sampler_config.resolve())))

    extract_command = [
        python,
        "-m",
        "panodeon.sampler.video_cli",
        "extract",
        "--video",
        str(settings.video_path.resolve()),
        "--selected",
        str(sampled_output / "selected_frames.csv"),
        "--output",
        str(output / "frames"),
        "--format",
        settings.image_format,
        "--jpeg-quality",
        str(settings.jpeg_quality),
    ]

    return (
        PipelineStep("中間動画生成", tuple(proxy_command), 0.0, 0.2),
        PipelineStep("軌跡抽出", tuple(stella_command), 0.2, 0.72),
        PipelineStep("フレーム選定", tuple(sample_command), 0.72, 0.84),
        PipelineStep("画像抽出", tuple(extract_command), 0.84, 1.0),
    )


def _write_event_log(output_dir: Path, run_id: str, line: str) -> None:
    primary = output_dir / "workflow_events.jsonl"
    for attempt in range(5):
        try:
            _append_event_line(primary, line)
            return
        except PermissionError:
            if attempt < 4:
                time.sleep(0.05 * (attempt + 1))

    fallback = output_dir / f"workflow_events.{run_id}.jsonl"
    try:
        _append_event_line(fallback, line)
    except OSError:
        pass


def _append_event_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


class PipelineCancelled(RuntimeError):
    pass


class PipelineExecutor:
    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def run(
        self,
        settings: PipelineSettings,
        on_stage: Callable[[PipelineStep], None],
        on_line: Callable[[PipelineStep, str], None],
    ) -> None:
        settings.validate()
        self._cancelled.clear()
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        state = load_workflow_state(settings.output_dir)
        steps = build_pipeline_steps(settings)
        run_id = uuid.uuid4().hex
        self._emit_event(
            settings.output_dir,
            steps[0],
            on_line,
            run_id,
            "pipeline_start",
            progress=0.0,
            total_steps=len(steps),
            message="全工程を開始",
        )
        for index, step in enumerate(steps, start=1):
            if self._cancelled.is_set():
                raise PipelineCancelled("中止")
            on_stage(step)
            key = step_key(step.name)
            common = {
                "stage": key,
                "stage_name": step.name,
                "stage_index": index,
                "total_steps": len(steps),
            }
            self._emit_event(
                settings.output_dir,
                step,
                on_line,
                run_id,
                "stage_start",
                progress=0.0,
                message="工程を開始",
                **common,
            )
            try:
                signature = step_signature(settings, key)
                if step_is_complete(settings, key, state, signature):
                    self._emit_event(
                        settings.output_dir,
                        step,
                        on_line,
                        run_id,
                        "skip",
                        progress=1.0,
                        message="完了済み結果を再利用",
                        **common,
                    )
                    mark_step_complete(settings.output_dir, state, key, signature)
                    continue
                self._run_step(step, on_line)
                mark_step_complete(settings.output_dir, state, key, signature)
                self._emit_event(
                    settings.output_dir,
                    step,
                    on_line,
                    run_id,
                    "stage_complete",
                    progress=1.0,
                    message="工程が完了",
                    **common,
                )
            except PipelineCancelled:
                self._emit_event(
                    settings.output_dir,
                    step,
                    on_line,
                    run_id,
                    "stage_cancelled",
                    progress=0.0,
                    message="工程を中止",
                    **common,
                )
                raise
            except Exception as exc:
                self._emit_event(
                    settings.output_dir,
                    step,
                    on_line,
                    run_id,
                    "stage_error",
                    progress=0.0,
                    message=str(exc),
                    **common,
                )
                raise
        self._emit_event(
            settings.output_dir,
            steps[-1],
            on_line,
            run_id,
            "pipeline_complete",
            progress=1.0,
            total_steps=len(steps),
            message="全工程が完了",
        )


    @staticmethod
    def _emit_event(
        output_dir: Path,
        step: PipelineStep,
        on_line: Callable[[PipelineStep, str], None],
        run_id: str,
        event: str,
        **fields: object,
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "event": event,
            **fields,
        }
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        _write_event_log(output_dir, run_id, line)
        on_line(step, line)

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            process.terminate()

    def _run_step(
        self,
        step: PipelineStep,
        on_line: Callable[[PipelineStep, str], None],
    ) -> None:
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            step.command,
            cwd=project_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
            env=environment,
        )
        with self._lock:
            self._process = process
        tail: list[str] = []
        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                tail.append(line)
                tail = tail[-20:]
                on_line(step, line)
            returncode = process.wait()
        finally:
            with self._lock:
                self._process = None
        if self._cancelled.is_set():
            raise PipelineCancelled("中止")
        if returncode != 0:
            detail = "\n".join(tail)
            raise RuntimeError(f"{step.name}に失敗: exit code {returncode}\n{detail}")
