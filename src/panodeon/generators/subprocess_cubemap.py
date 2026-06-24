from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from time import perf_counter
from typing import Callable

import numpy as np

from .base import JobCancelled, MaskOptions, MaskResult

CubemapProgress = Callable[[int, int, str], None]

_STDOUT_EOF = object()


class _WorkerCrash(RuntimeError):
    """The worker process died before completing the current request."""


class SubprocessCubemapGenerator:
    def __init__(self, model_path: Path, provider_name: str | None, timeout_sec: int = 1800) -> None:
        self.model_path = Path(model_path)
        self.provider_name = provider_name or "CPUExecutionProvider"
        self.timeout_sec = timeout_sec

    def generate(
        self,
        image: np.ndarray,
        options: MaskOptions,
        progress: CubemapProgress | None = None,
    ) -> MaskResult:
        start = perf_counter()
        mask, metadata = self._run_child(image, options, self.provider_name, progress)
        if metadata.get("returncode", 0) != 0:
            raise RuntimeError(str(metadata.get("error", "Cubemap subprocess failed")))
        return MaskResult(
            mask=mask,
            strategy="cubemap",
            elapsed_sec=perf_counter() - start,
            metadata=metadata,
        )

    def _run_child(
        self,
        image: np.ndarray,
        options: MaskOptions,
        provider_name: str,
        progress: CubemapProgress | None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        with tempfile.TemporaryDirectory(prefix="panodeon_cubemap_") as temp_dir_text:
            temp_dir = Path(temp_dir_text)
            image_path = temp_dir / "image.npy"
            mask_path = temp_dir / "mask.npy"
            np.save(image_path, image)
            command = [
                sys.executable,
                "-m",
                "panodeon.tools.generate_cubemap_mask",
                "--model",
                str(self.model_path),
                "--provider",
                provider_name,
                "--image",
                str(image_path),
                "--mask",
                str(mask_path),
                "--options",
                json.dumps(mask_options_to_json(options), separators=(",", ":")),
            ]
            process = subprocess.Popen(
                command,
                cwd=str(project_root()),
                env=child_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output_lines: list[str] = []
            stdout_done_marker = object()
            stdout_queue: queue.Queue[str | object] = queue.Queue()

            def read_stdout() -> None:
                if process.stdout is not None:
                    for line in process.stdout:
                        stdout_queue.put(line)
                stdout_queue.put(stdout_done_marker)

            threading.Thread(target=read_stdout, daemon=True).start()
            deadline = perf_counter() + self.timeout_sec
            stdout_done = False
            returncode: int | None = None

            while True:
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    process.kill()
                    returncode = process.wait(timeout=5)
                    metadata = {
                        "provider": provider_name,
                        "returncode": returncode,
                        "error": f"cubemap subprocess timed out after {self.timeout_sec} seconds",
                    }
                    return np.zeros(image.shape[:2], dtype=np.uint8), metadata
                try:
                    message = stdout_queue.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    message = None

                if message is stdout_done_marker:
                    stdout_done = True
                elif isinstance(message, str):
                    progress_message = progress_message_from_line(message)
                    if progress_message is not None:
                        if progress is not None:
                            progress(*progress_message)
                    elif message.strip():
                        output_lines.append(message.rstrip())

                returncode = process.poll()
                if returncode is not None and stdout_done:
                    break

            if returncode is None:
                returncode = process.wait(timeout=0)
            metadata: dict[str, object] = {
                "provider": provider_name,
                "returncode": returncode,
            }
            if returncode != 0:
                output = "\n".join(output_lines).strip()
                metadata["error"] = output or describe_returncode(returncode)
                return np.zeros(image.shape[:2], dtype=np.uint8), metadata
            return np.load(mask_path).astype(np.uint8), metadata


class PersistentCubemapGenerator:
    """Runs cubemap inference in a long-lived child process.

    The child loads the ONNX model once and serves jobs over stdin/stdout,
    so repeated generate() calls skip the per-image process spawn and model
    load. If the child dies (e.g. access violation in onnxruntime) it is
    restarted; a job interrupted on a reused worker is retried once on a
    fresh one. Call shutdown() when done to terminate the child and remove
    the temp directory.
    """

    def __init__(
        self,
        model_path: Path,
        provider_name: str | None,
        timeout_sec: int = 1800,
        startup_timeout_sec: int = 300,
    ) -> None:
        self.model_path = Path(model_path)
        self.provider_name = provider_name or "CPUExecutionProvider"
        self.timeout_sec = timeout_sec
        self.startup_timeout_sec = startup_timeout_sec
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[object] | None = None
        self._temp_dir: str | None = None

    def generate(
        self,
        image: np.ndarray,
        options: MaskOptions,
        progress: CubemapProgress | None = None,
        cancel_event: threading.Event | None = None,
    ) -> MaskResult:
        start = perf_counter()
        with self._lock:
            reused = self._is_alive()
            try:
                mask = self._run_job(image, options, progress, cancel_event)
            except _WorkerCrash as crash:
                self._stop_locked(kill=True)
                if not reused:
                    raise RuntimeError(str(crash)) from None
                # The worker may have died from earlier accumulated state;
                # give the job one retry on a freshly started worker.
                mask = self._run_job(image, options, progress, cancel_event)
        return MaskResult(
            mask=mask,
            strategy="cubemap",
            elapsed_sec=perf_counter() - start,
            metadata={"provider": self.provider_name, "persistent": True},
        )

    def shutdown(self) -> None:
        with self._lock:
            self._stop_locked()
            if self._temp_dir is not None:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                self._temp_dir = None

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    def _is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _temp_path(self, name: str) -> Path:
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="panodeon_cubemap_")
        return Path(self._temp_dir) / name

    def _ensure_process(self, cancel_event: threading.Event | None = None) -> None:
        if self._is_alive():
            return
        self._stop_locked(kill=True)
        lines: queue.Queue[object] = queue.Queue()
        command = [
            sys.executable,
            "-m",
            "panodeon.tools.cubemap_worker",
            "--model",
            str(self.model_path),
            "--provider",
            self.provider_name,
        ]
        process = subprocess.Popen(
            command,
            cwd=str(project_root()),
            env=child_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def read_stdout() -> None:
            if process.stdout is not None:
                for line in process.stdout:
                    lines.put(line)
            lines.put(_STDOUT_EOF)

        threading.Thread(target=read_stdout, daemon=True).start()
        self._process = process
        self._lines = lines
        self._wait_for_message("ready", self.startup_timeout_sec, progress=None, cancel_event=cancel_event)

    def _run_job(
        self,
        image: np.ndarray,
        options: MaskOptions,
        progress: CubemapProgress | None,
        cancel_event: threading.Event | None = None,
    ) -> np.ndarray:
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled("cubemap job cancelled")
        self._ensure_process(cancel_event)
        image_path = self._temp_path("image.npy")
        mask_path = self._temp_path("mask.npy")
        np.save(image_path, image)
        job = {
            "type": "job",
            "image": str(image_path),
            "mask": str(mask_path),
            "options": mask_options_to_json(options),
        }
        process = self._process
        if process is None or process.stdin is None:
            raise _WorkerCrash("cubemap worker is not running")
        try:
            process.stdin.write(json.dumps(job, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise _WorkerCrash(f"cubemap worker pipe closed: {exc}") from exc
        self._wait_for_message("done", self.timeout_sec, progress, cancel_event=cancel_event)
        return np.load(mask_path).astype(np.uint8)

    def _wait_for_message(
        self,
        expected: str,
        timeout_sec: float,
        progress: CubemapProgress | None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        lines = self._lines
        if lines is None:
            raise _WorkerCrash("cubemap worker is not running")
        deadline = perf_counter() + timeout_sec
        noise: list[str] = []
        while True:
            if cancel_event is not None and cancel_event.is_set():
                # The child may be mid-inference; kill it so cancellation is
                # prompt. The next job simply starts a fresh worker.
                self._stop_locked(kill=True)
                raise JobCancelled("cubemap job cancelled")
            remaining = deadline - perf_counter()
            if remaining <= 0:
                self._stop_locked(kill=True)
                raise RuntimeError(f"cubemap worker timed out after {timeout_sec} seconds")
            try:
                message = lines.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if message is _STDOUT_EOF:
                returncode = self._process.wait() if self._process is not None else -1
                detail = "\n".join(noise).strip()
                raise _WorkerCrash(detail or describe_returncode(returncode))
            text = str(message)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                if text.strip():
                    noise.append(text.rstrip())
                continue
            kind = data.get("type")
            if kind == "progress":
                parsed = progress_message_from_line(text)
                if parsed is not None and progress is not None:
                    progress(*parsed)
            elif kind == "error":
                raise RuntimeError(str(data.get("message", "cubemap worker job failed")))
            elif kind == expected:
                return

    def _stop_locked(self, kill: bool = False) -> None:
        process = self._process
        self._process = None
        self._lines = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            if kill:
                process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass


def mask_options_to_json(options: MaskOptions) -> dict[str, object]:
    return {
        "strategy": options.strategy,
        "selected_classes": list(options.selected_classes),
        "score_threshold": options.score_threshold,
        "dilate_px": options.dilate_px,
        "feather_px": options.feather_px,
        "cube_face_size": options.cube_face_size,
        "cube_fov_deg": options.cube_fov_deg,
        "mask_threshold": options.mask_threshold,
    }


def mask_options_from_json(data: dict[str, object]) -> MaskOptions:
    return MaskOptions(
        strategy=str(data.get("strategy", "cubemap")),
        selected_classes=tuple(str(value) for value in data.get("selected_classes", [])),
        score_threshold=float(data.get("score_threshold", 0.35)),
        dilate_px=int(data.get("dilate_px", 8)),
        feather_px=int(data.get("feather_px", 5)),
        cube_face_size=int(data.get("cube_face_size", 1024)),
        cube_fov_deg=float(data.get("cube_fov_deg", 105.0)),
        mask_threshold=float(data.get("mask_threshold", 0.4)),
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[2])
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not current else f"{src_path}{os.pathsep}{current}"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    return env


def progress_message_from_line(line: str) -> tuple[int, int, str] | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if data.get("type") != "progress":
        return None
    try:
        return int(data["current"]), int(data["total"]), str(data["face"])
    except (KeyError, TypeError, ValueError):
        return None


def describe_returncode(returncode: int) -> str:
    if returncode == 3221225477:
        return "access violation in cubemap child process (0xC0000005)"
    return f"exit code {returncode}"
