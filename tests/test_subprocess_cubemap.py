from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import threading

import numpy as np
import pytest

from panodeon.generators.base import JobCancelled, MaskOptions
from panodeon.generators.subprocess_cubemap import (
    PersistentCubemapGenerator,
    SubprocessCubemapGenerator,
    describe_returncode,
    mask_options_from_json,
    mask_options_to_json,
    progress_message_from_line,
)


def test_mask_options_json_roundtrip() -> None:
    options = MaskOptions(
        strategy="cubemap",
        selected_classes=("body", "head"),
        score_threshold=0.5,
        cube_face_size=512,
    )

    restored = mask_options_from_json(mask_options_to_json(options))

    assert restored.strategy == "cubemap"
    assert restored.selected_classes == ("body", "head")
    assert restored.score_threshold == 0.5
    assert restored.cube_face_size == 512


def test_subprocess_cubemap_reports_child_failure(monkeypatch) -> None:
    calls: list[list[str]] = []

    class FakeProcess:
        stdout = iter(())

        def poll(self):
            return 3221225477

        def wait(self, timeout=None):
            return 3221225477

    def fake_popen(command, **kwargs):
        calls.append(command)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    generator = SubprocessCubemapGenerator(Path("model.onnx"), "CUDAExecutionProvider")

    try:
        generator.generate(np.zeros((4, 8, 3), dtype=np.uint8), MaskOptions())
    except RuntimeError as exc:
        assert "0xC0000005" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert len(calls) == 1
    assert calls[0][calls[0].index("--provider") + 1] == "CUDAExecutionProvider"


def test_subprocess_cubemap_emits_progress(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, command):
            mask_path = Path(command[command.index("--mask") + 1])
            np.save(mask_path, np.full((4, 8), 255, dtype=np.uint8))
            self.stdout = iter(['{"type":"progress","current":3,"total":6,"face":"front"}\n'])

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: FakeProcess(command))
    generator = SubprocessCubemapGenerator(Path("model.onnx"), "CPUExecutionProvider")
    messages: list[tuple[int, int, str]] = []

    result = generator.generate(
        np.zeros((4, 8, 3), dtype=np.uint8),
        MaskOptions(),
        progress=lambda current, total, face: messages.append((current, total, face)),
    )

    assert messages == [(3, 6, "front")]
    assert result.mask.shape == (4, 8)


def test_describe_returncode_names_access_violation() -> None:
    assert "0xC0000005" in describe_returncode(3221225477)


def test_progress_message_from_line() -> None:
    line = '{"type":"progress","current":2,"total":6,"face":"right"}\n'

    assert progress_message_from_line(line) == (2, 6, "right")
    assert progress_message_from_line("onnxruntime warning\n") is None


class FakeWorkerProcess:
    """Speaks the cubemap worker protocol; `crash_on_job` makes the given
    1-based job number die without responding, `silent` accepts jobs but
    never answers them."""

    def __init__(self, crash_on_job: int = 0, silent: bool = False):
        self.crash_on_job = crash_on_job
        self.silent = silent
        self.jobs_received = 0
        self.stdin = self
        self._out: queue.Queue[str | None] = queue.Queue()
        self._returncode: int | None = None
        self._out.put('{"type":"ready"}\n')
        self.stdout = self._iter_lines()

    # --- stdin interface ---
    def write(self, text: str) -> None:
        job = json.loads(text)
        if job.get("type") != "job":
            return
        self.jobs_received += 1
        if self.silent:
            return
        if self.jobs_received == self.crash_on_job:
            self._returncode = 3221225477
            self._out.put(None)
            return
        np.save(Path(job["mask"]), np.full((4, 8), 255, dtype=np.uint8))
        self._out.put('{"type":"progress","current":1,"total":6,"face":"front"}\n')
        self._out.put('{"type":"done"}\n')

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if self._returncode is None:
            self._returncode = 0
            self._out.put(None)

    # --- process interface ---
    def _iter_lines(self):
        while True:
            item = self._out.get()
            if item is None:
                return
            yield item

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout=None) -> int:
        return self._returncode if self._returncode is not None else 0

    def kill(self) -> None:
        self.close()


def test_persistent_worker_reuses_single_process(monkeypatch) -> None:
    processes: list[FakeWorkerProcess] = []

    def fake_popen(command, **kwargs):
        processes.append(FakeWorkerProcess())
        return processes[-1]

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    generator = PersistentCubemapGenerator(Path("model.onnx"), "CPUExecutionProvider")
    messages: list[tuple[int, int, str]] = []

    first = generator.generate(
        np.zeros((4, 8, 3), dtype=np.uint8),
        MaskOptions(),
        progress=lambda current, total, face: messages.append((current, total, face)),
    )
    second = generator.generate(np.zeros((4, 8, 3), dtype=np.uint8), MaskOptions())
    generator.shutdown()

    assert len(processes) == 1
    assert processes[0].jobs_received == 2
    assert first.mask.shape == (4, 8)
    assert second.mask.shape == (4, 8)
    assert messages == [(1, 6, "front")]
    assert first.metadata["persistent"] is True


def test_persistent_worker_retries_job_after_crash(monkeypatch) -> None:
    processes: list[FakeWorkerProcess] = []

    def fake_popen(command, **kwargs):
        # first worker dies on its second job; replacement works fine
        processes.append(FakeWorkerProcess(crash_on_job=2 if not processes else 0))
        return processes[-1]

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    generator = PersistentCubemapGenerator(Path("model.onnx"), "CPUExecutionProvider")

    first = generator.generate(np.zeros((4, 8, 3), dtype=np.uint8), MaskOptions())
    second = generator.generate(np.zeros((4, 8, 3), dtype=np.uint8), MaskOptions())
    generator.shutdown()

    assert len(processes) == 2
    assert first.mask.shape == (4, 8)
    assert second.mask.shape == (4, 8)


def test_persistent_worker_cancels_before_job_starts(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: FakeWorkerProcess())
    generator = PersistentCubemapGenerator(Path("model.onnx"), "CPUExecutionProvider")
    event = threading.Event()
    event.set()

    with pytest.raises(JobCancelled):
        generator.generate(np.zeros((4, 8, 3), dtype=np.uint8), MaskOptions(), cancel_event=event)
    generator.shutdown()


def test_persistent_worker_cancels_job_in_flight(monkeypatch) -> None:
    processes: list[FakeWorkerProcess] = []

    def fake_popen(command, **kwargs):
        processes.append(FakeWorkerProcess(silent=True))
        return processes[-1]

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    generator = PersistentCubemapGenerator(Path("model.onnx"), "CPUExecutionProvider")
    event = threading.Event()
    threading.Timer(0.3, event.set).start()

    with pytest.raises(JobCancelled):
        generator.generate(np.zeros((4, 8, 3), dtype=np.uint8), MaskOptions(), cancel_event=event)
    # the in-flight worker is killed so cancellation takes effect immediately
    assert processes[0].poll() is not None
    generator.shutdown()


def test_persistent_worker_raises_when_fresh_worker_dies(monkeypatch) -> None:
    def fake_popen(command, **kwargs):
        return FakeWorkerProcess(crash_on_job=1)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    generator = PersistentCubemapGenerator(Path("model.onnx"), "CPUExecutionProvider")

    with pytest.raises(RuntimeError, match="0xC0000005"):
        generator.generate(np.zeros((4, 8, 3), dtype=np.uint8), MaskOptions())
    generator.shutdown()
