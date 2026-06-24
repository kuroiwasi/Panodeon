from __future__ import annotations

import threading
from typing import Callable

from PySide6.QtCore import QObject, Signal, Slot

from panodeon.generators.base import JobCancelled


class TaskWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, task: Callable[["TaskWorker"], object]) -> None:
        super().__init__()
        self._task = task
        self._cancel_event = threading.Event()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise JobCancelled("task cancelled")

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(self._task(self))
        except JobCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
