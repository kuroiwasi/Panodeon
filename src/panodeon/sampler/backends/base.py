from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class BackendRequest:
    video_path: Path
    work_dir: Path
    trajectory_path: Path
    backend_config_path: Path


class PoseBackend(Protocol):
    """Command boundary for an external pose-estimation process."""

    @property
    def name(self) -> str: ...

    def command(self, request: BackendRequest) -> Sequence[str]: ...
