"""Trajectory-based frame selection for 360-degree video (vendored from 360_frame_sampler).

The public dataclasses and ``sample_trajectory`` are re-exported lazily so that
importing lightweight submodules (``workflow``, ``resume``, ``events``) — as the GUI
does — does not pull in the heavy ``av`` / ``cv2`` stack required only by the actual
sampling pipeline. The pipeline itself runs out-of-process, so the GUI can start even
when ``av`` is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "ContinuityConfig",
    "OrbConfig",
    "SamplerConfig",
    "SamplingConfig",
    "SamplingResult",
    "SelectionRecord",
    "TrajectoryRecord",
    "sample_trajectory",
]

__version__ = "0.1.0"

_MODEL_NAMES = {
    "ContinuityConfig",
    "OrbConfig",
    "SamplerConfig",
    "SamplingConfig",
    "SelectionRecord",
    "TrajectoryRecord",
}
_PIPELINE_NAMES = {"SamplingResult", "sample_trajectory"}


def __getattr__(name: str):
    if name in _MODEL_NAMES:
        from . import models

        return getattr(models, name)
    if name in _PIPELINE_NAMES:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import (
        ContinuityConfig,
        OrbConfig,
        SamplerConfig,
        SamplingConfig,
        SelectionRecord,
        TrajectoryRecord,
    )
    from .pipeline import SamplingResult, sample_trajectory
