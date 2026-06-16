from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from colmap_mask.generators.cubemap import CubemapGenerator
from colmap_mask.generators.subprocess_cubemap import mask_options_from_json
from colmap_mask.inference.deim_wholebody import DeimWholebodySegmenter
from colmap_mask.inference.providers import resolve_execution_providers


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True)
    args = parser.parse_args()

    providers = resolve_execution_providers(args.provider)
    segmenter = DeimWholebodySegmenter(Path(args.model), providers=providers)
    generator = CubemapGenerator(segmenter)
    emit({"type": "ready"})

    def progress(current: int, total: int, face: str) -> None:
        emit({"type": "progress", "current": current, "total": total, "face": face})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError:
            continue
        if job.get("type") == "shutdown":
            break
        if job.get("type") != "job":
            continue
        image = None
        result = None
        try:
            image = np.load(Path(str(job["image"])))
            options = mask_options_from_json(job.get("options", {}))
            result = generator.generate(image, options, progress=progress)
            np.save(Path(str(job["mask"])), result.mask.astype(np.uint8))
            emit({"type": "done", "elapsed_sec": result.elapsed_sec})
        except Exception as exc:  # noqa: BLE001 - report to parent, keep serving
            emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            # release per-job buffers before idling for the next job
            image = None
            result = None
            gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
