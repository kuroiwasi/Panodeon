from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from panodeon.generators.cubemap import CubemapGenerator
from panodeon.generators.subprocess_cubemap import mask_options_from_json
from panodeon.inference.deim_wholebody import DeimWholebodySegmenter
from panodeon.inference.providers import resolve_execution_providers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--options", required=True)
    args = parser.parse_args()

    image = np.load(Path(args.image))
    options = mask_options_from_json(json.loads(args.options))
    providers = resolve_execution_providers(args.provider)
    segmenter = DeimWholebodySegmenter(Path(args.model), providers=providers)

    def progress(current: int, total: int, face: str) -> None:
        print(
            json.dumps({"type": "progress", "current": current, "total": total, "face": face}),
            flush=True,
        )

    result = CubemapGenerator(segmenter).generate(image, options, progress=progress)
    np.save(Path(args.mask), result.mask.astype(np.uint8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
