from __future__ import annotations

import argparse
from pathlib import Path

from colmap_mask.tools import download_model, setup_colmap

THIRD_PARTY_DIR = Path("third_party")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up everything under third_party/ (COLMAP binaries and ONNX models)."
    )
    parser.add_argument("--skip-colmap", action="store_true", help="Do not download COLMAP.")
    parser.add_argument("--skip-models", action="store_true", help="Do not download ONNX models.")
    parser.add_argument("--colmap-version", default=setup_colmap.COLMAP_VERSION)
    parser.add_argument(
        "--nocuda",
        action="store_true",
        help="Download the no-CUDA COLMAP build instead of the default CUDA build.",
    )
    parser.add_argument("--model-url", default=download_model.DEFAULT_URL)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download COLMAP even if it is already present.",
    )
    parser.add_argument("--third-party-dir", type=Path, default=THIRD_PARTY_DIR)
    args = parser.parse_args()

    third_party = args.third_party_dir

    if not args.skip_colmap:
        print("== Setting up COLMAP ==")
        setup_colmap.download_colmap(
            version=args.colmap_version,
            nocuda=args.nocuda,
            dest_dir=third_party / "colmap",
            force=args.force,
        )
    else:
        print("Skipping COLMAP.")

    if not args.skip_models:
        print("== Setting up ONNX models ==")
        model_dir = third_party / "models"
        download_model.download(args.model_url, model_dir)
        print(f"Models downloaded to {model_dir}")
    else:
        print("Skipping models.")

    print("third_party setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
