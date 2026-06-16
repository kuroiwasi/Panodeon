from __future__ import annotations

import argparse
import tarfile
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/488_DEIMv2-Wholebody49/resources.tar.gz"


def download(url: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "resources.tar.gz"
        urllib.request.urlretrieve(url, archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            safe_extract(archive, output_dir)


def safe_extract(archive: tarfile.TarFile, output_dir: Path) -> None:
    root = output_dir.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe archive member: {member.name}")
    archive.extractall(root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DEIMv2 Wholebody49 resources.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    download(args.url, args.output_dir)
    print(f"Downloaded to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
