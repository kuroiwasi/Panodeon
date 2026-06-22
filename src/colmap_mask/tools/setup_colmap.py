from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

COLMAP_VERSION = "4.0.4"
RELEASE_URL_TEMPLATE = (
    "https://github.com/colmap/colmap/releases/download/{version}/{asset}"
)


def asset_name(nocuda: bool) -> str:
    variant = "nocuda" if nocuda else "cuda"
    return f"colmap-x64-windows-{variant}.zip"


def release_url(version: str, nocuda: bool) -> str:
    return RELEASE_URL_TEMPLATE.format(version=version, asset=asset_name(nocuda))


def download_colmap(
    version: str = COLMAP_VERSION,
    nocuda: bool = False,
    dest_dir: Path = Path("third_party") / "colmap",
    force: bool = False,
) -> None:
    dest_dir = Path(dest_dir)
    colmap_exe = dest_dir / "bin" / "colmap.exe"
    if colmap_exe.exists() and not force:
        print(f"COLMAP already present at {dest_dir}, skipping (use --force to re-download).")
        return

    url = release_url(version, nocuda)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "colmap.zip"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, archive_path)

        extract_dir = tmp_path / "extracted"
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, extract_dir)

        source = locate_colmap_root(extract_dir)
        if force and dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest_dir))

    if not colmap_exe.exists():
        raise FileNotFoundError(f"Expected COLMAP binary not found at {colmap_exe}")
    print(f"COLMAP {version} installed to {dest_dir}")


def safe_extract(archive: zipfile.ZipFile, output_dir: Path) -> None:
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for member in archive.namelist():
        target = (root / member).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe archive member: {member}")
    archive.extractall(root)


def locate_colmap_root(extract_dir: Path) -> Path:
    """Return the directory containing bin/colmap.exe within the extracted zip.

    COLMAP release zips wrap everything in a top-level ``colmap-x64-windows-*``
    folder; this strips it so the result can be moved directly to third_party/colmap.
    """
    if (extract_dir / "bin" / "colmap.exe").exists():
        return extract_dir
    candidates = [p for p in extract_dir.iterdir() if p.is_dir()]
    for candidate in candidates:
        if (candidate / "bin" / "colmap.exe").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate bin/colmap.exe in extracted archive under {extract_dir}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a prebuilt COLMAP Windows release.")
    parser.add_argument("--colmap-version", default=COLMAP_VERSION)
    parser.add_argument(
        "--nocuda",
        action="store_true",
        help="Download the no-CUDA build instead of the default CUDA build.",
    )
    parser.add_argument("--dest", type=Path, default=Path("third_party") / "colmap")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove any existing install and re-download.",
    )
    args = parser.parse_args()
    download_colmap(
        version=args.colmap_version,
        nocuda=args.nocuda,
        dest_dir=args.dest,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
