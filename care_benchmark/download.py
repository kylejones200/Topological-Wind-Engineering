"""Download CARE to Compare dataset from Zenodo."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://zenodo.org/records/15846963/files/CARE_To_Compare.zip"
DEFAULT_DEST = _REPO_ROOT / "data" / "care" / "CARE_To_Compare.zip"


def _expected_size(url: str) -> int | None:
    try:
        head = requests.head(url, timeout=60, allow_redirects=True)
        if head.ok and head.headers.get("Content-Length"):
            return int(head.headers["Content-Length"])
    except requests.RequestException:
        return None
    return None


def _download_with_resume(dest_zip: Path, url: str, max_retries: int = 20) -> None:
    expected = _expected_size(url)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        downloaded = dest_zip.stat().st_size if dest_zip.exists() else 0
        if expected and downloaded >= expected:
            print(f"Download complete: {dest_zip} ({downloaded / 1e9:.2f} GB)", flush=True)
            return

        headers = {}
        mode = "wb"
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"
            mode = "ab"
            print(
                f"Resuming download (attempt {attempt}/{max_retries}) from "
                f"{downloaded / 1e9:.2f} GB",
                flush=True,
            )
        else:
            print(f"Downloading CARE dataset (~5.5 GB) to {dest_zip}", flush=True)

        try:
            with requests.get(url, stream=True, headers=headers, timeout=180) as response:
                if response.status_code not in (200, 206):
                    response.raise_for_status()
                with open(dest_zip, mode) as handle:
                    last_log = time.time()
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        if time.time() - last_log > 30:
                            size = dest_zip.stat().st_size
                            pct = f"{100 * size / expected:.1f}%" if expected else "?"
                            print(f"  ... {size / 1e9:.2f} GB ({pct})", flush=True)
                            last_log = time.time()
            if expected is None or dest_zip.stat().st_size >= expected:
                return
        except (requests.RequestException, OSError) as exc:
            print(f"Download interrupted: {exc}", flush=True)
            time.sleep(min(30, attempt * 5))

    raise RuntimeError(f"Failed to download CARE dataset after {max_retries} attempts")


def download_care(dest_zip: Path = DEFAULT_DEST, url: str = DEFAULT_URL) -> Path:
    dest_zip = Path(dest_zip)
    _download_with_resume(dest_zip, url)

    extract_root = dest_zip.parent / "CARE_To_Compare"
    if not (extract_root / "Wind Farm A" / "event_info.csv").is_file():
        print(f"Extracting to {extract_root}", flush=True)
        with zipfile.ZipFile(dest_zip, "r") as zf:
            zf.extractall(dest_zip.parent)
    return extract_root


def main():
    parser = argparse.ArgumentParser(description="Download CARE to Compare dataset")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    args = parser.parse_args()
    root = download_care(args.dest, args.url)
    print(f"CARE dataset ready at {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
