"""Download classic 3x3 natural-image patch DCT data.

The canonical javaPlex repository has historically hosted these Matlab files
under ``TDA/image_patches/data``. If those raw URLs move, this script fails loudly
and prints the locations it tried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASETS = {
    "n50000Dct.mat": [
        "https://raw.githubusercontent.com/appliedtopology/javaplex/master/data/natural_images/n50000Dct.mat",
        "https://github.com/appliedtopology/javaplex/raw/master/data/natural_images/n50000Dct.mat",
    ],
    "nk300c30Dct.mat": [
        "https://raw.githubusercontent.com/appliedtopology/javaplex/master/data/natural_images/nk300c30Dct.mat",
        "https://github.com/appliedtopology/javaplex/raw/master/data/natural_images/nk300c30Dct.mat",
    ],
    "nk15c30Dct.mat": [
        "https://raw.githubusercontent.com/appliedtopology/javaplex/master/data/natural_images/nk15c30Dct.mat",
        "https://github.com/appliedtopology/javaplex/raw/master/data/natural_images/nk15c30Dct.mat",
    ],
}


def _download(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "DiffusionGeometryFribourg/tda"})
    with urlopen(request, timeout=60) as response:
        if getattr(response, "status", 200) >= 400:
            raise HTTPError(url, response.status, response.reason, response.headers, None)
        with path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mat_variables(path: Path) -> List[Dict[str, object]]:
    mat = loadmat(path)
    variables = []
    for key, value in mat.items():
        if key.startswith("__") or not isinstance(value, np.ndarray):
            continue
        variables.append(
            {
                "name": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        )
    if not variables:
        raise ValueError(f"{path} contains no non-private Matlab array variables.")
    return variables


def fetch_one(
    name: str, urls: List[str], output_dir: Path, force: bool
) -> Dict[str, object]:
    path = output_dir / name
    if path.exists() and not force:
        print(f"{name}: already present; validating.")
        source_url = urls[0]
    else:
        errors = []
        tmp_path = path.with_suffix(path.suffix + ".part")
        for url in urls:
            print(f"{name}: downloading {url}")
            try:
                _download(url, tmp_path)
                tmp_path.replace(path)
                source_url = url
                break
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                errors.append(f"{url}: {exc}")
                if tmp_path.exists():
                    tmp_path.unlink()
        else:
            tried = "\n  ".join(errors)
            raise RuntimeError(
                f"Could not download {name}. Tried:\n  {tried}\n"
                "The canonical javaPlex natural_images source may have moved. "
                "Check https://github.com/appliedtopology/javaplex/tree/master/data/natural_images "
                "or a released javaPlex Matlab examples archive, then rerun with --force "
                "or place the file manually in TDA/image_patches/data/."
            )

    variables = _mat_variables(path)
    info = {
        "file": name,
        "source_url": source_url,
        "byte_size": path.stat().st_size,
        "sha256": _sha256(path),
        "matlab_variables": variables,
    }
    print(f"{name}: {info['byte_size']} bytes, sha256={info['sha256']}")
    return info


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "Classic natural-image 3x3 high-contrast patch DCT data used in javaPlex/applied-topology examples.",
        "datasets": [],
    }
    for name, urls in DATASETS.items():
        manifest["datasets"].append(fetch_one(name, urls, args.output_dir, args.force))

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("TDA/image_patches/data"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
