"""Fetch the frozen public inputs listed in downloads.json."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("downloads.json")


def verify(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"{path.name} differs from the frozen input. "
            f"Expected SHA256 {expected}, received {actual}. "
            "The local file or upstream version may have changed."
        )


def obtain(source: dict) -> None:
    target = ROOT / source["path"]
    if target.is_file():
        verify(target, source["sha256"])
        print(f"Verified {source['path']}", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {source['path']}", flush=True)
    request = urllib.request.Request(
        source["url"], headers={"User-Agent": "granular-fluctuations-replication/1.0"}
    )
    with TemporaryDirectory(prefix=".download-", dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        with urllib.request.urlopen(request, timeout=120) as response, staged.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        verify(staged, source["sha256"])
        staged.replace(target)
    print(f"Verified {source['path']}", flush=True)


def main() -> None:
    sources = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=list(sources), help="Fetch selected inputs.")
    args = parser.parse_args()
    selected = args.only or list(sources)
    failed = []
    for name in selected:
        try:
            obtain(sources[name])
        except (OSError, ValueError) as error:
            failed.append(name)
            print(f"Failed {name}: {error}", file=sys.stderr, flush=True)
    if failed:
        parser.exit(1, f"Verified {len(selected) - len(failed)}/{len(selected)} public inputs; failed: {', '.join(failed)}\n")
    print(f"Verified all {len(selected)} selected public inputs", flush=True)


if __name__ == "__main__":
    main()
