"""Check release file hashes and compare its code and inputs to the research copy."""

import argparse
import hashlib
import json
from pathlib import Path


IGNORED = {".cache", "__pycache__", ".ruff_cache", ".matplotlib-cache", ".pytest_cache"}
MANIFEST = "SHA256SUMS"


def files(root: Path, folder: str) -> dict[str, Path]:
    result = {}
    for path in sorted((root / folder).rglob("*")):
        if IGNORED.intersection(path.parts) or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise ValueError(f"Release inputs must be regular files: {path}")
        if path.is_file() and path.suffix not in {".pyc", ".pyo", ".lic"}:
            result[path.relative_to(root).as_posix()] = path
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, help="Research project root for byte comparisons.")
    parser.add_argument("--write", action="store_true", help="Write SHA256SUMS after comparing source files.")
    args = parser.parse_args()
    root = args.release.resolve()
    paths = {**files(root, "code"), **files(root, "data")}
    if not paths:
        parser.error("The release must contain code and its selected inputs")
    hashes = {name: sha256(path) for name, path in paths.items()}
    if args.source:
        source = args.source.resolve()
        source_code = files(source, "code")
        released_code = files(root, "code")
        if source_code.keys() != released_code.keys():
            missing = sorted(source_code.keys() - released_code.keys())
            extra = sorted(released_code.keys() - source_code.keys())
            raise ValueError(f"Code file sets differ. Missing {missing}. Extra {extra}.")
        different = [name for name, digest in hashes.items()
                     if not (source / name).is_file() or sha256(source / name) != digest]
        if different:
            raise ValueError(f"Files differ from the research copy: {different}")
        print(f"Research copy matches all {len(source_code)} code files and {len(paths) - len(source_code)} input files.")
    if args.write:
        if not args.source:
            parser.error("--write requires --source")
        (root / MANIFEST).write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())))
    else:
        expected = {}
        for line in (root / MANIFEST).read_text().splitlines():
            digest, name = line.split("  ", 1)
            expected[name] = digest
        differing = [name for name, digest in expected.items() if hashes.get(name) != digest]
        downloads = root / "data/downloads.json"
        allowed = {entry["path"]: entry["sha256"] for entry in json.loads(downloads.read_text()).values()} if downloads.is_file() else {}
        differing.extend(name for name in hashes.keys() - expected.keys() if hashes[name] != allowed.get(name))
        if differing:
            raise ValueError(f"Release hashes differ: {sorted(differing)}")
    print(f"Verified {len(hashes)} SHA-256 hashes.")


if __name__ == "__main__":
    main()
