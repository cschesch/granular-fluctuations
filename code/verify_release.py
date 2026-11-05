"""Verify bundled file checksums; optionally compare an exact release to its source."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re


IGNORED = {".cache", "__pycache__", ".ruff_cache", ".matplotlib-cache", ".pytest_cache"}
MANIFEST = "SHA256SUMS"
DOCUMENTS = ("README.md", "CITATION.cff", "MAINTAINERS.md")


def files(root: Path, folder: str) -> dict[str, Path]:
    result = {}
    for path in sorted((root / folder).rglob("*")):
        if IGNORED.intersection(path.parts) or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise ValueError(f"Release files must be regular files: {path}")
        if path.is_file() and path.suffix not in {".pyc", ".pyo", ".lic"}:
            result[path.relative_to(root).as_posix()] = path
    return result


def release_files(root: Path) -> dict[str, Path]:
    result = {**files(root, "code"), **files(root, "data"), **files(root, ".github")}
    for name in DOCUMENTS:
        if (root / name).is_file():
            result[name] = root / name
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != name or "\\" in name):
        raise ValueError(f"Invalid manifest path: {name!r}")
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path is a symlink or escapes the release: {name}")
    return path


def read_manifest(root: Path) -> dict[str, str]:
    expected = {}
    for number, line in enumerate((root / MANIFEST).read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"Invalid {MANIFEST} entry at line {number}")
        digest, name = match.groups()
        manifest_path(root, name)
        if name in expected:
            raise ValueError(f"Duplicate manifest path: {name}")
        expected[name] = digest
    if not expected:
        raise ValueError(f"{MANIFEST} is empty")
    return expected


def verify(root: Path, *, exact: bool = False) -> dict[str, str]:
    """Check listed files; additional inputs are allowed unless exact is requested."""
    expected = read_manifest(root)
    differing = []
    for name, digest in expected.items():
        path = manifest_path(root, name)
        if not path.is_file() or sha256(path) != digest:
            differing.append(name)
    if exact:
        differing.extend(release_files(root).keys() - expected.keys())
    if differing:
        raise ValueError(f"Release files differ: {sorted(differing)}")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--exact", action="store_true", help="Also reject unlisted code, data and release documents.")
    parser.add_argument("--source", type=Path, help="Require an exact release and compare it with the research project.")
    parser.add_argument("--write", action="store_true", help="Write checksums after comparing with --source.")
    args = parser.parse_args()
    if args.write and not args.source:
        parser.error("--write requires --source")
    root = args.release.resolve()
    if args.write:
        paths = release_files(root)
        if not paths:
            parser.error("The release contains no files")
        hashes = {name: sha256(manifest_path(root, name)) for name in paths}
    else:
        hashes = verify(root, exact=args.exact or bool(args.source))
    if args.source:
        source = args.source.resolve()
        source_code, released_code = files(source, "code"), files(root, "code")
        if source_code.keys() != released_code.keys():
            missing = sorted(source_code.keys() - released_code.keys())
            extra = sorted(released_code.keys() - source_code.keys())
            raise ValueError(f"Code file sets differ. Missing {missing}. Extra {extra}.")
        different = [name for name, digest in hashes.items()
                     if not (source / name).is_file() or sha256(source / name) != digest]
        if different:
            raise ValueError(f"Files differ from the research copy: {different}")
        print(f"Research copy matches all {len(hashes)} selected release files.")
    if args.write:
        (root / MANIFEST).write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())))
    print(f"File integrity: verified {len(hashes)} SHA-256 hashes. Numerical results are checked separately.")


if __name__ == "__main__":
    main()
