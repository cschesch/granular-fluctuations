"""Shared locations for generated analysis data."""

from pathlib import Path
import os
import shutil


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
DERIVED_CACHE = CODE_DIR / ".cache" / "derived"


def find_program(name: str, override: str | None = None) -> str | None:
    """Find an explicit executable, its environment setting, or a local install."""
    if configured := override or os.environ.get(name.upper()):
        return shutil.which(configured)
    names = ("stata-mp", "stata-se", "stata") if name == "stata" else (name,)
    for candidate in names:
        if path := shutil.which(candidate):
            return path
    patterns = {
        "matlab": "/Applications/MATLAB*.app/bin/matlab",
        "stata": "/Applications/Stata*/Stata*.app/Contents/MacOS/stata-*",
        "octave": "/usr/local/bin/octave",
    }
    pattern = patterns.get(name)
    if pattern:
        for path in sorted(Path("/").glob(pattern.lstrip("/")), reverse=name == "matlab"):
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
    return None


def matlab_string(path: Path) -> str:
    return str(path).replace("'", "''")


def available(label: str, *paths: Path) -> bool:
    missing = [str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
               for path in paths if not path.exists()]
    if missing:
        print(f"[skip] {label}: missing {', '.join(missing)}", flush=True)
    return not missing


def load_available(sources):
    """Load independent panels, skipping missing source files only."""
    for label, load in sources:
        try:
            yield load()
        except FileNotFoundError as error:
            print(f"[skip] {label}: {error}", flush=True)


def data_file(path: Path) -> Path:
    """Read the preserved source in plain or losslessly compressed form."""
    compressed = path.with_suffix(path.suffix + ".gz")
    return compressed if not path.exists() and compressed.is_file() else path


def upstream(study: str) -> Path:
    """Extract a pinned author ZIP once into an immutable cache directory."""
    import hashlib
    import json
    import shutil
    import tempfile
    import zipfile

    source = ROOT / "data" / study / "source.zip"
    if not source.is_file():
        return source.parent / "replication_package"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = source.with_suffix(".json")
    provenance = json.loads(metadata.read_text()) if metadata.is_file() else {}
    if provenance.get("sha256", digest) != digest:
        raise ValueError(f"Author archive checksum changed: {source}")
    target = CODE_DIR / ".cache/upstream" / study / digest
    if target.is_dir():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = provenance.get("extraction_prefix", "")
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=target.parent) as temporary:
        staged = Path(temporary) / "source"
        staged.mkdir()
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.startswith(prefix):
                    continue
                name = Path(member.filename[len(prefix):])
                if not name.parts or name.parts[0] == "__MACOSX" or name.name == ".DS_Store":
                    continue
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError(f"Unsafe path in author archive: {member.filename}")
                destination = staged / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as original, destination.open("wb") as output:
                    shutil.copyfileobj(original, output)
        try:
            staged.rename(target)
        except OSError:
            if not target.is_dir():
                raise
    return target
