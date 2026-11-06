"""Content fingerprints and archival writes for reconstructed Stella runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from project_paths import ROOT


def source_path(path: Path, base: Path) -> str:
    """Use project-relative names while retaining external calibration paths."""
    resolved, base = path.resolve(), base.resolve()
    return str(resolved.relative_to(base)) if resolved.is_relative_to(base) else str(resolved)


def file_fingerprints(paths: list[Path], base: Path) -> dict[str, str]:
    result = {}
    for path in paths:
        resolved = path.resolve()
        name = source_path(resolved, base)
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = digest.hexdigest()
    return result


def verify_fingerprints(expected: dict[str, str], base: Path) -> None:
    """Stop rather than attach a current label to stale upstream inputs."""
    actual = file_fingerprints([base / name for name in expected], base)
    changed = [name for name, digest in expected.items() if actual[name] != digest]
    if changed:
        raise ValueError(f"Stale Stella inputs; rebuild upstream stages: {changed}")


def archive_existing(path: Path) -> Path | None:
    """Preserve prior results under archive/, keeping source paths and hashes."""
    if not path.exists():
        return None
    path = path.resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_relative_to(ROOT.resolve()):
        relative = path.relative_to(ROOT.resolve())
    else:
        relative = Path("external") / path.relative_to(path.anchor)
    archive = ROOT / "archive/stella_results" / relative.parent / f"{path.stem}_{digest[:16]}{path.suffix}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copy2(path, archive)
    provenance = archive.with_suffix(archive.suffix + ".provenance.json")
    if not provenance.exists():
        provenance.write_text(json.dumps({
            "original_path": str(path),
            "archived_path": str(archive),
            "sha256": digest,
            "purpose": "Preserved before replacement of a canonical Stella result",
        }, indent=2) + "\n")
    return archive
