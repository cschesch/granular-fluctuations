#!/usr/bin/env python3
"""Build preserved Census X-12 v0.3 without changing its Fortran algorithms.

The discontinued Census download is preserved by FreeBSD. Verify the archive
against FreeBSD's HTTPS-published original checksum before extraction. Only
compiler/legacy-Fortran compatibility flags differ from the shipped makefile.
This establishes an actual X-12 engine; the author's exact build is not known.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tarfile

from project_paths import ROOT
ARCHIVE = ROOT / "data/stella2015/public_inputs/omegav03src.tar.gz"
BUILD = ROOT / "code/.cache/tools/x12a_v03"
URL = "http://distcache.freebsd.org/ports-distfiles/x12arima/omegav03src.tar.gz"
SHA256 = "ca818383da41205ebaa9542a59d4b89c95ad700bc1a7082bb64f7cb25c696a0a"
CHECKSUM_SOURCE = ("https://raw.githubusercontent.com/freebsd/freebsd-ports/"
    "e59c88cece035a155debaf8996ce8f20adabd78b/math/x12arima/distinfo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=shutil.which("gfortran"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not args.compiler:
        parser.error("GNU Fortran is required; pass --compiler.")
    if not ARCHIVE.exists():
        parser.error(f"Missing bundled X-12 source archive: {ARCHIVE}")
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("X-12 source archive checksum mismatch.")
    source = BUILD / "source"
    source.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(source, filter="data")
    compile_flags = "-O2 -std=legacy -fallow-argument-mismatch"
    link_flags = ""
    native_apple_tools = False
    # Intel Homebrew GCC may run under Rosetta while Xcode tools are arm64.
    # Invoke the native Apple assembler/linker, retaining GCC's x86_64 target.
    if platform.system() == "Darwin":
        probe = subprocess.run(["/usr/bin/arch", "-arm64", "/usr/bin/true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode == 0:
            native_apple_tools = True
            tools = BUILD / "toolchain"
            tools.mkdir(exist_ok=True)
            for name in ["as", "ld"]:
                wrapper = tools / name
                wrapper.write_text('#!/bin/sh\nexec /usr/bin/arch -arm64 /usr/bin/' + name + ' "$@"\n')
                wrapper.chmod(0o755)
            sdk = subprocess.check_output(["/usr/bin/arch", "-arm64", "/usr/bin/xcrun",
                "--show-sdk-path"], text=True).strip()
            compile_flags += f" -B{tools}/"
            link_flags = f"-B{tools}/ -isysroot {sdk}"
    command = ["make", "-f", "makefile.lnx", f"-j{args.jobs}",
        f"FC={args.compiler}", f"LINKER={args.compiler}",
        f"FFLAGS={compile_flags}", f"LDFLAGS={link_flags}"]
    with (BUILD / "build.log").open("w") as log:
        subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
    executable = source / "x12a"
    import numpy as np
    import pandas as pd
    from applications.stella.prepare_stella_sales import adjust_one
    t = np.arange(200)
    synthetic = np.exp(5 + .006*t + .1*np.sin(2*np.pi*t/4) + .03*np.sin(t*.37))
    _, adjusted, _ = adjust_one("x12_build_smoke", synthetic, pd.Period("1962Q1"),
        executable, "sax12")
    verified = bool(len(adjusted) == len(synthetic) and np.isfinite(adjusted).all()
                    and (adjusted > 0).all())
    if not verified:
        raise RuntimeError("Built X-12 failed its D11 smoke test.")
    record = {"archive": str(ARCHIVE.relative_to(ROOT)), "archive_sha256": SHA256,
        "archive_url": URL, "checksum_source_https": CHECKSUM_SOURCE,
        "executable": str(executable.relative_to(ROOT)),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output([args.compiler, "--version"], text=True).splitlines()[0],
        "command": command, "native_apple_tool_wrappers": native_apple_tools,
        "fortran_algorithm_edits": [],
        "verified": verified, "D11_smoke_test_observations": len(adjusted),
        "build_parameters": (source / "build.prm").read_text(),
        "caveat": "Original Census X-12 v0.3; Stella's exact executable build is unknown."}
    (BUILD / "build_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
