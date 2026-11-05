"""Discover and run tests in temporary checkouts, without changing analysis caches."""

import argparse
import ast
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


CODE = Path(__file__).resolve().parent
GROUPS = ("fast", "numerical", "integration")


def test_metadata(path):
    metadata = {"TEST_GROUP": "fast", "TEST_CACHE_INPUTS": ()}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in metadata:
                    metadata[target.id] = ast.literal_eval(node.value)
    if metadata["TEST_GROUP"] not in GROUPS:
        raise ValueError(f"Unknown test group in {path}: {metadata['TEST_GROUP']}")
    return metadata


def run_module(name, report):
    started = time.perf_counter()
    module = importlib.import_module(f"tests.{name}")
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    functions = [value for key, value in vars(module).items()
                 if key.startswith("test_") and inspect.isfunction(value)
                 and value.__module__ == module.__name__]
    suite.addTests(unittest.FunctionTestCase(function) for function in functions)
    if not suite.countTestCases() and callable(getattr(module, "main", None)):
        suite.addTest(unittest.FunctionTestCase(module.main))
    if not suite.countTestCases():
        raise ValueError(f"No tests discovered in {name}")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024**2 if sys.platform == "darwin" else 1024)
    payload = {
        "module": name, "tests_run": result.testsRun,
        "status": "passed" if result.wasSuccessful() else "failed",
        "skipped": [reason for _, reason in result.skipped],
        "failures": len(result.failures), "errors": len(result.errors),
        "seconds": time.perf_counter() - started, "peak_memory_mb": peak_mb,
    }
    if len(result.skipped) == result.testsRun:
        payload["status"] = "skipped"
    report.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


def copy_code(destination):
    shutil.copytree(CODE, destination, ignore=shutil.ignore_patterns(
        ".cache", "__pycache__", ".ruff_cache", ".pytest_cache", "*.pyc", "*.lic",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("all", *GROUPS), nargs="?", default="all")
    parser.add_argument("--list", action="store_true", help="List tests and groups without importing them.")
    parser.add_argument("--only", nargs="+", help="Run only these test module names.")
    parser.add_argument("--report", type=Path, default=CODE / ".cache/test_report.json")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_module(args.worker, args.report)
    discovered = [(path, test_metadata(path)) for path in sorted((CODE / "tests").glob("test_*.py"))]
    if args.only:
        unknown = set(args.only) - {path.stem for path, _ in discovered}
        if unknown:
            parser.error(f"Unknown test modules: {sorted(unknown)}")
    selected = [(path, metadata) for path, metadata in discovered
                if (args.group == "all" or metadata["TEST_GROUP"] == args.group)
                and (not args.only or path.stem in args.only)]
    if not selected:
        parser.error("No tests match the requested group and module names")
    if args.list:
        for path, metadata in selected:
            print(f"{metadata['TEST_GROUP']:12} {path.stem}")
        return 0
    reports = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="granularity-tests-") as temporary:
        root = Path(temporary)
        for group in GROUPS:
            tests = [(path, metadata) for path, metadata in selected if metadata["TEST_GROUP"] == group]
            if not tests:
                continue
            checkout = root / group
            isolated = checkout / "code"
            copy_code(isolated)
            if group == "integration" and (CODE.parent / "data").is_dir():
                (checkout / "data").symlink_to(CODE.parent / "data", target_is_directory=True)
            for _, metadata in tests:
                for pattern in metadata["TEST_CACHE_INPUTS"]:
                    for source in CODE.glob(pattern):
                        if source.is_file():
                            target = isolated / source.relative_to(CODE)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, target)
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="2",
                       MPLCONFIGDIR=str(checkout / ".matplotlib"), PYTHONDONTWRITEBYTECODE="1")
            for path, _ in tests:
                print(f"\n[{group}] {path.stem}", flush=True)
                record = checkout / f"{path.stem}.json"
                result = subprocess.run([sys.executable, str(isolated / "run_tests.py"),
                                         "--worker", path.stem, "--report", str(record)],
                                        cwd=isolated, env=env)
                payload = json.loads(record.read_text()) if record.is_file() else {
                    "module": path.stem, "status": "failed",
                    "reason": f"Test process exited with status {result.returncode} before reporting",
                }
                payload["group"] = group
                if result.returncode:
                    payload["status"] = "failed"
                reports.append(payload)
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "seconds": time.perf_counter() - started, "modules": reports}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    counts = {status: sum(row["status"] == status for row in reports) for status in ("passed", "skipped", "failed")}
    skipped_tests = sum(len(row.get("skipped", [])) for row in reports)
    print(f"\nTest modules: {counts['passed']} passed, {counts['skipped']} skipped, {counts['failed']} failed; "
          f"{skipped_tests} individual tests skipped. Report: {args.report}")
    return int(bool(counts["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
