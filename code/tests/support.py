"""Explicit prerequisites for integration checks."""

from unittest import SkipTest


def require_files(*paths):
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SkipTest("Missing prepared inputs: " + ", ".join(missing))
