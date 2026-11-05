"""File-integrity and numerical-reference checks using temporary fixtures only."""

import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import check_results
import verify_release


class IntegrityChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "code").mkdir()
        self.code = self.root / "code/example.py"
        self.code.write_text("print('example')\n")
        self.line = f"{verify_release.sha256(self.code)}  code/example.py\n"
        (self.root / "SHA256SUMS").write_text(self.line)

    def test_extra_data_are_allowed_by_default(self):
        extra = self.root / "data/licensed.csv"
        extra.parent.mkdir()
        extra.write_text("user-supplied input\n")
        self.assertEqual(len(verify_release.verify(self.root)), 1)
        with self.assertRaisesRegex(ValueError, "licensed.csv"):
            verify_release.verify(self.root, exact=True)

    def test_changed_or_missing_listed_files_fail(self):
        self.code.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "example.py"):
            verify_release.verify(self.root)
        self.code.unlink()
        with self.assertRaisesRegex(ValueError, "example.py"):
            verify_release.verify(self.root)

    def test_manifest_rejects_escapes_duplicates_and_symlinks(self):
        digest = "0" * 64
        for name in ("../outside", "/tmp/outside", "code/../../outside", "code/./example.py"):
            with self.subTest(name=name):
                (self.root / "SHA256SUMS").write_text(f"{digest}  {name}\n")
                with self.assertRaisesRegex(ValueError, "manifest path"):
                    verify_release.verify(self.root)
        (self.root / "SHA256SUMS").write_text(self.line * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify_release.verify(self.root)
        link = self.root / "code/link.py"
        link.symlink_to(self.code)
        (self.root / "SHA256SUMS").write_text(f"{digest}  code/link.py\n")
        with self.assertRaisesRegex(ValueError, "symlink"):
            verify_release.verify(self.root)

    def test_release_writer_includes_documentation_and_workflows(self):
        source, release = self.root / "research", self.root / "release"
        for name in ("code/run.py", "data/raw.csv", "README.md", "CITATION.cff",
                     "MAINTAINERS.md", ".github/workflows/check.yml"):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + "\n")
        shutil.copytree(source, release)
        with patch.object(sys, "argv", ["verify", "--source", str(source), "--release", str(release), "--write"]):
            with contextlib.redirect_stdout(io.StringIO()):
                verify_release.main()
        self.assertEqual(len(verify_release.verify(release, exact=True)), 6)
        (release / "data/extra.csv").write_text("extra")
        with patch.object(sys, "argv", ["verify", "--source", str(source), "--release", str(release)]):
            with self.assertRaisesRegex(ValueError, "extra.csv"):
                verify_release.main()


class NumericalChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.reference = self.root / "reference.json"
        checks = [{"entity": entity, "file": "summary.pkl", "values": {"estimate": .2, "N": 10},
                   "absolute_tolerances": {"estimate": 1e-6}} for entity in ("A", "B")]
        self.reference.write_text(json.dumps({"checks": checks}))
        self.original_reference = self.reference.read_bytes()

    def write(self, rows):
        pd.DataFrame(rows, columns=("Entity", "estimate", "N")).to_pickle(self.cache / "summary.pkl")

    def run_check(self, strict=False):
        arguments = ["check", "--cache", str(self.cache), "--reference", str(self.reference),
                     "--report", str(self.root / "result_checks.json")]
        if strict:
            arguments.append("--strict")
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            status = check_results.main()
        self.assertEqual(self.reference.read_bytes(), self.original_reference)
        return status, json.loads((self.root / "result_checks.json").read_text())

    def test_tolerance_and_exact_dimensions(self):
        self.write([("A", .2 + .5e-6, 10), ("B", .2, 10)])
        status, report = self.run_check(strict=True)
        self.assertEqual((status, report["status"]), (0, "passed"))
        self.write([("A", .2, 11), ("B", .2, 10)])
        status, report = self.run_check()
        self.assertEqual((status, report["status"]), (1, "failed"))

    def test_missing_results_are_partial_or_not_checked(self):
        status, report = self.run_check()
        self.assertEqual((status, report["status"]), (0, "not_checked"))
        self.write([("A", .2, 10)])
        status, report = self.run_check()
        self.assertEqual((status, report["status"]), (0, "partial"))
        self.assertEqual(self.run_check(strict=True)[0], 1)

    def test_empty_summary_is_unavailable_but_malformed_rows_fail(self):
        for frame in (pd.DataFrame(), pd.DataFrame(columns=("Entity", "estimate", "N"))):
            frame.to_pickle(self.cache / "summary.pkl")
            status, report = self.run_check()
            self.assertEqual((status, report["status"]), (0, "not_checked"))
            self.assertEqual(report["counts"]["unavailable"], 2)
            self.assertEqual(self.run_check(strict=True)[0], 1)
        pd.DataFrame({"Entity": ["A"]}).to_pickle(self.cache / "summary.pkl")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.run_check()

    def test_report_identifies_the_exact_checked_bytes_and_time(self):
        self.write([("A", .2, 10), ("B", .2, 10)])
        contents = (self.cache / "summary.pkl").read_bytes()
        before = datetime.now(timezone.utc)
        _, report = self.run_check()
        checked_at = datetime.fromisoformat(report["checked_at"])
        self.assertLessEqual(before, checked_at)
        self.assertLessEqual(checked_at, datetime.now(timezone.utc))
        self.assertEqual(report["reference_sha256"], hashlib.sha256(self.original_reference).hexdigest())
        self.assertEqual(report["result_sha256"], {"summary.pkl": hashlib.sha256(contents).hexdigest()})
        self.write([("A", .3, 10), ("B", .2, 10)])
        status, changed = self.run_check()
        self.assertEqual(status, 1)
        self.assertNotEqual(changed["result_sha256"], report["result_sha256"])
        self.assertEqual(changed["reference_sha256"], report["reference_sha256"])

    def test_mismatch_and_nonfinite_values_cannot_pass(self):
        for value in (.3, float("inf")):
            self.write([("A", value, 10), ("B", .2, 10)])
            self.assertEqual(self.run_check()[0], 1)
        self.write([("A", float("nan"), 10), ("B", .2, 10)])
        self.assertEqual(self.run_check(strict=True)[0], 1)


if __name__ == "__main__":
    unittest.main()
