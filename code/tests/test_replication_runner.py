"""Partial, strict and failed runs must remain distinguishable from completed runs."""

import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import run_replication as runner
import paper_outputs


class ReplicationStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.code = self.root / "code"
        self.code.mkdir()
        self.output = self.root / "output"
        self.cache = self.code / ".cache/run_outputs"
        self.report = self.output / "run_report.json"
        for name, value in (("CODE_DIR", self.code), ("OUT", self.output), ("CACHE", self.cache)):
            patcher = patch.object(runner, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.dict("os.environ", {})
        patcher.start()
        self.addCleanup(patcher.stop)

    def execute(self, analyses, strict=False):
        with contextlib.redirect_stdout(io.StringIO()):
            status = runner.execute([("test", a) for a in analyses], self.report, strict=strict)
        return status, json.loads(self.report.read_text())

    def test_missing_input_is_partial_and_visible(self):
        analysis = runner.Analysis("missing", (), (self.root / "missing.csv",), ("missing.tex",))
        code, report = self.execute([analysis])
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["analyses"][0]["status"], "skipped")
        self.assertIn("Unavailable", (self.output / "missing.tex").read_text())

    def test_strict_missing_input_fails_and_stops(self):
        missing = runner.Analysis("missing", (), (self.root / "missing.csv",))
        code, report = self.execute([missing, runner.Analysis("later", ())], strict=True)
        self.assertNotEqual(code, 0)
        self.assertEqual(report["analyses"][1]["status"], "not_run")

    def test_command_skip_and_failure_are_reported(self):
        (self.code / "partial.py").write_text('print("[skip] public panel: absent data")\n')
        (self.code / "failure.py").write_text('raise SystemExit(3)\n')
        partial = runner.Analysis("partial", (("partial",),))
        code, report = self.execute([partial])
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "partial")
        self.assertIn("absent data", report["analyses"][0]["reasons"][0])
        code, _ = self.execute([partial], strict=True)
        self.assertNotEqual(code, 0)
        code, report = self.execute([runner.Analysis("failed", (("failure",),))])
        self.assertEqual(code, 3)
        self.assertEqual(report["status"], "failed")

    def test_success_is_complete(self):
        (self.code / "success.py").write_text('print("done")\n')
        code, report = self.execute([runner.Analysis("success", (("success",),))])
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["analyses"][0]["commands"][0]["returncode"], 0)

    def test_stella_missing_software_skips_but_locator_errors_fail(self):
        analysis = runner.Analysis("Stella", ())
        with patch.object(runner, "find_program", return_value=None), \
                patch("network_solver.find_gensys", return_value=self.root), \
                patch("applications.stella.prepare_stella_sales.locate_x13", return_value=self.root):
            code, report = self.execute([analysis])
            self.assertEqual((code, report["status"]), (0, "partial"))
            self.assertIn("Octave", report["analyses"][0]["reasons"][0])
            self.assertEqual(self.execute([analysis], strict=True)[0], 2)
        with patch("network_solver.find_gensys", side_effect=ValueError("invalid metadata")):
            with self.assertRaisesRegex(ValueError, "invalid metadata"):
                self.execute([analysis])
        self.assertEqual(json.loads(self.report.read_text())["status"], "failed")

    def test_strict_artifact_check_rejects_a_partial_section_set(self):
        self.output.mkdir()
        self.report.write_text(json.dumps({"status": "complete", "analyses": [{"section": "simulations"}]}))
        with patch.object(paper_outputs, "RUN_REPORT", self.report), \
                patch.object(paper_outputs, "check_outputs"), \
                patch.object(sys, "argv", ["paper_outputs", "check", "--strict", "--manuscript", str(self.root / "absent.tex")]):
            with self.assertRaisesRegex(RuntimeError, "does not cover the full paper"):
                paper_outputs.main()

    def test_empirical_refreshes_quarterly_prerequisite(self):
        raw = self.root / "sales.csv"
        raw.touch()
        quarterly = runner.Analysis("Quarterly Compustat", ())
        appendix = runner.Analysis("Appendix", ())
        sections = dict(applications=[quarterly], empirical=[appendix])
        with patch.object(runner, "wrds_path", return_value=raw):
            self.assertEqual(list(runner.requested_analyses(sections, "empirical")),
                             [("prerequisites", quarterly), ("empirical", appendix)])
            self.cache.mkdir(parents=True)
            for name in ("compustat_qtr_summary.pkl", "compustat_qtr_draws_subsampling.pkl"):
                (self.cache / name).touch()
            self.assertEqual(list(runner.requested_analyses(sections, "empirical")),
                             [("prerequisites", quarterly), ("empirical", appendix)])


if __name__ == "__main__":
    unittest.main()
