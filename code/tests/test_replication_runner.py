"""Partial, strict and failed runs must remain distinguishable from completed runs."""

import contextlib
import hashlib
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
        for name, value in (("ROOT", self.root), ("CODE_DIR", self.code), ("OUT", self.output), ("CACHE", self.cache)):
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

    def test_report_is_current_during_commands_and_history_survives_next_run(self):
        self.output.mkdir()
        self.report.write_text('{"status": "complete"}')
        snapshots = []

        def inspect_report(command, log):
            current = json.loads(self.report.read_text())
            snapshots.append(current)
            self.assertEqual(current["status"], "running")
            history = log.parent / "run_report.json"
            self.assertEqual(current, json.loads(history.read_text()))
            return dict(returncode=0, skipped=[], log=str(log))

        analyses = [runner.Analysis(name, ((name,),)) for name in ("first", "second")]
        with patch.object(runner, "run", side_effect=inspect_report):
            _, complete = self.execute(analyses)
        self.assertEqual([r["status"] for r in snapshots[0]["analyses"]], ["running", "not_run"])
        self.assertEqual([r["status"] for r in snapshots[1]["analyses"]], ["completed", "running"])
        history = next((self.cache / "logs").glob("*/run_report.json"))
        self.assertEqual(complete, json.loads(history.read_text()))
        self.execute([runner.Analysis("missing", (), (self.root / "missing",))])
        self.assertEqual(complete, json.loads(history.read_text()))
        self.assertEqual(len(list((self.cache / "logs").glob("*/run_report.json"))), 2)

    def test_interruption_leaves_completed_and_pending_analyses_distinct(self):
        good = dict(returncode=0, skipped=[])
        analyses = [runner.Analysis(name, ((name,),)) for name in ("first", "stopped", "later")]
        with patch.object(runner, "run", side_effect=[good, KeyboardInterrupt("stopped")]):
            with self.assertRaises(KeyboardInterrupt):
                self.execute(analyses)
        report = json.loads(self.report.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual([r["status"] for r in report["analyses"]], ["completed", "failed", "not_run"])

    def test_report_replacement_failure_preserves_previous_json(self):
        self.output.mkdir()
        self.report.write_text('{"status": "running"}\n')
        with patch.object(runner.os, "replace", side_effect=OSError("replacement failed")):
            with self.assertRaises(OSError):
                runner.write_report({"status": "complete"}, self.report)
        self.assertEqual(json.loads(self.report.read_text()), {"status": "running"})
        self.assertEqual(list(self.output.iterdir()), [self.report])

    def test_code_revision_uses_git_or_release_manifest(self):
        (self.root / ".git").mkdir()
        with patch.object(runner.subprocess, "run") as git:
            git.return_value.stdout = "abc123\n"
            self.assertEqual(runner.code_revision(), {"git_revision": "abc123"})
        (self.root / ".git").rmdir()
        contents = b"release checksum list\n"
        (self.root / "SHA256SUMS").write_bytes(contents)
        self.assertEqual(runner.code_revision(),
                         {"release_manifest_sha256": hashlib.sha256(contents).hexdigest()})
        (self.root / ".git").mkdir()
        for error in (FileNotFoundError("git"), runner.subprocess.CalledProcessError(128, "git")):
            with patch.object(runner.subprocess, "run", side_effect=error):
                self.assertEqual(runner.code_revision(),
                                 {"release_manifest_sha256": hashlib.sha256(contents).hexdigest()})

    def test_stella_missing_software_skips_but_locator_errors_fail(self):
        analysis = runner.Analysis("Stella", ())
        with patch.object(runner, "find_program", return_value=None), \
                patch("applications.foerster.prepare.prepared_ready", return_value=True), \
                patch("network_solver.find_gensys", return_value=self.root), \
                patch("applications.stella.prepare_stella_sales.locate_x13", return_value=self.root):
            code, report = self.execute([analysis])
            self.assertEqual((code, report["status"]), (0, "partial"))
            self.assertIn("Octave", report["analyses"][0]["reasons"][0])
            self.assertEqual(self.execute([analysis], strict=True)[0], 2)
        with patch("applications.foerster.prepare.prepared_ready", return_value=True), \
                patch("network_solver.find_gensys", side_effect=ValueError("invalid metadata")):
            with self.assertRaisesRegex(ValueError, "invalid metadata"):
                self.execute([analysis])
        self.assertEqual(json.loads(self.report.read_text())["status"], "failed")

    def test_unvalidated_foerster_cache_blocks_both_consumers(self):
        for name in ("Foerster", "Stella", "Stella dense network"):
            with self.subTest(name=name), \
                    patch("applications.foerster.prepare.prepared_ready", return_value=False), \
                    patch.object(runner, "run") as run:
                code, report = self.execute([runner.Analysis(name, (("must_not_run",),))])
                self.assertEqual((code, report["status"]), (0, "partial"))
                self.assertIn("preparation record", report["analyses"][0]["reasons"][0])
                run.assert_not_called()

    def test_stella_dense_reuses_prepared_sales_but_requires_dynamic_software(self):
        analysis = runner.Analysis("Stella dense network", ())
        with patch("applications.foerster.prepare.prepared_ready", return_value=True), \
                patch("network_solver.find_gensys", return_value=self.root), \
                patch("applications.stella.prepare_stella_sales.locate_x13",
                      side_effect=AssertionError("Prepared sales must not require X-12")):
            with patch.object(runner, "find_program", return_value="octave"):
                self.assertEqual(runner.missing_requirements(analysis), [])
            with patch.object(runner, "find_program", return_value=None):
                self.assertIn("Octave", runner.missing_requirements(analysis)[0])

    def test_unvalidated_atalay_cache_blocks_estimation(self):
        from applications.atalay import prepare

        for name, check in (("Atalay", "benchmark_ready"),
                            ("Atalay elasticity profile", "profile_ready"),
                            ("Atalay calibration comparison", "fits_ready")):
            with self.subTest(name=name), patch.object(prepare, check, return_value=False), \
                    patch.object(runner, "run") as run:
                code, report = self.execute([runner.Analysis(name, (("must_not_run",),))])
                self.assertEqual((code, report["status"]), (0, "partial"))
                run.assert_not_called()

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
