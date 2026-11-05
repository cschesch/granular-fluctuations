"""Check source-cache migration and stage isolation without Stata or MATLAB."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from scipy.io import savemat

from applications.atalay import prepare


class PreparationChecks(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.derived = self.root / "derived"
        self.work = self.derived / "replication_package"
        self.work.mkdir(parents=True)
        self.archive = self.root / "source.zip"
        with ZipFile(self.archive, "w") as archive:
            archive.writestr("io_read.do", "ssc install hprescott\n")
        self.digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        for key, value in (("ARCHIVE", self.archive), ("ARCHIVE_SHA256", self.digest),
                           ("DERIVED", self.derived), ("WORK", self.work)):
            patcher = patch.object(prepare, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        (self.work / "io_read.do").write_bytes(prepare.adapted_stata_source())
        for name in prepare.STATA_FILES:
            (self.work / name).write_text("existing prepared input\n")
        (self.derived / "source.json").write_text(json.dumps(
            {"original_archive_sha256": self.digest, "fingerprint": "earlier-code-hash"}))
        self.grid = self.derived / "elasticity_surface/valuable_grid.mat"
        self.grid.parent.mkdir()
        self.grid.write_bytes(b"preserve expensive output")

    def test_legacy_cache_adoption_preserves_grid_and_records_inputs(self):
        self.assertTrue(prepare.source_ready())
        record = json.loads((self.derived / "source.json").read_text())
        self.assertEqual(record["source_version"], prepare.SOURCE_VERSION)
        self.assertEqual(set(record["stata_files"]), set(prepare.STATA_FILES))
        self.assertNotIn("fingerprint", record)
        self.assertEqual(self.grid.read_bytes(), b"preserve expensive output")
        self.assertTrue(prepare.source_ready())
        (self.work / prepare.STATA_FILES[0]).write_text("changed input\n")
        with self.assertRaisesRegex(ValueError, "Stata inputs changed"):
            prepare.source_ready()

    def test_legacy_adoption_requires_exact_stata_adaptation(self):
        (self.work / "io_read.do").write_text("unreviewed modification\n")
        with self.assertRaisesRegex(ValueError, "source adaptation"):
            prepare.source_ready()

    def test_source_version_change_is_explicit_even_with_cached_outputs(self):
        self.assertTrue(prepare.source_ready())
        with patch.object(prepare, "SOURCE_VERSION", prepare.SOURCE_VERSION + 1):
            with self.assertRaisesRegex(ValueError, "version changed"):
                prepare.source_ready()
        self.assertTrue(self.grid.is_file())

    def test_failed_surface_cell_is_not_reported_as_complete(self):
        m = np.round(np.r_[.005, .01, .02, .03, .04 + .03 * np.arange(35)], 6)
        m[np.isclose(m, 1)] = .999
        q = np.sort(np.r_[.03 * np.arange(1, 51), .999])
        ok = np.ones((len(m), len(q)), dtype=bool)
        ok[-1, -1] = False
        path = self.grid.parent / "atalay_surface_epsD_667.mat"
        savemat(path, dict(eps_D=2/3, eps_M_grid=m, eps_Q_grid=q, rows_done=len(m), ok=ok))
        with self.assertRaisesRegex(RuntimeError, "failed cells"):
            prepare.surface_ready(667)

    def test_failed_stata_run_preserves_source_and_grid(self):
        with patch.object(prepare.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "stata")):
            with self.assertRaises(subprocess.CalledProcessError):
                prepare.prepare_stata("stata")
        self.assertEqual((self.work / prepare.STATA_FILES[0]).read_text(), "existing prepared input\n")
        self.assertTrue(self.grid.is_file())

    def test_successful_stata_run_replaces_only_source_working_copy(self):
        def create_outputs(*args, **kwargs):
            for name in prepare.STATA_FILES:
                (kwargs["cwd"] / name).write_text("new prepared input\n")
        with patch.object(prepare.subprocess, "run", side_effect=create_outputs):
            prepare.prepare_stata("stata")
        self.assertEqual((self.work / prepare.STATA_FILES[0]).read_text(), "new prepared input\n")
        self.assertEqual(self.grid.read_bytes(), b"preserve expensive output")
        self.assertTrue(prepare.source_ready())

    def test_completed_surface_needs_no_program_or_benchmark(self):
        with patch.object(sys, "argv", ["prepare", "--section", "surface"]), \
                patch.object(prepare, "fits_ready", return_value=True), \
                patch.object(prepare, "benchmark_ready") as benchmark, \
                patch.object(prepare, "profile_ready") as profile, \
                patch.object(prepare, "find_program") as find_program, \
                contextlib.redirect_stdout(io.StringIO()):
            prepare.main()
        benchmark.assert_not_called()
        profile.assert_not_called()
        find_program.assert_not_called()
        self.assertTrue(self.grid.is_file())

    def test_profile_failure_and_smoke_grid_cannot_count_as_complete(self):
        from applications.atalay.run_atalay_elasticity_profile import build_profile

        directory = self.derived / "elasticity_profile/eps_M"
        directory.mkdir(parents=True)
        index = directory / "grid_index.csv"
        index.write_text("sweep,value,status,file\neps_M,0.1,failed,\n")
        self.assertFalse(prepare.profile_ready("eps_M"))
        with self.assertRaisesRegex(RuntimeError, "filter failed"):
            build_profile(directory.parent, 10, "CLARABEL")
        (directory / "one.mat").touch()
        index.write_text("sweep,value,status,file\neps_M,0.1,ok,one.mat\n")
        self.assertFalse(prepare.profile_ready("eps_M"))


if __name__ == "__main__":
    unittest.main()
