"""Check preparation fingerprints and checkpoint isolation without Stata or MATLAB."""

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
import pandas as pd
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
        self.scripts = self.root / "code"
        self.scripts.mkdir()
        for name in ("export_atalay_elasticity_profile.m", "export_atalay_elasticity_surface.m", "fit_surface.m"):
            (self.scripts / name).write_bytes((prepare.HERE / name).read_bytes())
        with ZipFile(self.archive, "w") as archive:
            archive.writestr("io_read.do", "ssc install hprescott\n")
        self.digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        for key, value in (("ARCHIVE", self.archive), ("ARCHIVE_SHA256", self.digest),
                           ("DERIVED", self.derived), ("WORK", self.work), ("HERE", self.scripts)):
            patcher = patch.object(prepare, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        (self.work / "io_read.do").write_bytes(prepare.adapted_stata_source())
        for name in prepare.STATA_FILES:
            (self.work / name).write_text("existing prepared input\n")
        for name in prepare.FILTER_CODE:
            (self.work / name).write_text("original computation\n")
        (self.derived / "source.json").write_text(json.dumps(
            {"original_archive_sha256": self.digest, "fingerprint": "earlier-code-hash"}))
        self.grid = self.derived / "elasticity_surface/valuable_grid.mat"
        self.grid.parent.mkdir()
        self.grid.write_bytes(b"preserve expensive output")

    def mark(self, stage, key="", complete=True):
        prepare.write_stage(stage, key, prepare.stage_inputs(stage, key), complete)

    def write_surface(self, tag=667, *, complete=True, failed=False):
        m = np.round(np.r_[.005, .01, .02, .03, .04 + .03 * np.arange(35)], 6)
        m[np.isclose(m, 1)] = .999
        q = np.sort(np.r_[.03 * np.arange(1, 51), .999])
        ok = np.ones((len(m), len(q)), dtype=bool)
        if failed:
            ok[-1, -1] = False
        path = self.grid.parent / f"atalay_surface_epsD_{tag}.mat"
        savemat(path, dict(eps_D={667: 2/3, 999: .999, 1333: 4/3}[tag],
                          eps_M_grid=m, eps_Q_grid=q, rows_done=len(m) if complete else 1, ok=ok))
        return path, m, q

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
        self.write_surface(failed=True)
        self.mark("surface", 667)
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
        self.mark("profile", "eps_M")
        self.assertFalse(prepare.profile_ready("eps_M"))
        with self.assertRaisesRegex(RuntimeError, "filter failed"):
            build_profile(directory.parent, 10, "CLARABEL")
        (directory / "one.mat").touch()
        index.write_text("sweep,value,status,file\neps_M,0.1,ok,one.mat\n")
        self.assertFalse(prepare.profile_ready("eps_M"))

    def stage_records(self):
        return {(stage, key): prepare.stage_inputs(stage, key) for stage, key in (
            ("benchmark", ""), ("profile", "eps_M"), ("profile", "eps_Q"),
            ("surface", 667), ("surface", 999), ("fits", 667), ("fits", 999))}

    def test_exporters_and_fit_code_invalidate_only_dependent_stages(self):
        self.write_surface(667)
        self.write_surface(999)
        dependencies = {
            "export_atalay_elasticity_profile.m": {("benchmark", ""), ("profile", "eps_M"), ("profile", "eps_Q")},
            "export_atalay_elasticity_surface.m": {("surface", 667), ("surface", 999), ("fits", 667), ("fits", 999)},
            "fit_surface.m": {("fits", 667), ("fits", 999)},
        }
        for name, expected in dependencies.items():
            with self.subTest(script=name):
                before = self.stage_records()
                path = self.scripts / name
                path.write_text(path.read_text() + "\nchanged_computation = 1;\n")
                after = self.stage_records()
                self.assertEqual({key for key in before if before[key] != after[key]}, expected)

    def test_plot_and_readme_changes_do_not_invalidate_preparation(self):
        self.write_surface(667)
        self.write_surface(999)
        before = self.stage_records()
        (self.scripts / "run_atalay_elasticity_surface.py").write_text("new plotting code\n")
        (self.scripts / "README.md").write_text("new documentation\n")
        self.assertEqual(before, self.stage_records())

    def test_source_changes_invalidate_stages_and_tag_settings_are_isolated(self):
        self.write_surface(667)
        self.write_surface(999)
        for path in (self.work / "filter_full.m", self.work / prepare.STATA_FILES[0], self.archive):
            before = self.stage_records()
            path.write_bytes(path.read_bytes() + b"changed source\n")
            after = self.stage_records()
            self.assertTrue(all(before[key] != after[key] for key in before))
        before = self.stage_records()
        with patch.object(prepare, "SURFACE_VALUES", ((667, "0.667"), (999, "0.999"), (1333, "4/3"))):
            after = self.stage_records()
        self.assertEqual({key for key in before if before[key] != after[key]}, {("surface", 667), ("fits", 667)})

    def test_matching_partial_surface_resumes_and_completion_is_validated(self):
        path, _, _ = self.write_surface(complete=False)
        self.mark("surface", 667, complete=False)
        original = path.read_bytes()
        inputs = prepare.begin_stage("surface", 667)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(prepare.surface_ready(667))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            prepare.finish_stage("surface", 667, inputs)
        self.assertFalse(prepare.read_stage("surface", 667)["complete"])
        self.write_surface()
        prepare.finish_stage("surface", 667, inputs)
        self.assertTrue(prepare.surface_ready(667))

    def test_missing_or_changed_fingerprint_preserves_and_excludes_old_checkpoints(self):
        for recorded in (False, True):
            with self.subTest(recorded=recorded):
                path, _, _ = self.write_surface(complete=False)
                partial = path.with_name(path.name + ".partial.mat")
                partial.write_bytes(b"unfinished checkpoint")
                if recorded:
                    self.mark("surface", 667, complete=False)
                else:
                    prepare.stage_marker("surface", 667).unlink(missing_ok=True)
                exporter = self.scripts / "export_atalay_elasticity_surface.m"
                exporter.write_text(exporter.read_text() + "\nchanged = 1;\n")
                old = path.read_bytes()
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare.begin_stage("surface", 667)
                self.assertFalse(path.exists())
                self.assertFalse(partial.exists())
                preserved = list((self.derived / "superseded").glob(f"*/elasticity_surface/{path.name}"))
                self.assertTrue(any(item.read_bytes() == old for item in preserved))
                self.assertTrue(self.grid.is_file())

    def test_surface_change_invalidates_fits_and_stale_fit_rows_cannot_resume(self):
        path, m, q = self.write_surface()
        self.mark("surface", 667)
        fit = self.grid.parent / "one_factor_667.csv"
        mm, qq = np.meshgrid(m, q, indexing="ij")
        table = pd.DataFrame(dict(eps_D=2/3, eps_M=mm.ravel(), eps_Q=qq.ravel(),
                                  one_factor=.5, fit_limit=0))
        table.to_csv(fit, index=False)
        self.mark("fits", 667)
        self.assertTrue(prepare.fits_ready(667))
        original = fit.read_bytes()
        prepare.begin_stage("fits", 667)
        self.assertEqual(fit.read_bytes(), original)
        path.write_bytes(path.read_bytes() + b"changed surface bytes")
        self.assertFalse(prepare.fits_ready(667))
        with contextlib.redirect_stdout(io.StringIO()):
            prepare.begin_stage("fits", 667)
        self.assertFalse(fit.exists())
        self.assertTrue(path.is_file())
        preserved = list((self.derived / "superseded").glob("*/elasticity_surface/one_factor_667.csv"))
        self.assertEqual(preserved[0].read_bytes(), original)

    def test_completed_legacy_surface_is_not_implicitly_adopted(self):
        path, _, _ = self.write_surface()
        original = path.read_bytes()
        self.assertFalse(prepare.surface_ready(667))
        self.assertFalse(prepare.stage_marker("surface", 667).exists())
        self.assertEqual(path.read_bytes(), original)

    def test_invalid_marker_is_not_a_current_stage(self):
        marker = prepare.stage_marker("benchmark")
        marker.parent.mkdir(parents=True)
        for text in ("invalid JSON", "[]", "null", "0"):
            marker.write_text(text)
            self.assertEqual(prepare.read_stage("benchmark"), {})
            self.assertFalse(prepare.benchmark_ready())

    def test_python_preparation_version_invalidates_only_its_stage(self):
        self.write_surface(667)
        self.write_surface(999)
        before = self.stage_records()
        with patch.dict(prepare.PREPARATION_VERSIONS, {"benchmark": 2}):
            after = self.stage_records()
        self.assertEqual({key for key in before if before[key] != after[key]}, {("benchmark", "")})

    def test_benchmark_settings_and_structure_are_checked(self):
        frame = pd.DataFrame(np.ones((2, 30)), columns=[f"shock_{i:02d}" for i in range(1, 31)])
        frame.insert(0, "year", [1999, 2000])
        frame.to_csv(self.derived / "atalay_benchmark_productivity_shocks.csv", index=False)
        np.savetxt(self.derived / "atalay_benchmark_network_variance_matrix.csv", np.eye(30), delimiter=",")
        self.mark("benchmark")
        self.assertTrue(prepare.benchmark_ready())
        with patch.object(prepare, "BENCHMARK_EPS_M", .2):
            self.assertFalse(prepare.benchmark_ready())
        np.savetxt(self.derived / "atalay_benchmark_network_variance_matrix.csv", np.eye(29), delimiter=",")
        self.assertFalse(prepare.benchmark_ready())

    def test_preparation_records_inputs_before_work_and_finishes_after_checks(self):
        surface, m, q = self.write_surface()
        fit = self.grid.parent / "one_factor_667.csv"
        fit.write_text("unverified legacy fit\n")

        def run(program, statements):
            if "export_atalay_elasticity_surface.m" in statements:
                self.assertFalse(surface.exists())
                self.assertFalse(prepare.read_stage("surface", 667)["complete"])
                self.write_surface()
            else:
                self.assertFalse(fit.exists())
                self.assertTrue(prepare.surface_ready(667))
                self.assertFalse(prepare.read_stage("fits", 667)["complete"])
                mm, qq = np.meshgrid(m, q, indexing="ij")
                pd.DataFrame(dict(eps_D=2/3, eps_M=mm.ravel(), eps_Q=qq.ravel(),
                                  one_factor=.5, fit_limit=0)).to_csv(fit, index=False)

        with patch.object(prepare, "run_matlab", side_effect=run) as runner, \
                contextlib.redirect_stdout(io.StringIO()):
            prepare.prepare_surface("matlab", 667, "2/3")
            self.assertTrue(prepare.fits_ready(667))
            self.assertEqual(runner.call_count, 2)
            prepare.prepare_surface("matlab", 667, "2/3")
            self.assertEqual(runner.call_count, 2)


if __name__ == "__main__":
    unittest.main()
