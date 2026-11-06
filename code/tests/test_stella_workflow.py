"""Stella provenance and constructor inputs stay separate from generated results."""

import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from applications.stella import stella_provenance as provenance
from applications.stella import validate_stella_calibration_source as constructor


class StellaWorkflowTests(unittest.TestCase):
    def test_archive_preserves_source_locations_and_distinct_contents(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(provenance, "ROOT", root):
                for directory in ("code/.cache/customer", "code/.cache/dense"):
                    source = root / directory / "diagnostics.json"
                    source.parent.mkdir(parents=True)
                    source.write_text("old result")
                    archived = provenance.archive_existing(source)
                    self.assertEqual(archived.parent, root / "archive/stella_results" / directory)
                    self.assertEqual(archived.read_bytes(), source.read_bytes())
                    self.assertEqual(provenance.archive_existing(source), archived)
                    metadata = json.loads(archived.with_suffix(".json.provenance.json").read_text())
                    self.assertEqual(metadata["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
                    self.assertEqual(metadata["original_path"], str(source.resolve()))
                    source.write_text("new result")
                    self.assertNotEqual(provenance.archive_existing(source), archived)
                    self.assertEqual(archived.read_text(), "old result")
                    self.assertFalse((source.parent / "stella_result_archive").exists())

    def test_external_calibration_paths_are_recorded_and_archived(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            source = Path(temporary) / "external/diagnostics.json"
            source.parent.mkdir()
            source.write_text("external result")
            with patch.object(provenance, "ROOT", root):
                archived = provenance.archive_existing(source)
            self.assertTrue(archived.is_relative_to(root / "archive/stella_results/external"))
            self.assertEqual(archived.read_bytes(), source.read_bytes())
            expected = provenance.file_fingerprints([source], root)
            self.assertEqual(list(expected), [str(source.resolve())])
            provenance.verify_fingerprints(expected, root)
            source.write_text("changed result")
            with self.assertRaisesRegex(ValueError, "Stale Stella inputs"):
                provenance.verify_fingerprints(expected, root)

    def test_dense_constructor_needs_no_customer_links(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            calibration = Path(temporary) / "external_calibration"
            calibration.mkdir()
            (calibration / "diagnostics.json").write_text('{"network_calibration": "dense"}')
            pd.DataFrame({"gvkey": ["000001", "000002"], "stella_sid": ["1", "2"]}).to_csv(
                calibration / "firm_mapping.csv", index=False)
            pd.DataFrame({"gvkey": ["000001", "000002"], "first_time": [1, 1],
                          "last_time": [10, 10], "ratio_emp": [.1, .2]}).to_csv(
                calibration / "firm_employment_ratios.csv", index=False)
            source = root / "source/filtering"
            source.mkdir(parents=True)
            (source / "matrix_construction_Icalibration.m").write_text("% source fixture")
            for name in ("matrix_io", "alpha", "matrix_capital"):
                np.savetxt(calibration / f"{name}.csv", np.eye(2), delimiter=",")

            def original_constructor(*args, cwd, **kwargs):
                self.assertFalse((cwd / "comp_data.csv").exists())
                for name in ("matrix_io", "alpha", "matrix_capital"):
                    shutil.copyfile(calibration / f"{name}.csv", cwd / f"{name}.csv")

            with patch.object(constructor, "ROOT", root), \
                    patch.object(constructor, "OFFICIAL", source.parent), \
                    patch.object(constructor, "OFFICIAL_IO", calibration / "matrix_io.csv"), \
                    patch.object(constructor, "OFFICIAL_SURPLUS", calibration / "alpha.csv"), \
                    patch.object(constructor, "resolve_edges", side_effect=AssertionError("unused customer data")), \
                    patch.object(constructor.subprocess, "run", side_effect=original_constructor):
                result = constructor.validate(calibration, "octave", export_inputs=True)
            self.assertTrue(result["all_pass"])
            self.assertEqual(result["calibration_directory"], str(calibration.resolve()))
            self.assertEqual({p.name for p in (calibration / "source_constructor_inputs").iterdir()},
                             {"matrix_io_input.csv", "valadd.csv", "tid.csv"})


if __name__ == "__main__":
    unittest.main()
