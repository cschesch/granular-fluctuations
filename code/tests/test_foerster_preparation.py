"""Foerster caches require matching inputs, calculation code, and output hashes."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from applications.foerster import prepare as foerster


class FoersterPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.code = self.root / "code"
        self.cache = self.root / "cache"
        self.gensys = self.root / "gensys"
        self.gensys.mkdir()
        for name in foerster.GENSYS_FILES:
            (self.gensys / name).write_text(f"function {Path(name).stem}; end\n")
        for name in foerster.SOURCE_INPUTS:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.script = self.code / "applications/foerster/prepare.py"
        self.script.parent.mkdir(parents=True)
        self.script.write_text('"""Preparation."""\nvalue = 1\n')
        (self.code / "network_solver.py").write_text("value = 2\n")
        (self.code / "network_dynamic_solver.m").write_text("value = 3;\n")
        for name, value in (("CODE_DIR", self.code), ("CACHE", self.cache),
                            ("MODEL_INPUTS", self.cache / "model_inputs"),
                            ("EXPORTS", self.cache / "exports"), ("__file__", str(self.script))):
            self.patch(foerster, name, value)
        self.patch(foerster, "upstream", return_value=self.source)
        self.gensys_locator = patch("network_solver.find_gensys", return_value=self.gensys)
        self.find_gensys = self.gensys_locator.start()
        self.addCleanup(self.gensys_locator.stop)

    def patch(self, target, name, *args, **kwargs):
        patcher = patch.object(target, name, *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def populate_cache(self):
        for directory, names in ((foerster.MODEL_INPUTS, foerster.MODEL_FILES),
                                 (foerster.EXPORTS, foerster.OUTPUTS)):
            directory.mkdir(parents=True, exist_ok=True)
            for name in names:
                (directory / name).write_text(name + "\n")
        record = foerster.input_fingerprint(self.source)
        record["gensys"] = foerster.gensys_fingerprint(self.gensys)
        record["outputs"] = foerster.output_fingerprint(foerster.MODEL_INPUTS, foerster.EXPORTS)
        (self.cache / "preparation.json").write_text(json.dumps(record))

    def cache_bytes(self):
        return {str(path.relative_to(self.cache)): path.read_bytes()
                for path in self.cache.rglob("*") if path.is_file()}

    def run_prepare(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = foerster.prepare()
        return result, output.getvalue()

    def test_valid_cache_reused_without_octave_or_gensys(self):
        self.populate_cache()
        before = self.cache_bytes()
        self.find_gensys.side_effect = FileNotFoundError("not installed")
        self.patch(foerster, "find_program", side_effect=AssertionError("runtime lookup before reuse"))
        result, message = self.run_prepare()
        self.assertTrue(result)
        self.assertIn("reused", message)
        self.assertEqual(self.cache_bytes(), before)

    def test_changed_inputs_or_code_skip_without_touching_cache(self):
        self.patch(foerster, "find_program", return_value=None)
        for path in (self.source / "Data/IP98.xls", self.source / "Data/IO98.xls",
                     self.source / "Data/CapFlow98.xls", self.script,
                     self.code / "network_solver.py", self.code / "network_dynamic_solver.m"):
            with self.subTest(path=path):
                self.populate_cache()
                before = self.cache_bytes()
                original = path.read_text()
                path.write_text(original + ("\nchanged = 1\n" if path.suffix == ".py" else "changed\n"))
                try:
                    self.assertFalse(foerster.prepared_ready())
                    result, message = self.run_prepare()
                    self.assertFalse(result)
                    self.assertIn("[skip]", message)
                    self.assertEqual(self.cache_bytes(), before)
                finally:
                    path.write_text(original)

    def test_missing_or_corrupted_outputs_are_not_reused(self):
        for directory, names in ((foerster.MODEL_INPUTS, foerster.MODEL_FILES),
                                 (foerster.EXPORTS, foerster.OUTPUTS)):
            for name in names:
                for corruption in ("missing", "changed"):
                    with self.subTest(file=name, corruption=corruption):
                        self.populate_cache()
                        path = directory / name
                        if corruption == "missing":
                            path.unlink()
                        else:
                            path.write_text("corrupted\n")
                        self.assertFalse(foerster.prepared_ready())

    def test_legacy_or_invalid_stamp_is_not_upgraded(self):
        self.populate_cache()
        stamp = self.cache / "preparation.json"
        record = json.loads(stamp.read_text())
        del record["outputs"]
        self.patch(foerster, "find_program", return_value=None)
        for content in (json.dumps(record), "{invalid", "[]"):
            with self.subTest(content=content):
                stamp.write_text(content)
                before = self.cache_bytes()
                self.assertFalse(self.run_prepare()[0])
                self.assertEqual(self.cache_bytes(), before)

    def test_changed_installed_gensys_or_helpers_invalidate_cache(self):
        for name in foerster.GENSYS_FILES:
            with self.subTest(file=name):
                self.populate_cache()
                self.assertTrue(foerster.prepared_ready())
                path = self.gensys / name
                original = path.read_text()
                path.write_text("changed\n")
                self.assertFalse(foerster.prepared_ready())
                path.write_text(original)

    def test_missing_solver_helper_allows_reuse_but_not_rebuild(self):
        self.populate_cache()
        (self.gensys / "qzdiv.m").unlink()
        self.assertTrue(foerster.prepared_ready())
        (self.source / "Data/IP98.xls").write_text("changed\n")
        before = self.cache_bytes()
        self.patch(foerster, "find_program", return_value="octave")
        result, message = self.run_prepare()
        self.assertFalse(result)
        self.assertIn("qzdiv.m", message)
        self.assertEqual(self.cache_bytes(), before)

    def test_python_comment_and_docstring_changes_preserve_fingerprint(self):
        self.populate_cache()
        self.script.write_text('# explanation\n"""Updated explanation."""\nvalue = 1\n')
        self.assertTrue(foerster.prepared_ready())

    def mock_calculation(self, solver):
        self.patch(foerster, "find_program", return_value="octave")
        self.patch(foerster, "structural_inputs", return_value=(np.eye(2),) * 3)
        levels = np.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.]])
        self.patch(foerster, "industrial_production", return_value=(
            np.arange(4), levels, np.full((4, 2), .5),
            pd.DataFrame({"epsilon_column": ["eps_001", "eps_002"]})))
        patcher = patch("network_solver.run_solver", side_effect=solver)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def test_new_preparation_records_all_outputs_and_can_be_reused(self):
        def solver(inputs, exports, octave, gensys):
            for name, value in (("pii_a", 1.), ("xi", .1), ("cap_rho", .2)):
                np.savetxt(exports / f"{name}.csv", np.eye(2) * value, delimiter=",")

        mocked_solver = self.mock_calculation(solver)
        result, message = self.run_prepare()
        self.assertTrue(result)
        self.assertIn("prepared 4 quarters and 2 sectors", message)
        self.assertTrue(foerster.prepared_ready())
        record = json.loads((self.cache / "preparation.json").read_text())
        self.assertEqual(len(record["outputs"]), len(foerster.OUTPUTS) + len(foerster.MODEL_FILES))
        self.assertTrue(self.run_prepare()[0])
        mocked_solver.assert_called_once()

    def test_failed_rebuild_preserves_existing_cache(self):
        self.populate_cache()
        (self.source / "Data/IP98.xls").write_text("changed\n")
        before = self.cache_bytes()

        def failing_solver(inputs, exports, octave, gensys):
            (exports / "pii_a.csv").write_text("incomplete\n")
            raise RuntimeError("solver failed")

        self.mock_calculation(failing_solver)
        with self.assertRaisesRegex(RuntimeError, "solver failed"):
            self.run_prepare()
        self.assertEqual(self.cache_bytes(), before)
        self.assertFalse(foerster.prepared_ready())

    def test_failed_promotion_cannot_leave_a_valid_stamp(self):
        self.populate_cache()
        (self.source / "Data/IP98.xls").write_text("changed\n")
        self.patch(foerster, "find_program", return_value="octave")

        def staged_outputs(source, model_inputs, exports, octave, gensys):
            for directory, names in ((model_inputs, foerster.MODEL_FILES),
                                     (exports, foerster.OUTPUTS)):
                for name in names:
                    (directory / name).write_text("new values\n")
            return 4, 2

        self.patch(foerster, "write_preparation", side_effect=staged_outputs)
        replace = Path.replace

        def interrupted_replace(path, destination):
            if destination.parent == foerster.EXPORTS:
                raise OSError("promotion interrupted")
            return replace(path, destination)

        with patch.object(Path, "replace", interrupted_replace):
            with self.assertRaisesRegex(OSError, "promotion interrupted"):
                self.run_prepare()
        self.assertFalse((self.cache / "preparation.json").exists())
        self.assertFalse(foerster.prepared_ready())


if __name__ == "__main__":
    unittest.main()
