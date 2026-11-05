"""Full-content cache invalidation and explicit selection of licensed inputs."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mgp_core
import wrds_inputs


class CacheFingerprintChecks(unittest.TestCase):
    def test_changed_tail_and_middle_invalidate_same_size_input(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            for size, offset in ((3 * 2**19, 5 * 2**18), (4 * 2**20, 2 * 2**20)):
                with self.subTest(size=size):
                    path.write_bytes(b"a" * size)
                    before = mgp_core.clean_panel_cache_path("panel", {"version": 1}, [path])
                    stat = path.stat()
                    with path.open("r+b") as stream:
                        stream.seek(offset)
                        stream.write(b"b")
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                    after = mgp_core.clean_panel_cache_path("panel", {"version": 1}, [path])
                    self.assertNotEqual(before, after)
                    self.assertEqual(mgp_core._content_fingerprint(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_cleaning_version_invalidates_existing_cache(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "input.csv"
            source.write_text("x\n1\n")
            with patch.object(mgp_core, "PANEL_CACHE_DIR", directory):
                old = mgp_core.clean_panel_cache_path("panel", {"version": 1}, [source])
                mgp_core.write_clean_panel_cache(old, {"cleaned": "old rule"})
                new = mgp_core.clean_panel_cache_path("panel", {"version": 2}, [source])
                self.assertNotEqual(old, new)
                self.assertIsNone(mgp_core.read_clean_panel_cache(new))
                self.assertTrue(old.exists())


class InputSelectionChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directory = self.root / "data/compustat"
        self.directory.mkdir(parents=True)
        self.paper = self.directory / "compustat_annual_wrds_2025-01-01.csv.gz"
        self.newer = self.directory / "compustat_annual_wrds_2026-01-01.csv.gz"
        self.paper.write_bytes(b"paper input")
        self.newer.write_bytes(b"new input")
        pin = {"file": self.paper.name, "sha256": hashlib.sha256(self.paper.read_bytes()).hexdigest(),
               "extracted_utc": "2025-01-01", "analysis_cutoff": "2024-12-31"}
        (self.directory / "paper_inputs.json").write_text(json.dumps({"datasets": {"compustat_annual": pin}}))
        self.root_patch = patch.object(wrds_inputs, "ROOT", self.root)
        self.env_patch = patch.dict(os.environ, {"WRDS_INPUTS": "paper", "REPLICATION_INPUT_LOG": "",
                                                "REPLICATION_VERIFY_INPUTS": ""})
        self.root_patch.start()
        self.env_patch.start()
        wrds_inputs._SELECTED.clear()
        wrds_inputs._VERIFIED.clear()

    def tearDown(self):
        self.root_patch.stop()
        self.env_patch.stop()
        wrds_inputs._SELECTED.clear()
        wrds_inputs._VERIFIED.clear()
        self.temporary.cleanup()

    def test_paper_default_does_not_substitute_newest_or_inventory(self):
        inventory = {"datasets": [{"dataset": "compustat_annual", "file": str(self.newer)}]}
        (self.root / "data/wrds_inventory.json").write_text(json.dumps(inventory))
        self.assertEqual(wrds_inputs.wrds_path("compustat_annual"), self.paper)
        self.paper.unlink()
        self.assertEqual(wrds_inputs.wrds_path("compustat_annual"), self.paper)

    def test_latest_is_explicit_and_can_change_within_process(self):
        self.assertEqual(wrds_inputs.wrds_path("compustat_annual"), self.paper)
        with patch.dict(os.environ, {"WRDS_INPUTS": "latest"}):
            self.assertEqual(wrds_inputs.wrds_path("compustat_annual"), self.newer)
        records = wrds_inputs.selected_inputs()
        self.assertEqual([row["mode"] for row in records], ["paper", "latest"])

    def test_verification_rejects_changed_paper_and_records_new_input(self):
        wrds_inputs.wrds_path("compustat_annual")
        wrds_inputs.verify_inputs([self.paper])
        self.assertTrue(wrds_inputs.selected_inputs()[0]["matches_paper"])
        self.paper.write_bytes(b"changed input")
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            wrds_inputs.verify_inputs([self.paper])
        wrds_inputs.wrds_path("compustat_annual", mode="latest")
        wrds_inputs.verify_inputs([self.newer])
        self.assertFalse(wrds_inputs.selected_inputs()[-1]["matches_paper"])

    def test_subprocess_log_records_path_and_verified_hash(self):
        log = self.root / "run/inputs.jsonl"
        with patch.dict(os.environ, {"REPLICATION_INPUT_LOG": str(log)}):
            wrds_inputs.wrds_path("compustat_annual")
            wrds_inputs.verify_inputs([self.paper])
        records = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(records[-1]["type"], "input")
        self.assertEqual(records[-1]["sha256"], hashlib.sha256(self.paper.read_bytes()).hexdigest())
        self.assertEqual(records[-1]["file"], str(self.paper.relative_to(self.root)))

    def test_data_free_checkout_returns_missing_filename(self):
        (self.directory / "paper_inputs.json").unlink()
        selected = wrds_inputs.wrds_path("compustat_annual")
        self.assertEqual(selected, self.directory / "compustat_annual.csv.gz")
        self.assertFalse(selected.exists())

    def test_optional_selection_verification_rejects_changed_file(self):
        self.paper.write_bytes(b"changed input")
        with patch.dict(os.environ, {"REPLICATION_VERIFY_INPUTS": "1"}):
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                wrds_inputs.wrds_path("compustat_annual")

    def test_verify_all_selected_inputs(self):
        wrds_inputs.wrds_path("compustat_annual")
        wrds_inputs.verify_inputs()
        self.assertTrue(wrds_inputs.selected_inputs()[0]["matches_paper"])


if __name__ == "__main__":
    unittest.main()
