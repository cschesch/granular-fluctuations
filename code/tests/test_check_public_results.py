"""Reject changed or unapproved raw inputs before public-data estimation."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from check_public_results import PUBLIC_INPUTS, copy_inputs


class PublicInputChecks(unittest.TestCase):
    def test_reference_inputs_are_verified_and_copied_exclusively(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source", Path(directory) / "copy"
            expected = {}
            for name in PUBLIC_INPUTS:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                expected[name] = hashlib.sha256(name.encode()).hexdigest()
            (source / "code").mkdir()
            reference = source / "code/reference_results.json"
            reference.write_text(json.dumps({"checks": [{"inputs": expected}]}))
            (source / "data/private.csv").write_text("must not be copied")
            self.assertEqual(copy_inputs(source, destination), expected)
            actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
            self.assertEqual(actual, PUBLIC_INPUTS)
            (source / next(iter(PUBLIC_INPUTS))).write_text("changed input")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                copy_inputs(source, destination)
            expected["data/private.csv"] = hashlib.sha256(b"must not be copied").hexdigest()
            reference.write_text(json.dumps({"checks": [{"inputs": expected}]}))
            with self.assertRaisesRegex(ValueError, "five allowlisted inputs"):
                copy_inputs(source, destination)
