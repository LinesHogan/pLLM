#!/usr/bin/env python3
"""Unit tests for lightweight runtime import boundaries."""

from __future__ import annotations

import subprocess
import sys
import unittest


class RuntimeImportBoundariesUnitTest(unittest.TestCase):
    def test_importing_port_contracts_does_not_import_vllm(self) -> None:
        # Checked in a subprocess: dropping "vllm" from this process'
        # sys.modules would force a re-import later, and vLLM 0.25+ registers
        # process-global torch opaque types at import time (re-registration
        # raises), which would poison the rest of the suite.
        code = (
            "import importlib, json, sys; "
            "importlib.import_module('tllm.ports.base'); "
            "print(json.dumps({'vllm': 'vllm' in sys.modules}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(completed.stdout.strip().splitlines()[-1], '{"vllm": false}')


if __name__ == "__main__":
    unittest.main()
