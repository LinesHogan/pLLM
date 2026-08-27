#!/usr/bin/env python3
"""The native compiler facade must not load the legacy stack on import.

`import tllm.workload` / `import tllm.schedule` are compiler-only entry
points.  They must not pull the legacy Program/workgraph frontend or the
runtime into the process; the compat bridge and the physical lowering
helpers resolve lazily (PEP 562) and are gone at legacy retirement.
"""

from __future__ import annotations

import subprocess
import sys
import unittest


def _run(code: str) -> str:
    # Checked in a subprocess: unloading modules mid-suite would poison
    # later tests (vLLM registers process-global torch opaque types).
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip().splitlines()[-1]


class NativeImportBoundariesUnitTest(unittest.TestCase):
    def test_importing_workload_facade_loads_no_legacy_or_runtime(self) -> None:
        code = (
            "import sys; "
            "import tllm.workload; "
            "legacy = [m for m in sys.modules "
            "if m == 'tllm.program' or m.startswith('tllm.program.') "
            "or m == 'tllm.workgraph' or m.startswith('tllm.workgraph.') "
            "or m == 'tllm.runtime' or m.startswith('tllm.runtime.')]; "
            "print('OK' if not legacy else 'FAIL ' + ','.join(sorted(legacy)))"
        )
        self.assertEqual(_run(code), "OK")

    def test_importing_schedule_package_loads_no_legacy(self) -> None:
        code = (
            "import sys; "
            "import tllm.schedule; "
            "legacy = [m for m in sys.modules "
            "if m == 'tllm.program' or m.startswith('tllm.program.') "
            "or m == 'tllm.workgraph' or m.startswith('tllm.workgraph.')]; "
            "print('OK' if not legacy else 'FAIL ' + ','.join(sorted(legacy)))"
        )
        self.assertEqual(_run(code), "OK")

    def test_lazy_facade_names_still_resolve(self) -> None:
        code = (
            "from tllm.workload import lower_program, dependence_effect_fingerprint; "
            "from tllm.schedule import compile_schedule, lower_physical_flow_plan; "
            "from tllm import TllmRuntime, flow, BaseConsumer; "
            "print('OK')"
        )
        self.assertEqual(_run(code), "OK")


if __name__ == "__main__":
    unittest.main()
