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

    def test_coordinator_compile_loads_no_legacy(self) -> None:
        # R2-5 acceptance: the coordinator and one full compile (including the
        # mechanical physical lowering) must not pull the legacy
        # Program/workgraph stack into the process.  The physical descriptor
        # vocabulary now lives in tllm.contracts.physical_plan.
        code = """
import sys
from tllm.schedule import (
    CompileOptions,
    MainWorkloadTimeline,
    TimelineAnchor,
    WorkloadCompilerCoordinator,
)
from tllm.workload import Between, declare_workload, provided, sink, task, workload


@workload(id="boundary.probe")
class Probe:
    @task(
        inputs={"value": provided("value")},
        outputs={"result_value": sink("result")},
        timing=Between("model.execute", "step.end"),
    )
    def run(self, value: object) -> object:
        return value


timeline = MainWorkloadTimeline(
    backend_id="vllm",
    backend_version="0.27.1",
    runner_family="mrv2",
    graph_mode="eager",
    anchors=(TimelineAnchor("model.execute", 0), TimelineAnchor("step.end", 1)),
    regions=(),
    static_headroom_bytes=64,
)
coordinator = WorkloadCompilerCoordinator(
    (declare_workload(Probe()),),
    options=CompileOptions(memory_budget_bytes=1 << 20),
)
compilation = coordinator.compile(timeline)
legacy = [
    module
    for module in sys.modules
    if module == "tllm.program" or module.startswith("tllm.program.")
    or module == "tllm.workgraph" or module.startswith("tllm.workgraph.")
]
print("OK" if compilation.decision.signature and not legacy else "FAIL " + ",".join(sorted(legacy)))
"""
        self.assertEqual(_run(code), "OK")


if __name__ == "__main__":
    unittest.main()
