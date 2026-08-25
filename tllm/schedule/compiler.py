"""Mechanical Python bridge to the Rust schedule compiler."""

from __future__ import annotations

import platform
import sys
from dataclasses import replace
from typing import Any, cast

from tllm.schedule.contracts import (
    CompileOptions,
    MainWorkloadTimeline,
    ScheduleDecision,
    ScheduleGraph,
)


def compiler_environment_fingerprint() -> str:
    """Hash static compiler/device facts without consulting live memory state."""

    import torch

    device = "cpu"
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(index)
        device = f"cuda:{capability[0]}.{capability[1]}:{torch.cuda.get_device_name(index)}"
    payload = {
        "cuda": str(torch.version.cuda),
        "device": device,
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": str(torch.__version__),
    }
    from tllm.schedule.contracts import structural_key

    return structural_key(payload)


def effective_compile_options(options: CompileOptions) -> CompileOptions:
    if not isinstance(options, CompileOptions):
        raise TypeError("options must be CompileOptions")
    if options.compiler_fingerprint:
        return options
    return replace(
        options, compiler_fingerprint=compiler_environment_fingerprint()
    )


def compile_schedule(
    graph: ScheduleGraph,
    timeline: MainWorkloadTimeline,
    options: CompileOptions = CompileOptions(),
) -> ScheduleDecision:
    if not isinstance(graph, ScheduleGraph):
        raise TypeError("compile_schedule.graph must be ScheduleGraph")
    if not isinstance(timeline, MainWorkloadTimeline):
        raise TypeError("compile_schedule.timeline must be MainWorkloadTimeline")
    if not isinstance(options, CompileOptions):
        raise TypeError("compile_schedule.options must be CompileOptions")
    options = effective_compile_options(options)
    from tllm import _core

    payload = cast(Any, _core).compile_schedule(graph.to_json(), timeline.to_json(), options.to_json())
    return ScheduleDecision.from_json(payload)


def validate_schedule(
    graph: ScheduleGraph,
    timeline: MainWorkloadTimeline,
    decision: ScheduleDecision,
    options: CompileOptions = CompileOptions(),
) -> None:
    if not isinstance(decision, ScheduleDecision):
        raise TypeError("validate_schedule.decision must be ScheduleDecision")
    options = effective_compile_options(options)
    from tllm import _core

    cast(Any, _core).validate_schedule(
        graph.to_json(),
        timeline.to_json(),
        options.to_json(),
        decision.to_json(),
    )


__all__ = [
    "compile_schedule",
    "compiler_environment_fingerprint",
    "effective_compile_options",
    "validate_schedule",
]
