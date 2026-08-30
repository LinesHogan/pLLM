#!/usr/bin/env python3
"""GPU oracle for eager versus signed native optimizer-batch replay."""

# pyright: strict

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.esamp.workload import ESampConfig, ESampWorkload
from tllm.contracts.hidden_locator import HiddenLocatorCapability
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime import residual_runtime
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_bindings import SamplerRuntimeBinding
from tllm.schedule import (
    AsyncBatchCaptureMode,
    HostCommandKind,
    MainWorkloadTimeline,
)


_ROWS = 3
_HIDDEN_SIZE = 4
_STEPS = 3
_SOURCE_PATH = "model.model.layers[0].input_layernorm"
_TARGET_PATH = "model.model.layers[-1].input_layernorm"


@dataclass(frozen=True, slots=True)
class _StepBatch:
    source: torch.Tensor
    target: torch.Tensor
    sample_indices: torch.Tensor
    sample_ids: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _RunEvidence:
    workload: ESampWorkload
    losses: tuple[float, ...]
    versions: tuple[int, ...]
    update_counts: tuple[int, ...]
    parameter_steps: tuple[tuple[tuple[str, torch.Tensor], ...], ...]
    batch_phases: tuple[tuple[str, int], ...]
    eager_count: int
    replay_count: int
    fallback_count: int


def _timeline(
    graph_mode: Literal["eager", "piecewise"],
) -> MainWorkloadTimeline:
    capabilities = BackendCapabilities(
        backend_id="gpu-native-optimizer-oracle",
        runner_family="mrv2",
        sampler_stages=frozenset(),
        native_min_p=False,
        tuple_sampler_return=False,
        split_execute_and_sample=True,
        graph_safety=GraphSafety.CAPTURE_SAFE,
        hidden_locators=frozenset(
            {HiddenLocatorCapability.BLOCK_OUTPUT_DECODE}
        ),
    )
    return export_main_workload_timeline(
        capabilities,
        graph_mode=graph_mode,
        static_headroom_bytes=64 << 20,
    )


def _make_workload(
    device: torch.device,
) -> tuple[ESampWorkload, TllmRuntime[ESampWorkload]]:
    workload = ESampWorkload(
        ESampConfig(
            enabled=True,
            intervention=False,
            distiller_hidden_dim=2,
            distiller_lr=1e-3,
            output_layernorm=False,
            training_row_selection="none",
        )
    )
    runtime = TllmRuntime(
        workloads=(workload,),
        workload_memory_budget_bytes=64 << 20,
    )
    runtime.cold_specialize_workloads(
        RuntimeSpecializationSpec(
            rows=_ROWS,
            hidden_size=_HIDDEN_SIZE,
            dtype=torch.float32,
            device=device,
        )
    )
    return workload, runtime


def _input_batches(device: torch.device) -> tuple[_StepBatch, ...]:
    base = torch.arange(
        _ROWS * _HIDDEN_SIZE,
        dtype=torch.float32,
        device=device,
    ).reshape(_ROWS, _HIDDEN_SIZE)
    return tuple(
        _StepBatch(
            source=(base + step).mul(0.125),
            target=torch.flip(base, dims=(1,)).sub(step).mul(0.0625),
            sample_indices=torch.tensor(
                [0, 1, step % 2],
                dtype=torch.long,
                device=device,
            ),
            sample_ids=(0, 1, step % 2),
        )
        for step in range(_STEPS)
    )


def _parameter_snapshot(
    workload: ESampWorkload,
) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple(
        (name, parameter.detach().clone())
        for name, parameter in workload.distiller.named_parameters()
    )


def _assert_parameters_close(
    eager: tuple[tuple[str, torch.Tensor], ...],
    captured: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    eager_names = tuple(name for name, _parameter in eager)
    captured_names = tuple(name for name, _parameter in captured)
    if eager_names != captured_names:
        raise AssertionError(
            "eager and captured models expose different parameter sets"
        )
    for (name, expected), (_captured_name, actual) in zip(
        eager,
        captured,
        strict=True,
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda message: f"parameter {name!r}: {message}",
        )


def _run(
    graph_mode: Literal["eager", "piecewise"],
    workload: ESampWorkload,
    runtime: TllmRuntime[ESampWorkload],
    batches: tuple[_StepBatch, ...],
) -> _RunEvidence:
    compilation = runtime.compile_workloads(_timeline(graph_mode))
    if compilation.physical is None:
        raise AssertionError("native workload compilation has no physical plan")
    tail_submit = next(
        command
        for advance in compilation.decision.runtime_advance_plan
        for command in advance.host_commands
        if command.kind is HostCommandKind.SUBMIT_ASYNC
    )
    expected_capture = (
        AsyncBatchCaptureMode.OPTIMIZER_TRANSITION
        if graph_mode == "piecewise"
        else AsyncBatchCaptureMode.DISABLED
    )
    if tail_submit.async_capture_mode is not expected_capture:
        raise AssertionError(
            "Rust signed the wrong optimizer batch capture mode: "
            f"expected={expected_capture.value!r}, "
            f"actual={tail_submit.async_capture_mode.value!r}"
        )

    source_locator = ResidualLocator(0, "block_output", "decode")
    target_locator = ResidualLocator(-1, "block_output", "decode")
    state = ResidualRuntimeState()
    state.residual_bindings = {
        _SOURCE_PATH: ResidualPathBinding(source_locator, _SOURCE_PATH),
        _TARGET_PATH: ResidualPathBinding(target_locator, _TARGET_PATH),
    }
    bindings = runtime.workload_execution_bindings
    if bindings is None:
        raise AssertionError("native ESamp runtime produced no execution bindings")
    sampler = workload.native_workload_resources().sampler
    match sampler:
        case SamplerRuntimeBinding(stage_identity=stage_identity):
            pass
        case _:
            raise AssertionError("ESamp did not expose its typed sampler input")

    losses: list[float] = []
    versions: list[int] = []
    updates: list[int] = []
    parameter_steps: list[tuple[tuple[str, torch.Tensor], ...]] = []
    execution = None
    prior_runtime = residual_runtime.RUNTIME
    residual_runtime.RUNTIME = state
    try:
        residual_runtime.publish_workload_compilation(
            compilation,
            bindings=bindings,
        )
        execution = state.workload_execution
        if execution is None:
            raise AssertionError("native ESamp execution was not installed")
        try:
            anchors = tuple(
                dict.fromkeys(
                    region.placement_anchor
                    for region in compilation.decision.execution_regions
                )
            )
            for ordinal, batch in enumerate(batches, start=1):
                request_ids = tuple(
                    f"{graph_mode}-request-{row}" for row in range(_ROWS)
                )
                state.decode_count = _ROWS
                state.decode_request_ids = list(request_ids)
                state.decode_prompt_idxs = list(range(_ROWS))
                state.decode_sample_idxs = list(batch.sample_ids)
                state.tap_decode_hidden = {
                    _SOURCE_PATH: batch.source.clone(),
                    _TARGET_PATH: batch.target.clone(),
                }
                stage_identity(
                    ordinal,
                    request_ids,
                    batch.sample_ids,
                    batch.sample_indices,
                )

                execution.prepare_step(ordinal)
                for anchor in anchors:
                    execution.advance_anchor(anchor)
                execution.finish_step(ordinal)
                execution.synchronize()

                diagnostics = execution.read_resource_diagnostics()["esamp"]
                if diagnostics["loss_count"] != 1:
                    raise AssertionError(
                        f"{graph_mode} step {ordinal} committed loss more than once"
                    )
                if diagnostics["update_count"] != ordinal:
                    raise AssertionError(
                        f"{graph_mode} step {ordinal} committed update more than once"
                    )
                if workload.state_version != ordinal:
                    raise AssertionError(
                        f"{graph_mode} step {ordinal} advanced version incorrectly"
                    )
                stats = execution.read_and_reset_stats(sync=True)
                if stats.loss_count != 1:
                    raise AssertionError(
                        f"{graph_mode} step {ordinal} reported duplicate loss stats"
                    )
                losses.append(stats.loss_avg)
                updates.append(int(diagnostics["update_count"]))
                versions.append(workload.state_version)
                parameter_steps.append(_parameter_snapshot(workload))

            report = execution.region_cost_report(block=True)
            batch_phases = tuple(
                sorted(
                    (reading.phase, reading.samples)
                    for reading in report.readings.values()
                    if reading.region_id == tail_submit.async_artifact_key
                )
            )
            stats = execution.executor.stats
            evidence = _RunEvidence(
                workload=workload,
                losses=tuple(losses),
                versions=tuple(versions),
                update_counts=tuple(updates),
                parameter_steps=tuple(parameter_steps),
                batch_phases=batch_phases,
                eager_count=stats.eager_count,
                replay_count=stats.replay_count,
                fallback_count=stats.fallback_count,
            )
        finally:
            execution.close()
    finally:
        residual_runtime.RUNTIME = prior_runtime
    return evidence


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    eager_workload, eager_runtime = _make_workload(device)
    captured_workload, captured_runtime = _make_workload(device)
    captured_workload.distiller.load_state_dict(
        eager_workload.distiller.state_dict(),
        strict=True,
    )
    _assert_parameters_close(
        _parameter_snapshot(eager_workload),
        _parameter_snapshot(captured_workload),
    )
    torch.testing.assert_close(
        captured_workload.optimizer.state_dict(),
        eager_workload.optimizer.state_dict(),
    )

    batches = _input_batches(device)
    eager = _run("eager", eager_workload, eager_runtime, batches)
    captured = _run(
        "piecewise",
        captured_workload,
        captured_runtime,
        batches,
    )

    torch.testing.assert_close(
        torch.tensor(captured.losses),
        torch.tensor(eager.losses),
        rtol=1e-5,
        atol=1e-6,
    )
    if eager.versions != (1, 2, 3) or captured.versions != (1, 2, 3):
        raise AssertionError("state version did not advance once per logical step")
    if (
        eager.update_counts != (1, 2, 3)
        or captured.update_counts != (1, 2, 3)
    ):
        raise AssertionError("optimizer update did not commit once per logical step")
    for eager_step, captured_step in zip(
        eager.parameter_steps,
        captured.parameter_steps,
        strict=True,
    ):
        _assert_parameters_close(eager_step, captured_step)
    _assert_parameters_close(
        _parameter_snapshot(eager.workload),
        _parameter_snapshot(captured.workload),
    )
    torch.testing.assert_close(
        captured.workload.optimizer.state_dict(),
        eager.workload.optimizer.state_dict(),
        rtol=1e-5,
        atol=1e-6,
    )
    if captured.batch_phases != (
        ("capture", 1),
        ("replay", 2),
        ("warmup", 1),
    ):
        raise AssertionError(
            "piecewise execution did not run optimizer warmup, capture, and replay: "
            f"{captured.batch_phases!r}"
        )
    if eager.batch_phases:
        raise AssertionError("eager execution unexpectedly installed a batch graph")
    if eager.fallback_count != 0 or captured.fallback_count != 0:
        raise AssertionError("native optimizer parity used a fallback path")
    if captured.eager_count <= 0 or captured.replay_count <= 0:
        raise AssertionError("captured execution lacks mixed warmup/replay evidence")

    torch.cuda.synchronize(device)
    print(
        json.dumps(
            {
                "steps": _STEPS,
                "eager_losses": eager.losses,
                "captured_losses": captured.losses,
                "batch_phases": dict(captured.batch_phases),
                "eager_counts": {
                    "eager": eager.eager_count,
                    "replay": eager.replay_count,
                    "fallback": eager.fallback_count,
                },
                "captured_counts": {
                    "eager": captured.eager_count,
                    "replay": captured.replay_count,
                    "fallback": captured.fallback_count,
                },
                "versions": captured.versions,
                "updates": captured.update_counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
