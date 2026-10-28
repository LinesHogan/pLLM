"""Request parameter overlays on a real vLLM MRv2 model: dense, low-rank, imported and route-keyed.

Each overlay must be applied to every row, update only its own request (or route),
and leave no lineage, pending release or fallback behind.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass, replace
from functools import partial
from unittest.mock import patch

import torch
from vllm import LLM, SamplingParams

import pllm
from pllm import TensorUpdate
from pllm.backends.vllm.runtime import VllmLlm
from pllm.config import ModelConfig, RunnerKind, VllmDType
from pllm.contracts.parameter_overlay import (
    ParameterModuleKind,
    ParameterOverlayOperator,
    ParameterRole,
    ParameterTarget,
)
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.execution.scan_execution import PreparedScan
from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload

TARGET = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
PROMPTS = ["The sky is", "Water freezes at"]
SAMPLING = SamplingParams(max_tokens=2, temperature=0.0, n=1, ignore_eos=True)


def low_rank_factors(keys=()) -> pllm.RequestState:
    shapes = {
        "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),
        "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),
    }
    return pllm.RequestState(shapes, keys=keys) if keys else pllm.RequestState(shapes)


@dataclass(frozen=True)
class RequestLowRankTTT(pllm.Workload):
    """Per-request LoRA trained on the same captured operand pairs as the dense path."""

    @property
    def factors(self) -> pllm.RequestState:
        return low_rank_factors()

    @pllm.overlay(
        visibility=pllm.Visibility.forward_pinned(),
        state="factors",
        targets=(TARGET,),
        operator=ParameterOverlayOperator.LOW_RANK_ADD,
        factors=("a", "b"),
        scale=0.5,
    )
    def transition(
        self,
        hidden: torch.Tensor = pllm.operand("input"),
        desired: torch.Tensor = pllm.operand("target"),
        base: torch.Tensor = pllm.operand("weight"),
        a: torch.Tensor = pllm.parameter("factors", "a"),
        b: torch.Tensor = pllm.parameter("factors", "b"),
    ) -> TensorUpdate:
        output = hidden @ (b @ a).T
        error = hidden @ base.T + 0.5 * output - desired
        gradient_b = 2 * error.T @ (hidden @ a.T) / hidden.shape[0]
        gradient_a = 2 * (error @ b).T @ hidden / hidden.shape[0]
        a.add_(-1e-3 * gradient_a)
        b.add_(-1e-3 * gradient_b)
        return TensorUpdate(
            output,
            output.new_ones((), dtype=torch.bool),
            output.new_ones((), dtype=torch.int64),
            output.new_full((), hidden.shape[0], dtype=torch.int64),
        )


@dataclass(frozen=True)
class RoutedLowRankTTT(RequestLowRankTTT):
    """Route-keyed per-adapter LoRA; the routing policy is ordinary host code."""

    @property
    def factors(self) -> pllm.RequestState:
        return low_rank_factors((pllm.state.RequestKeyField.ROUTE,))


@dataclass(frozen=True)
class ImportedLowRank(pllm.Workload):
    """Per-request LoRA factors produced outside the engine (e.g. by a hypernetwork)."""

    @property
    def factors(self) -> pllm.RequestState:
        return replace(low_rank_factors(), imported=True)

    @property
    def adapter(self) -> pllm.model.Overlay:
        return pllm.model.Overlay(
            "factors", (TARGET,), operator=ParameterOverlayOperator.LOW_RANK_ADD, factors=("a", "b"), scale=0.5
        )


def open_llm(
    workload,
    seqs=2,
    tokens=64,
    model_length=64,
    *,
    cuda_graphs=False,
    chunked=False,
    runner=RunnerKind.MRV2,
    **execution,
) -> VllmLlm[object, object]:
    options = {"execution": pllm.ExecutionOptions(**execution)} if execution else {}
    config = ModelConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype=VllmDType.BFLOAT16,
        max_model_len=model_length,
        max_num_seqs=seqs,
        max_num_batched_tokens=tokens,
        gpu_memory_utilization=0.25,
        cuda_graphs=cuda_graphs,
        enable_chunked_prefill=chunked,
        runner=runner,
        seed=7,
        **options,
    )
    # Separate engine starts profile different CUDA/library warmup overhead.
    # A common public vLLM byte budget makes the strict/batched comparison
    # about overlay execution rather than that profiling variation.
    with patch("pllm.backends.vllm.runtime.LLM", partial(LLM, kv_cache_memory_bytes=4 << 30)):
        handle = pllm.make_llm(workload=workload, config=config)
    if not isinstance(handle, VllmLlm):
        raise RuntimeError("public make_llm did not route the parameter overlay to vLLM")
    return handle


def texts(outputs) -> tuple[str, ...]:
    return tuple(output.outputs[0].text for output in outputs)


def _check(handle: VllmLlm[object, object], label: str) -> tuple[str, ...]:
    prompts = (
        ["The sky is blue. " * 5, "Water freezes when it is cold. " * 4]
        if label.startswith(("dense", "gated"))
        else PROMPTS
    )
    if label.startswith("gated"):
        prompts = ["The sky is blue. " * 100, "Water freezes when it is cold. " * 80]
    outputs = texts(handle.generate(prompts, SAMPLING))
    diagnostics = handle.deployment.execution.diagnostics()
    if (
        not all(outputs)
        or diagnostics.update_count < 1
        or diagnostics.trained_rows < 1
        or handle.runner.require_session().apply_count < 2
        or handle.runner.require_session().updated_apply_count < 1
        or diagnostics.active_lineages != 0
        or diagnostics.pending_state_releases != 0
        or diagnostics.fallback_count != 0
    ):
        raise RuntimeError(
            f"{label} MRv2 parameter-overlay lifecycle evidence is incomplete: "
            f"outputs={outputs!r}, diagnostics={diagnostics}, "
            f"apply={handle.runner.require_session().apply_count}, "
            f"updated_apply={handle.runner.require_session().updated_apply_count}"
        )
    return outputs


def _check_cancelled_projection(handle):
    """A captured real projection retains its pinned page through cancellation."""
    from pllm.contracts.hidden_locator import HiddenPhase
    from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
    from pllm.contracts.request_identity import RequestIdentity

    execution = handle.deployment.execution
    from pllm.backends.vllm.overlay_graph import HybridParameterOverlay

    overlay = handle.runner.require_session()._overlay
    if isinstance(overlay, HybridParameterOverlay):
        overlay = overlay.pinned
    reader = overlay.readers[0]
    identity = RequestIdentity("cancelled-projection", 41, 0)
    stream = torch.cuda.current_stream()
    (ref,) = execution.admit((identity,), stream)
    rows = PackedPhaseRowBatch(
        handle.attachment.physical.plan_epoch, 0, 1, (PackedPhaseRow(0, 0, 0, HiddenPhase.PREFILL, identity),)
    )
    invocation = execution.begin(rows.rows, stream)
    overlay.reserve()
    overlay.prepare(rows, invocation.context, execution.directory)
    hidden = reader.captured.new_ones((1, reader.captured.shape[1]))
    reader.forward(hidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = reader.forward(hidden)[0]
    slot = execution.arena.field_slots((invocation.context.pin,), (ref,), reader.binding.fields[0].field)[1][0]
    address = reader.pools[0][slot].data_ptr()
    stream.synchronize()
    with patch.object(torch.cuda.Event, "synchronize", side_effect=AssertionError("host cancellation wait")):
        torch.cuda._sleep(100_000_000)
        graph.replay()
        execution.cancel_step(invocation)
        overlay.clear()
        execution.retire((identity,))
        assert execution.arena.pending_pin_releases > 0
        other = RequestIdentity("replacement-projection", 42, 0)
        (replacement,) = execution.admit((other,), stream)
        pin = execution.arena.pin_exact((replacement,), stream)
        new_slot = execution.arena.field_slots((pin,), (replacement,), reader.binding.fields[0].field)[1][0]
        assert reader.pools[0][new_slot].data_ptr() != address
        execution.arena.release_pin(pin)
    stream.synchronize()
    assert torch.isfinite(output).all()
    execution.retire((other,), wait=True)
    execution.arena.collect_completed(wait=True)
    assert execution.arena.pending_pin_releases == 0


def _check_graph(handle):
    """Pinned graph operands reach training as row counts and request order change."""
    from pllm.backends.vllm.overlay_graph import PinnedParameterOverlay, _ProjectionReader

    overlay = handle.runner.require_session()._overlay
    assert isinstance(overlay, PinnedParameterOverlay)
    reader = overlay.readers[0]
    assert all(
        pool.untyped_storage()
        == handle.deployment.execution.arena.field_pool(reader.binding.resource, view.field).untyped_storage()
        for pool, view in zip(reader.pools, reader.binding.fields, strict=True)
    )
    pointers = (reader.table.data_ptr(), reader.captured.data_ptr(), reader.pools[0].data_ptr())
    prompts = ["The sky is blue. " * 8, "Water freezes when it is cold. " * 2]
    with patch.object(_ProjectionReader, "emit", side_effect=AssertionError("warm replay called Python projection")):
        first = texts(handle.generate(prompts, SAMPLING))
        assert texts(handle.generate(list(reversed(prompts)), SAMPLING)) == first[::-1]
        assert texts(handle.generate(prompts[:1], SAMPLING)) == first[:1]
        assert all(texts(handle.generate(["The sky is blue. " * 5, "Water is cold. " * 2], SAMPLING)))
    assert overlay.take_captures() == ()
    assert pointers == (reader.table.data_ptr(), reader.captured.data_ptr(), reader.pools[0].data_ptr())
    assert handle.diagnostics().update_count > 0 and handle.diagnostics().active_lineages == 0
    _check_cancelled_projection(handle)
    return first


def _check_hybrid(handle):
    """Exact eligibility selects source execution; inactive decode never calls Python projections."""
    from vllm.config import CUDAGraphMode

    from pllm.backends.vllm.overlay_graph import HybridParameterOverlay, _ProjectionReader
    from pllm.contracts.hidden_locator import HiddenPhase

    overlay = handle.runner.require_session()._overlay
    assert isinstance(overlay, HybridParameterOverlay)
    manager = handle.runner.cudagraph_manager
    dispatch = manager.dispatch
    modes = []

    def record(num_reqs, num_tokens, uniform_token_count, num_active_loras):
        result = dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras)
        modes.append((manager.eager, result.cg_mode))
        return result

    prompts = ["The sky is blue. " * 18, "Water is cold. " * 8]
    with (
        patch.object(manager, "dispatch", record),
        patch.object(_ProjectionReader, "emit", side_effect=AssertionError("warm hybrid called Python pinned reader")),
    ):
        first = texts(handle.generate(prompts, SAMPLING))
        assert texts(handle.generate(list(reversed(prompts)), SAMPLING)) == first[::-1]
        texts(handle.generate(["Hi"], SAMPLING))  # One-token prefill must also use Exact, even at C=0.
    assert all(mode is CUDAGraphMode.NONE if training else mode is CUDAGraphMode.FULL for training, mode in modes)
    assert any(training for training, _ in modes)
    if HiddenPhase.DECODE in overlay.phases:
        assert all(training for training, _ in modes)
    else:
        assert any(not training for training, _ in modes)
        _check_cancelled_projection(handle)
    diagnostics = handle.diagnostics()
    assert diagnostics.update_count > 0 and diagnostics.active_lineages == diagnostics.pending_state_releases == 0
    assert overlay.take_captures() == ()
    return first


def _key(prompt_index: int) -> RequestStateKey:
    return RequestStateKey.project(RequestIdentity("", prompt_index, 0), (RequestKeyField.PROMPT_IDX,))


def _check_imported(handle: VllmLlm[object, object]) -> tuple[str, ...]:
    (layout,) = handle.attachment.inputs.plan.tensor_layout_request.layouts
    down_shape, up_shape = (tuple(field.shape) for field in layout.layout.fields[:2])
    device = handle.device
    # No import: every row reads its zero-initialized request state, identical to the base model.
    baseline = texts(handle.generate(PROMPTS, SAMPLING))
    if not all(baseline):
        raise RuntimeError("imported overlay baseline generation is incomplete")
    # One batch, one imported request: the second row must keep the exact base outputs.
    generator = torch.Generator().manual_seed(11)
    down = torch.randn(down_shape, generator=generator).to(device=device, dtype=torch.bfloat16)
    up = torch.randn(up_shape, generator=generator).to(device=device, dtype=torch.bfloat16)
    update = pllm.state.StateTensorUpdate((down.unsqueeze(0), up.unsqueeze(0)), (_key(2),))
    mixed = texts(handle.generate(PROMPTS, SAMPLING, prompt_indices=(2, 3), auxiliary=update))
    if mixed[0] == baseline[0] or mixed[1] != baseline[1]:
        raise RuntimeError("imported factors must move exactly their own request's outputs")
    # Imported zero factors are an exact no-op on fresh requests.
    zeros = (
        torch.zeros((2, *down_shape), device=device, dtype=torch.bfloat16),
        torch.zeros((2, *up_shape), device=device, dtype=torch.bfloat16),
    )
    update = pllm.state.StateTensorUpdate(zeros, (_key(4), _key(5)))
    if texts(handle.generate(PROMPTS, SAMPLING, prompt_indices=(4, 5), auxiliary=update)) != baseline:
        raise RuntimeError("imported zero factors must reproduce the base model exactly")
    diagnostics = handle.deployment.execution.diagnostics()
    if (
        diagnostics.update_count != 3
        or diagnostics.active_lineages != 0
        or diagnostics.pending_state_releases != 0
        or diagnostics.fallback_count != 0
    ):
        raise RuntimeError("imported overlay lifecycle evidence is incomplete")
    return mixed


def _check_routed(handle: VllmLlm[object, object]) -> tuple[str, ...]:
    prompts = ["2 plus 2 is", "The sum of 1 and 1 is", "def main():"]
    try:
        handle.generate(prompts, SAMPLING)
    except ValueError as exc:
        if "route" not in str(exc):
            raise
    else:
        raise RuntimeError("route-keyed state accepted generation without routes")
    routes = tuple("math" if re.search(r"\d|plus|sum", prompt) else "code" for prompt in prompts)
    outputs = texts(handle.generate(prompts, SAMPLING, routes=routes))
    diagnostics = handle.deployment.execution.diagnostics()
    # Three retired requests, two route lineages still resident: the two math
    # prompts shared one adapter, and the dormant pool outlives its requests.
    if (
        diagnostics.active_lineages != 2
        or diagnostics.update_count < 1
        or handle.runner.require_session().apply_count < 3
        or handle.runner.require_session().updated_apply_count < 1
        or diagnostics.pending_state_releases != 0
        or diagnostics.fallback_count != 0
    ):
        raise RuntimeError("routed overlay lifecycle evidence is incomplete")
    return outputs


def main() -> int:
    from pllm.workloads.prefill_adapters import PrefillAdapters

    adapter_tokens = []
    for chunked in (False, True):
        with open_llm(PrefillAdapters(chunk_size=3, rank=2), chunked=chunked, tokens=7 if chunked else 64) as handle:
            outputs = handle.generate(["The sky is blue. " * 5, "Water freezes at " * 7], SAMPLING)
            adapter_tokens.append(texts(outputs))
            diagnostics = handle.diagnostics()
            assert diagnostics.update_count == 5 * sum((len(output.prompt_token_ids) + 2) // 3 for output in outputs)
            assert diagnostics.trained_rows == 5 * sum(len(output.prompt_token_ids) for output in outputs)
            assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
    assert adapter_tokens[0] == adapter_tokens[1]
    dense = InPlaceTTTWorkload(
        InPlaceTTTConfig((TARGET,), chunk_size=1, learning_rate=1e-4, update_prefill=True, update_decode=False)
    )
    headwise = GatedDeltaWorkload(
        GatedDeltaConfig(
            (ParameterTarget(0, ParameterModuleKind.ATTENTION, ParameterRole.QUERY_PROJECTION_WEIGHT),), chunk_size=64
        )
    )
    scenarios = (
        (
            "dense_strict",
            lambda: open_llm(dense, memory_budget_bytes=1 << 30, allow_numerical_transforms=False),
            lambda h: _check(h, "dense strict"),
        ),
        (
            "dense_512",
            lambda: open_llm(dense, memory_budget_bytes=512 << 20),
            lambda h: _check(h, "dense window"),
        ),
        (
            "dense_1536",
            lambda: open_llm(dense, memory_budget_bytes=1536 << 20),
            lambda h: _check(h, "dense tree"),
        ),
        (
            "gated_exact",
            lambda: open_llm(
                headwise,
                tokens=2048,
                model_length=1024,
                memory_budget_bytes=256 << 20,
                allow_numerical_transforms=True,
            ),
            lambda h: _check(h, "gated strict"),
        ),
        (
            "gated",
            lambda: open_llm(
                replace(headwise, config=replace(headwise.config, visibility=pllm.Visibility.forward_pinned())),
                tokens=2048,
                model_length=1024,
                memory_budget_bytes=256 << 20,
            ),
            lambda h: _check(h, "gated tree"),
        ),
        (
            "graph_head",
            lambda: open_llm(
                replace(
                    headwise,
                    config=replace(headwise.config, chunk_size=16, visibility=pllm.Visibility.forward_pinned()),
                ),
                tokens=128,
                model_length=128,
                cuda_graphs=True,
                capture_memory_bytes=256 << 20,
                memory_budget_bytes=1 << 30,
            ),
            lambda h: _check_graph(h),
        ),
        (
            "graph_head_strict",
            lambda: open_llm(
                replace(
                    headwise,
                    config=replace(headwise.config, chunk_size=16, visibility=pllm.Visibility.forward_pinned()),
                ),
                tokens=128,
                model_length=128,
                cuda_graphs=True,
                capture_memory_bytes=256 << 20,
                memory_budget_bytes=1 << 30,
                allow_numerical_transforms=False,
            ),
            _check_graph,
        ),
        (
            "graph_dense",
            lambda: open_llm(
                dense,
                tokens=128,
                model_length=128,
                cuda_graphs=True,
                capture_memory_bytes=512 << 20,
                memory_budget_bytes=2 << 30,
            ),
            _check_graph,
        ),
        (
            "hybrid_head",
            lambda: open_llm(
                replace(headwise, config=replace(headwise.config, chunk_size=16)),
                tokens=128,
                model_length=128,
                cuda_graphs=True,
                capture_memory_bytes=512 << 20,
                memory_budget_bytes=2 << 30,
            ),
            _check_hybrid,
        ),
        (
            "hybrid_decode_training",
            lambda: open_llm(
                replace(headwise, config=replace(headwise.config, chunk_size=16, update_decode=True)),
                tokens=128,
                model_length=128,
                cuda_graphs=True,
                capture_memory_bytes=512 << 20,
                memory_budget_bytes=2 << 30,
            ),
            _check_hybrid,
        ),
        (
            "hybrid_dense",
            lambda: open_llm(
                replace(dense, config=replace(dense.config, visibility=pllm.Exact())),
                tokens=128,
                model_length=128,
                cuda_graphs=True,
                capture_memory_bytes=512 << 20,
                memory_budget_bytes=2 << 30,
            ),
            _check_hybrid,
        ),
        # The process-wide pool retains the larger hybrid captures above.
        (
            "graph_low_rank",
            lambda: open_llm(
                RequestLowRankTTT(),
                tokens=128,
                model_length=128,
                cuda_graphs=True,
                capture_memory_bytes=512 << 20,
                memory_budget_bytes=1 << 30,
            ),
            _check_graph,
        ),
        ("low_rank", lambda: open_llm(RequestLowRankTTT()), lambda h: _check(h, "request low-rank")),
        ("imported", lambda: open_llm(ImportedLowRank(), memory_budget_bytes=1 << 30), _check_imported),
        (
            "graph_imported",
            lambda: open_llm(
                ImportedLowRank(), cuda_graphs=True, capture_memory_bytes=512 << 20, memory_budget_bytes=1 << 30
            ),
            _check_imported,
        ),
        ("routed", lambda: open_llm(RoutedLowRankTTT(), seqs=3, tokens=96), _check_routed),
    )
    results: dict[str, tuple[str, ...]] = {}
    dense_cache = None
    for name, open_handle, check in scenarios:
        handle = open_handle()
        try:
            cache = handle.runner.kv_cache_config
            before = (cache.num_blocks, tuple(tensor.size for tensor in cache.kv_cache_tensors))
            forms: set[tuple[str, int]] = set()
            execute = PreparedScan.windows

            def record(plan, *args, tracing=False):
                assert plan.analysis is not None
                if not tracing:
                    prefix = "right_matrix:" if plan.analysis.kind == "right" else ""
                    forms.add((prefix + plan.form.value, plan.window))
                return execute(plan, *args, tracing=tracing)

            with patch.object(PreparedScan, "windows", record):
                results[name] = check(handle)
            if name == "dense_1536" and not any(form == "hillis_steele" for form, _ in forms):
                raise RuntimeError("large-budget dense overlay did not exercise a tree window")
            if name == "gated" and not any(form.startswith("right_matrix:") for form, _ in forms):
                raise RuntimeError("Gated delta overlay did not exercise a right-matrix scan")
            if name == "gated_exact" and not any(form.startswith("right_matrix:") for form, _ in forms):
                raise RuntimeError("Exact overlay did not select its right-matrix prefix")
            after = (cache.num_blocks, tuple(tensor.size for tensor in cache.kv_cache_tensors))
            if before != after:
                raise RuntimeError("overlay execution changed the vLLM KV allocation")
            if name == "dense_strict":
                dense_cache = before
            if name.startswith(("dense_", "gated")) and before != dense_cache:
                raise RuntimeError("additive scan changed vLLM KV-cache capacity")
            if name.startswith(("dense_", "gated")):
                print(
                    json.dumps(
                        {
                            "scenario": name,
                            "scan_forms": sorted(forms),
                            "kv_blocks": before[0],
                            "kv_bytes": sum(before[1]),
                            "task_workspace": [
                                (str(f.task), f.workspace_bytes)
                                for f in handle.attachment.inputs.plan.timeline.task_resources
                            ],
                        }
                    ),
                    flush=True,
                )
        finally:
            handle.close()
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
