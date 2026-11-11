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
from pllm import UpdateOutput
from pllm.backends.vllm.runtime import VllmLlm
from pllm.config import ModelConfig
from pllm.contracts.parameter_overlay import (
    Projection,
)
from pllm.contracts.request_identity import RequestIdentity
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.scan_execution import PreparedScan
from pllm.testing_gpu import check_probe
from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload
from test.dense_update import DenseUpdate, DenseUpdateConfig

TARGET = Projection("down_proj", 0)
PROMPTS = ["The sky is", "Water freezes at"]
SAMPLING = SamplingParams(max_tokens=2, temperature=0.0, n=1, ignore_eos=True)


def low_rank_factors(key_fields=()) -> pllm.RequestState:
    shapes = {
        "a": pllm.Field(lambda in_features, out_features: (2, in_features)),
        "b": pllm.Field(lambda in_features, out_features: (out_features, 2)),
    }
    return pllm.RequestState(shapes, key_fields=key_fields) if key_fields else pllm.RequestState(shapes)


@dataclass(frozen=True)
class RequestLowRankTTT(pllm.Workload):
    """Per-request LoRA trained on the same captured operand pairs as the dense path."""

    @property
    def factors(self) -> pllm.RequestState:
        return low_rank_factors()

    @pllm.train_in_forward(
        visibility="forward_pinned",
        state="factors",
        targets=(TARGET,),
        operator="low_rank",
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
    ) -> UpdateOutput:
        output = (hidden @ a.to(hidden.dtype).T) @ b.to(hidden.dtype).T
        error = hidden @ base.T + 0.5 * output - desired
        gradient_b = 2 * error.T @ (hidden @ a.to(hidden.dtype).T) / hidden.shape[0]
        gradient_a = 2 * (error @ b.to(hidden.dtype)).T @ hidden / hidden.shape[0]
        a.add_(-1e-3 * gradient_a)
        b.add_(-1e-3 * gradient_b)
        return UpdateOutput(
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
        return low_rank_factors(("route",))


@dataclass(frozen=True)
class ImportedLowRank(pllm.Workload):
    """Per-request LoRA factors produced outside the engine (e.g. by a hypernetwork)."""

    @property
    def factors(self) -> pllm.RequestState:
        return replace(low_rank_factors(), init=pllm.Imported())

    @property
    def adapter(self) -> pllm.model.Adapter:
        return pllm.model.Adapter("factors", (TARGET,), operator="low_rank", factors=("a", "b"), scale=0.5)


def open_llm(
    workload,
    seqs=2,
    tokens=64,
    model_length=64,
    *,
    cuda_graphs=False,
    chunked=False,
    **execution,
) -> VllmLlm:
    options = {"execution": pllm.ExecutionOptions(**execution)} if execution else {}
    config = ModelConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        max_model_len=model_length,
        max_num_seqs=seqs,
        max_num_batched_tokens=tokens,
        gpu_memory_utilization=0.25,
        cuda_graphs=cuda_graphs,
        enable_chunked_prefill=chunked,
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


def _check(handle: VllmLlm, label: str) -> tuple[str, ...]:
    prompts = (
        ["The sky is blue. " * 5, "Water freezes when it is cold. " * 4]
        if label.startswith(("dense", "gated"))
        else PROMPTS
    )
    if label.startswith("gated"):
        prompts = ["The sky is blue. " * 100, "Water freezes when it is cold. " * 80]
    outputs = texts(handle.generate(prompts, SAMPLING))
    diagnostics = handle.diagnostics()
    if (
        not all(outputs)
        or diagnostics.update_count < 1
        or diagnostics.trained_rows < 1
        or handle.runner.require_session().apply_count < 2
        or handle.runner.require_session().updated_apply_count < 1
        or diagnostics.active_states != 0
        or diagnostics.pending_state_releases != 0
    ):
        raise RuntimeError(
            f"{label} MRv2 parameter-overlay lifecycle evidence is incomplete: "
            f"outputs={outputs!r}, diagnostics={diagnostics}, "
            f"apply={handle.runner.require_session().apply_count}, "
            f"updated_apply={handle.runner.require_session().updated_apply_count}"
        )
    return outputs


def _projection_output(reader, hidden):
    """Run the model site's producer so projection operands exist during capture."""
    return reader.forward(hidden)[0] if reader.mlp is None else reader.forward_mlp(hidden)


def _check_cancelled_projection(handle):
    """A captured real projection retains its pinned page through cancellation."""
    from pllm.contracts.hidden_locator import HiddenPhase
    from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch

    execution = handle.runner.require_session().deployment.execution
    overlay = handle.runner.require_session().parameter_overlay
    reader = overlay.readers[0]
    identity = RequestIdentity("cancelled-projection", 41, 0)
    stream = torch.cuda.current_stream()
    (ref,) = execution.admit((identity,), stream)
    rows = PackedPhaseRowBatch.from_rows(0, 1, (PackedPhaseRow(0, 0, 0, HiddenPhase.PREFILL, identity),))
    invocation = execution.begin(rows.rows, stream)
    overlay.reserve()
    overlay.prepare_pinned(rows, invocation, execution.directory, None)
    features = reader.captured.shape[1] if reader.mlp is None else reader.width
    hidden = reader.captured.new_ones((1, features))
    _projection_output(reader, hidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _projection_output(reader, hidden)
    slot = execution.arena.field_slots((invocation.pin,), StateRefs.of((ref,)), reader.binding.fields[0].field)[1][0]
    address = reader.pools[0][slot].data_ptr()
    stream.synchronize()
    with patch.object(torch.cuda.Event, "synchronize", side_effect=AssertionError("host cancellation wait")):
        torch.cuda._sleep(100_000_000)
        graph.replay()
        execution.cancel_step(invocation)
        overlay.clear()
        execution.retire((identity,))
        other = RequestIdentity("replacement-projection", 42, 0)
        (replacement,) = execution.admit((other,), stream)
        pin = execution.arena.pin_exact(StateRefs.of((replacement,)), stream)
        new_slot = execution.arena.field_slots((pin,), StateRefs.of((replacement,)), reader.binding.fields[0].field)[1][
            0
        ]
        assert reader.pools[0][new_slot].data_ptr() != address
        execution.arena.release_pin(pin)
    stream.synchronize()
    assert torch.isfinite(output).all()
    execution.retire((other,), wait=True)
    execution.arena.collect_completed(wait=True)
    assert execution.arena.pending_pin_releases == 0


def _check_graph(handle):
    """Pinned graph operands reach training as row counts and request order change."""
    from pllm.backends.vllm.parameter_overlay import ProjectionReader

    overlay = handle.runner.require_session().parameter_overlay
    assert overlay.pinned and not overlay.exact
    reader = overlay.readers[0]
    arena = handle.runner.require_session().deployment.execution.arena
    assert all(
        pool.untyped_storage() == arena.field_pool(reader.binding.resource, view.field).untyped_storage()
        for pool, view in zip(reader.pools, reader.binding.fields, strict=True)
    )
    pointers = (reader.table.data_ptr(), reader.captured.data_ptr(), reader.pools[0].data_ptr())
    prompts = ["The sky is blue. " * 8, "Water freezes when it is cold. " * 2]
    with patch.object(ProjectionReader, "emit", side_effect=AssertionError("warm replay called Python projection")):
        first = texts(handle.generate(prompts, SAMPLING))
        assert texts(handle.generate(list(reversed(prompts)), SAMPLING)) == first[::-1]
        assert texts(handle.generate(prompts[:1], SAMPLING)) == first[:1]
        assert all(texts(handle.generate(["The sky is blue. " * 5, "Water is cold. " * 2], SAMPLING)))
    assert overlay.take_operands() == ()
    assert pointers == (reader.table.data_ptr(), reader.captured.data_ptr(), reader.pools[0].data_ptr())
    assert handle.diagnostics().update_count > 0 and handle.diagnostics().active_states == 0
    _check_cancelled_projection(handle)
    return first


def _check_hybrid(handle):
    """Prefill selects source execution; decode replays the bound projection kernels."""
    from vllm.config import CUDAGraphMode

    from pllm.backends.vllm.parameter_overlay import ProjectionReader

    overlay = handle.runner.require_session().parameter_overlay
    assert overlay.pinned and overlay.exact
    manager = handle.runner.cudagraph_manager
    dispatch = manager.dispatch
    modes = []

    def record(*args, **kwargs):
        result = dispatch(*args, **kwargs)
        modes.append((manager.eager, result.cg_mode))
        return result

    prompts = ["The sky is blue. " * 18, "Water is cold. " * 8]
    with (
        patch.object(manager, "dispatch", record),
        patch.object(ProjectionReader, "emit", side_effect=AssertionError("warm hybrid called Python pinned reader")),
    ):
        first = texts(handle.generate(prompts, SAMPLING))
        assert texts(handle.generate(list(reversed(prompts)), SAMPLING)) == first[::-1]
        texts(handle.generate(["Hi"], SAMPLING))  # One-token prefill must also use Exact, even at C=0.
    assert all(mode is CUDAGraphMode.NONE if training else mode is CUDAGraphMode.FULL for training, mode in modes)
    assert any(training for training, _ in modes)
    assert any(not training for training, _ in modes)
    if overlay.readers[0].decode is None:
        _check_cancelled_projection(handle)
    diagnostics = handle.diagnostics()
    assert diagnostics.update_count > 0 and diagnostics.active_states == diagnostics.pending_state_releases == 0
    assert overlay.take_operands() == ()
    return first


def _check_imported(handle: VllmLlm) -> tuple[str, ...]:
    (layout,) = handle.attachment.plan.tensor_layout_request.layouts
    down_shape, up_shape = (tuple(field.shape) for field in layout.layout.fields[:2])
    device = handle.device
    # No import: every row reads its zero-initialized request state, identical to the base model.
    baseline = texts(handle.generate(PROMPTS, SAMPLING))
    if not all(baseline):
        raise RuntimeError("imported overlay baseline generation is incomplete")
    # One batch, one imported request: the second row must keep the exact base outputs.
    generator = torch.Generator().manual_seed(11)
    down = torch.randn(down_shape, generator=generator).to(device=device, dtype=torch.float32)
    up = torch.randn(up_shape, generator=generator).to(device=device, dtype=torch.float32)
    update = pllm.io.StateImport((torch.stack((down, torch.zeros_like(down))), torch.stack((up, torch.zeros_like(up)))))
    mixed = texts(handle.generate(PROMPTS, SAMPLING, prompt_indices=(2, 3), batch=update))
    if mixed[0] == baseline[0] or mixed[1] != baseline[1]:
        raise RuntimeError("imported factors must move exactly their own request's outputs")
    # Imported zero factors are an exact no-op on fresh requests.
    zeros = (
        torch.zeros((2, *down_shape), device=device, dtype=torch.float32),
        torch.zeros((2, *up_shape), device=device, dtype=torch.float32),
    )
    update = pllm.io.StateImport(zeros)
    if texts(handle.generate(PROMPTS, SAMPLING, prompt_indices=(4, 5), batch=update)) != baseline:
        raise RuntimeError("imported zero factors must reproduce the base model exactly")
    diagnostics = handle.diagnostics()
    if diagnostics.update_count != 4 or diagnostics.active_states != 0 or diagnostics.pending_state_releases != 0:
        raise RuntimeError("imported overlay lifecycle evidence is incomplete")
    return mixed


def _check_routed(handle: VllmLlm) -> tuple[str, ...]:
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
    diagnostics = handle.diagnostics()
    # Three retired requests, two route lineages still resident: the two math
    # prompts shared one adapter, and the dormant pool outlives its requests.
    if (
        diagnostics.active_states != 2
        or diagnostics.update_count < 1
        or handle.runner.require_session().apply_count < 3
        or handle.runner.require_session().updated_apply_count < 1
        or diagnostics.pending_state_releases != 0
    ):
        raise RuntimeError("routed overlay lifecycle evidence is incomplete")
    return outputs


def checkpoint_ttt() -> None:
    """A tiny learned checkpoint exercises loading, exact decode, mixed requests and fresh state."""
    from tempfile import TemporaryDirectory

    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload

    with TemporaryDirectory(prefix="pllm-ttt-") as directory:
        config = Qwen3Config(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=151936,
            tie_word_embeddings=True,
            ttt_mode=True,
            ttt_layers=[0, 1],
            ttt_chunk=4,
            ttt_lr=0.01,
            ttt_target="hidden_states",
            ttt_proj=True,
        )
        torch.manual_seed(41)
        Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(directory)
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").save_pretrained(directory)
        weights = load_file(str(Path(directory) / "model.safetensors"))
        weights.update(
            {
                f"model.layers.{layer}.mlp.ttt_{name}.weight": torch.randn(shape, dtype=torch.bfloat16) * 0.01
                for layer in (0, 1)
                for name, shape in (("conv", (128, 1, 5)), ("proj", (128, 128)))
            }
        )
        save_file(weights, str(Path(directory) / "model.safetensors"))
        del weights
        for track, dtype in (
            ("T1", torch.float32),
            ("T1", torch.bfloat16),
            ("T2", torch.float32),
            ("T2", torch.bfloat16),
        ):
            workload = InPlaceTTTWorkload(InPlaceTTTConfig.from_checkpoint(directory, track=track), dtype=dtype)
            model_config = ModelConfig(
                directory,
                max_num_seqs=2,
                max_model_len=32,
                max_num_batched_tokens=32,
                execution=pllm.ExecutionOptions(
                    allow_numerical_transforms=False, cuda_graph_pool_budget_bytes=512 << 20
                ),
            )
            with (
                patch("pllm.backends.vllm.runtime.LLM", partial(LLM, kv_cache_memory_bytes=32 << 20)),
                pllm.make_llm(model_config, workload=workload) as handle,
            ):
                check_probe(workload, model_config, handle)
                prompts = [{"prompt_token_ids": [100, 101, 102]}, {"prompt_token_ids": [103] * 8}]
                sampling = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
                first = handle.generate(prompts, sampling)
                second = handle.generate(prompts, sampling)
                assert texts(first) == texts(second)
                diagnostics = handle.diagnostics()
                assert diagnostics.update_count == 12 and diagnostics.trained_rows == 48
                assert diagnostics.active_states == 1 and diagnostics.pending_state_releases == 0
                assert not diagnostics.eager_reason


def community_t2(checkpoint: str) -> None:
    """Report clipping on community weights; these hidden-target weights establish no T2 quality claim.

    Run explicitly with --community-t2 CHECKPOINT; ordinary verification uses the tiny fixture.
    """
    from time import perf_counter

    from pllm.execution.overlay_task import ChunkUpdate
    from pllm.workloads import InPlaceTTTConfig, InPlaceTTTWorkload

    options = InPlaceTTTConfig.from_checkpoint(checkpoint, track="T2")
    measurements = []
    execute = ChunkUpdate.contiguous

    def record(update, arguments, plan, offset):
        before = arguments["delta"].clone()
        result = execute(update, arguments, plan, offset)
        if result.update_count.item():
            source = arguments["targets"]
            wanted = (
                torch.nn.functional.conv1d(source.T.unsqueeze(0), arguments["conv"], padding=2, groups=source.shape[1])
                .squeeze(0)
                .T
                @ arguments["proj"]
            )
            raw_norm = (wanted.float().T @ arguments["inputs"].float()).norm().item()
            applied_norm = (arguments["delta"] - before).norm().item()
            assert raw_norm > 1e-5
            assert abs(applied_norm - options.learning_rate * 1e-5) < 1e-9
            layer = options.targets[len(measurements)].layer
            measurements.append({"layer": layer, "raw_norm": raw_norm, "applied_norm": applied_norm})
        return result

    config = ModelConfig(
        checkpoint,
        max_model_len=options.chunk_size + 8,
        max_num_seqs=1,
        max_num_batched_tokens=options.chunk_size + 8,
        cuda_graphs=False,
        execution=pllm.ExecutionOptions(allow_numerical_transforms=False, execution_peak_budget_bytes=5 << 30),
    )
    with (
        patch("pllm.backends.vllm.runtime.LLM", partial(LLM, kv_cache_memory_bytes=512 << 20)),
        pllm.make_llm(config, workload=InPlaceTTTWorkload(options)) as handle,
        patch.object(ChunkUpdate, "contiguous", record),
    ):
        started = perf_counter()
        outputs = handle.generate(
            [{"prompt_token_ids": [100 + i % 100 for i in range(options.chunk_size)]}],
            SamplingParams(temperature=0, max_tokens=3, ignore_eos=True),
        )
        diagnostics = handle.diagnostics()
        assert len(outputs[0].outputs[0].token_ids) == 3
        assert len(measurements) == diagnostics.update_count == len(options.targets)
        assert diagnostics.pending_state_releases == 0
        print(
            json.dumps(
                {
                    "track": "T2",
                    "dtype": "float32 delta / bfloat16 model",
                    "eta": options.learning_rate,
                    "tau": 1e-5,
                    "chunk": options.chunk_size,
                    "generated": 3,
                    "seconds": perf_counter() - started,
                    "clipping": measurements,
                }
            ),
            flush=True,
        )


def main() -> int:
    from pllm.workloads.prefill_adapters import PrefillAdaptersWorkload

    adapter_tokens = []
    for chunked in (False, True):
        with open_llm(
            PrefillAdaptersWorkload(chunk_size=3, rank=2), chunked=chunked, tokens=7 if chunked else 64
        ) as handle:
            outputs = handle.generate(["The sky is blue. " * 5, "Water freezes at " * 7], SAMPLING)
            adapter_tokens.append(texts(outputs))
            diagnostics = handle.diagnostics()
            assert diagnostics.update_count == 5 * sum((len(output.prompt_token_ids) + 2) // 3 for output in outputs)
            assert diagnostics.trained_rows == 5 * sum(len(output.prompt_token_ids) for output in outputs)
            assert diagnostics.active_states == diagnostics.pending_state_releases == 0
    assert adapter_tokens[0] == adapter_tokens[1]
    dense = DenseUpdate(DenseUpdateConfig((TARGET,), chunk_size=1, learning_rate=1e-4, tokens="prefill"))
    headwise = GatedDeltaWorkload(GatedDeltaConfig((Projection("q_proj", 0),), chunk_size=64))
    scenarios = (
        (
            "dense_strict",
            lambda: open_llm(dense, execution_peak_budget_bytes=1 << 30, allow_numerical_transforms=False),
            lambda h: _check(h, "dense strict"),
        ),
        (
            "dense_768",
            lambda: open_llm(dense, execution_peak_budget_bytes=768 << 20),
            lambda h: _check(h, "dense window"),
        ),
        (
            "dense_1536",
            lambda: open_llm(dense, execution_peak_budget_bytes=1536 << 20),
            lambda h: _check(h, "dense tree"),
        ),
        (
            "gated_exact",
            lambda: open_llm(
                headwise,
                tokens=2048,
                model_length=1024,
                execution_peak_budget_bytes=256 << 20,
                allow_numerical_transforms=True,
            ),
            lambda h: _check(h, "gated strict"),
        ),
        (
            "gated",
            lambda: open_llm(
                replace(headwise, config=replace(headwise.config, visibility="forward_pinned")),
                tokens=2048,
                model_length=1024,
                execution_peak_budget_bytes=256 << 20,
            ),
            lambda h: _check(h, "gated tree"),
        ),
        (
            "graph_head",
            lambda: open_llm(
                replace(
                    headwise,
                    config=replace(headwise.config, chunk_size=16, visibility="forward_pinned"),
                ),
                tokens=128,
                model_length=128,
                cuda_graphs=True,
                cuda_graph_pool_budget_bytes=256 << 20,
                execution_peak_budget_bytes=1 << 30,
            ),
            lambda h: _check_graph(h),
        ),
        (
            "graph_head_strict",
            lambda: open_llm(
                replace(
                    headwise,
                    config=replace(headwise.config, chunk_size=16, visibility="forward_pinned"),
                ),
                tokens=128,
                model_length=128,
                cuda_graphs=True,
                cuda_graph_pool_budget_bytes=256 << 20,
                execution_peak_budget_bytes=1 << 30,
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
                cuda_graph_pool_budget_bytes=512 << 20,
                execution_peak_budget_bytes=3 << 30,
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
                cuda_graph_pool_budget_bytes=512 << 20,
                execution_peak_budget_bytes=2 << 30,
            ),
            _check_hybrid,
        ),
        (
            "hybrid_decode_training",
            lambda: open_llm(
                replace(headwise, config=replace(headwise.config, chunk_size=16, tokens="all")),
                tokens=128,
                model_length=128,
                cuda_graphs=True,
                cuda_graph_pool_budget_bytes=512 << 20,
                execution_peak_budget_bytes=2 << 30,
            ),
            _check_hybrid,
        ),
        (
            "hybrid_dense",
            lambda: open_llm(
                replace(dense, config=replace(dense.config, visibility="exact")),
                tokens=128,
                model_length=128,
                cuda_graphs=True,
                cuda_graph_pool_budget_bytes=512 << 20,
                execution_peak_budget_bytes=3 << 30,
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
                cuda_graph_pool_budget_bytes=512 << 20,
                execution_peak_budget_bytes=1 << 30,
            ),
            _check_graph,
        ),
        ("low_rank", lambda: open_llm(RequestLowRankTTT()), lambda h: _check(h, "request low-rank")),
        ("imported", lambda: open_llm(ImportedLowRank(), execution_peak_budget_bytes=1 << 30), _check_imported),
        (
            "graph_imported",
            lambda: open_llm(
                ImportedLowRank(),
                cuda_graphs=True,
                cuda_graph_pool_budget_bytes=512 << 20,
                execution_peak_budget_bytes=1 << 30,
            ),
            _check_imported,
        ),
        ("routed", lambda: open_llm(RoutedLowRankTTT(), seqs=3, tokens=96), _check_routed),
    )
    results: dict[str, tuple[str, ...]] = {}
    dense_cache = None
    for name, open_handle, check in scenarios:
        allocated = torch.cuda.memory_allocated()
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
            if name == "dense_1536" and forms:
                raise RuntimeError("large dense carry materialized per-chunk weights")
            if name == "gated" and not any(form == "right_matrix:hillis_steele" for form, _ in forms):
                raise RuntimeError("Gated delta overlay did not exercise a right-matrix tree window")
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
                                (str(f.task), f.workspace_bytes) for f in handle.attachment.plan.timeline.task_resources
                            ],
                        }
                    ),
                    flush=True,
                )
        finally:
            handle.close()
        assert torch.cuda.memory_allocated() - allocated < 256 << 20, f"{name}: closed engine retains GPU storage"
    print(results)
    return 0


def preemption_replays():
    """Real KV pressure replays an Exact recurrence and releases its request state."""
    with (
        patch(
            "pllm.backends.vllm.runtime.LLM", partial(LLM, num_gpu_blocks_override=10, kv_cache_memory_bytes=64 << 20)
        ),
        pllm.make_llm(
            workload=GatedDeltaWorkload(
                GatedDeltaConfig(
                    (Projection("q_proj", 0),),
                    chunk_size=7,
                    tokens="all",
                )
            ),
            config=ModelConfig(
                "Qwen/Qwen2.5-0.5B-Instruct",
                # One of the ten blocks is vLLM's null block; vLLM 0.32 refuses a max_model_len beyond the others.
                max_model_len=144,
                max_num_seqs=8,
                max_num_batched_tokens=64,
                enable_chunked_prefill=True,
                seed=7,
                execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=256 << 20),
            ),
        ) as handle,
    ):
        session = handle.runner.require_session()
        with patch.object(session, "preempt_requests", wraps=session.preempt_requests) as preempt:
            outputs = handle.generate(
                ["The number is"] * 8, SamplingParams(max_tokens=80, temperature=0, ignore_eos=True)
            )
        assert any(call.args[0] for call in preempt.call_args_list)
        assert all(len(output.outputs[0].token_ids) == 80 for output in outputs)
        assert handle.diagnostics().active_states == 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--community-t2":
        community_t2(sys.argv[2])
        raise SystemExit(0)
    checkpoint_ttt()
    preemption_replays()
    raise SystemExit(main())
