"""Real-engine contract checks for workload authors, exposed by pllm.testing.check_on_gpu."""

from __future__ import annotations

# Diagnostics deliberately inspect the installed session and in-process engine.
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from traceback import extract_tb
from typing import Concatenate, ParamSpec
from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

import pllm
from pllm.backends.vllm.checkpoint import load_options
from pllm.backends.vllm.model_shape import ModelShape
from pllm.backends.vllm.overlay_graph import GraphOptions, graph_options
from pllm.backends.vllm.runtime import VllmLlm, probe_backend, vllm_engine
from pllm.config import ExecutionOptions, ModelConfig
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.host_batch import CausalTrainBatch, map_host_batch
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedRows, PackedSamplerRowBatch
from pllm.contracts.parameter_overlay import Projection, ProjectionOperands
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId, TaskId
from pllm.contracts.state_ref import StateRef
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.overlay_task import ProjectionTransition
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.state_execution import ProjectionResiduals, StateTaskContext
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_declaration import TensorOutput
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_program import read_batch
from pllm.explanation import explain, explain_facts
from pllm.testing_contracts import CheckResult, CheckSkip, ContractError, check_contracts
from pllm.usage import Workload

_Allocation = ParamSpec("_Allocation")

# Sequence continuations avoid near-tied prose tokens whose greedy choice changes with vLLM batch geometry.
CHECK_PROMPTS = ["Continue the sequence: 1, 2, 3,", "Continue the alphabet: A, B, C,"]


def check_probe(workload: Workload, config: ModelConfig, handle: VllmLlm[TensorFrame, TensorOutput]) -> None:
    """Compare config-only and loaded-model reports through the same pure computation."""
    expected = explain(workload, config)
    runner = handle.runner
    requirements = workload.backend_requirements()
    shape = ModelShape.of_model(runner.decoder, requirements.parameter_targets, runner.dtype)
    facts = probe_backend(requirements, runner, publish_lag=config.execution.publish_lag)
    # The installed state already consumed memory; replay planning against its load-time capacity.
    assert handle.compile_facts is not None
    facts = replace(facts, state_page_budget_bytes=handle.compile_facts.state_page_budget_bytes)
    actual = explain_facts(workload, config, shape, facts, runner.hidden_layouts)
    failures: list[str] = []
    if expected != actual:
        failures.append("[explain/probe] config and loaded-model costs differ; report a pllm planning bug")
    if expected.kv_capacity:
        model = runner.decoder.kv_writable("attention capacity checks")
        for capacity in expected.kv_capacity:
            attention = model.model.layers[capacity.layout.layer].self_attn.attn
            if capacity.layout.keep == "none":
                assert attention.kv_sharing_target_layer_name is not None
                continue
            spec = attention.get_kv_cache_spec(runner.vllm_config)
            assert spec is not None
            blocks = spec.max_memory_usage_bytes(runner.vllm_config) // spec.page_size_bytes
            if abs(blocks - capacity.blocks) > 1:
                failures.append(
                    f"[explain/KV] layer {capacity.layout.layer}: {capacity.blocks} predicted blocks vs {blocks} native blocks"
                )
    if expected.eager_reason != handle.diagnostics().eager_reason:
        failures.append("[explain/graphs] engine graph mode differs from explain; report a pllm admission bug")
    if sum(state.reserved_bytes for state in actual.states) != handle.attachment.plan.states.reserved_bytes:
        failures.append("[explain/state] allocated state reservations differ; report a pllm allocation bug")
    if not requirements.parameter_targets:
        plan = handle.attachment.plan
        if (
            actual.peak_memory_bytes != plan.schedule.peak_memory_bytes
            or tuple(anchor.reserved_bytes for anchor in actual.anchors) != plan.timeline.reserved_bytes
        ):
            failures.append("[explain/reservations] installed plan differs from the probe; report a pllm planning bug")
    if failures:
        raise ContractError(tuple(failures))


class SynchronizationCheck:
    """Check pllm anchors and model-site staging after two decode warmup steps at each packed width."""

    def __init__(self, handle: VllmLlm[TensorFrame, TensorOutput]) -> None:
        session = handle.runner.require_session()
        execution = session.deployment.execution
        self.prepare_tokens = session.prepare_tokens
        self.prepare_input = session.prepare_input
        self.token_checked = 0
        self.prepare_sites = None if session.model_sites is None else session.model_sites.prepare
        self.prepare_overlay = None if session.forward_overlay is None else session.forward_overlay.prepare
        self.execute = execution.execute_anchor
        self.execute_part = execution.execute_part
        self.steps: dict[int, set[int]] = {}
        self.checked: set[int] = set()
        self.input_steps: dict[int, set[int]] = {}
        self.input_checked: set[int] = set()

    @contextmanager
    def scope(self, context: StateTaskContext[TensorFrame]) -> Generator[None]:
        rows = context.selection.rows
        if not rows and context.value is not None:
            rows = context.value.step.rows.rows
        steps = self.steps.setdefault(len(rows), set())
        if rows and all(row.role is HiddenPhase.DECODE for row in rows):
            steps.add(context.invocation_id)
        enabled = len(steps) > 2 and context.invocation_id in steps
        mode = torch.cuda.get_sync_debug_mode()
        try:
            if enabled:
                torch.cuda.set_sync_debug_mode("error")
                self.checked.add(context.invocation_id)
            yield
        except RuntimeError as error:
            locations = "; ".join(
                f"{frame.filename}:{frame.lineno} ({frame.name})" for frame in extract_tb(error.__traceback__)[-5:]
            )
            raise RuntimeError(f"{error}; {locations}") from error
        finally:
            torch.cuda.set_sync_debug_mode(mode)

    def tokens(
        self, history: torch.Tensor, indices: tuple[int, ...], starts: torch.Tensor, positions: torch.Tensor, rows: int
    ) -> None:
        """Include the native history gather in the existing steady-decode sync check."""
        mode = torch.cuda.get_sync_debug_mode()
        try:
            if rows == len(indices) and len(self.steps.get(rows, set())) > 2:
                torch.cuda.set_sync_debug_mode("error")
                self.token_checked += 1
            self.prepare_tokens(history, indices, starts, positions, rows)
        finally:
            torch.cuda.set_sync_debug_mode(mode)

    def prepare(
        self, rows: PackedPhaseRowBatch, context: StateTaskContext[TensorFrame], directory: RequestStateDirectory
    ) -> None:
        """Include uploads and state staging between anchors in the model-site sync check."""
        assert self.prepare_sites is not None
        with self.scope(context):
            self.prepare_sites(rows, context, directory)

    def input(self, sampler: PackedSamplerRowBatch) -> None:
        """Check native step staging even when a stateless hook needs no task context."""
        rows = sampler.row_batch
        mode = torch.cuda.get_sync_debug_mode()
        try:
            steps = self.input_steps.setdefault(len(rows.rows), set())
            if rows.rows and all(row.role is HiddenPhase.DECODE for row in rows.rows):
                steps.add(rows.step_id)
            if len(steps) > 2 and rows.step_id in steps:
                torch.cuda.set_sync_debug_mode("error")
                self.input_checked.add(rows.step_id)
            self.prepare_input(sampler)
        finally:
            torch.cuda.set_sync_debug_mode(mode)

    def overlay(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext[TensorFrame],
        directory: RequestStateDirectory,
        transition: ProjectionTransition | None = None,
    ) -> None:
        """Include graph port and private-page staging in steady decode sync checks."""
        assert self.prepare_overlay is not None
        with self.scope(context):
            self.prepare_overlay(rows, context, directory, transition)

    def anchor(self, context: StateTaskContext[TensorFrame], anchor: AnchorId) -> None:
        with self.scope(context):
            self.execute(context, anchor)

    def part(
        self, context: StateTaskContext[TensorFrame], value: StepInput, task: TaskId, *, complete: bool
    ) -> ProjectionResiduals:
        with self.scope(context):
            return self.execute_part(context, value, task, complete=complete)


def _tokens(handle: VllmLlm[TensorFrame, TensorOutput], temperature: float = 0) -> tuple[tuple[int, ...], ...]:
    params = SamplingParams(temperature=temperature, seed=17, max_tokens=16, ignore_eos=True)
    return tuple(tuple(output.outputs[0].token_ids) for output in handle.generate(CHECK_PROMPTS, params))


@dataclass
class _Generation:
    """Token IDs and terminal request states, synchronized only by explicit diagnostics."""

    tokens: tuple[tuple[int, ...], ...] = ()
    states: dict[int, tuple[torch.Tensor, ...]] = field(default_factory=lambda: dict[int, tuple[torch.Tensor, ...]]())
    taps: dict[tuple[PackedHiddenResourceBinding, int, int], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[PackedHiddenResourceBinding, int, int], torch.Tensor]()
    )

    projections: dict[tuple[Projection, int, int], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=lambda: dict[tuple[Projection, int, int], tuple[torch.Tensor, torch.Tensor]]()
    )

    def operands(self, value: StepInput, reference: _Generation | None) -> StepInput:
        """Hold projection inputs fixed per token when the scheduler rebuilds a model-state history."""
        operands: list[ProjectionOperands] = []
        for operand in value.operands:
            rows = tuple(
                row for row in value.rows.rows if row.identity == operand.identity and row.phase is operand.phase
            )
            assert len(rows) == len(operand.hidden)
            inputs: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            for index, row in enumerate(rows):
                key = operand.target, row.identity.prompt_index % len(CHECK_PROMPTS), row.token_offset
                pair = operand.hidden[index], operand.desired[index]
                if reference is not None:
                    pair = reference.projections.get(key, pair)
                self.projections[key] = pair[0].clone(), pair[1].clone()
                inputs.append(pair[0])
                targets.append(pair[1])
            operands.append(replace(operand, hidden=torch.stack(inputs), desired=torch.stack(targets)))
        return replace(value, operands=tuple(operands))

    def hidden(self, value: StepInput, reference: _Generation | None) -> StepInput:
        """Keep per-token tap inputs fixed, as in the CPU schedule script, despite base-model rounding."""
        hidden = value.hidden
        assert hidden is not None
        for tap in hidden.taps:
            # Preserve the producer pool addresses that the workload CUDA graphs captured.
            storage = tap.storage
            for row in value.rows.rows:
                if row.recomputed:
                    continue
                if row.phase is HiddenPhase.PREFILL and row.token_offset != row.prompt_length - 1:
                    continue
                key = tap.resource, row.identity.prompt_index % len(CHECK_PROMPTS), row.token_offset
                slot = int(tap.slots[row.packed_row_index])
                if reference is not None and key in reference.taps:
                    storage[slot].copy_(reference.taps[key])
                self.taps[key] = storage[slot].clone()
        return replace(
            value,
            hidden_readiness=CudaStreamReadiness(torch.cuda.current_stream()),
        )

    def compare(self, replayed: _Generation, contract: str = "preempted") -> None:
        if self.tokens != replayed.tokens:
            raise ContractError(
                (
                    f"[{contract}] greedy tokens changed: {self.tokens} -> {replayed.tokens}; "
                    "rerun with VLLM_BATCH_INVARIANT=1 to exclude batch-width rounding, "
                    "then report a pllm recompute bug if they still differ",
                )
            )
        self.compare_states(replayed, contract)

    def compare_states(self, replayed: _Generation, contract: str) -> None:
        """Compare fixed-input state with dtype tolerances; floating transforms may change greedy tokens."""
        if self.states.keys() != replayed.states.keys():
            raise ContractError((f"[{contract}] request states disappeared; report a pllm retirement bug",))
        for request, fields in self.states.items():
            for index, (expected, actual) in enumerate(zip(fields, replayed.states[request], strict=True)):
                try:
                    torch.testing.assert_close(actual, expected)
                except AssertionError as error:
                    raise ContractError(
                        (f"[{contract}] request {request}, field {index}: {error}; report a pllm recompute bug",)
                    ) from error


def _state_tokens(handle: VllmLlm[TensorFrame, TensorOutput], reference: _Generation | None = None) -> _Generation:
    session = handle.runner.require_session()
    execution = session.deployment.execution
    retire = execution.retire
    execute_part = execution.execute_part
    result = _Generation()

    def snapshot(identities: tuple[RequestIdentity, ...], *, wait: bool = False) -> tuple[StateRef, ...]:
        for identity in identities:
            result.states[identity.prompt_index % len(CHECK_PROMPTS)] = tuple(
                tensor.detach().cpu()
                for ref in execution.latest((identity,))
                for tensor in execution.snapshot(ref, torch.cuda.current_stream(execution.device))
            )
        return retire(identities, wait=wait)

    def part(
        context: StateTaskContext[TensorFrame], value: StepInput, task: TaskId, *, complete: bool
    ) -> ProjectionResiduals:
        if value.overlay_updates and reference is not None:
            # Captured math has already run. Replay the fixed diagnostic operands from the
            # input pages, just as eager projection interception does before the authored step.
            math = execution.math.parameter_overlay
            assert math is not None
            before = read_batch(context, "input", math.resource).for_resource(math.resource)
            private = read_batch(context, "private", math.resource).for_resource(math.resource)
            for field in range(len(math.buffers) * len(math.targets)):
                pool, sources = before.field_slots(field)
                _, destinations = private.field_slots(field)
                for source, destination in zip(sources, destinations, strict=True):
                    if source != destination:
                        pool[destination].copy_(pool[source])
            value = replace(value, overlay_updates={})
        return execute_part(context, result.operands(value, reference), task, complete=complete)

    with patch.object(execution, "retire", snapshot), patch.object(execution, "execute_part", part):
        if session.hidden_taps is None:
            result.tokens = _tokens(handle)
        else:
            publish = session.hidden_taps.publish

            def hidden(value: StepInput, resources: tuple[PackedHiddenResourceBinding, ...]) -> StepInput:
                return result.hidden(
                    publish(value, resources), None if session.custom_forward is not None else reference
                )

            with patch.object(session.hidden_taps, "publish", hidden):
                result.tokens = _tokens(handle)
    return result


class _Eviction:
    """Make the real KV allocator fail once, then let vLLM evict and recompute normally."""

    def __init__(self) -> None:
        self.request: Request | None = None
        self.requests: dict[str, Request] = {}

    def wrap(
        self, allocate: Callable[Concatenate[Request, _Allocation], KVCacheBlocks | None]
    ) -> Callable[Concatenate[Request, _Allocation], KVCacheBlocks | None]:
        def allocation(request: Request, *args: _Allocation.args, **kwargs: _Allocation.kwargs) -> KVCacheBlocks | None:
            self.requests[request.request_id] = request
            if self.request is None and request.num_output_tokens >= 4:
                self.request = request
                return None
            return allocate(request, *args, **kwargs)

        return allocation


def _preemption(handle: VllmLlm[TensorFrame, TensorOutput], reference: _Generation) -> _Generation:
    manager = handle.require_llm().llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
    eviction = _Eviction()
    with patch.object(manager, "allocate_slots", new=eviction.wrap(manager.allocate_slots)):
        result = _state_tokens(handle, reference)
    if eviction.request is None or not any(request.num_preemptions for request in eviction.requests.values()):
        raise ContractError(("[preempted] no KV eviction occurred; report a pllm GPU checker bug",))
    return result


def release_models() -> None:
    """Drain finished diagnostic engines before vLLM samples free memory for the next one."""
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        # Compiled graph caches can retain a closed engine's model tensors.
        torch.compiler.reset()
        cleanup_dist_env_and_memory()
        torch.cuda.synchronize()


def _neutral_tokens(workload: Workload, config: ModelConfig) -> tuple[tuple[tuple[int, ...], ...], ...]:
    with pllm.make_llm(config, workload=workload) as handle:
        return (_tokens(handle), _tokens(handle, 0.8))


@contextmanager
def plain_model(workload: Workload, config: ModelConfig) -> Generator[VllmLlm[None, None]]:
    """Load vanilla vLLM with the workload's checkpoint and graph/compile options."""
    report = explain(workload, config)
    requirements = workload.backend_requirements()
    options: GraphOptions = {} if report.eager_reason else graph_options(config, requirements)
    with VllmLlm[None, None](
        vllm_engine(
            replace(config, execution=ExecutionOptions()),
            {**options, "worker_cls": "auto", **load_options(requirements.checkpoint_names)},
            {},
            cuda_graphs=not bool(report.eager_reason),
        )
    ) as vanilla:
        yield vanilla


def _neutral(workload: Workload, config: ModelConfig) -> None:
    observed = _neutral_tokens(workload, config)
    release_models()
    with plain_model(workload, config) as vanilla:
        expected = tuple(
            tuple(
                tuple(output.outputs[0].token_ids)
                for output in vanilla.generate(
                    CHECK_PROMPTS,
                    SamplingParams(temperature=temp, seed=17, max_tokens=16, ignore_eos=True),
                    use_tqdm=False,
                )
            )
            for temp in (0, 0.8)
        )
    if observed != expected:
        raise ContractError(
            (
                "[neutral] generated token IDs differ from vanilla in the same graph mode; "
                "check that neutral= disables every model/sampler effect, then report a pllm bug if it does",
            )
        )


def _sequence_update(
    handle: VllmLlm[TensorFrame, TensorOutput], batch: CausalTrainBatch[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """Exercise the host-to-worker path and read the complete published optimizer state."""
    handle.generate(CHECK_PROMPTS, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True), batch=batch)
    execution = handle.runner.require_session().deployment.execution
    return tuple(
        tensor.detach().cpu()
        for ref in execution.directory.select(PackedRows.from_rows(())).refs
        for tensor in execution.snapshot(ref, torch.cuda.current_stream())
    )


def _sequence_trial(
    workload: Workload, config: ModelConfig, batch: CausalTrainBatch[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """End the engine's scope before freeing its model and opening the next engine."""
    with pllm.make_llm(config, workload=workload) as handle:
        return _sequence_update(handle, batch)


def _sequence_permutation(workload: Workload, config: ModelConfig) -> None:
    """Fresh engines give both sequence orders identical starting parameters and Adam moments."""
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    labels = torch.tensor([[-100, -100, 3, 4], [-100, 3, 2, -100]])
    batch = CausalTrainBatch(ids, labels, torch.ones_like(ids), torch.tensor([-0.5, 1.5]))
    expected = _sequence_trial(workload, config, batch)
    release_models()
    permuted = map_host_batch(batch, lambda tensor: tensor.flip(0))
    assert isinstance(permuted, CausalTrainBatch)
    actual = _sequence_trial(workload, config, permuted)
    try:
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=1e-4, atol=1e-5)
    except (AssertionError, ValueError) as error:
        raise ContractError(
            (f"[sequence permutation] sequences and weights permuted together changed the update: {error}",)
        ) from error


def _workload_checks(workload: Workload, config: ModelConfig) -> CheckResult:
    passed: list[str] = []
    skipped: list[CheckSkip] = []
    failures: list[str] = []
    report = explain(workload, config)
    with pllm.make_llm(config, workload=workload) as handle:
        try:
            check_probe(workload, config, handle)
            passed.append("explain/probe")
        except ContractError as error:
            failures.extend(error.failures)
        sync = SynchronizationCheck(handle)
        sites = handle.runner.require_session().model_sites
        overlay = handle.runner.require_session().forward_overlay
        execution = handle.runner.require_session().deployment.execution
        try:
            with (
                patch.object(execution, "execute_anchor", sync.anchor),
                patch.object(execution, "execute_part", sync.part),
                patch.object(handle.runner.require_session(), "prepare_tokens", sync.tokens),
                patch.object(sites, "prepare", sync.prepare) if sites is not None else nullcontext(),
                patch.object(overlay, "prepare", sync.overlay) if overlay is not None else nullcontext(),
                patch.object(handle.runner.require_session(), "prepare_input", sync.input),
            ):
                baseline = _state_tokens(handle)
            if (
                max(len(sync.checked), len(sync.input_checked)) < 8
                or workload.backend_requirements().input_token_history >= 0
                and sync.token_checked < 8
            ):
                raise ContractError(("[host syncs] fewer than eight steady decode steps; report a pllm checker bug",))
            passed.append("host syncs: eight steady decode steps")
        except (RuntimeError, ContractError) as error:
            failures.append(f"[host syncs] {error}; remove host reads/synchronization from the named per-step path")
            baseline = None
        if report.preemption == "reset" and report.recompute_rejection:
            skipped.append(CheckSkip("preempted", report.recompute_rejection))
        elif (
            any(state.scope == "instance" for state in report.states)
            and handle.attachment.plan.semantic_graph.trains_on_batch
        ):
            skipped.append(CheckSkip("preempted", "instance training needs a caller-provided host batch"))
        else:
            try:
                with (
                    patch.object(execution, "execute_anchor", sync.anchor),
                    patch.object(execution, "execute_part", sync.part),
                ):
                    replayed = _preemption(handle, baseline or _Generation())
                if baseline is not None:
                    baseline.compare(replayed)
                    passed.append("preempted: fixed per-token inputs, token IDs and terminal request state")
            except (RuntimeError, ContractError) as error:
                failures.append(f"[preempted] {error}")
    if (
        HiddenPhase.DECODE in workload.backend_requirements().exact_phases
        and not report.eager_reason
        and baseline is not None
    ):
        release_models()
        try:
            with pllm.make_llm(replace(config, cuda_graphs=False), workload=workload) as eager:
                baseline.compare_states(_state_tokens(eager, baseline), "exact decode")
            passed.append("exact decode: eager terminal request state on fixed per-token inputs")
        except (RuntimeError, ContractError) as error:
            failures.append(f"[exact decode] {error}")
    if failures:
        raise ContractError(tuple(failures))
    return CheckResult(tuple(passed), tuple(skipped))


def check_on_gpu(workload: Workload, config: ModelConfig, *, neutral: Workload | None = None) -> CheckResult:
    """Load real engines to check planning, eight steady decode anchors, replay and neutral sampling.

    Neutral checks compare greedy and seeded sampled token IDs bitwise in the workload's graph mode.
    Forced KV allocation failure exercises vLLM's real eviction/recompute path. Tap and projection inputs are recorded
    and replayed per token, as in the CPU script, to exclude base-model rounding differences; custom forwards
    regenerate their readouts. Untrained early exits may need VLLM_BATCH_INVARIANT=1 to exclude
    batch-width rounding during eviction. Causal batches also check that permuting sequences and
    weights preserves an AdamW update.
    Workloads with custom ops replay CPU-script method and cell inputs on CUDA and compare outputs
    with dtype-default tolerances. The fake decoder excludes model arithmetic from that comparison.
    Declared replay rejections are reported as explicit skips.
    Run in an isolated process with a CUDA device; multiprocessing and parallel workers are unsupported.
    """
    if (
        config.multiprocessing
        or max(config.tensor_parallel_size, config.pipeline_parallel_size, config.data_parallel_size) > 1
    ):
        raise ValueError("check_on_gpu needs multiprocessing=False and all parallel sizes=1")
    if config.max_num_seqs < 2 or config.max_model_len < 32:
        raise ValueError("check_on_gpu needs max_num_seqs >= 2 and max_model_len >= 32 for its fixed script")
    passed: list[str] = []
    skipped: list[CheckSkip] = []
    failures: list[str] = []
    if explain(workload, config).custom_ops:
        try:
            replay = check_contracts(workload, config, device_equivalence=True)
            unexecuted = {"CPU schedules", "adaptive decoder"}
            if "custom-residual method math / CPU schedules" not in replay.passed:
                unexecuted.add("model-site execution")
            missing = tuple(skip for skip in replay.skipped if skip.property in unexecuted)
            if missing:
                skipped.extend(CheckSkip("device equivalence", skip.reason) for skip in missing)
            else:
                passed.append("device equivalence: CPU check script method outputs replayed on CUDA")
        except (AssertionError, RuntimeError, ValueError, TypeError) as error:
            failures.append(f"[device equivalence] {error}")
    release_models()
    try:
        result = _workload_checks(workload, config)
        passed.extend(result.passed)
        skipped.extend(result.skipped)
    except ContractError as error:
        failures.extend(error.failures)
    finally:
        release_models()
    if any(member.host_batch for member in explain(workload, config).members):
        if config.execution.max_host_training_rows < 2:
            skipped.append(CheckSkip("sequence permutation", "set max_host_training_rows >= 2"))
        else:
            try:
                _sequence_permutation(workload, config)
                passed.append("sequence permutation: aligned sequences and weights preserve the AdamW update")
            except ContractError as error:
                failures.extend(error.failures)
            finally:
                release_models()
    if neutral is None:
        skipped.append(CheckSkip("neutral", "pass neutral= with the author's disabled settings"))
    else:
        try:
            _neutral(neutral, config)
            passed.append("neutral: greedy and seeded sampling bitwise")
        except ContractError as error:
            failures.extend(error.failures)
        finally:
            release_models()
    if failures:
        raise ContractError(tuple(failures))
    return CheckResult(tuple(passed), tuple(skipped))
