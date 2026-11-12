"""Real-engine contract checks for workload authors, exposed by pllm.testing.check_on_gpu."""

from __future__ import annotations

# Diagnostics deliberately inspect the installed session and in-process engine.
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from multiprocessing import get_context
from multiprocessing.connection import Connection
from traceback import format_exc
from typing import Concatenate, ParamSpec
from unittest.mock import patch

import torch
import vllm.envs as vllm_envs
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

import pllm
from pllm.backends.vllm.checkpoint import load_options
from pllm.backends.vllm.compat import enable_batch_invariance
from pllm.backends.vllm.model_shape import ModelShape
from pllm.backends.vllm.overlay_graph import GraphOptions, graph_options
from pllm.backends.vllm.runtime import VllmLlm, probe_backend, vllm_engine
from pllm.config import ExecutionOptions, ModelConfig
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.host_batch import CausalTrainBatch, map_host_batch
from pllm.contracts.semantic_graph import StateEffectKind, StateScope
from pllm.execution.step_input import StepInput
from pllm.execution.trace import TraceEventKind, TraceSelection
from pllm.explanation import explain, explain_facts
from pllm.testing_contracts import CheckResult, CheckSkip, ContractError, affects_model, check_contracts
from pllm.workload import Workload
from pllm.workload_transport import decode_workload, encode_workload

_Allocation = ParamSpec("_Allocation")

# Sequence continuations avoid near-tied prose tokens whose greedy choice changes with vLLM batch geometry.
CHECK_PROMPTS = ["Continue the sequence: 1, 2, 3,", "Continue the alphabet: A, B, C,"]


def check_probe(workload: Workload, config: ModelConfig, handle: VllmLlm) -> None:
    """Compare config-only and loaded-model reports through the same pure computation."""
    expected = explain(workload, config)
    runner = handle.runner
    requirements = workload.backend_requirements()
    shape = ModelShape.of_model(runner.decoder, requirements.parameter_targets, runner.dtype)
    facts = probe_backend(requirements, runner)
    # The installed state already consumed memory; replay planning against its load-time capacity.
    assert handle.compile_facts is not None
    facts = replace(facts, state_page_budget_bytes=handle.compile_facts.state_page_budget_bytes)
    actual = explain_facts(workload, config, shape, facts, runner.hidden_layouts)
    failures: list[str] = []
    if expected != actual:
        failures.append("[explain/probe] config and loaded-model costs differ; report a pLLM planning bug")
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
        failures.append("[explain/graphs] engine graph mode differs from explain; report a pLLM admission bug")
    if sum(state.reserved_bytes for state in actual.states) != handle.attachment.plan.states.reserved_bytes:
        failures.append("[explain/state] allocated state reservations differ; report a pLLM allocation bug")
    if not requirements.parameter_targets:
        plan = handle.attachment.plan
        if (
            actual.peak_memory_bytes != plan.schedule.peak_memory_bytes
            or tuple(anchor.reserved_bytes for anchor in actual.anchors) != plan.timeline.reserved_bytes
        ):
            failures.append("[explain/reservations] installed plan differs from the probe; report a pLLM planning bug")
    if failures:
        raise ContractError(tuple(failures))


def _tokens(handle: VllmLlm, temperature: float = 0) -> tuple[tuple[int, ...], ...]:
    params = SamplingParams(temperature=temperature, seed=17, max_tokens=16, ignore_eos=True)
    return tuple(tuple(output.outputs[0].token_ids) for output in handle.generate(CHECK_PROMPTS, params))


# Three key groups of different lengths; batched together they are chunked and mix prefill with decode rows.
_KEY_GROUPS = (*CHECK_PROMPTS, "Continue the even numbers: 2, 4, 6,")
_REVERSED = (2, 1, 0)
_TRACE = TraceSelection(states=True, max_events=65536)


@dataclass(frozen=True)
class _Run:
    """What one generate call let each (prompt, sample) observe: token IDs, exports and terminal request state."""

    tokens: dict[tuple[int, int], tuple[int, ...]]
    values: dict[tuple[int, int, str], torch.Tensor]

    def merged(self, other: _Run) -> _Run:
        return _Run(self.tokens | other.tokens, self.values | other.values)

    def differences(self, other: _Run, contract: str, cause: str) -> list[str]:
        """Token IDs bitwise; float values within dtype tolerance, since chunking reorders a request's own sums. An
        export row present in one run only is not compared: max_rows drops rows of crowded steps."""
        failures = [
            f"[{contract}] prompt {prompt} sample {sample}: token IDs differ from token "
            f"{next(index for index, (a, b) in enumerate(zip(tokens, theirs)) if a != b)}; {cause}"
            for (prompt, sample), tokens in self.tokens.items()
            if tokens != (theirs := other.tokens[prompt, sample])
        ]
        for key in self.values.keys() | other.values.keys():
            prompt, sample, name = key
            if name.startswith("export ") and not (key in self.values and key in other.values):
                continue
            try:
                assert key in self.values and key in other.values, "missing in one run"
                torch.testing.assert_close(other.values[key], self.values[key])
            except AssertionError as error:
                failures.append(
                    f"[{contract}] prompt {prompt} sample {sample}, {name}: {str(error).splitlines()[0]}; {cause}"
                )
        return failures


def _run(handle: VllmLlm, prompts: tuple[int, ...], params: SamplingParams) -> _Run:
    """Generate the key groups `prompts` (indices into _KEY_GROUPS, also their prompt identities)."""
    seen = max((event.sequence for event in handle.trace_snapshot().events), default=0)
    outputs = handle.generate([_KEY_GROUPS[index] for index in prompts], params, prompt_indices=prompts)
    tokens = {
        (prompt, sample): tuple(output.token_ids)
        for prompt, request in zip(prompts, outputs, strict=True)
        for sample, output in enumerate(request.outputs)
    }
    trace = handle.trace_snapshot()
    assert not trace.dropped
    values: dict[tuple[int, int, str], torch.Tensor] = {}
    for event in trace.events:
        if event.sequence <= seen or event.tensor_data is None:
            continue
        if event.kind is TraceEventKind.RETIRE:
            (identity,) = event.identities
            values[identity.prompt_index, identity.sample_index, event.tensor_name or ""] = event.tensor()
        else:
            for identity, position, value in zip(event.identities, event.positions, event.tensor(), strict=True):
                values[identity.prompt_index, identity.sample_index, f"export {event.tensor_name} @{position}"] = value
    return _Run(tokens, values)


def _isolation_skip(plan: PlanBundle) -> str | None:
    """Why a request's outputs may legitimately depend on the requests generated with it."""
    graph = plan.semantic_graph
    written = {
        effect.resource for task in graph.tasks for effect in task.effects if effect.kind is not StateEffectKind.READ
    }
    shared = sorted(
        str(state.resource)
        for state in graph.states
        if state.scope is StateScope.INSTANCE and state.resource in written
    )
    if shared:
        return (
            f"InstanceState {', '.join(shared)} is written while generating; solo and batched runs train it differently"
        )
    if any(task.read_distance for task in plan.schedule.tasks):
        return "publish_lag reads see versions by step count, which batching changes"
    return None


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


def _preemption(handle: VllmLlm, params: SamplingParams) -> _Run:
    manager = handle.require_llm().llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
    eviction = _Eviction()
    with patch.object(manager, "allocate_slots", new=eviction.wrap(manager.allocate_slots)):
        result = _run(handle, _REVERSED, params)
    if eviction.request is None or not any(request.num_preemptions for request in eviction.requests.values()):
        raise ContractError(("[preempted] no KV eviction occurred; report a pLLM GPU checker bug",))
    return result


@contextmanager
def plain_model(workload: Workload, config: ModelConfig) -> Generator[VllmLlm]:
    """Load vanilla vLLM with the workload's checkpoint and graph/compile options."""
    report = explain(workload, config)
    requirements = workload.backend_requirements()
    options: GraphOptions = {} if report.eager_reason else graph_options(config, requirements)
    llm, hooks = vllm_engine(
        replace(config, execution=ExecutionOptions()),
        {**options, "worker_cls": "auto", **load_options(requirements.checkpoint_names)},
        {},
        cuda_graphs=not bool(report.eager_reason),
    )
    with VllmLlm(llm, compile_hooks=hooks) as vanilla:
        yield vanilla


def _neutral_distribution(workload: Workload, config: ModelConfig) -> str:
    """Allow BF16 fusion rounding: mean/P99 KL <= .001/.01 nats, TV <= .01/.05.

    The 7B RTX 4090 calibration includes neutral attention, activation, residual and
    projection hooks. Full-vocabulary distances expose changes hidden by greedy token IDs.
    """
    from pllm.testing_distribution import check_distribution

    reports: list[str] = []
    for temperature in (0, 0.8):
        result = check_distribution(
            workload,
            config,
            temperature=temperature,
            max_mean_kl=1e-3,
            max_p99_kl=1e-2,
            max_mean_tv=1e-2,
            max_p99_tv=5e-2,
        )
        if not result.passed:
            raise ContractError(
                (f"[neutral] teacher-forced full-vocabulary distributions differ from vanilla:\n{result}",)
            )
        kl = (result.kl_plain_to_workload, result.kl_workload_to_plain)
        reports.append(
            f"T={temperature}: KL mean/P99 {max(item.mean for item in kl):.3g}/{max(item.p99 for item in kl):.3g}, "
            f"TV {result.tv.mean:.3g}/{result.tv.p99:.3g}"
        )
    return "neutral: teacher-forced full-vocabulary KL/TV vs vanilla; " + "; ".join(reports)


def _neutral(workload: Workload, config: ModelConfig) -> str:
    if not affects_model(workload, config):
        return "neutral: no model or sampler effects"
    requirements = workload.backend_requirements()
    if not explain(workload, config).eager_reason:
        compilation = graph_options(config, requirements).get("compilation_config")
        if compilation is not None and compilation.backend == "inductor":
            return _neutral_distribution(workload, config)
    with pllm.make_llm(config, workload=workload) as handle:
        observed = _tokens(handle), _tokens(handle, 0.8)
    with plain_model(workload, config) as vanilla:
        expected = _tokens(vanilla), _tokens(vanilla, 0.8)
    if observed != expected:
        raise ContractError(
            (
                "[neutral] generated token IDs differ from vanilla with the same model kernels; "
                "check that neutral= disables every model/sampler effect, then report a pLLM bug if it does",
            )
        )
    return "neutral: greedy and seeded sampling bitwise vs vanilla with the same kernels"


def _sequence_update(handle: VllmLlm, batch: CausalTrainBatch[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Exercise the host-to-worker path and read the complete published optimizer state."""
    handle.generate(CHECK_PROMPTS, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True), batch=batch)
    execution = handle.runner.require_session().deployment.execution
    return tuple(
        tensor.detach().cpu()
        for ref in execution.directory.select(StepInput.of_batch(batch).rows.rows).refs
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


_ISOLATION = (
    "vLLM runs batch-invariant kernels here, so the output depends on the requests generated alongside: "
    "keep workload methods and custom kernels row-local and batch-invariant, else report a pLLM routing bug"
)


def _workload_checks(workload: Workload, config: ModelConfig) -> CheckResult:
    """Run each key group alone, then all of them batched (reversed, chunked, mixed) and once more with a forced
    eviction, and compare what every request observed."""
    passed: list[str] = []
    skipped: list[CheckSkip] = []
    failures: list[str] = []
    report = explain(workload, config)
    requirements = workload.backend_requirements()
    params = SamplingParams(
        n=min(2, requirements.sample_limit or 2), temperature=0.8, seed=17, max_tokens=16, ignore_eos=True
    )
    batched = None
    with pllm.make_llm(config, workload=workload) as handle:
        try:
            check_probe(workload, config, handle)
            passed.append("explain/probe")
        except ContractError as error:
            failures.extend(error.failures)
        if requirements.feeds:
            reason = "a fed state's values follow another process's timing; its protocol tests cover it"
            return CheckResult(tuple(passed), (CheckSkip("GPU schedules", reason),))
        handle.select_trace(_TRACE)
        driver = handle.runner.require_session().driver
        driver.check_host_syncs()
        try:
            solo = _run(handle, (0,), params).merged(_run(handle, (1,), params)).merged(_run(handle, (2,), params))
            batched = _run(handle, _REVERSED, params)
            if driver.steady_steps < 8:
                raise ContractError(("[host syncs] fewer than eight steady decode steps; report a pLLM checker bug",))
            passed.append("host syncs: eight steady decode steps")
        except (RuntimeError, ContractError) as error:
            failures.append(f"[host syncs] {error}; remove host reads/synchronization from the named per-step path")
            solo = None
        isolation = _isolation_skip(handle.attachment.plan)
        if isolation is not None:
            skipped.append(CheckSkip("solo vs batched", isolation))
        elif solo is not None and batched is not None:
            differences = solo.differences(batched, "solo vs batched", _ISOLATION)
            failures.extend(differences)
            if not differences:
                passed.append("solo vs batched: reversed, chunked and mixed batches match each key group alone")
        if report.preemption == "reset" and report.recompute_rejection:
            skipped.append(CheckSkip("preempted", report.recompute_rejection))
        elif (
            any(state.scope == "instance" for state in report.states)
            and handle.attachment.plan.semantic_graph.trains_on_batch
        ):
            skipped.append(CheckSkip("preempted", "instance training needs a caller-provided host batch"))
        elif batched is not None:
            try:
                evicted = _preemption(handle, params)
                differences = batched.differences(evicted, "preempted", "report a pLLM recompute bug")
                failures.extend(differences)
                if not differences:
                    passed.append("preempted: a forced eviction and recompute match the batched run")
            except (RuntimeError, ContractError) as error:
                failures.append(f"[preempted] {error}")
    if HiddenPhase.DECODE in requirements.exact_phases and not report.eager_reason and batched is not None:
        try:
            with pllm.make_llm(replace(config, cuda_graphs=False), workload=workload) as eager:
                eager.select_trace(_TRACE)
                differences = batched.differences(
                    _run(eager, _REVERSED, params), "exact decode", "the captured graph differs from eager execution"
                )
            failures.extend(differences)
            if not differences:
                passed.append("exact decode: eager execution matches the captured graphs")
        except (RuntimeError, ContractError) as error:
            failures.append(f"[exact decode] {error}")
    if failures:
        raise ContractError(tuple(failures))
    return CheckResult(tuple(passed), tuple(skipped))


def check_on_gpu(workload: Workload, config: ModelConfig, *, neutral: Workload | None = None) -> CheckResult:
    """Load real engines to check planning, eight steady decode anchors, schedule independence and neutral sampling.

    The reference is the workload itself: three prompts are generated one at a time, then together in reverse
    order (chunked, prefill mixed with decode), then together again while vLLM's KV allocator fails once so a
    request is evicted and recomputed. Every request's token IDs must match bitwise, its exports and terminal
    request state within dtype tolerance (chunking and recompute reorder a request's own sums). Exact-decode
    workloads also compare eager execution with the captured graphs. The check runs in a child process whose
    engines use vLLM's batch-invariant kernels (VLLM_BATCH_INVARIANT=1); a crash there is reported as a failed
    check with the child's output. Instance states written while generating and publish_lag reads make solo and
    batched runs legitimately differ: that comparison is reported as a skip, as are declared replay rejections.
    Neutral checks compare greedy and seeded token IDs bitwise when model kernels match vanilla.
    Inductor model hooks can change fusion: those compare teacher-forced full-vocabulary distributions
    on greedy and seeded continuations, with mean/P99 KL <= .001/.01 nats (both directions) and
    mean/P99 TV <= .01/.05. Successful checks report the comparison and measured distances.
    Causal batches also check that permuting sequences and weights preserves an AdamW update.
    Workloads with custom ops replay CPU-script method and cell inputs on CUDA and compare outputs
    with dtype-default tolerances. The fake decoder excludes model arithmetic from that comparison.
    Needs a CUDA device; multiprocessing and parallel workers are unsupported.
    """
    if (
        config.multiprocessing
        or max(config.tensor_parallel_size, config.pipeline_parallel_size, config.data_parallel_size) > 1
    ):
        raise ValueError("check_on_gpu needs multiprocessing=False and all parallel sizes=1")
    if config.max_num_seqs < 2 or config.max_model_len < 32:
        raise ValueError("check_on_gpu needs max_num_seqs >= 2 and max_model_len >= 32 for its fixed script")
    context = get_context("spawn")
    received, sent = context.Pipe(duplex=False)
    payload = encode_workload(workload), None if neutral is None else encode_workload(neutral), config
    child = context.Process(target=_check_in_child, args=(*payload, sent), name="pllm-check-on-gpu")
    # The child inherits this environment when it starts; the parent's is restored right after. Graphs compiled
    # with batch-invariant kernels get their own vLLM cache: 0.27.1 fails loading the other mode's cached graphs.
    child_environment = {
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_CACHE_ROOT": os.path.join(vllm_envs.VLLM_CACHE_ROOT, "batch-invariant"),
    }
    previous = {name: os.environ.get(name) for name in child_environment}
    os.environ.update(child_environment)
    try:
        child.start()
    finally:
        for name, value in previous.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value
    sent.close()
    try:
        outcome = received.recv()
    except EOFError:
        child.join()
        outcome = (f"[GPU check process] exited with code {child.exitcode} before reporting; its output shows why",)
    child.join()
    if isinstance(outcome, CheckResult):
        return outcome
    raise ContractError(outcome)


def _check_in_child(workload: str, neutral: str | None, config: ModelConfig, results: Connection) -> None:
    """Run the check with batch-invariant kernels and send its result, or every failure, to the parent."""
    _kernels = enable_batch_invariance()  # registered while the library lives
    try:
        results.send(
            run_gpu_checks(decode_workload(workload), config, None if neutral is None else decode_workload(neutral))
        )
    except ContractError as error:
        results.send(error.failures)
    except BaseException:
        results.send((f"[GPU check process] {format_exc()}",))


def run_gpu_checks(workload: Workload, config: ModelConfig, neutral: Workload | None) -> CheckResult:
    """check_on_gpu's checks in this process, which enabled batch invariance (enable_batch_invariance) before its
    first engine; check_on_gpu runs them in a child process started that way."""
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
    try:
        result = _workload_checks(workload, config)
        passed.extend(result.passed)
        skipped.extend(result.skipped)
    except ContractError as error:
        failures.extend(error.failures)
    if any(member.host_batch for member in explain(workload, config).members):
        if config.execution.max_host_training_rows < 2:
            skipped.append(CheckSkip("sequence permutation", "set max_host_training_rows >= 2"))
        else:
            try:
                _sequence_permutation(workload, config)
                passed.append("sequence permutation: aligned sequences and weights preserve the AdamW update")
            except ContractError as error:
                failures.extend(error.failures)
    if neutral is None:
        skipped.append(CheckSkip("neutral", "pass neutral= with the author's disabled settings"))
    else:
        try:
            passed.append(_neutral(neutral, config))
        except ContractError as error:
            failures.extend(error.failures)
    if failures:
        raise ContractError(tuple(failures))
    return CheckResult(tuple(passed), tuple(skipped))
