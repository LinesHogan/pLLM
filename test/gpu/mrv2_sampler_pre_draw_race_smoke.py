#!/usr/bin/env python3
"""GPU smoke: the MRv2 sampler wrapper fences side work before consumption.

A signed side-stream Region (window ``model.execute`` -> ``sampler.pre_draw``)
runs a deliberately long kernel chain and then publishes its result into a
shared CUDA buffer.  The real ``make_mrv2_sample_wrapper`` path is driven with
a provider whose ``prepare_step`` reads that buffer to build its intervention
logits.  Because the wrapper dispatches ``sampler.pre_draw`` (which joins the
side stream onto the main stream) *before* ``prepare_step``, the provider
observes the completed buffer; the old (racy) order read it while the side
chain was still running and produced a different draw.

Asserts on hardware:

- the producing Region is signed ``side_forward`` and runs off the default
  stream;
- ``dispatch_sampler_pre_draw`` runs before ``provider.prepare_step`` on the
  real wrapper path;
- the sampled tokens are derived from the *completed* buffer contents.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch

from tllm.backends.vllm import sampler_mrv2
from tllm.runtime import residual_runtime
from tllm.runtime.sampler_bridge.provider import SamplerInterventionStage
from tllm.runtime.sampler_bridge.types import CandidateModifierState
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_execution import WorkloadExecutionBindings
from tllm.runtime.workload_bindings import RuntimeOutputBinding, RuntimePortRegistry
from tllm.schedule import (
    BackendObservableEvent,
    CompileOptions,
    MainWorkloadTimeline,
    TimelineAnchor,
    WorkloadCompilerCoordinator,
)
from tllm.workload import Between, declare_workload, result, sink, task, temporary, workload

_VOCAB = 8
_ROWS = 2
_WINNER = 5


def _timeline() -> MainWorkloadTimeline:
    """The production vLLM timeline shape (schema v2, signed backend events).

    ``sampler.pre_draw`` advances zero Region nodes for this workload — the
    side-stream join owed to the sampler provider must still happen there.
    """
    return MainWorkloadTimeline(
        backend_id="vllm",
        backend_version="0.27.1",
        runner_family="mrv2",
        graph_mode="eager",
        anchors=(
            TimelineAnchor("step.begin", 0, backend_events=()),
            TimelineAnchor(
                "prepare.pre",
                1,
                backend_events=(BackendObservableEvent.PREPARE_INPUTS_PRE,),
            ),
            TimelineAnchor(
                "prepare.post",
                2,
                backend_events=(BackendObservableEvent.PREPARE_INPUTS_POST,),
            ),
            TimelineAnchor(
                "model.execute",
                3,
                backend_events=(BackendObservableEvent.EXECUTE_MODEL_PRE,),
            ),
            TimelineAnchor(
                "logits.post",
                4,
                visible_values=("model.hidden", "model.logits"),
                backend_events=(BackendObservableEvent.EXECUTE_MODEL_POST,),
            ),
            TimelineAnchor(
                "sampler.pre_draw",
                5,
                visible_values=("model.logits",),
                backend_events=(
                    BackendObservableEvent.SAMPLE_TOKENS_PRE,
                    BackendObservableEvent.SAMPLER_PRE_DRAW,
                ),
            ),
            TimelineAnchor(
                "token.commit",
                6,
                visible_values=("sample.token",),
                backend_events=(BackendObservableEvent.SAMPLE_TOKENS_POST,),
            ),
            TimelineAnchor(
                "step.end",
                7,
                backend_events=(BackendObservableEvent.SAMPLE_TOKENS_POST,),
            ),
            TimelineAnchor("next.prepare", 8, backend_events=()),
            TimelineAnchor("generate.end", 9, backend_events=()),
        ),
        regions=(),
        static_headroom_bytes=64,
        capabilities=("eager_only",),
        schema_version=2,
    )


class _Observer:
    def __init__(self, device: torch.device) -> None:
        self.buffer = torch.zeros((_ROWS, _VOCAB), device=device)
        self.on_side_stream: bool | None = None
        self.outputs: list[tuple[int, torch.Tensor]] = []


class _RaceProvider:
    """Reads the side-Region buffer to build a pre-filter intervention."""

    def __init__(self, observer: _Observer) -> None:
        self._observer = observer

    def is_active(self) -> bool:
        return True

    def intervention_stage(self) -> SamplerInterventionStage:
        return SamplerInterventionStage.PRE_FILTER

    def prepare_step(self, view: Any) -> CandidateModifierState:
        rows = int(view.logits.shape[0])
        # Main-stream read of the buffer the side Region publishes.  Without
        # the pre-draw join this races the side chain and sees zeros.
        total = self._observer.buffer.sum()
        reweighted = view.logits.clone()
        reweighted[:, _WINNER] += total
        return CandidateModifierState(
            beta=1.0,
            backend="pre_filter_dense",
            affected_row_ids=torch.arange(rows, device=view.logits.device, dtype=torch.long),
            pred_hidden=view.source_hidden.detach(),
            lm_head_weight=torch.empty((0, 0), device=view.logits.device, dtype=view.logits.dtype),
            lm_head_bias=None,
            precomputed_dense_logits=(2.0 * view.logits) - reweighted,
        )

    def after_sample(self, view: Any, sampled_token_ids: torch.Tensor) -> None:
        return None


class _SamplerStates:
    def __init__(self, device: torch.device) -> None:
        self.temperature = SimpleNamespace(gpu=torch.ones((_ROWS,), device=device))
        self.seeds = SimpleNamespace(gpu=torch.zeros((_ROWS,), device=device, dtype=torch.long))

    def get_top_k_top_p(self, expanded_idx_mapping: Any, idx_mapping_np: Any) -> tuple[None, None]:
        return None, None


class _SamplerDouble:
    """Minimal MRv2 sampler protocol double (temperature == 1 passthrough)."""

    def __init__(self, device: torch.device) -> None:
        self.logprobs_mode = "raw_logprobs"
        self.use_fp64_gumbel = False
        self.sampling_states = _SamplerStates(device)

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        *args: Any,
        skip_top_k_top_p: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        return logits / self.sampling_states.temperature.gpu.unsqueeze(1)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    observer = _Observer(device)

    @workload(id="gpu.mrv2-pre-draw-race-smoke")
    class SideProducer:
        @task(
            outputs={"value": temporary()},
            timing=Between("model.execute", "logits.post"),
        )
        def produce(self) -> torch.Tensor:
            return torch.full((_ROWS, _VOCAB), 3.0, device=device)

        @task(
            inputs={"value": result("produce", "value")},
            outputs={"output": sink("answer")},
            timing=Between("logits.post", "sampler.pre_draw"),
        )
        def consume(self, value: torch.Tensor) -> torch.Tensor:
            observer.on_side_stream = (
                torch.cuda.current_stream().cuda_stream
                != torch.cuda.default_stream().cuda_stream
            )
            # Keep the side stream busy so an unfenced main-stream read loses.
            busy = torch.randn(2048, 2048, device=device)
            for _ in range(50):
                busy = busy @ busy * 1e-6
            observer.buffer.copy_(value + 4.0)
            return observer.buffer.clone()

    declaration = declare_workload(SideProducer())
    compilation = WorkloadCompilerCoordinator(
        (declaration,),
        options=CompileOptions(
            memory_budget_bytes=1 << 20,
            compile_policy="eager",
            allow_side_stream=True,
        ),
    ).compile(_timeline())
    regions = compilation.decision.execution_regions
    assert len(regions) == 2, f"expected two Regions, got {len(regions)}"
    by_task = {region.tasks[0]: region for region in regions}
    region = by_task["consume"]
    assert region.stream is not None and region.stream.value == "side_forward", (
        f"consume Region must be signed side_forward, got {region.stream}"
    )

    runtime = ResidualRuntimeState()
    runtime.decode_count = _ROWS
    runtime.source_resolved_path = "layers.0"
    runtime.tap_decode_hidden["layers.0"] = torch.ones((_ROWS, 4), device=device)
    runtime.decode_prompt_idx_tensor = torch.tensor([0, 1])
    runtime.decode_sample_idx_tensor = torch.tensor([0, 1])
    runtime.decode_prompt_idxs = [0, 1]
    runtime.decode_sample_idxs = [0, 1]
    runtime.decode_request_ids = ["req-a", "req-b"]
    runtime.event_step_id = 5
    provider = _RaceProvider(observer)
    runtime.consumer = SimpleNamespace(sampler_modifier_provider=lambda: provider)

    bindings = WorkloadExecutionBindings(
        declarations=(declaration,),
        ports=RuntimePortRegistry(
            outputs={
                "answer": RuntimeOutputBinding(
                    lambda value, step_id: observer.outputs.append(
                        (step_id, value)
                    )
                )
            }
        ),
    )
    runner = SimpleNamespace(model=object())
    sampler = _SamplerDouble(device)
    wrapped = sampler_mrv2.make_mrv2_sample_wrapper(
        sampler=sampler,
        runtime=runtime,
        runner=runner,
        orig_sample=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("orig path must not run while the provider is active")
        ),
    )

    order: list[str] = []
    real_dispatch = sampler_mrv2.dispatch_sampler_pre_draw

    def _recording_dispatch(**kwargs: Any) -> None:
        order.append("dispatch")
        real_dispatch(**kwargs)

    raw_logits = torch.zeros((_ROWS, _VOCAB), device=device)

    def _fake_gumbel(filtered: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return filtered.argmax(dim=-1)

    with mock.patch.object(residual_runtime, "RUNTIME", runtime):
        residual_runtime.publish_workload_compilation(compilation, bindings=bindings)
        residual_runtime.prepare_physical_workgraph_step(5)
        assert residual_runtime.advance_workload_anchor(by_task["produce"].placement_anchor)
        assert residual_runtime.advance_workload_anchor(region.placement_anchor)

        with (
            mock.patch.object(sampler_mrv2, "dispatch_sampler_pre_draw", _recording_dispatch),
            mock.patch(
                "vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p",
                side_effect=lambda logits, top_k, top_p: logits,
            ),
            mock.patch(
                "vllm.v1.worker.gpu.sample.gumbel.gumbel_sample",
                side_effect=_fake_gumbel,
            ),
        ):
            provider_prepare = provider.prepare_step

            def _recording_prepare(view: Any) -> CandidateModifierState:
                order.append("provider")
                return provider_prepare(view)

            provider.prepare_step = _recording_prepare  # type: ignore[method-assign]
            sampled, _ = wrapped(
                raw_logits,
                torch.arange(_ROWS, device=device, dtype=torch.long),
                None,
                torch.arange(_ROWS, device=device, dtype=torch.long),
                torch.zeros(_ROWS, device=device, dtype=torch.long),
                torch.arange(_ROWS, device=device, dtype=torch.long),
            )

        execution = runtime.workload_execution
        assert execution is not None
        execution.finish_step(5)

    torch.cuda.synchronize()
    assert order == ["dispatch", "provider"], f"pre-draw fence must precede consumption: {order}"
    assert observer.on_side_stream is True, "consume Region must run on a side stream"
    expected = torch.full((_ROWS,), _WINNER, device=device, dtype=torch.long)
    assert torch.equal(sampled.to(device=device, dtype=torch.long), expected), (
        f"provider consumed an incomplete side-Region buffer: sampled {sampled.tolist()}"
    )
    assert len(observer.outputs) == 1
    print("mrv2 pre-draw race smoke OK: fence precedes provider consumption on hardware")


if __name__ == "__main__":
    main()
