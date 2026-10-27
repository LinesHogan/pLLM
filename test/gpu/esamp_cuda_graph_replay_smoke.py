"""Captured CUDA-graph steps on the GPU: ESamp, request memory and instance means with changing sampler inputs.

Reuses the CPU tests' ESamp and distillation oracles on CUDA, then checks that a
captured instance-mean step replays (no Python math) with changing row counts, reused
sampler buffers and publish lags 0-2.
"""

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

import pllm
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding, PackedHiddenSnapshot, PackedHiddenSnapshotBatch
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import SamplerFeedback
from pllm.contracts.semantic_graph import AnchorId, AnchorSymbol, SchedulerTimeline, TimelineAnchor
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.post_sample import PostSampleOverlap
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.step_input import TensorStepInput
from pllm.execution.tensor_task import TensorTask
from test.backend_facts import backend_facts
from test.workloads.test_esamp import _esamp_native_oracle, _ordinary_request_distillation


class MeanFit(pllm.Workload):
    """A shared parameter trained against changing sampler buffers, with masked row means."""

    weights = pllm.InstanceState({"value": (4,)}, optimizer=pllm.AdamW(lr=0.03), dtype=torch.float64)

    @pllm.guide_sampling(beta=0.1)
    def predict(
        self,
        hidden: torch.Tensor = pllm.hidden(0),
        weight: torch.Tensor = pllm.parameter("weights", "value"),
    ) -> torch.Tensor:
        return hidden * weight

    @pllm.train(state="weights", reduction="mean")
    def loss(
        self,
        hidden: torch.Tensor = pllm.hidden(0),
        weight: torch.Tensor = pllm.parameter("weights", "value"),
        target: torch.Tensor = pllm.base_logits(),
        token: torch.Tensor = pllm.sampled_tokens(),
    ) -> torch.Tensor:
        return (hidden * weight - target - token[:, None]).square().mean(1)


def mean_capture(publish_lag: int = 0) -> None:
    """Mean padding and instance gradients agree, even when sampler pointers and row counts change."""
    device = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(device)
    resource = PackedHiddenResourceBinding("0", HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, "model.layers[0]")
    timeline = SchedulerTimeline(
        tuple(
            TimelineAnchor(AnchorSymbol(name), i)
            for i, name in enumerate(("model.execute", "sampler.pre_draw", "token.commit", "next.prepare"))
        ),
        ("compute/cuda@1", "state/dense_page_cuda@1"),
    )
    workload = MeanFit()
    provider = PageBundleProviderFacts("test.mean@1", 6, StateDeviceKind.CUDA)
    facts = backend_facts(4, (resource,), timeline, provider, max_lineages=3)
    facts = workload.specialize(
        facts, pllm.ExecutionOptions(capture_memory_bytes=16 * 1024**2, publish_lag=publish_lag)
    )
    declaration = workload.declaration(facts)
    # One admitted lineage: the instance state.
    plan = compile_page_bundle_plan(
        (declaration.graph, declaration.layouts),
        facts.timeline,
        facts.providers,
        1,
        facts.state_memory_budget_bytes,
        device,
    )
    execution = workload.bind_math(facts, plan).bind(PageBundleArena(plan), AnchorId(1))
    assert plan.timeline.visibility is not None
    overlap = PostSampleOverlap(execution.arena, publish_lag, torch.cuda.Stream(device=device)) if publish_lag else None
    execution.admit((), stream)
    parameter = torch.nn.Parameter(torch.zeros(4, device=device, dtype=torch.float64))
    optimizer = torch.optim.AdamW((parameter,), lr=0.03, weight_decay=0, capturable=True, foreach=True)
    hidden = torch.empty((3, 4), device=device, dtype=torch.float32)
    target_buffer = torch.empty((3, 4), device=device)
    token_buffer = torch.empty(3, device=device, dtype=torch.long)
    for step, count in enumerate((1, 3, 2)):
        hidden.copy_(torch.arange(12, device=device, dtype=torch.float32).view(3, 4) / 10)
        rows = PackedPhaseRowBatch(
            1,
            step,
            count,
            tuple(PackedPhaseRow(i, i, step, HiddenPhase.DECODE, RequestIdentity(str(i), i, 0)) for i in range(count)),
        )
        sampler = PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, i) for i in range(count)))
        value = TensorStepInput(
            rows,
            sampler,
            PackedHiddenSnapshotBatch(rows, (PackedHiddenSnapshot(resource, hidden[:count], hidden),)),
            hidden_readiness=CudaStreamReadiness(stream),
        )
        target = target_buffer[:count].fill_(step * 0.2)
        tokens = token_buffer[:count]
        tokens.copy_(torch.arange(count, device=device, dtype=torch.long))
        oracle = (hidden[:count] * parameter - target - tokens[:, None]).square().mean()
        optimizer.zero_grad()
        oracle.backward()
        optimizer.step()
        guard = (
            patch.object(TensorTask, "differentiate", side_effect=AssertionError("Python math during replay"))
            if step
            else patch.object(TensorTask, "differentiate", TensorTask.differentiate)
        )
        with guard:
            invocation = execution.start_step(value, stream, history=() if overlap is None else overlap.history)
            assert invocation.context.pin.refs[0].version_id == max(0, step - publish_lag)
            for anchor in plan.timeline.anchors:
                execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
                if anchor.ordinal == 1:
                    execution.complete_sampler(invocation, stream, feedback=SamplerFeedback(sampler, target, tokens))
                    if overlap is not None:
                        overlap.copy_inputs(invocation.context)
                        execution.resume_latest(invocation, overlap.stream)
                        overlap.retain(invocation.context.pin.refs)
                        # Model taps and sampler outputs may be overwritten while training reads
                        # its private snapshot, including during captured replay with fewer rows.
                        hidden.fill_(999)
                        target.fill_(999)
                        tokens.fill_(999)
            result = execution.finish_step(invocation)
        if overlap is not None:
            stream.wait_stream(overlap.stream)
        torch.testing.assert_close(result.loss, oracle.detach())
        torch.testing.assert_close(execution.snapshot(result.publication.after_refs[0], stream)[0], parameter)
    # Incomplete sampler coverage is a boundary error, never an out-of-bounds graph read.
    missing = PackedSamplerRowBatch(rows, (PackedSamplerRowLink(0, 0),))
    invocation = execution.start_step(replace(value, sampler_rows=missing), stream)
    with pytest.raises(ValueError):
        for anchor in plan.timeline.anchors:
            execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
            if anchor.ordinal == 1:
                execution.complete_sampler(
                    invocation, stream, feedback=SamplerFeedback(missing, target[:1], tokens[:1])
                )
    torch.testing.assert_close(execution.snapshot(result.publication.after_refs[0], stream)[0], parameter)
    if overlap is not None:
        overlap.clear()
    execution.retire(())


def main() -> None:
    """Run the shared eager/captured step against independent numerical oracles."""
    for normalized in (False, True):
        _esamp_native_oracle(torch.device("cuda", 0), normalized)
    _ordinary_request_distillation()
    for lag in (0, 1, 2):
        mean_capture(lag)
    print("generic step capture: ESamp, request memory, instance mean and changing sampler buffers agree")


if __name__ == "__main__":
    main()
