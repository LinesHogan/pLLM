"""Generic step capture: ESamp, request memory, instance means and changing sampler inputs."""

import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

import pllm
from pllm.authoring import TensorStepInput
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding, PackedHiddenSnapshot, PackedHiddenSnapshotBatch
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import SamplerFeedback
from pllm.contracts.semantic_graph import AnchorId, AnchorSymbol, SchedulerTimeline, TimelineAnchor
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.tensor_task import TensorTask
from test.backend_facts import backend_facts
from test.workloads.test_esamp import _esamp_native_oracle, _ordinary_request_distillation


class MeanFit(pllm.Workload):
    """A shared parameter trained against changing sampler buffers, with masked row means."""

    weights = pllm.InstanceState({"value": (4,)}, optimizer=pllm.AdamW(lr=0.03), dtype=torch.float64)

    @pllm.on(pllm.Phase.BEFORE_SAMPLE)
    @pllm.guide_sampling(beta=0.1)
    def predict(
        self,
        hidden: Annotated[torch.Tensor, pllm.Hidden(0)],
        weight: Annotated[torch.Tensor, pllm.Parameter("weights", "value")],
    ) -> torch.Tensor:
        return hidden * weight

    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.train(state="weights", reduction="mean")
    def loss(
        self,
        hidden: Annotated[torch.Tensor, pllm.Hidden(0)],
        weight: Annotated[torch.Tensor, pllm.Parameter("weights", "value")],
        target: Annotated[torch.Tensor, pllm.BaseLogits()],
        token: Annotated[torch.Tensor, pllm.SampledTokens()],
    ) -> torch.Tensor:
        return (hidden * weight - target - token[:, None]).square().mean(1)


def mean_capture() -> None:
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
    facts = backend_facts(
        4, (resource,), timeline, PageBundleProviderFacts("test.mean@1", 6, StateDeviceKind.CUDA), max_lineages=3
    )
    facts = workload.specialize(facts, pllm.ExecutionOptions(capture=True, capture_memory_bytes=16 * 1024**2))
    declaration = workload.declaration(facts)
    plan = compile_page_bundle_plan(
        (declaration.graph, declaration.layouts),
        facts.timeline,
        facts.providers,
        1,
        facts.state_memory_budget_bytes,
        device,
    )
    execution = workload.bind_math(facts, plan).bind(PageBundleArena(plan), AnchorId(1))
    execution.admit((), stream)
    parameter = torch.nn.Parameter(torch.zeros(4, device=device, dtype=torch.float64))
    optimizer = torch.optim.AdamW((parameter,), lr=0.03, weight_decay=0, capturable=True, foreach=True)
    hidden = torch.arange(12, device=device, dtype=torch.float32).view(3, 4) / 10
    for step, count in enumerate((1, 3, 2)):
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
        target = torch.full((count, 4), step * 0.2, device=device)
        tokens = torch.arange(count, device=device, dtype=torch.long)
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
            invocation = execution.start_step(value, stream)
            for anchor in plan.timeline.anchors:
                execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
                if anchor.ordinal == 1:
                    execution.complete_sampler(invocation, stream, feedback=SamplerFeedback(sampler, target, tokens))
            result = execution.finish_step(invocation)
        torch.testing.assert_close(result.loss, oracle.detach())
        torch.testing.assert_close(execution.snapshot(result.publication.after_refs[0], stream)[0], parameter)
    # Incomplete sampler coverage is a boundary error, never an out-of-bounds graph read.
    missing = PackedSamplerRowBatch(rows, (PackedSamplerRowLink(0, 0),))
    invocation = execution.start_step(replace(value, sampler_rows=missing), stream)
    with pytest.raises(ValueError, match="cover every operation row"):
        for anchor in plan.timeline.anchors:
            execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
            if anchor.ordinal == 1:
                execution.complete_sampler(
                    invocation, stream, feedback=SamplerFeedback(missing, target[:1], tokens[:1])
                )
    torch.testing.assert_close(execution.snapshot(result.publication.after_refs[0], stream)[0], parameter)
    execution.retire(())
    underfunded = replace(
        facts,
        timeline=replace(
            facts.timeline,
            task_resources=tuple(replace(fact, resident_bytes=1) for fact in facts.timeline.task_resources),
        ),
    )
    declaration = workload.declaration(underfunded)
    plan = compile_page_bundle_plan(
        (declaration.graph, declaration.layouts),
        underfunded.timeline,
        facts.providers,
        1,
        facts.state_memory_budget_bytes,
        device,
    )
    with pytest.raises(ValueError, match="resident bytes"):
        workload.bind_math(underfunded, plan)


def main() -> None:
    """Run the shared eager/captured step against independent numerical oracles."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        _esamp_native_oracle(monkeypatch, torch.device("cuda", 0))
    _ordinary_request_distillation()
    mean_capture()
    print("generic step capture: ESamp, COW, memory, instance mean, changing sampler buffers and preflight agree")


if __name__ == "__main__":
    main()
