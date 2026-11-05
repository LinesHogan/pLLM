"""Captured CUDA-graph steps on the GPU: ESamp, request memory and instance means with changing sampler inputs.

Reuses the CPU tests' ESamp and distillation oracles on CUDA, then checks that a
captured instance-mean step replays (no Python math) with changing row counts, reused
sampler buffers and publish lags 0-2.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

import pllm
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import HiddenRowBatch, HiddenRows, PackedHiddenResourceBinding
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import SamplerFeedback
from pllm.contracts.semantic_graph import AnchorId
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_task import TensorTask
from pllm.testing import backend_facts


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
    timeline = model_timeline(device)
    workload = MeanFit()
    provider = PageBundleProviderFacts("test.mean@1", 6, StateDeviceKind.CUDA)
    facts = backend_facts(4, (resource,), timeline, provider, max_lineages=3)
    facts, declaration = workload.specialize(
        facts, pllm.ExecutionOptions(capture_memory_bytes=16 * 1024**2, publish_lag=publish_lag)
    )
    # One admitted lineage: the instance state.
    plan = compile_page_bundle_plan(
        (declaration.graph, declaration.layouts),
        facts.timeline,
        facts.providers,
        1,
        facts.state_memory_budget_bytes,
        device,
    )
    execution = workload.bind_math(facts, plan, declaration).bind(PageBundleArena(plan), AnchorId(3))
    assert plan.timeline.visibility is not None
    overlap = execution.post_sample
    execution.admit((), stream)
    parameter = torch.nn.Parameter(torch.zeros(4, device=device, dtype=torch.float64))
    optimizer = torch.optim.AdamW((parameter,), lr=0.03, weight_decay=0, capturable=True, foreach=True)
    hidden = torch.empty((3, 4), device=device, dtype=torch.float32)
    target_buffer = torch.empty((3, 4), device=device)
    token_buffer = torch.empty(3, device=device, dtype=torch.long)
    for step, count in enumerate((1, 3, 2)):
        hidden.copy_(torch.arange(12, device=device, dtype=torch.float32).view(3, 4) / 10)
        rows = PackedPhaseRowBatch.from_rows(
            step,
            count,
            tuple(PackedPhaseRow(i, i, step, HiddenPhase.DECODE, RequestIdentity(str(i), i, 0)) for i in range(count)),
        )
        sampler = PackedSamplerRowBatch.from_links(rows, tuple(PackedSamplerRowLink(i, i) for i in range(count)))
        value = StepInput(
            rows,
            sampler,
            HiddenRowBatch(rows, (HiddenRows(resource, hidden, np.arange(count, dtype=np.int64)),)),
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
            assert invocation.pin.refs[0].version_id == max(0, step - publish_lag)
            for anchor in plan.timeline.anchors:
                execution.advance_to(invocation, anchor.symbol)
                if anchor.symbol == "sampler.pre_draw":
                    execution.complete_sampler(invocation, stream, feedback=SamplerFeedback(sampler, target, tokens))
            if overlap is not None:
                # Sampler outputs may be overwritten after the handoff copy.
                target.fill_(999)
                tokens.fill_(999)
            result = execution.finish_step(invocation, stream)
        torch.testing.assert_close(result.loss, oracle.detach())
        torch.testing.assert_close(execution.snapshot(result.publication.after_refs[0], stream)[0], parameter)
    if overlap is not None:
        execution.release_history()
    execution.retire(())


def main() -> None:
    """Run the shared eager/captured step against independent numerical oracles."""
    for lag in (0, 1, 2):
        mean_capture(lag)
    print("generic step capture: ESamp, request memory, instance mean and changing sampler buffers agree")


if __name__ == "__main__":
    main()
