"""Evidence 25: independent gated-residual/AdamW oracle and exact request views."""

import pytest
import torch
import torch.nn.functional as F

from tllm.canonical.ptts.dense_page_planning import DensePageCpuProviderFacts
from tllm.canonical.ptts.dense_page import DensePageStateProvider, DensePageTransitionBatch, DensePageRoot
from tllm.canonical.workloads.esamp_request import (
    ESampRequestWorkload,
)
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestCompileOptions, ESampRequestConfig, ESampRequestPackedHiddenBindings,
    ESampRequestSpecialization, ESampRequestStepInput,
)
from tllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from tllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding, PackedHiddenSnapshot, PackedHiddenSnapshotBatch,
)
from tllm.contracts.packed_rows import (
    PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink,
)
from tllm.contracts.request_identity import RequestIdentity, RequestKeyField
from tllm.contracts.semantic_graph import (
    AnchorId, AnchorSymbol, SchedulerTimeline, TimelineAnchor,
)
from tllm.contracts.state_representation import StateDeviceKind
from tllm.contracts.tensor_layout import TensorDType


def _reference(source: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    up, gate, down, bias, scale, shift = parameters.split((8, 8, 8, 2, 4, 4))
    residual = ((source @ up.reshape(4, 2)) * F.silu(
        source @ gate.reshape(4, 2) + bias,
    )) @ down.reshape(2, 4)
    return F.layer_norm(source + residual, (4,)) * scale + shift


def test_esamp_predicts_before_training_and_isolates_request_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = ESampRequestPackedHiddenBindings(*(
        PackedHiddenResourceBinding(str(layer), HiddenLayerKind.INDEX,
            HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]")
        for layer in (0, -1)
    ))
    config = ESampRequestConfig(
        rank=2, intervention=True, intervention_beta=0.2, learning_rate=0.03,
        weight_decay=0.1, key_fields=tuple(RequestKeyField),
    )
    timeline = SchedulerTimeline(
        tuple(TimelineAnchor(AnchorSymbol(name), index) for index, name in enumerate((
            "model.execute", "sampler.pre_draw", "token.commit", "next.prepare",
        ))), ("compute/cpu@1", "state/dense_page_cpu@1"),
    )
    compiled = ESampRequestWorkload(config).compile(
        ESampRequestSpecialization(4, TensorDType.FLOAT32, 7, bindings, StateDeviceKind.CPU, 0),
        ESampRequestCompileOptions(timeline, (DensePageCpuProviderFacts("test.esamp@1", 3),), 2, 10000),
    )
    execution = compiled.execution
    assert execution.device.type == "cpu"
    identities = (RequestIdentity("shared-id", 0, 0), RequestIdentity("shared-id", 0, 1))
    latest = list(execution.admit(identities))
    parameters = [execution.snapshot(ref)[0].requires_grad_() for ref in latest]
    optimizers = [torch.optim.AdamW((parameter,), lr=config.learning_rate,
        betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)
        for parameter in parameters]
    generator = torch.Generator().manual_seed(7)
    for parameter in parameters:
        expected = torch.cat((*(torch.empty(8).normal_(0.0, 0.02, generator=generator)
            for _ in range(3)), torch.ones(2), torch.ones(4), torch.zeros(4)))
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

    source = torch.tensor([[0.2, -0.1, 0.4, 0.3], [-0.5, 0.7, 0.1, -0.2], [0.1, 0.3, -0.2, 0.8]])
    target = torch.tensor([[0.6, 0.0, -0.3, 0.2], [0.4, -0.2, 0.8, 0.1], [-0.2, 0.5, 0.1, 0.4]])
    for step_id, owners in enumerate(((0, 1, 0), (0,))):
        count = len(owners)
        rows = PackedPhaseRowBatch(1, step_id, count, tuple(
            PackedPhaseRow(index, index, identities[owner].request_id, step_id,
                HiddenPhase.DECODE, identities[owner], index)
            for index, owner in enumerate(owners)
        ))
        tensors = {bindings.source: source[:count], bindings.target: target[:count]}
        step = ESampRequestStepInput(rows,
            PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, i) for i in range(count))),
            PackedHiddenSnapshotBatch(rows, tuple(PackedHiddenSnapshot(resource, tensors[resource])
                for resource in bindings.resources)),
        )
        before = [execution.snapshot(ref) for ref in latest]
        for last_anchor in (0, 1):
            cancelled = execution.start_step(step)
            for anchor in range(last_anchor + 1):
                execution.execute_anchor(cancelled, AnchorId(anchor))
            execution.cancel_step(cancelled)
            with pytest.raises(RuntimeError, match="active"):
                execution.cancel_step(cancelled)

        def reject_publication(
            self: DensePageStateProvider, batch: DensePageTransitionBatch,
        ) -> tuple[DensePageRoot, ...]:
            raise RuntimeError("injected publication failure")

        with monkeypatch.context() as fault:
            fault.setattr(DensePageStateProvider, "publish_transition_batch", reject_publication)
            failed = execution.start_step(step)
            execution.execute_anchor(failed, AnchorId(0))
            execution.execute_anchor(failed, AnchorId(1))
            execution.complete_sampler(failed)
            with pytest.raises(RuntimeError, match="injected publication failure"):
                execution.execute_anchor(failed, AnchorId(2))
        assert execution.free_state_pages == 4
        for ref, root in zip(latest, before, strict=True):
            for actual, expected in zip(execution.snapshot(ref), root, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        handle = execution.start_step(step)
        execution.execute_anchor(handle, AnchorId(0))
        guidance = execution.sampler_guidance(handle)
        expected_prediction = torch.cat(tuple(
            _reference(source[i:i + 1], parameters[owner]).detach()
            for i, owner in enumerate(owners)
        ))
        if step_id == 0:
            assert guidance.affected_row_ids == ()
        else:
            assert guidance.affected_row_ids == (0,)
            torch.testing.assert_close(guidance.pred_hidden, expected_prediction)
        # No update or version change is allowed before sampler completion.
        for ref, root in zip(latest, before, strict=True):
            for actual, expected in zip(execution.snapshot(ref), root, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="sampler completion"):
            execution.execute_anchor(handle, AnchorId(2))
        execution.execute_anchor(handle, AnchorId(1))
        execution.complete_sampler(handle)
        execution.execute_anchor(handle, AnchorId(2))
        result = execution.finish_step(handle)
        assert result.guidance is guidance
        for index, owner in enumerate(sorted(set(owners))):
            selected = [i for i, selected_owner in enumerate(owners) if selected_owner == owner]
            optimizer = optimizers[owner]
            optimizer.zero_grad()
            loss = F.mse_loss(_reference(source[selected], parameters[owner]), target[selected])
            loss.backward()
            torch.testing.assert_close(result.losses[index], loss.detach())
            optimizer.step()
            assert result.before_refs[index] == latest[owner]
            latest[owner] = result.after_refs[index]
            assert latest[owner].version_id == step_id + 1
            root = execution.snapshot(latest[owner])
            state = optimizer.state[parameters[owner]]
            for actual, expected in zip(root,
                (parameters[owner], parameters[owner].grad, state["step"], state["exp_avg"], state["exp_avg_sq"]),
                strict=True,
            ):
                torch.testing.assert_close(actual, expected)
        if step_id == 1:
            for actual, expected in zip(execution.snapshot(latest[1]), before[1], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(guidance.pred_hidden, expected_prediction)
        with pytest.raises(RuntimeError, match="not registered"):
            execution.finish_step(handle)

    execution.retire(identities)
    diagnostics = execution.diagnostics()
    assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0
    assert diagnostics.loss_count == diagnostics.update_count == 3
    assert diagnostics.completed_sampler_pins == 4
    assert execution.free_state_pages == 6
    assert execution.plan.fallback_count == 0
