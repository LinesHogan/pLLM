"""TLM evidence 11/24: independent factor AdamW and causal objective semantics.

The real HF/PEFT model path is covered by test/gpu/vllm_tlm_smoke.py.
"""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from pllm.authoring import CausalTrainBatch
from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentTopology,
    BackendPlacementProbe,
)
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget, ParameterTargetShape
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import TaskResourceFact, TaskSymbol
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.kernels.adamw import AdamWConfig
from pllm.workloads.tlm import TLMConfig, TLMMode, TLMWorkload


class TinyCausalModel:
    device = torch.device("cpu")
    tensor_dtype = TensorDType.FLOAT64
    fingerprint = "a" * 64
    scale = 1.0
    model_source = ("tiny", "test")
    targets = tuple(
        ParameterTargetShape(ParameterTarget(0, ParameterModuleKind.ATTENTION, role), 3, 3)
        for role in (ParameterRole.QUERY_PROJECTION_WEIGHT, ParameterRole.VALUE_PROJECTION_WEIGHT)
    )

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(23)
        self.initial_factors = tuple(
            torch.randn(shape, generator=generator, dtype=torch.float64) * 0.1
            for shape in ((2, 3), (3, 2), (2, 3), (3, 2))
        )
        self.embedding = torch.randn(5, 3, generator=generator, dtype=torch.float64)
        self.base = torch.randn(2, 3, 3, generator=generator, dtype=torch.float64)
        self.head = torch.randn(3, 5, generator=generator, dtype=torch.float64)

    def __deepcopy__(self, memo):
        raise AssertionError("binding must retain the original frozen causal model")

    @property
    def resident_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*self.initial_factors, self.embedding, self.base, self.head)
        )

    def logits(self, batch: CausalTrainBatch, factors: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
        hidden = self.embedding[batch.input_ids]
        weights = (
            self.base
            if factors is None
            else torch.stack(
                tuple(self.base[index] + factors[2 * index + 1] @ factors[2 * index] for index in range(2))
            )
        )
        return torch.tanh((hidden @ weights[0].T) + (hidden @ weights[1].T)) @ self.head


def _facts(model, *, reserved=None, task_resources=()) -> BackendCompileFacts:
    """Single-device facts reserving the frozen model's resident bytes (or ``reserved``)."""
    kind, ordinal = StateDeviceKind(model.device.type), model.device.index or 0
    timeline = RuntimeFacts(model.device).timeline(("optimizer/adamw@1", "transition/full_root@1"))
    reserved = model.resident_bytes if reserved is None else reserved
    timeline = replace(timeline, reserved_bytes=(reserved,) * len(timeline.anchors), task_resources=task_resources)
    probe = BackendPlacementProbe(
        "test.tlm@1",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        AttachmentTopology.IN_PROCESS,
        AttachmentGraphMode.EAGER,
        kind,
        ordinal,
        1,
        1,
        model.tensor_dtype,
        model.targets[0].input_size,
        8,
        (),
        (),
    )
    provider = PageBundleProviderFacts("test.tlm@1", 4, kind, ordinal)
    return BackendCompileFacts(replace(probe, targets=model.targets), timeline, (provider,), 2, 32768, 7)


def _compile(model, config: TLMConfig):
    return TLMWorkload(config, model).compile_backend(_facts(model)).execution


def _reference(model, parameters, batch, config):
    hidden = model.embedding[batch.input_ids]
    q = model.base[0] + parameters[1] @ parameters[0]
    v = model.base[1] + parameters[3] @ parameters[2]
    logits = torch.tanh(F.linear(hidden, q) + F.linear(hidden, v)) @ model.head
    selector = (
        torch.tanh(F.linear(hidden, model.base[0]) + F.linear(hidden, model.base[1])) @ model.head
        if config.mode is TLMMode.OFFLINE
        else logits.detach()
    )
    target = batch.labels[:, 1:]
    valid = target != config.ignore_index

    def ce(value):
        token = -value[:, :-1].log_softmax(-1).gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return (token * valid).sum(-1) / valid.sum(-1).clamp_min(1)

    sentence = ce(selector).detach()
    chosen = sentence > config.threshold
    return (ce(logits) * config.lamb * (sentence - config.threshold).exp() * chosen).sum() / chosen.sum().clamp_min(1)


def test_global_multifactor_adamw_matches_oracle_and_publishes_one_version() -> None:
    model = TinyCausalModel()
    config = TLMConfig(optimizer=AdamWConfig(0.01, (0.8, 0.95), 1e-7, 0.1), threshold=0.0)
    execution = _compile(model, config)
    (ref,) = execution.admit(())
    first, second = RequestIdentity("first", 0, 0), RequestIdentity("second", 1, 0)
    execution.admit((first,))
    old_pin = execution.directory.inference_pin(first)
    before = execution.arena.snapshot(old_pin, ref)
    reference = tuple(torch.nn.Parameter(value.clone()) for value in model.initial_factors)
    optimizer = torch.optim.AdamW(reference, lr=0.01, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.1, foreach=False)
    batch = CausalTrainBatch(torch.tensor([[0, 1, 2, 3], [2, 4, 1, 0]]), torch.tensor([[0, 1, -100, 3], [2, 4, 1, 0]]))
    for version in (1, 2):
        loss = _reference(model, reference, batch, config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        result = execution.step(batch)
        (current,) = result.publication.after_refs
        assert current.version_id == version and result.selected_count == 2
        if version == 1:
            execution.admit((second,))
            second_pin = execution.directory.inference_pin(second)
        assert execution.directory.inference_pin(first) is old_pin
        assert tuple(old_pin.refs) == (ref,)
        actual = execution.snapshot(current)
        for index, parameter in enumerate(reference):
            fields = actual[index * 5 : (index + 1) * 5]
            state = optimizer.state[parameter]
            for value, expected in zip(
                fields, (parameter, parameter.grad, state["step"], state["exp_avg"], state["exp_avg_sq"]), strict=True
            ):
                torch.testing.assert_close(value, expected)
        torch.testing.assert_close(result.loss, loss.detach())
        for actual_old, expected in zip(execution.arena.snapshot(old_pin, ref), before, strict=True):
            torch.testing.assert_close(actual_old, expected)
    assert second_pin.refs[0].version_id == 1  # A later request pins the version current at its admission.
    execution.retire((first, second))
    with pytest.raises(AssertionError):
        execution.arena.snapshot(old_pin, ref)
    execution.retire(())
    assert execution.diagnostics().active_lineages == 0


def test_tlm_matches_objective_online_offline_graph_and_qv_publication() -> None:
    model = TinyCausalModel()
    if torch.cuda.is_available():
        gpu_model = TinyCausalModel()
        gpu_model.device = torch.device("cuda", 0)
        gpu_model.initial_factors = tuple(t.to(gpu_model.device) for t in gpu_model.initial_factors)
        online = TLMWorkload(TLMConfig(mode=TLMMode.ONLINE), gpu_model)
        attachment = online.plan_backend(online.specialize(_facts(gpu_model), ExecutionOptions()))
        assert attachment.inputs.request.overlay_bindings
        assert all(task.lane == "main" for task in attachment.inputs.plan.schedule_physical.tasks)
    batch = CausalTrainBatch(torch.tensor([[0, 2, 3]]), torch.tensor([[0, 2, 3]]))
    for mode in TLMMode:
        config = TLMConfig(mode=mode, threshold=0.0)
        limits = (TaskResourceFact(TaskSymbol("update"), 4096, input_shapes=((1, 3),)),)
        with pytest.raises(ValueError):  # The frozen model must fit the reserved bytes.
            TLMWorkload(config, model).plan_backend(_facts(model, reserved=model.resident_bytes - 1))
        facts = _facts(model, task_resources=limits)
        execution = TLMWorkload(config, model).compile_backend(facts).execution
        execution.admit(())
        tokens = torch.zeros((2, 3), dtype=torch.int64)
        with pytest.raises(ValueError):  # Larger than the planned input shape.
            execution.step(CausalTrainBatch(tokens, tokens))
        anchors = {
            execution.arena.plan.semantic_graph.tasks[task.task_id].symbol: task.anchor_id
            for task in execution.arena.plan.schedule_decision.tasks
        }
        timeline = {item.symbol: item.ordinal for item in execution.arena.plan.timeline.anchors}
        assert anchors["update"] <= anchors["publish"]
        assert (anchors["publish"] < timeline["model.execute"]) == (mode is TLMMode.OFFLINE)
        assert (anchors["update"] >= timeline["token.commit"]) == (mode is TLMMode.ONLINE)
        expected = _reference(model, model.initial_factors, batch, config)
        result = execution.step(batch)
        torch.testing.assert_close(result.loss, expected)
        assert result.publication is not None and result.selected_count == 1
        execution.retire(())
        skipped = _compile(model, replace(config, threshold=1e6))
        (initial,) = skipped.admit(())
        result = skipped.step(batch)
        assert result.publication is None and result.selected_count == 0
        assert skipped.arena.latest(initial.lineage_id) == initial
        skipped.retire(())
