"""TLM evidence 11/24: independent factor AdamW and causal objective semantics."""

from dataclasses import replace
from unittest.mock import patch
from weakref import ref as weakref

import pytest
import torch
from torch.nn import functional as F

from pllm.authoring import BackendCompileFacts, CausalTrainBatch, ParameterTargetShape
from pllm.contracts.auxiliary import validate_auxiliary
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.backend_attachment import compile_backend_attachment, validate_backend_attachment
from pllm.kernels.adamw import AdamWConfig
from pllm.execution.ptts.page_bundle import PageBundleArena, PageWrite
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.contracts.semantic_graph import AnchorId, TaskResourceFact, TaskSymbol
from test.backend_facts import assert_auxiliary_preflight
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.workloads.tlm import TLMConfig, TLMMode, TLMWorkload
from pllm.workloads.tlm import HfQvModel, HfQvConfig
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode, AttachmentTopology, BackendPlacementProbe,
)
from pllm.contracts.tensor_layout import TensorDType
from pllm.contracts.state_representation import StateDeviceKind


class TinyCausalModel:
    device = torch.device("cpu")
    tensor_dtype = TensorDType.FLOAT64
    fingerprint = "a" * 64
    scale = 1.0
    model_source = ("tiny", "test")
    targets = tuple(ParameterTargetShape(ParameterTarget(0, ParameterModuleKind.ATTENTION, role), 3, 3)
        for role in (ParameterRole.QUERY_PROJECTION_WEIGHT, ParameterRole.VALUE_PROJECTION_WEIGHT))

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(23)
        self.initial_factors = tuple(torch.randn(shape, generator=generator, dtype=torch.float64) * 0.1
            for shape in ((2, 3), (3, 2), (2, 3), (3, 2)))
        self.embedding = torch.randn(5, 3, generator=generator, dtype=torch.float64)
        self.base = torch.randn(2, 3, 3, generator=generator, dtype=torch.float64)
        self.head = torch.randn(3, 5, generator=generator, dtype=torch.float64)

    def __deepcopy__(self, memo):
        raise AssertionError("binding must retain the original frozen causal model")

    @property
    def resident_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size()
            for tensor in (*self.initial_factors, self.embedding, self.base, self.head))

    def logits(self, batch: CausalTrainBatch, factors: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
        hidden = self.embedding[batch.input_ids]
        weights = self.base if factors is None else torch.stack(tuple(
            self.base[index] + factors[2 * index + 1] @ factors[2 * index]
            for index in range(2)))
        return torch.tanh((hidden @ weights[0].T) + (hidden @ weights[1].T)) @ self.head


def _compile(model: TinyCausalModel, config: TLMConfig):
    workload = TLMWorkload(config, model)
    timeline = RuntimeFacts(model.device).timeline(("optimizer/adamw@1", "transition/full_root@1"))
    timeline = replace(timeline, reserved_bytes=(model.resident_bytes,) * len(timeline.anchors))
    return workload.compile_backend(BackendCompileFacts(
        BackendPlacementProbe("test.tlm@1", "a" * 64, "b" * 64, "c" * 64,
            AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind(model.device.type),
            model.device.index or 0, 1, 1, model.tensor_dtype, model.targets[0].input_size, 8, (), ()),
        timeline,
        (PageBundleProviderFacts("test.tlm@1", 4, StateDeviceKind(model.device.type), model.device.index or 0),),
        2, 32768, 7, model.targets)).execution


def _reference(model, parameters, batch, config):
    hidden = model.embedding[batch.input_ids]
    q = model.base[0] + parameters[1] @ parameters[0]
    v = model.base[1] + parameters[3] @ parameters[2]
    logits = torch.tanh(F.linear(hidden, q) + F.linear(hidden, v)) @ model.head
    selector = (torch.tanh(F.linear(hidden, model.base[0]) + F.linear(hidden, model.base[1])) @ model.head
        if config.mode is TLMMode.OFFLINE else logits.detach())
    target = batch.labels[:, 1:]
    valid = target != config.ignore_index
    def ce(value):
        token = -value[:, :-1].log_softmax(-1).gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return (token * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    sentence = ce(selector).detach()
    chosen = sentence > config.threshold
    return (ce(logits) * config.lamb * (sentence - config.threshold).exp() * chosen).sum() / chosen.sum().clamp_min(1)


def test_global_multifactor_adamw_matches_oracle_and_publishes_one_version(monkeypatch) -> None:
    model = TinyCausalModel()
    config = TLMConfig(optimizer=AdamWConfig(0.01, (0.8, 0.95), 1e-7, 0.1), threshold=0.0)
    execution = _compile(model, config)
    prepare = PageBundleArena.prepare_transition
    def copied_optimizer_fields(arena, *args, **kwargs):
        assert kwargs["copy_fields"] == tuple(index for index in range(5 * len(model.initial_factors)) if index % 5 != 1)
        return prepare(arena, *args, **kwargs)
    monkeypatch.setattr(PageBundleArena, "prepare_transition", copied_optimizer_fields)
    (ref,) = execution.admit(())
    first, second = RequestIdentity("first", 0, 0), RequestIdentity("second", 1, 0)
    execution.admit((first,))
    old_pin = execution.directory.inference_pin(first)
    with pytest.raises(RuntimeError, match="inference requests"):
        execution.retire(())
    before = execution.arena.snapshot(old_pin, ref)
    reference = tuple(torch.nn.Parameter(value.clone()) for value in model.initial_factors)
    optimizer = torch.optim.AdamW(reference, lr=0.01, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.1, foreach=False)
    batch = CausalTrainBatch(torch.tensor([[0, 1, 2, 3], [2, 4, 1, 0]]),
        torch.tensor([[0, 1, -100, 3], [2, 4, 1, 0]]))
    for version in (1, 2):
        loss = _reference(model, reference, batch, config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        def fail_publish(self: PageBundleArena, write: PageWrite):
            raise RuntimeError("injected multifactor publication failure")
        with monkeypatch.context() as patch:
            patch.setattr(PageBundleArena, "publish", fail_publish)
            with pytest.raises(RuntimeError, match="multifactor"):
                execution.step(batch)
        assert execution.arena.latest(ref.lineage_id).version_id == version - 1
        invocation = execution.start_step(batch)
        for anchor in execution.arena.plan.timeline.anchors:
            execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
            if invocation.context.write is not None:
                assert invocation.context.updated_refs == invocation.context.write.after_refs
            if anchor.ordinal == execution.sampler_anchor_id:
                execution.complete_sampler(invocation)
        result = execution.finish_step(invocation)
        assert result.publication is not None
        assert invocation.context.updated_refs == result.publication.after_refs
        (current,) = result.publication.after_refs
        assert current.version_id == version and result.selected_count == 2
        if version == 1:
            execution.admit((second,))
            second_pin = execution.directory.inference_pin(second)
        assert execution.directory.inference_pin(first) is old_pin
        assert old_pin.refs == (ref,)
        actual = execution.snapshot(current)
        for index, parameter in enumerate(reference):
            fields = actual[index * 5:(index + 1) * 5]
            state = optimizer.state[parameter]
            for value, expected in zip(fields, (parameter, parameter.grad, state["step"], state["exp_avg"], state["exp_avg_sq"]), strict=True):
                torch.testing.assert_close(value, expected)
        torch.testing.assert_close(result.loss, loss.detach())
        for actual_old, expected in zip(execution.arena.snapshot(old_pin, ref), before, strict=True):
            torch.testing.assert_close(actual_old, expected)
    assert second_pin.refs[0].version_id == 1
    assert execution.diagnostics().inference_pins == 2
    execution.retire((first, second))
    assert execution.diagnostics().inference_pins == 0
    with pytest.raises(RuntimeError, match="pin"):
        execution.arena.snapshot(old_pin, ref)
    execution.retire(())
    assert execution.diagnostics().active_lineages == 0
    assert execution.diagnostics().update_count == 2
    assert execution.diagnostics().fallback_count == 0


def test_tlm_matches_objective_online_offline_graph_and_qv_publication() -> None:
    model = TinyCausalModel()
    batch = CausalTrainBatch(torch.tensor([[0, 2, 3]]), torch.tensor([[0, 2, 3]]))
    for mode in TLMMode:
        config = TLMConfig(mode=mode, threshold=0.0)
        facts = BackendCompileFacts(
            BackendPlacementProbe("test.qv@1", "a" * 64, "b" * 64, "c" * 64,
                AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind.CPU,
                0, 1, 1, model.tensor_dtype, 3, 8, (), ()),
            RuntimeFacts(model.device).timeline(("optimizer/adamw@1", "transition/full_root@1")),
            (PageBundleProviderFacts("test.qv@1", 4),), 2, 32768, 7, model.targets)
        limits = TaskResourceFact(TaskSymbol("update"), 4096, input_shapes=((1, 3),))
        facts = replace(facts, timeline=replace(facts.timeline, task_resources=(limits,),
            reserved_bytes=(model.resident_bytes,) * len(facts.timeline.anchors)))
        assert TLMWorkload(config, model).backend_requirements().resident_bytes == model.resident_bytes
        with pytest.raises(ValueError, match="model residency"):
            TLMWorkload(config, model).plan_backend(replace(facts, timeline=replace(facts.timeline,
                reserved_bytes=(model.resident_bytes - 1,) * len(facts.timeline.anchors))))
        deployment = TLMWorkload(config, model).compile_backend(facts)
        execution = deployment.execution
        signed = deployment.attachment
        validate_auxiliary(batch, signed.inputs.plan, SharedTensor.describe)
        assert_auxiliary_preflight(signed, replace(batch, labels=batch.labels.float()))
        metadata = TinyCausalModel()
        metadata.device = torch.device("meta")
        metadata.initial_factors = tuple(torch.empty_like(factor, device="meta") for factor in model.initial_factors)
        cold = TLMWorkload(config, metadata)
        assert cold.plan_backend(facts) == signed
        with pytest.raises(ValueError, match="materialized"):
            cold.bind_math(facts, signed.inputs.plan)
        first, second = signed.inputs.request.overlay_bindings
        changed = replace(signed.inputs, request=replace(signed.inputs.request,
            overlay_bindings=(replace(first, scale=2.0), second)))
        assert compile_backend_attachment(changed) != signed.decision
        with pytest.raises(ValueError):
            validate_backend_attachment(changed, signed.decision)
        execution.admit(())
        for shape in ((2, 3), (1, 4)):
            tokens = torch.zeros(shape, dtype=torch.int64)
            oversized = CausalTrainBatch(tokens, tokens)
            assert_auxiliary_preflight(signed, oversized)
            with pytest.raises(ValueError, match="signed input bounds"):
                execution.step(oversized)
        with pytest.raises(ValueError, match="signed .* input bounds"):
            TLMWorkload(config, model).plan_backend(replace(facts,
                timeline=replace(facts.timeline, task_resources=(replace(limits, input_shapes=()),))))
        anchors = {execution.arena.plan.semantic_graph.tasks[task.task_id].symbol: task.anchor_id
            for task in execution.arena.plan.schedule_decision.tasks}
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
        assert skipped.diagnostics().update_count == 0
        skipped.retire(())

    for device in (torch.device("cpu"), *((torch.device("cuda", 0),) if torch.cuda.is_available() else ())):
        _hf_matches_reference(device)


def _hf_matches_reference(device: torch.device) -> None:
    # Actual HF/PEFT forward/backward, compared against an independently optimized
    # model. Only the mathematical state is copied into the test reference.
    from copy import deepcopy
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    torch.manual_seed(17)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=11, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, attention_dropout=0.0, tie_word_embeddings=True))
    base.to(device=device, dtype=torch.float32)
    hf = HfQvModel(base, HfQvConfig("tiny.qwen2", "test", True, False,
        2, 4, 0.0, TensorDType.FLOAT32))
    assert base.model.embed_tokens.weight is base.lm_head.weight
    assert hf.resident_bytes == sum(tensor.numel() * tensor.element_size()
        for tensor in (*hf.model.parameters(), *hf.model.buffers()))
    with torch.device("meta"):
        metadata = HfQvModel(Qwen2ForCausalLM(base.config), hf.config)
    assert metadata.resident_bytes == hf.resident_bytes and metadata.fingerprint == hf.fingerprint
    assert tuple((target.input_size, target.output_size) for target in hf.targets) == ((8, 8), (8, 4))
    frozen = tuple(value.clone() for value in hf.initial_factors)
    batch = CausalTrainBatch(torch.tensor([[1, 2, 3, 4], [2, 3, 4, 0]], device=device),
        torch.tensor([[1, 2, 3, 4], [2, 3, 4, -100]], device=device),
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], device=device))
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    selection_logits = []
    original_logits = hf.logits
    original_forward = hf.model.forward
    def bounded_forward(*args, **kwargs):
        assert kwargs["output_hidden_states"] is False and kwargs["output_attentions"] is False
        result = original_forward(*args, **kwargs)
        assert result.hidden_states is result.attentions is result.past_key_values is None
        return result
    def observe_logits(batch, factors):
        if factors is not None:
            assert all(tensor() is None for tensor in selection_logits), "full offline logits survived into adapted forward"
            selection_logits.clear()
        value = original_logits(batch, factors)
        if factors is None:
            selection_logits.append(weakref(value))
        return value
    for mode in TLMMode:
        config = TLMConfig(mode=mode, threshold=0.0)
        execution = _compile(hf, config)
        execution.admit((), stream)
        reference_model = deepcopy(hf.model)
        parameters = [parameter.requires_grad_(True) for name, parameter in reference_model.named_parameters()
            if ".lora_" in name]
        optimizer = torch.optim.AdamW(parameters, lr=config.optimizer.lr, betas=config.optimizer.betas,
            eps=config.optimizer.eps, weight_decay=config.optimizer.weight_decay, foreach=False)
        for _ in range(2):
            logits = reference_model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                use_cache=False, return_dict=True).logits
            if mode is TLMMode.OFFLINE:
                with torch.no_grad(), reference_model.disable_adapter():
                    selector = reference_model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                        use_cache=False, return_dict=True).logits
            else:
                selector = logits.detach()
            labels = batch.labels[:, 1:]
            valid = labels != -100
            def sentence_ce(value):
                losses = F.cross_entropy(value[:, :-1].reshape(-1, 11), labels.reshape(-1),
                    ignore_index=-100, reduction="none").reshape_as(labels)
                return losses.sum(-1) / valid.sum(-1)
            loss = (sentence_ce(logits) * config.lamb * sentence_ce(selector).detach().exp()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with patch.object(hf, "logits", observe_logits), patch.object(hf.model, "forward", bounded_forward):
                result = execution.step(batch, stream)
            torch.testing.assert_close(result.loss, loss.detach())
            (version,) = result.publication.after_refs
            values = execution.snapshot(version, stream)
            for index, parameter in enumerate(parameters):
                torch.testing.assert_close(values[5 * index], parameter)
                torch.testing.assert_close(values[5 * index + 1], parameter.grad)
        for original, expected in zip(hf.initial_factors, frozen, strict=True):
            torch.testing.assert_close(original, expected)
        execution.retire(())
        assert execution.diagnostics().active_lineages == 0
