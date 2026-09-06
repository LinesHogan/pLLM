"""TLM evidence 11/24: independent factor AdamW and causal objective semantics."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from tllm.canonical.backend_program import ParameterTargetShape
from tllm.canonical.kernels.adamw import AdamWConfig
from tllm.canonical.ptts.page_bundle import PageBundleArena, PageWrite
from tllm.canonical.ptts.page_bundle_planning import PageBundleProviderFacts
from tllm.canonical.runtime_facts import CanonicalRuntimeFacts
from tllm.canonical.workloads.tlm import TLMConfig, TLMMode, TLMTrainBatch, TLMWorkload
from tllm.canonical.workloads.tlm import HfQvModel, HfQvConfig
from tllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
from tllm.contracts.tensor_layout import TensorDType
from tllm.contracts.state_representation import StateDeviceKind


class TinyCausalModel:
    device = torch.device("cpu")
    tensor_dtype = TensorDType.FLOAT64
    fingerprint = "a" * 64
    targets = tuple(ParameterTargetShape(ParameterTarget(0, ParameterModuleKind.ATTENTION, role), 3, 3)
        for role in (ParameterRole.QUERY_PROJECTION_WEIGHT, ParameterRole.VALUE_PROJECTION_WEIGHT))

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(23)
        self.initial_factors = tuple(torch.randn(shape, generator=generator, dtype=torch.float64) * 0.1
            for shape in ((2, 3), (3, 2), (2, 3), (3, 2)))
        self.embedding = torch.randn(5, 3, generator=generator, dtype=torch.float64)
        self.base = torch.randn(2, 3, 3, generator=generator, dtype=torch.float64)
        self.head = torch.randn(3, 5, generator=generator, dtype=torch.float64)

    def logits(self, batch: TLMTrainBatch, factors: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
        hidden = self.embedding[batch.input_ids]
        weights = self.base if factors is None else torch.stack(tuple(
            self.base[index] + factors[2 * index + 1] @ factors[2 * index]
            for index in range(2)))
        return torch.tanh((hidden @ weights[0].T) + (hidden @ weights[1].T)) @ self.head


def _compile(model: TinyCausalModel, config: TLMConfig):
    return TLMWorkload(config, model).compile(
        CanonicalRuntimeFacts(model.device).timeline(("optimizer/adamw@1", "transition/full_root@1")),
        (PageBundleProviderFacts("test.tlm@1", 4, StateDeviceKind(model.device.type), model.device.index or 0),),
        state_memory_budget_bytes=32768)


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
    (ref,) = execution.admit(())
    old_pin = execution.arena.pin_exact((ref,))
    before = execution.arena.snapshot(old_pin, ref)
    reference = tuple(torch.nn.Parameter(value.clone()) for value in model.initial_factors)
    optimizer = torch.optim.AdamW(reference, lr=0.01, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.1, foreach=False)
    batch = TLMTrainBatch(torch.tensor([[0, 1, 2, 3], [2, 4, 1, 0]]),
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
        result = execution.step(batch)
        assert result.publication is not None
        (current,) = result.publication.after_refs
        assert current.version_id == version and result.selected_count == 2
        actual = execution.snapshot(current)
        for index, parameter in enumerate(reference):
            fields = actual[index * 5:(index + 1) * 5]
            state = optimizer.state[parameter]
            for value, expected in zip(fields, (parameter, parameter.grad, state["step"], state["exp_avg"], state["exp_avg_sq"]), strict=True):
                torch.testing.assert_close(value, expected)
        torch.testing.assert_close(result.loss, loss.detach())
        for actual_old, expected in zip(execution.arena.snapshot(old_pin, ref), before, strict=True):
            torch.testing.assert_close(actual_old, expected)
    execution.arena.wait_release(execution.arena.release_pin(old_pin))
    execution.retire(())
    assert execution.diagnostics().active_lineages == 0
    assert execution.diagnostics().update_count == 2
    assert execution.diagnostics().fallback_count == 0


def test_tlm_matches_objective_online_offline_graph_and_qv_publication() -> None:
    model = TinyCausalModel()
    batch = TLMTrainBatch(torch.tensor([[0, 2, 3]]), torch.tensor([[0, 2, 3]]))
    for mode in TLMMode:
        config = TLMConfig(mode=mode, threshold=0.0)
        execution = _compile(model, config)
        execution.admit(())
        anchors = {task.task_symbol: task.anchor_id for task in execution.arena.plan.schedule_decision.tasks}
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
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, attention_dropout=0.0))
    base.to(device=device, dtype=torch.float32)
    hf = HfQvModel(base, HfQvConfig("tiny.qwen2", "test", True, False,
        2, 4, 0.0, TensorDType.FLOAT32))
    assert tuple((target.input_size, target.output_size) for target in hf.targets) == ((8, 8), (8, 4))
    frozen = tuple(value.clone() for value in hf.initial_factors)
    batch = TLMTrainBatch(torch.tensor([[1, 2, 3, 4], [2, 3, 4, 0]], device=device),
        torch.tensor([[1, 2, 3, 4], [2, 3, 4, -100]], device=device),
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], device=device))
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
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
