"""Safetensors deployment agrees with initializers, signs bytes, and validates before GPU allocation."""

import json
from dataclasses import dataclass, replace
from hashlib import sha256

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import pllm
from pllm import testing
from pllm.execution.method_dependencies import check_dependencies


def fill_probe(fields, generator):
    """Initialize the same logical parameters as the saved field and module probes."""
    fields["w"].copy_(torch.arange(1.0, 5.0))
    fields["b"].fill_(0.5)
    if "hidden" in fields:
        fields["hidden"].fill_(7)


class MetaProbe(nn.Linear):
    """A probe whose initializer must never allocate real random parameters."""

    def __init__(self, width):
        super().__init__(width, 1)

    def reset_parameters(self):
        if not self.weight.is_meta:
            raise AssertionError("pretrained module ran real random initialization")
        super().reset_parameters()


@dataclass(frozen=True)
class FieldProbe(pllm.Workload):
    """The tutorial's frozen probe as plain fields, also used to test request and trained state."""

    state: pllm.InstanceState

    @pllm.export(name="score", max_rows=4, max_cols=1)
    def score(self, h=pllm.final_hidden(), w=pllm.parameter("state", "w"), b=pllm.parameter("state", "b")):
        return torch.sigmoid((h.float() * w).sum(-1, keepdim=True) + b)


@dataclass(frozen=True)
class ModuleProbe(pllm.Workload):
    """The same probe deployed as a strict module state dict."""

    state: pllm.InstanceState

    @pllm.export(name="score", max_rows=4, max_cols=1)
    def score(self, h=pllm.final_hidden(), probe=pllm.module("state")):
        return torch.sigmoid(probe(h))


@dataclass(frozen=True)
class LogitLookup(pllm.Workload):
    """A token-only wide table lookup feeding an ordinary sampler adapter."""

    path: str
    buckets: int = 8
    placement: str = "host"
    gate: float = 1.0

    @property
    def memory(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {"table": lambda m: (self.buckets, m.vocab_size)}, init=pllm.Pretrained(self.path, placement=self.placement)
        )

    def lookup(self, ids=pllm.input_tokens(history=1), table=pllm.parameter("memory", "table")):
        return table[(ids[:, 0] + ids[:, 1]).remainder(self.buckets)].float() * self.gate

    @pllm.add_logits()
    def add(self, value=pllm.result("lookup")):
        return value * 1.0


@pytest.mark.parametrize("state_type", (pllm.InstanceState, pllm.RequestState))
@pytest.mark.parametrize("optimizer", (None, pllm.AdamW(lr=1e-3)))
@pytest.mark.parametrize("module", (False, True))
def test_pretrained_equals_initialize_and_signs_file_bytes(tmp_path, state_type, optimizer, module):
    path = tmp_path / "probe.safetensors"
    values = (
        {"weight": torch.arange(1.0, 5.0)[None], "bias": torch.tensor([0.5])}
        if module
        else {"w": torch.arange(1.0, 5.0), "b": torch.tensor([0.5])}
    )
    carry = {"hidden": (1,)} if optimizer and not module else {}
    if carry:
        values["hidden"] = torch.tensor([7.0])
    save_file(values, path)
    if module:
        state = state_type(
            init=pllm.FromModule(lambda m: MetaProbe(m.hidden_size), pretrained=str(path)), optimizer=optimizer
        )
        workload = ModuleProbe(state)
    else:
        state = state_type(
            {"w": lambda m: (m.hidden_size,), "b": pllm.Field((1,))},
            init=pllm.Pretrained(str(path)),
            optimizer=optimizer,
            carry=carry,
        )
        workload = FieldProbe(state)
    reference = FieldProbe(
        state_type({"w": (4,), "b": (1,)}, init=pllm.Computed(fill_probe), optimizer=optimizer, carry=carry)
    )
    tap = testing.final_hidden_binding(1)
    facts = testing.backend_facts(4, (tap,), layer_count=1, memory_bytes=1 << 20)
    binding = workload.build(facts)
    assert sha256(path.read_bytes()).hexdigest() in binding.state_sources["state"]
    with testing.compile_execution(workload, facts) as execution:
        ids = (testing.RequestIdentity("a", 0, 0),)
        execution.admit(ids)
        (ref,) = execution.latest(ids)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5])
        root = execution.snapshot(ref)
        actual = root[0] if module or optimizer else torch.cat((root[0], root[1]))
        torch.testing.assert_close(actual, expected)
        if carry:
            torch.testing.assert_close(root[-1], torch.tensor([7.0]))
        testing.run_step(execution, testing.step_input(ids, {tap: torch.ones(1, 4)}))
        (event,) = (event for event in execution.trace.snapshot().events if event.tensor_name == "score")
        torch.testing.assert_close(event.tensor(), torch.sigmoid(torch.tensor([[10.5]])))
        execution.retire(ids)
    with testing.compile_execution(reference, facts) as initialized:
        initialized.admit(ids)
        reference_root = initialized.snapshot(initialized.latest(ids)[0])
        torch.testing.assert_close(reference_root[0] if optimizer else torch.cat(reference_root), actual)
        initialized.retire(ids)
    save_file({key: value + 1 for key, value in values.items()}, path)
    assert workload.build(facts).graph.execution_parameters_digest != binding.graph.execution_parameters_digest
    with pytest.raises(ValueError, match="changed after make_llm hashed it"):
        workload.bind_backend(facts, workload.plan_backend(facts, binding), binding)


def test_pretrained_boundary_messages_and_import_first_export(tmp_path):
    path = tmp_path / "probe.safetensors"
    state = pllm.InstanceState({"w": (4,), "b": (1,)}, init=pllm.Pretrained(str(path)))
    workload = FieldProbe(state)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "intermediate_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            }
        )
    )
    config = pllm.ModelConfig(str(tmp_path), max_num_seqs=2, max_num_batched_tokens=16)
    with pytest.raises(ValueError, match="does not exist; save it with safetensors.torch.save_file"):
        pllm.explain(workload, config)
    save_file({"w": torch.ones(4)}, path)
    with pytest.raises(ValueError, match="has no tensor 'b' for state 'state'"):
        pllm.explain(workload, config)
    save_file({"w": torch.ones(4), "b": torch.ones(2)}, path)
    with pytest.raises(ValueError, match="adjust the declared field shape"):
        pllm.explain(workload, config)
    module = ModuleProbe(
        pllm.InstanceState(init=pllm.FromModule(lambda m: MetaProbe(m.hidden_size), pretrained=str(path)))
    )
    for values in (
        {"weight": torch.ones(1, 4)},
        {"weight": torch.ones(1, 4), "bias": torch.ones(1), "extra": torch.ones(1)},
        {"weight": torch.ones(1, 3), "bias": torch.ones(1)},
    ):
        save_file(values, path)
        with pytest.raises(
            ValueError,
            match="cannot load module state 'state'.*",
        ):
            pllm.explain(module, config)
    save_file({"w": torch.ones(4), "b": torch.ones(1)}, path)
    report = pllm.explain(workload, config)
    assert report.states[0].source in str(report) and str(path) in report.states[0].source
    for captured in (nn.Linear(4, 1), torch.ones(4)):
        factory = lambda m: captured
        with pytest.raises(ValueError, match="pretrained="):
            check_dependencies(factory)
    assert testing.check(workload, config).passed
    imported = replace(workload, state=replace(state, init=pllm.Imported()))
    tap = testing.final_hidden_binding(1)
    facts = testing.backend_facts(4, (tap,), layer_count=1, memory_bytes=1 << 20)
    with testing.compile_execution(imported, facts) as execution:
        ids = (testing.RequestIdentity("a", 0, 0),)
        execution.admit(ids)
        testing.run_step(
            execution,
            testing.step_input(
                ids, {tap: torch.ones(1, 4)}, batch=pllm.io.StateImport((torch.zeros(4), torch.ones(1)))
            ),
        )
        (event,) = (event for event in execution.trace.snapshot().events if event.tensor_name == "score")
        torch.testing.assert_close(event.tensor(), torch.sigmoid(torch.ones(1, 1)))
        execution.retire(ids)


def test_pretrained_uses_model_side_initializer_paths(tmp_path):
    from pllm.contracts.parameter_overlay import Projection
    from test.dense_update import DenseUpdate, DenseUpdateConfig
    from test.workloads.test_model_sites import FastWeightMemory

    @dataclass(frozen=True)
    class SavedDense(DenseUpdate):
        path: str = ""

        @property
        def dense_fast_weights(self) -> pllm.RequestState:
            return replace(super().dense_fast_weights, init=pllm.Pretrained(self.path))

    @dataclass(frozen=True)
    class SavedMemory(FastWeightMemory):
        path: str = ""

        @property
        def memory(self) -> pllm.RequestState:
            return replace(super().memory, init=pllm.Pretrained(self.path))

    path = str(tmp_path / "memory.safetensors")
    target = Projection("down_proj", 0)
    facts = testing.backend_facts(
        3,
        memory_bytes=1 << 20,
        layer_count=2,
        targets={target: (3, 2)},
    )
    for workload, key, shape in (
        (SavedDense(DenseUpdateConfig((target,), chunk_size=2), path=path), "delta", (2, 3)),
        (SavedMemory(path=path), "M", (3, 3)),
    ):
        expected = torch.arange(shape[0] * shape[1], dtype=torch.float64).reshape(shape)
        save_file({key: expected}, path)
        with testing.compile_execution(workload, facts) as execution:
            identities = (testing.RequestIdentity("a", 0, 0),)
            (ref,) = execution.admit(identities)
            torch.testing.assert_close(execution.snapshot(ref)[0].double(), expected)
            execution.retire(identities)
