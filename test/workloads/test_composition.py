"""Effect composition: one table of every pair of authoring effects, admitted or rejected naming the conflict, and a
CPU step showing that composed effects compute exactly what each computes alone."""

import itertools
from dataclasses import dataclass, field, replace

import pytest
import torch

import pllm
from pllm import testing
from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.parameter_overlay import ProjectionShape
from pllm.execution.runtime_facts import model_timeline

TARGET = pllm.model.Projection("down_proj", 1)
BINDING = testing.hidden_binding(0, 3)
CAPABILITIES = ("model/custom_forward@1", "transition/full_root@1", "optimizer/adamw@1", "state/fenced_reclaim@1")
FACTS = testing.backend_facts(
    4,
    (BINDING,),
    model_timeline(torch.device("cpu"), CAPABILITIES),
    memory_bytes=1 << 20,
    layer_count=3,
    vocab_size=4,
    targets=(ProjectionShape(TARGET, 4, 4),),
)
FACTS = replace(FACTS, probe=replace(FACTS.probe, causal_domain=CausalModelDomain(2, 3, 16), capabilities=CAPABILITIES))


class Train(pllm.Workload):
    @property
    def trained(self) -> pllm.RequestState:
        return pllm.RequestState({"w": (4,)}, optimizer=pllm.AdamW(lr=0.01))

    @pllm.train(state="trained")
    def train_rows(self, h=pllm.hidden(0), w=pllm.parameter("trained", "w")):
        return (h - w).square().mean(-1)


class TrainOther(pllm.Workload):
    """A second, independent request trainer."""

    @property
    def other(self) -> pllm.RequestState:
        return pllm.RequestState({"w": (4,)}, optimizer=pllm.AdamW(lr=0.02))

    @pllm.train(state="other")
    def train_other(self, h=pllm.hidden(0), w=pllm.parameter("other", "w")):
        return (2 * h - w).abs().mean(-1)


class TrainBatch(pllm.Workload):
    @property
    def batch_state(self) -> pllm.InstanceState:
        return pllm.InstanceState({"w": (4,)}, optimizer=pllm.AdamW(lr=0.01))

    @pllm.train(state="batch_state")
    def train_batch(self, h=pllm.batch("hidden"), w=pllm.parameter("batch_state", "w")):
        return (h - w).square().mean(-1)


class InForward(pllm.Workload):
    @property
    def fast(self) -> pllm.RequestState:
        return pllm.RequestState({"w": pllm.Field((4, 4))})

    @pllm.train_in_forward(state="fast", targets=(TARGET,), weight="w", chunk=2, visibility="exact")
    def overlay_update(self, w=pllm.parameter("fast", "w"), x=pllm.operand("input"), y=pllm.operand("target")):
        return w + y.T @ x * 0.01, x @ w.T


class Adapt(pllm.Workload):
    @property
    def adapter_state(self) -> pllm.RequestState:
        return pllm.RequestState({"w": (4, 4)})

    @property
    def adapter(self) -> pllm.model.Adapter:
        return pllm.model.Adapter("adapter_state", (TARGET,), weight="w")


class Patch(pllm.Workload):
    @property
    def patch_state(self) -> pllm.InstanceState:
        return pllm.InstanceState({"v": (4,)})

    @property
    def patch(self) -> pllm.model.ActivationPatch:
        return pllm.model.ActivationPatch("patch_state", (1,), operator="add")


class Forward(pllm.Workload):
    @pllm.custom_forward()
    def route(self, layers):
        return layers()[-1]


class Positions(Forward):
    @property
    def position(self) -> pllm.PositionState:
        return pllm.PositionState()


class AddLogits(pllm.Workload):
    @pllm.add_logits()
    def logits(self, h=pllm.hidden(0)):
        return h * 0.01


class Guide(pllm.Workload):
    @pllm.guide_sampling(beta=0.2)
    def guide(self, h=pllm.hidden(0)):
        return h * 0.01


class Export(pllm.Workload):
    @pllm.export()
    def export_hidden(self, h=pllm.hidden(0)):
        return h


class Write(pllm.Workload):
    @property
    def written(self) -> pllm.InstanceState:
        return pllm.InstanceState({"v": (4,)})

    @pllm.write(state="written", fields=("v",))
    def write_value(self, v=pllm.parameter("written", "v")):
        return (v + 1,)


class Feed(Patch):
    """A fed state written into the activation patch's state: the one supported feed chain."""

    @property
    def source(self) -> pllm.InstanceState:
        return pllm.InstanceState({"v": (4,)}, feed="ordered")

    @pllm.write(state="patch_state", fields=("v",))
    def transform(self, v=pllm.parameter("source", "v")):
        return (v * 2,)


EFFECTS = {
    effect.__name__: effect
    for effect in (
        Train,
        TrainOther,
        TrainBatch,
        InForward,
        Adapt,
        Patch,
        Forward,
        Positions,
        AddLogits,
        Guide,
        Export,
        Write,
        Feed,
    )
}
CUSTOM_FORWARD_TAPS = "which a @pllm.custom_forward does not produce"
FEED_STATES = "may declare only the fed states"
FEED_READERS = "conditional reader requires only its publication"
POSITIONS = "PositionState needs a @pllm.custom_forward without"
# Every rejected combination and the conflict its message names; every other single effect and pair is admitted.
REJECTED = {
    "Write": "the workload has no effect on the model",
    "Train+TrainBatch": "one training mode per workload",
    "TrainOther+TrainBatch": "one training mode per workload",
    "Train+Forward": CUSTOM_FORWARD_TAPS,
    "Train+Positions": CUSTOM_FORWARD_TAPS,
    "TrainOther+Forward": CUSTOM_FORWARD_TAPS,
    "TrainOther+Positions": CUSTOM_FORWARD_TAPS,
    "Forward+AddLogits": CUSTOM_FORWARD_TAPS,
    "Forward+Guide": CUSTOM_FORWARD_TAPS,
    "Forward+Export": CUSTOM_FORWARD_TAPS,
    "Positions+AddLogits": CUSTOM_FORWARD_TAPS,
    "Positions+Guide": CUSTOM_FORWARD_TAPS,
    "Positions+Export": CUSTOM_FORWARD_TAPS,
    "Adapt+Forward": "corrects projections for the model's own packed rows",
    "Adapt+Positions": "corrects projections for the model's own packed rows",
    "InForward+Adapt": "does not yet combine with a pllm.model.Adapter",
    "InForward+Positions": POSITIONS,
    "Patch+Positions": POSITIONS,
    "Positions+Feed": POSITIONS,
    "AddLogits+Guide": "both send a result to vLLM's sampler",
    "Train+Feed": FEED_STATES,
    "TrainOther+Feed": FEED_STATES,
    "TrainBatch+Feed": FEED_STATES,
    "InForward+Feed": FEED_STATES,
    "Adapt+Feed": FEED_STATES,
    "Write+Feed": FEED_STATES,
    "AddLogits+Feed": FEED_READERS,
    "Guide+Feed": FEED_READERS,
    "Export+Feed": FEED_READERS,
}
CASES = tuple("+".join(names) for count in (1, 2) for names in itertools.combinations(EFFECTS, count))


@pytest.mark.parametrize("case", CASES)
def test_effects_compose_unless_they_conflict(case: str) -> None:
    chosen = tuple(EFFECTS[name] for name in case.split("+"))
    # Feed extends Patch and Positions extends Forward; such a pair is the subclass alone.
    bases = tuple(cls for cls in chosen if not any(other is not cls and issubclass(other, cls) for other in chosen))
    workload = type(f"Composed_{case.replace('+', '_')}", bases, {})()
    expected = REJECTED.get(case)
    if expected is None:
        facts, declaration = workload.specialize(FACTS, pllm.ExecutionOptions())
        workload.plan_backend(facts, declaration)
        return
    with pytest.raises(ValueError, match=expected):
        facts, declaration = workload.specialize(FACTS, pllm.ExecutionOptions())
        workload.plan_backend(facts, declaration)


@dataclass(frozen=True)
class Exported(pllm.Workload):
    events: list[pllm.io.TraceEvent] = field(default_factory=lambda: [], compare=False)

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        self.events.append(event)


def _step(workload: pllm.Workload) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """One guided CPU step over two decode rows: every state afterwards, the guidance and the exported rows."""
    facts = replace(
        FACTS, timeline=model_timeline(torch.device("cpu"), ("transition/full_root@1", "optimizer/adamw@1"))
    )
    execution = testing.compile_execution(workload, facts)
    identities = (testing.RequestIdentity("a", 0, 0), testing.RequestIdentity("b", 1, 0))
    refs = execution.admit(identities)
    hidden = torch.tensor([[0.5, -1.0, 2.0, 0.25], [1.5, 0.75, -0.5, 1.0]])
    guidance, result = testing.run_guided_step(execution, testing.step_input(identities, {BINDING: hidden}))
    execution.trace.drain()  # exports reach on_export on a delivery thread
    states: dict[str, torch.Tensor] = {}
    for ref in refs:
        latest = execution.arena.latest(ref.lineage_id)
        resource = str(execution.arena.root(latest).resource)
        index = sum(key.startswith(resource) for key in states)
        states[f"{resource}:{index}"] = torch.cat(tuple(value.flatten() for value in execution.snapshot(latest)))
    states["guidance"] = guidance.values
    assert isinstance(workload, Exported)
    return states, [event.tensor() for event in workload.events if event.tensor_name is not None]


def test_composed_effects_match_each_effect_alone() -> None:
    composed_states, composed_exports = _step(
        type("All", (Exported, Train, TrainOther, Export, Write, AddLogits), {})()
    )
    expected: dict[str, torch.Tensor] = {}
    exports: list[torch.Tensor] = []
    for effects in ((Train, AddLogits), (TrainOther, AddLogits), (Write, AddLogits), (Export, AddLogits)):
        states, observed = _step(type("One", (Exported, *effects), {})())
        expected.update(states)
        exports.extend(observed)
    assert composed_states.keys() == expected.keys()
    assert all(value.count_nonzero() for value in composed_states.values())  # every effect changed its state
    for key, value in expected.items():
        torch.testing.assert_close(composed_states[key], value, rtol=0, atol=0)
    assert len(composed_exports) == len(exports) == 1
    torch.testing.assert_close(composed_exports[0], exports[0], rtol=0, atol=0)
