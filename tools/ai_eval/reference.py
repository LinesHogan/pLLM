"""Hidden executable reference workloads used to validate this evaluation harness, never given to the author agent."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

import pllm
from pllm.execution.state_execution import StateExecution
from pllm.execution.tensor_binding import TensorOutput
from pllm.execution.tensor_frame import TensorFrame


def direction(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Paired population contrast."""
    return positive.mean(0) - negative.mean(0)


@dataclass(frozen=True)
class CAA(pllm.Workload):
    """Apply a population direction at the MLP residual."""

    alpha: float = 0.7
    steer = pllm.InstanceState({"vector": pllm.Field(lambda i, o: (i,))}, imported=True, dtype=torch.float32)

    @pllm.custom_residual(site="mlp", layers=(1,))
    def intervene(
        self, x: torch.Tensor, y: torch.Tensor, vector: torch.Tensor = pllm.parameter("steer", "vector")
    ) -> torch.Tensor:
        """Add the steering vector to the residual sum."""
        return (x + y + self.alpha * vector).to(x.dtype)


@dataclass(frozen=True)
class ITI(pllm.Workload):
    """Add already projected head directions to the attention residual."""

    alpha: float = 0.6
    heads = pllm.InstanceState({"directions": pllm.Field(lambda i, o: (2, i))}, imported=True, dtype=torch.float32)

    @pllm.custom_residual(site="attention", layers=(1,))
    def intervene(
        self, x: torch.Tensor, y: torch.Tensor, directions: torch.Tensor = pllm.parameter("heads", "directions")
    ) -> torch.Tensor:
        """Sum the selected heads in residual coordinates."""
        return (x + y + self.alpha * directions.sum(0)).to(x.dtype)


@dataclass(frozen=True)
class DoLa(pllm.Workload):
    """Select the most divergent premature distribution per row."""

    tau: float = 0.1

    @pllm.add_logits()
    def contrast(self, logits: torch.Tensor = pllm.layer_logits((0, 1, -1))) -> torch.Tensor:
        """Return an additive sampler delta."""
        lp = logits.log_softmax(-1)
        final, early = lp[:, -1:], lp[:, :-1]
        mix = ((final.exp() + early.exp()) / 2).log()
        jsd = (final.exp() * (final - mix) + early.exp() * (early - mix)).sum(-1)
        chosen = early[torch.arange(len(logits), device=logits.device), jsd.argmax(-1)]
        mask = final[:, 0].exp() >= self.tau * final[:, 0].exp().amax(-1, keepdim=True)
        return torch.where(mask, final[:, 0] - chosen, -torch.inf) - logits[:, -1]


class PromptLoRA(pllm.Workload):
    """Generate a request overlay at prompt completion."""

    factors = pllm.RequestState(
        {"a": pllm.Field(lambda i, o: (2, i)), "b": pllm.Field(lambda i, o: (o, 2))}, dtype=torch.float32
    )
    adapter = pllm.model.Adapter("factors", (pllm.model.Projection("q_proj", 1),), operator="low_rank")

    @pllm.on(pllm.Phase.AFTER_FORWARD)
    @pllm.write(state="factors", fields=("a_0", "b_0"))
    def produce(
        self,
        h: torch.Tensor = pllm.hidden(0, tokens="prefill"),
        a: torch.Tensor = pllm.parameter("factors", "a_0"),
        b: torch.Tensor = pllm.parameter("factors", "b_0"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Publish both factors together without editing templates."""
        return torch.stack((h.tanh(), (2 * h).tanh()), 1).to(a), torch.stack((h, -h), -1).div(h.shape[-1]).to(b)


def initialize_student(fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
    """A learns a nonzero basis; B initially preserves the base path."""
    fields["a"].fill_(0.05)
    fields["b"].zero_()


class ESamp(pllm.Workload):
    """A request-local hidden predictor with decode-time distillation."""

    student = pllm.RequestState(
        {"a": lambda m: (2, m.hidden_size), "b": lambda m: (m.hidden_size, 2)},
        dtype=torch.float32,
        optimizer=pllm.AdamW(lr=0.01),
        initialize=initialize_student,
    )

    @pllm.guide_sampling(beta=0.5)
    def predict(
        self,
        x: torch.Tensor = pllm.hidden(0),
        a: torch.Tensor = pllm.parameter("student", "a"),
        b: torch.Tensor = pllm.parameter("student", "b"),
    ) -> torch.Tensor:
        """Apply batched low-rank factors."""
        return ((x.unsqueeze(1) @ a.transpose(-1, -2)) @ b.transpose(-1, -2)).squeeze(1)

    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.train(state="student")
    def loss(
        self,
        x: torch.Tensor = pllm.hidden(0),
        target: torch.Tensor = pllm.hidden(2),
        a: torch.Tensor = pllm.parameter("student", "a"),
        b: torch.Tensor = pllm.parameter("student", "b"),
    ) -> torch.Tensor:
        """Per-row MSE; the base model is detached."""
        prediction = ((x.unsqueeze(1) @ a.transpose(-1, -2)) @ b.transpose(-1, -2)).squeeze(1)
        return (prediction - target.detach()).square().mean(-1)


class Memory(pllm.Workload):
    """Chunk-local fast weights owned by the framework."""

    memory = pllm.RequestState({"M": lambda m: (m.hidden_size, m.hidden_size)}, dtype=torch.float32)

    @pllm.custom_residual(site="mlp", layers=(1,), state="memory", chunk=2)
    def remember(
        self, x: torch.Tensor, y: torch.Tensor, M: torch.Tensor = pllm.parameter("memory", "M")
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the chunk-entry matrix and return its next value."""
        k = torch.nn.functional.normalize(x.float(), dim=-1)
        r = k @ M.T
        return (x.float() + y.float() + 0.5 * r).to(x.dtype), M - 0.3 / len(x) * (r - y.float()).T @ k


@dataclass(frozen=True)
class LoopedDepth(pllm.Workload):
    """Reuse the tail while each row remains active."""

    threshold: float = 0.4

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        """A bounded loop with monotone row exit masks."""
        h = layers(0, 1)[0]
        active = torch.ones_like(h[:, :1], dtype=torch.bool)
        for _ in range(3):
            active = active & (h.square().mean(-1, keepdim=True) > self.threshold)
            h = layers(1, hidden=h, active=active)[-1]
        return h


class TTRL(pllm.Workload):
    """A shared reward head for completion feedback."""

    head = pllm.InstanceState({"w": lambda m: (m.hidden_size,)}, dtype=torch.float32, optimizer=pllm.AdamW(lr=0.01))

    @pllm.train(state="head")
    def loss(
        self,
        hidden: torch.Tensor = pllm.batch("hidden"),
        targets: torch.Tensor = pllm.batch("targets"),
        weights: torch.Tensor = pllm.batch("weights"),
        w: torch.Tensor = pllm.parameter("head", "w"),
    ) -> pllm.UpdateOutput:
        """Normalize over valid answers and suppress empty updates."""
        total = weights.sum()
        loss = (weights * (hidden @ w - targets).square()).sum() / total.clamp_min(1)
        changed = total > 0
        return pllm.UpdateOutput(loss, changed, changed.to(torch.int64), (weights > 0).sum())


def vote(answers: list[str], groups: list[int]) -> tuple[list[float], list[float]]:
    """Choose a stable majority within each prompt."""
    labels = [answer.strip().lower() for answer in answers]
    winners: dict[int, str] = {}
    for group in dict.fromkeys(groups):
        counts: dict[str, int] = {}
        for label, key in zip(labels, groups, strict=True):
            if key == group and label:
                counts[label] = counts.get(label, 0) + 1
        winners[group] = max(counts, key=lambda label: counts[label]) if counts else ""
    return (
        [float(bool(label) and label == winners[group]) for label, group in zip(labels, groups, strict=True)],
        [float(bool(label)) for label in labels],
    )


def feedback(
    execution: StateExecution[pllm.host.StepInput, TensorFrame, TensorOutput],
    hidden: torch.Tensor,
    answers: list[str],
    groups: list[int],
) -> TensorOutput:
    """Feed one completion batch to an admitted host execution."""
    targets, weights = vote(answers, groups)
    batch = pllm.host.WeightedRowBatch(hidden, hidden.new_tensor(targets), hidden.new_tensor(weights))
    return execution.step(pllm.host.StepInput.of_batch(batch))


WORKLOADS = {
    "caa": CAA,
    "dola": DoLa,
    "iti": ITI,
    "prompt_lora": PromptLoRA,
    "esamp": ESamp,
    "memory": Memory,
    "looped_depth": LoopedDepth,
    "ttrl": TTRL,
}
