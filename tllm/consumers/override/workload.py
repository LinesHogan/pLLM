"""OverRIDE iterative logit reweighting expressed as three signed tasks."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.kit.workload import Workload
from tllm.workload import (
    AdamSpec,
    Before,
    Between,
    NumericConstraint,
    State,
    WorkloadConfig,
    config_field,
    extern,
    hidden,
    optimize,
    read,
    result,
    sampler,
    task,
    temporary,
    workload,
)


class OverRIDEConfig(WorkloadConfig):
    enabled: bool = config_field(default=True)
    hidden_size: int = config_field(
        default=0, constraint=NumericConstraint.NONNEGATIVE
    )
    vocab_size: int = config_field(
        default=151936, constraint=NumericConstraint.POSITIVE
    )
    rank: int = config_field(default=16, constraint=NumericConstraint.POSITIVE)
    num_iterations: int = config_field(
        default=10, constraint=NumericConstraint.POSITIVE
    )
    lambd: float = config_field(default=0.8)
    learning_rate: float = config_field(
        default=1e-3, constraint=NumericConstraint.POSITIVE
    )
    source_layer_path: str = config_field(default="model.model.norm")


class OverRIDEHeads(torch.nn.Module):
    def __init__(
        self,
        head_count: int,
        rank: int,
        hidden_size: int,
        vocab_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.down = torch.nn.Parameter(
            torch.empty((head_count, rank, hidden_size), device=device, dtype=dtype)
        )
        torch.nn.init.normal_(self.down, mean=0.0, std=0.02)
        self.up = torch.nn.Parameter(
            torch.zeros((head_count, vocab_size, rank), device=device, dtype=dtype)
        )

    def forward(self, hidden: torch.Tensor, head_idx: torch.Tensor) -> torch.Tensor:
        down = self.down.index_select(0, head_idx)
        up = self.up.index_select(0, head_idx)
        return torch.bmm(
            torch.bmm(hidden.unsqueeze(1), down.transpose(1, 2)),
            up.transpose(1, 2),
        ).squeeze(1)


def iteration_from_request_id(request_id: str) -> int:
    prefix, separator, _suffix = request_id.partition("_")
    return int(prefix) + 1 if separator and prefix.isdigit() else 1


@workload(id="override", config=OverRIDEConfig, enabled=OverRIDEConfig.enabled)
class OverRIDEWorkload(Workload[OverRIDEConfig]):
    heads: State[OverRIDEHeads] = State.replica(
        optimizer=AdamSpec(lr=OverRIDEConfig.learning_rate)
    )

    @heads.initialize
    def init_heads(self, spec: RuntimeSpecializationSpec) -> OverRIDEHeads:
        hidden_size = self.config.hidden_size or spec.hidden_size
        return OverRIDEHeads(
            head_count=self.config.num_iterations - 1,
            rank=self.config.rank,
            hidden_size=hidden_size,
            vocab_size=self.config.vocab_size,
            device=spec.device,
            dtype=spec.dtype,
        )

    @task(
        inputs={"request_ids": sampler.request_ids()},
        outputs={
            "previous_heads": temporary(),
            "scales": temporary(),
            "train_rows": temporary(),
            "train_heads": temporary(),
        },
        timing=Between("model.execute", "sampler.pre_draw"),
        implementation=extern("tllm.override:select_iteration_heads@1"),
    )
    def select_heads(
        self, request_ids: tuple[str, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        iterations = tuple(iteration_from_request_id(value) for value in request_ids)
        head_count = self.config.num_iterations - 1
        previous = tuple(iteration - 2 for iteration in iterations)
        train_rows = tuple(
            row for row, iteration in enumerate(iterations) if iteration <= head_count
        )
        return (
            torch.tensor([max(index, 0) for index in previous], dtype=torch.long),
            torch.tensor(
                [[self.config.lambd if index >= 0 else 0.0] for index in previous]
            ),
            torch.tensor(train_rows, dtype=torch.long),
            torch.tensor([iterations[row] - 1 for row in train_rows], dtype=torch.long),
        )

    @task(
        inputs={
            "hidden": hidden(
                0,
                "block_output",
                "decode",
                path=OverRIDEConfig.source_layer_path,
            ),
            "previous_heads": result("select_heads", "previous_heads"),
            "scales": result("select_heads", "scales"),
        },
        outputs={
            "delta": sampler.add_logits(
                enabled=OverRIDEConfig.enabled, required=True
            ),
            "kept_hidden": temporary(),
        },
        effects=(read(heads),),
        timing=Between("model.execute", "sampler.pre_draw"),
    )
    def reweight(
        self,
        hidden: torch.Tensor,
        previous_heads: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            delta = self.heads(hidden, previous_heads.to(hidden.device))
            delta = delta * scales.to(device=hidden.device, dtype=hidden.dtype)
        return delta, hidden

    @task(
        inputs={
            "hidden": result("reweight", "kept_hidden"),
            "train_rows": result("select_heads", "train_rows"),
            "train_heads": result("select_heads", "train_heads"),
            "base_logits": sampler.logits(),
            "tokens": sampler.sampled_tokens(),
        },
        outputs={"loss": optimize(heads, at=Before("step.end"))},
        timing=Between("token.commit", "step.end"),
    )
    def train(
        self,
        hidden: torch.Tensor,
        train_rows: torch.Tensor,
        train_heads: torch.Tensor,
        base_logits: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        rows = train_rows.to(hidden.device)
        selected_hidden = hidden.index_select(0, rows)
        selected_logits = base_logits.index_select(0, rows)
        selected_tokens = tokens.index_select(0, rows)
        logits = selected_logits + self.heads(
            selected_hidden, train_heads.to(hidden.device)
        ).to(selected_logits.dtype)
        return F.cross_entropy(logits, selected_tokens)


__all__ = [
    "OverRIDEConfig",
    "OverRIDEHeads",
    "OverRIDEWorkload",
    "iteration_from_request_id",
]
