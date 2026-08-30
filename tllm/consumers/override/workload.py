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
    ConfigField,
    ConfigKind,
    NumericConstraint,
    State,
    TrainableState,
    WorkloadConfig,
    config_field,
    declared_config_field,
    extern,
    hidden,
    maybe_optimize,
    read,
    result,
    sampler,
    task,
    temporary,
    workload,
)


_OVERRIDE_ENABLED = ConfigField.boolean(True)
_OVERRIDE_LEARNING_RATE = ConfigField.floating(
    1e-3, NumericConstraint.POSITIVE
)


class OverRIDEConfig(WorkloadConfig):
    enabled: bool = declared_config_field(_OVERRIDE_ENABLED)
    hidden_size: int = config_field(
        default=0, kind=ConfigKind.INT, constraint=NumericConstraint.NONNEGATIVE
    )
    vocab_size: int = config_field(
        default=151936, kind=ConfigKind.INT, constraint=NumericConstraint.POSITIVE
    )
    rank: int = config_field(default=16, kind=ConfigKind.INT, constraint=NumericConstraint.POSITIVE)
    num_iterations: int = config_field(
        default=10, kind=ConfigKind.INT, constraint=NumericConstraint.POSITIVE
    )
    lambd: float = config_field(default=0.8, kind=ConfigKind.FLOAT)
    learning_rate: float = declared_config_field(_OVERRIDE_LEARNING_RATE)
    source_layer_path: str = config_field(default="model.model.norm", kind=ConfigKind.STR)


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


class OverRIDEWorkload(Workload[OverRIDEConfig]):
    heads: TrainableState[OverRIDEHeads] = State[OverRIDEHeads].replica(
        optimizer=AdamSpec.configured(_OVERRIDE_LEARNING_RATE)
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
        inputs={"sample_indices": sampler.sample_indices()},
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
        self, sample_indices: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if any(
            sample_idx < 0 or sample_idx >= self.config.num_iterations
            for sample_idx in sample_indices
        ):
            raise RuntimeError(
                "OverRIDE sample indices must be within the configured iterations"
            )
        head_count = self.config.num_iterations - 1
        previous = tuple(sample_idx - 1 for sample_idx in sample_indices)
        train_rows = tuple(
            row
            for row, sample_idx in enumerate(sample_indices)
            if sample_idx < head_count
        )
        return (
            torch.tensor([max(index, 0) for index in previous], dtype=torch.long),
            torch.tensor(
                [[self.config.lambd if index >= 0 else 0.0] for index in previous]
            ),
            torch.tensor(train_rows, dtype=torch.long),
            torch.tensor(
                [sample_indices[row] for row in train_rows],
                dtype=torch.long,
            ),
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
            if self.heads.down.shape[0] == 0:
                delta = hidden.new_zeros(
                    (hidden.shape[0], self.config.vocab_size)
                )
            else:
                delta = self.heads(hidden, previous_heads.to(hidden.device))
                delta = delta * scales.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
        return delta, hidden

    @task(
        inputs={
            "hidden": result("reweight", "kept_hidden"),
            "train_rows": result("select_heads", "train_rows"),
            "train_heads": result("select_heads", "train_heads"),
            "base_logits": sampler.logits(),
            "tokens": sampler.sampled_tokens(),
        },
        outputs={
            "loss": maybe_optimize(heads, at=Before("step.end"))
        },
        timing=Between("token.commit", "step.end"),
    )
    def train(
        self,
        hidden: torch.Tensor,
        train_rows: torch.Tensor,
        train_heads: torch.Tensor,
        base_logits: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        if train_rows.numel() == 0:
            return None
        rows = train_rows.to(hidden.device)
        selected_hidden = hidden.index_select(0, rows)
        selected_logits = base_logits.index_select(0, rows)
        selected_tokens = tokens.index_select(0, rows)
        logits = selected_logits + self.heads(
            selected_hidden, train_heads.to(hidden.device)
        ).to(selected_logits.dtype)
        return F.cross_entropy(logits, selected_tokens)


workload(
    OverRIDEWorkload,
    id="override",
    config=OverRIDEConfig,
    enabled=_OVERRIDE_ENABLED,
)


__all__ = [
    "OverRIDEConfig",
    "OverRIDEHeads",
    "OverRIDEWorkload",
]
