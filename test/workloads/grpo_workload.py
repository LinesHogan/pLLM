"""Small tutorial-style TTRL policy and GRPO subclass used by CPU and real-engine checks."""

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from vllm import RequestOutput, SamplingParams

import pllm
from pllm.backends.vllm.runtime import VllmLlm
from pllm.execution.tensor_declaration import TensorOutput
from pllm.execution.tensor_frame import TensorFrame
from pllm.host import CausalTrainBatch
from pllm.workloads.ttrl import boxed_label


@dataclass(frozen=True)
class TTRL(pllm.Workload):
    """Tutorial 05's majority-vote LoRA policy and sampling/feedback loop."""

    optimizer: pllm.AdamW = pllm.AdamW(1e-4)
    rank: int = 2
    alpha: int = 2

    @property
    def factors(self) -> pllm.InstanceState:
        """Train q/v factors while the decoder stays frozen."""
        return pllm.InstanceState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            optimizer=self.optimizer,
            initialize=self.initialize,
        )

    @property
    def adapter(self) -> pllm.model.Adapter:
        """Apply the same low-rank delta during training and generation."""
        return pllm.model.Adapter(
            "factors",
            (pllm.model.Projection("q_proj"), pllm.model.Projection("v_proj")),
            operator="low_rank",
            scale=self.alpha / self.rank,
        )

    def initialize(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        """Seed down factors and start up factors at zero, as in the tutorial."""
        for name, value in fields.items():
            if name.startswith("a_"):
                torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
            else:
                value.zero_()

    @pllm.on(pllm.Phase.BEFORE_FORWARD)
    @pllm.train(state="factors")
    def update(
        self,
        input_ids: torch.Tensor = pllm.batch("input_ids"),
        labels: torch.Tensor = pllm.batch("labels"),
        attention_mask: torch.Tensor | None = pllm.batch("attention_mask"),
        model: pllm.model.CausalModel = pllm.causal_model("adapter"),
    ) -> torch.Tensor:
        """Fit answer tokens selected by the majority vote."""
        logits = model(input_ids, attention_mask=attention_mask)[:, :-1].float()
        return F.cross_entropy(logits.transpose(1, 2), labels[:, 1:], ignore_index=-100)

    def advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Select majority answers for supervised TTRL."""
        return rewards

    def rewards(self, output: RequestOutput) -> torch.Tensor:
        """Score samples by agreement with their group's majority boxed answer."""
        labels = [boxed_label(sample.text) for sample in output.outputs]
        counts = Counter(label for label in labels if label)
        majority = counts.most_common(1)[0][0] if counts else ""
        return torch.tensor([float(bool(label) and label == majority) for label in labels])

    def training_batch(
        self, outputs: list[RequestOutput], *, max_rows: int, pad_token: int
    ) -> CausalTrainBatch[torch.Tensor] | None:
        """Pad answer-only labels and keep sample weights beside their token rows on the host."""
        rows: list[tuple[list[int], list[int], float]] = []
        for output in outputs:
            rewards = self.rewards(output)
            weights = self.advantages(rewards)
            assert output.prompt_token_ids is not None
            rows.extend(
                (list(output.prompt_token_ids), list(sample.token_ids), float(weight))
                for sample, weight in zip(output.outputs, weights, strict=True)
                if weight != 0 and sample.token_ids
            )
        rows = rows[:max_rows]
        if not rows:
            return None
        width = max(len(prompt) + len(answer) for prompt, answer, _ in rows)
        ids = torch.full((len(rows), width), pad_token, dtype=torch.int64)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for index, (prompt, answer, _) in enumerate(rows):
            length = len(prompt) + len(answer)
            ids[index, :length] = torch.tensor(prompt + answer)
            labels[index, len(prompt) : length] = torch.tensor(answer)
            mask[index, :length] = 1
        return CausalTrainBatch(ids, labels, mask, torch.tensor([weight for _, _, weight in rows]))

    def run(
        self,
        handle: VllmLlm[TensorFrame, TensorOutput],
        prompts: list[str],
        sampling: SamplingParams,
        *,
        rounds: int,
        max_rows: int,
        pad_token: int,
    ) -> list[CausalTrainBatch[torch.Tensor] | None]:
        """Sample once, then consume each round's feedback exactly once before sampling the next policy."""
        outputs = handle.generate(prompts, sampling)
        batches: list[CausalTrainBatch[torch.Tensor] | None] = []
        for _ in range(rounds):
            batch = self.training_batch(outputs, max_rows=max_rows, pad_token=pad_token)
            batches.append(batch)
            outputs = handle.generate(prompts, sampling, batch=batch)
        return batches


@dataclass(frozen=True)
class GRPO(TTRL):
    """One on-policy update with group-normalized majority rewards; no repeated epochs or clipping."""

    def advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Constant-reward groups carry no signal; population std also handles a one-sample group."""
        return (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-4)

    @pllm.on(pllm.Phase.BEFORE_FORWARD)
    @pllm.train(state="factors")
    def update(
        self,
        input_ids: torch.Tensor = pllm.batch("input_ids"),
        labels: torch.Tensor = pllm.batch("labels"),
        attention_mask: torch.Tensor | None = pllm.batch("attention_mask"),
        model: pllm.model.CausalModel = pllm.causal_model("adapter"),
        weights: torch.Tensor = pllm.batch("weights"),
    ) -> torch.Tensor:
        """Weight each sequence's mean answer-token negative log probability by its advantage."""
        logits = model(input_ids, attention_mask=attention_mask)[:, :-1].float()
        targets = labels[:, 1:]
        losses = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100, reduction="none")
        mask = targets != -100
        per_sequence = (losses * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        return (weights * per_sequence).mean()
