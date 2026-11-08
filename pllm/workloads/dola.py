"""DoLa (Chuang et al., 2024, "Decoding by Contrasting Layers"): contrast the final layer with the most divergent
early layer at every sampled token, through pllm.layer_logits and @pllm.add_logits."""

import math
from dataclasses import dataclass

import torch

import pllm


@dataclass(frozen=True)
class DoLaWorkload(pllm.Workload):
    """`layers` are the candidate early ("premature") layers; the final layer is the mature one.

    Per token, the candidate with the largest Jensen-Shannon divergence from the final layer is contrasted:
    log p_final - log p_early, over the tokens whose final probability is at least `plausibility` times the
    largest (the adaptive plausibility constraint); every other token is excluded.
    """

    layers: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14)
    plausibility: float = 0.1

    def lens(self) -> tuple[int, ...]:
        return (*self.layers, -1)

    @pllm.add_logits()
    def contrast(self, logits: torch.Tensor = pllm.layer_logits(lens)) -> torch.Tensor:
        final = logits[:, -1].log_softmax(-1)  # [rows, vocab]
        early = logits[:, :-1].log_softmax(-1)  # [rows, candidates, vocab]
        mixture = (0.5 * (final.exp().unsqueeze(1) + early.exp())).log()
        divergence = (final.exp().unsqueeze(1) * (final.unsqueeze(1) - mixture)).sum(-1) + (
            early.exp() * (early - mixture)
        ).sum(-1)
        chosen = early.gather(1, divergence.argmax(1).view(-1, 1, 1).expand(-1, 1, early.shape[-1])).squeeze(1)
        plausible = final >= final.max(-1, keepdim=True).values + math.log(self.plausibility)
        # The sampler adds the result to the model's logits, which are logits[:, -1].
        return torch.where(plausible, final - chosen, -torch.inf) - logits[:, -1]
