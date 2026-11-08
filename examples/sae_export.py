"""Observe Qwen2.5-0.5B layers 3/5/17 at every computed token, printing and appending independent pickle records.

Use a fresh sae_features.pkl per run; replace the encoder factory with trained SAE weights.
Each export encodes the hidden rows of every prompt and generated token a step computes (at most
max_num_batched_tokens, 512 by default) and keeps 16 features per row; each record carries the rows' token
positions. Close the model handle to finish delivery before reading the file in the execution process.
"""

import pickle
from dataclasses import dataclass

import torch

import pllm


@dataclass(frozen=True)
class SAE(pllm.Workload):
    """A frozen linear feature encoder shared by three observation taps."""

    encoder = pllm.InstanceState(init=pllm.FromModule(lambda m: torch.nn.Linear(m.hidden_size, 16, bias=False)))

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        """Print and persist the same sequence, layers, request identities, token positions and CPU value."""
        record = (event.sequence, event.step_id, event.tensor_name, event.identities, event.positions, event.tensor())
        print(record, flush=True)
        with open("sae_features.pkl", "ab") as output:
            pickle.dump(record, output)

    @pllm.export(name="sae.3", max_rows=512, max_cols=16, overflow="block")
    def layer3(
        self,
        h: torch.Tensor = pllm.hidden(3, tokens="all"),
        encoder: torch.nn.Linear = pllm.module("encoder"),
    ) -> torch.Tensor:
        """Encode every computed token's row at layer 3."""
        return torch.relu(encoder(h))

    @pllm.export(name="sae.5", max_rows=512, max_cols=16, overflow="block")
    def layer5(
        self,
        h: torch.Tensor = pllm.hidden(5, tokens="all"),
        encoder: torch.nn.Linear = pllm.module("encoder"),
    ) -> torch.Tensor:
        """Encode every computed token's row at layer 5."""
        return torch.relu(encoder(h))

    @pllm.export(name="sae.17", max_rows=512, max_cols=16, overflow="block")
    def layer17(
        self,
        h: torch.Tensor = pllm.hidden(17, tokens="all"),
        encoder: torch.nn.Linear = pllm.module("encoder"),
    ) -> torch.Tensor:
        """Encode every computed token's row at layer 17."""
        return torch.relu(encoder(h))
