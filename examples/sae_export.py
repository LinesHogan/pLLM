"""Observe Qwen2.5-0.5B layers 3/5/17, printing and appending independent pickle records.

Use a fresh sae_features.pkl per run; replace the encoder factory with trained SAE weights.
Each export reads full hidden rows and bounds its encoded output to eight rows and 16 features.
Close the model handle to finish delivery before reading the file in the execution process.
"""

import pickle
from dataclasses import dataclass

import torch

import pllm


@dataclass(frozen=True)
class SAE(pllm.Workload):
    """A frozen linear feature encoder shared by three observation taps."""

    encoder = pllm.InstanceState(module=lambda m: torch.nn.Linear(m.hidden_size, 16, bias=False))

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        """Print and persist the same sequence, layers, request identities and CPU value."""
        record = (event.sequence, event.step_id, event.tensor_name, event.identities, event.tensor())
        print(record, flush=True)
        with open("sae_features.pkl", "ab") as output:
            pickle.dump(record, output)

    @pllm.export(name="sae.3", max_rows=8, max_cols=16)
    def layer3(
        self,
        h: torch.Tensor = pllm.hidden(3),
        encoder: torch.nn.Linear = pllm.module("encoder"),
    ) -> torch.Tensor:
        """Encode decode rows at layer 3."""
        return torch.relu(encoder(h))

    @pllm.export(name="sae.5", max_rows=8, max_cols=16)
    def layer5(
        self,
        h: torch.Tensor = pllm.hidden(5),
        encoder: torch.nn.Linear = pllm.module("encoder"),
    ) -> torch.Tensor:
        """Encode decode rows at layer 5."""
        return torch.relu(encoder(h))

    @pllm.export(name="sae.17", max_rows=8, max_cols=16)
    def layer17(
        self,
        h: torch.Tensor = pllm.hidden(17),
        encoder: torch.nn.Linear = pllm.module("encoder"),
    ) -> torch.Tensor:
        """Encode decode rows at layer 17."""
        return torch.relu(encoder(h))
