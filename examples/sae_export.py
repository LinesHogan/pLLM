"""Observe Qwen2.5-0.5B layers 3/5/17, printing and appending independent pickle records.

Use a fresh sae_features.pkl per run; replace the encoder factory with trained SAE weights.
The 896-column input bound covers this model's hidden width; four rows fit the export byte limit.
One observation task reads all three taps: the middle output axis follows layers 3, 5, 17.
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

    def on_export(self, event: pllm.TraceEvent) -> None:
        """Print and persist the same sequence, layers, request identities and CPU value."""
        record = (event.sequence, event.step_id, event.tensor_name, event.identities, event.tensor())
        print(record, flush=True)
        with open("sae_features.pkl", "ab") as output:
            pickle.dump(record, output)

    @pllm.export(name="sae.3/5/17", max_rows=4, max_cols=896)
    def features(
        self,
        h3: torch.Tensor = pllm.hidden(3),
        h5: torch.Tensor = pllm.hidden(5),
        h17: torch.Tensor = pllm.hidden(17),
        encoder: torch.nn.Linear = pllm.module("encoder"),
    ) -> torch.Tensor:
        """Encode decode rows, keeping the layer axis explicit in every delivered value."""
        return torch.stack((torch.relu(encoder(h3)), torch.relu(encoder(h5)), torch.relu(encoder(h17))), dim=1)
