"""What a backend reports about the model and device before a workload is compiled against it."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pllm.contracts.backend_attachment import ModelProbe
from pllm.contracts.semantic_graph import SchedulerTimeline
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts


@dataclass(frozen=True, slots=True)
class BackendCompileFacts:
    probe: ModelProbe
    timeline: SchedulerTimeline
    providers: tuple[PageBundleProviderFacts, ...]
    max_admitted_lineages: int
    state_memory_budget_bytes: int
    initialization_seed: int
    vocab_size: int = 0
    layer_count: int = 0

    @property
    def device(self) -> torch.device:
        return (
            torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_initialized() else 0)
            if self.probe.device_kind.value == "cuda"
            else torch.device("cpu")
        )
