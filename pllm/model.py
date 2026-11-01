"""In-model interventions and their targets: adapters, activation patches, projections and the causal model.

Workloads use these together with `pllm.train_in_forward` and `pllm.on`.
"""

from pllm.contracts.parameter_overlay import Projection as Projection
from pllm.execution.causal_batch import CausalModel as CausalModel
from pllm.usage import ActivationPatch as ActivationPatch, Adapter as Adapter

__all__ = [
    "ActivationPatch",
    "Adapter",
    "CausalModel",
    "Projection",
]
