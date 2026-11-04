"""Places in, and changes to, the base model's computation.

Membership rule: a value names a place in the base model's computation or a change to it: adapters and
activation patches (read-only edits), the projections they target, the Layers and LayerOutputs a
@pllm.custom_forward method receives, and the CausalModel pllm.causal_model injects.
"""

from pllm.contracts.parameter_overlay import Projection as Projection
from pllm.execution.attention import PagedKV as PagedKV
from pllm.execution.causal_batch import CausalModel as CausalModel
from pllm.execution.token_program import LayerOutputs as LayerOutputs, Layers as Layers
from pllm.usage import ActivationPatch as ActivationPatch, Adapter as Adapter

__all__ = [
    "ActivationPatch",
    "Adapter",
    "CausalModel",
    "LayerOutputs",
    "Layers",
    "Projection",
    "PagedKV",
]
