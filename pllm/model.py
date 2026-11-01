"""In-model interventions and their targets: adapters, activation patches and the causal model.

Workloads use these together with `pllm.train_in_forward`, `pllm.on` and the Visibility presets.
"""

from pllm.authoring import HiddenSite as HiddenSite
from pllm.contracts.backend_attachment import (
    ActivationPatchOperator as ActivationPatchOperator,
    DecodeRefresh as DecodeRefresh,
    DecodeWindow as DecodeWindow,
    MissingPolicy as MissingPolicy,
)
from pllm.contracts.parameter_overlay import (
    ParameterModuleKind as ParameterModuleKind,
    ParameterOverlayOperator as ParameterOverlayOperator,
    ParameterRole as ParameterRole,
    ParameterTarget as ParameterTarget,
)
from pllm.execution.causal_batch import CausalModel as CausalModel
from pllm.usage import ActivationPatch as ActivationPatch, Adapter as Adapter

__all__ = [
    "CausalModel",
    "ParameterTarget",
    "ParameterModuleKind",
    "ParameterRole",
    "ParameterOverlayOperator",
    "HiddenSite",
    "ActivationPatch",
    "Adapter",
    "ActivationPatchOperator",
    "DecodeWindow",
    "DecodeRefresh",
    "MissingPolicy",
]
