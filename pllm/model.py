"""In-model interventions and their targets: parameter overlays, activation patches, replay.

Workloads use these together with `pllm.overlay`, `pllm.on` and the Visibility presets.
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
from pllm.usage import (
    Activation as Activation,
    Overlay as Overlay,
    ParameterEffect as ParameterEffect,
)

__all__ = [
    "ParameterEffect",
    "ParameterTarget",
    "ParameterModuleKind",
    "ParameterRole",
    "ParameterOverlayOperator",
    "HiddenSite",
    "Activation",
    "Overlay",
    "ActivationPatchOperator",
    "DecodeWindow",
    "DecodeRefresh",
    "MissingPolicy",
]
