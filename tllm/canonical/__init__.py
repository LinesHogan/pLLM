"""Clean-room native core; legacy execution stacks are forbidden dependencies."""

from tllm.canonical.host_runtime import CanonicalHostRuntimeFacts
from tllm.canonical.backend_attachment import (
    compile_backend_attachment,
    lower_backend_attachment,
    validate_backend_attachment,
    validate_backend_attachment_physical,
    validate_backend_attachment_reconstruction,
)
from tllm.canonical.state_representation import (
    compile_state_representation,
    lower_state_physical_plan,
    validate_state_physical_plan,
    validate_state_representation,
    validate_state_representation_reconstruction,
)
from tllm.canonical.tensor_layout import (
    intern_tensor_layouts,
    validate_tensor_layouts,
)

__all__ = [
    "compile_backend_attachment",
    "CanonicalHostRuntimeFacts",
    "compile_state_representation",
    "lower_backend_attachment",
    "lower_state_physical_plan",
    "validate_state_physical_plan",
    "validate_state_representation",
    "validate_state_representation_reconstruction",
    "validate_backend_attachment",
    "validate_backend_attachment_physical",
    "validate_backend_attachment_reconstruction",
    "intern_tensor_layouts",
    "validate_tensor_layouts",
]
