"""Explain which logical writer each model/update read sees, as derived by Rust."""

from pllm import _core
from pllm.contracts.visibility import VisibilitySource
from pllm.execution.codec import encode_boundary


def explain_visibility(source: VisibilitySource) -> str:
    """List symbolic readers and their writers; r=request, t=target, c=chunk."""
    return _core.explain_state_visibility(encode_boundary(source)).replace("; ", ";\n")
