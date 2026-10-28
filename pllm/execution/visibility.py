"""Explain which logical writer each model/update read sees, as derived by Rust."""

from pllm import _core
from pllm.contracts.codec import encode_boundary
from pllm.contracts.visibility import VisibilitySource


def explain_visibility(source: VisibilitySource) -> str:
    """List symbolic readers and their writers; r=request, t=target, c=chunk."""
    relation = _core.explain_state_visibility(encode_boundary(source)).replace("; ", ";\n")
    return relation + (
        "\nOverlay entry relation; layer_route update boundaries govern later private reads."
        if source.scope == "overlay"
        else ""
    )
