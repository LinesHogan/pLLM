"""Bind author state-read declarations through Rust before building model regions."""

from pllm import _core
from pllm.contracts.visibility import StateVisibility, VisibilitySource
from pllm.execution.codec import decode_boundary, encode_boundary


def bind_visibility(source: VisibilitySource) -> StateVisibility:
    """Derive normalized read routes, including planned-only Exact overlays."""
    return decode_boundary(_core.derive_state_visibility(encode_boundary(source)), StateVisibility)
