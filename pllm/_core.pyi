"""Type stub for the Rust extension module ``pllm._core`` (pyo3)."""

from __future__ import annotations

def intern_tensor_layouts(request_json: str) -> str: ...
def compile_schedule(graph_json: str, timeline_json: str, state_json: str | None = None) -> str: ...
def compile_state_schedule(
    graph_json: str,
    timeline_json: str,
    request_json: str,
) -> tuple[str, str]: ...
