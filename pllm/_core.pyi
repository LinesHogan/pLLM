"""Type stub for the Rust extension module ``pllm._core`` (pyo3)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

def intern_tensor_layouts(request_json: str) -> str: ...
def compile_schedule(graph_json: str, timeline_json: str, state_json: str | None = None) -> str: ...
def compile_state_schedule(
    graph_json: str,
    timeline_json: str,
    request_json: str,
) -> tuple[str, str]: ...

class RowLabeler:
    """Admission generations and bulk packed-row labeling owned by one session."""
    def admit(self) -> int: ...
    def retire(self, id: int) -> None: ...
    def prepare(
        self,
        scheduled: np.ndarray,
        prompt: np.ndarray,
        computed: np.ndarray,
        ids: npt.NDArray[np.int64],
        keep: bool,
    ) -> tuple[np.ndarray, npt.NDArray[np.int64], list[npt.NDArray[np.int64]], list[npt.NDArray[np.int64]]]: ...
