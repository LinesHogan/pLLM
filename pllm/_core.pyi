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
    def admit(self, words: npt.NDArray[np.int64]) -> int: ...
    def retire(self, id: int) -> None: ...
    def prepare(
        self,
        scheduled: np.ndarray,
        prompt: np.ndarray,
        computed: np.ndarray,
        ids: npt.NDArray[np.int64],
        keep: bool,
    ) -> tuple[
        np.ndarray,
        npt.NDArray[np.int64],
        list[npt.NDArray[np.int64]],
        list[npt.NDArray[np.int64]],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]: ...

_Int64Array = npt.NDArray[np.int64]

class ResourceProjection:
    def __init__(self, fields: _Int64Array) -> None: ...
    def bind(self, words: _Int64Array, resource: int, layout: int, lineage: int, row: int) -> None: ...
    def retire(self, resource: int, lineage: int) -> None: ...
    def project(
        self, words: _Int64Array, offsets: _Int64Array, requests: _Int64Array
    ) -> list[tuple[_Int64Array, _Int64Array]]: ...

def ref_positions(
    layouts: _Int64Array,
    lineages: _Int64Array,
    versions: _Int64Array,
    selected_layouts: _Int64Array,
    selected_lineages: _Int64Array,
    selected_versions: _Int64Array,
) -> _Int64Array: ...
def replace_versions(versions: _Int64Array, positions: _Int64Array, after: _Int64Array) -> _Int64Array: ...
def row_plan(
    rows: _Int64Array,
    packed: _Int64Array,
    links: _Int64Array,
    identity_words: _Int64Array,
    identity_offsets: _Int64Array,
    selection_requests: _Int64Array,
    state_indices: _Int64Array,
    versions: _Int64Array,
    instance: bool,
) -> tuple[_Int64Array, _Int64Array, _Int64Array]: ...
def fill_table(
    host: _Int64Array,
    columns: _Int64Array,
    sources: _Int64Array,
    updated: _Int64Array,
    destinations: _Int64Array,
    inputs: _Int64Array,
    rows: int,
    states: int,
) -> None: ...
def select_states(requests: _Int64Array, states: _Int64Array, selected: _Int64Array) -> _Int64Array: ...
def rewind_versions(
    lineages: _Int64Array,
    versions: _Int64Array,
    history_lineages: list[_Int64Array],
    history_versions: list[_Int64Array],
) -> _Int64Array: ...
def read_versions(
    versions: _Int64Array, owners: _Int64Array, positions: _Int64Array, after: _Int64Array, owner: int
) -> tuple[_Int64Array, _Int64Array]: ...
def refs_equal(
    layouts: _Int64Array,
    lineages: _Int64Array,
    versions: _Int64Array,
    other_layouts: _Int64Array,
    other_lineages: _Int64Array,
    other_versions: _Int64Array,
) -> bool: ...
