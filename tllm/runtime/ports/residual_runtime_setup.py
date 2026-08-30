#!/usr/bin/env python3
"""Generic residual runtime setup extracted from side-train hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Protocol, cast

import torch

from tllm.contracts.runtime_context import RunnerLike
from tllm.ports.base import WriteMode, WriteRows
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.ports import residual_bindings as _residual_bindings
from tllm.runtime.ports import residual_capture_buffers as _residual_capture_buffers
from tllm.runtime.ports import residual_capture_hooks as _residual_capture_hooks
from tllm.runtime.ports import residual_write as _residual_write
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.ports.residual_write import ResidualWriteLease, WriteLeaseKey
from tllm.runtime.ports.tap_handle import TapHandle

if TYPE_CHECKING:
    from tllm.schedule import ScheduleGraph
    from tllm.runtime.torch_adapter import CudaLongHostStaging


@dataclass(frozen=True)
class ResidualRuntimeSetup:
    resolved_layers: Dict[str, torch.nn.Module]
    runtime_bindings: Dict[str, ResidualPathBinding]
    hook_spec: tuple[tuple[str, ...], str, str]
    rows: int
    compact_rows: int
    capture_full_rows: bool
    hidden_size: int
    hidden_dtype: torch.dtype
    source_resolved: str
    target_resolved: str
    native_hidden_path_aliases: Dict[str, str]
    capture_native_hidden: bool = False


class ConfigLike(Protocol):
    tap_layer_paths: list[str]
    graph_scratch_rows: int


class RuntimeLike(Protocol):
    config: ConfigLike
    residual_raw_paths: dict[ResidualLocator, str]
    dispatch_plan: DispatchPlan | None
    decode_row_idx: torch.Tensor | None
    decode_valid_mask: torch.Tensor | None
    decode_prompt_idx_buf: torch.Tensor | None
    decode_sample_idx_buf: torch.Tensor | None
    decode_identity_staging: "CudaLongHostStaging | None"
    decode_logit_row_selector_buf: torch.Tensor | None
    decode_logit_row_selector_staging: "CudaLongHostStaging | None"
    decode_compact_row_idx: torch.Tensor | None
    decode_compact_count: int
    decode_compact_row_ids: tuple[int, ...]
    decode_count: int
    tap_layers: dict[str, torch.nn.Module]
    tap_scratch: dict[str, torch.Tensor]
    tap_decode_hidden: dict[str, torch.Tensor]
    tap_decode_hidden_compact: dict[str, torch.Tensor]
    capture_full_residual_rows: bool
    residual_bindings: dict[str, ResidualPathBinding]
    write_leases: Dict[WriteLeaseKey, ResidualWriteLease]
    tap_handles: Dict[str, TapHandle]
    source_resolved_path: str
    target_resolved_path: str
    launch_consumer_from_hooks: bool
    native_schedule_graph: "ScheduleGraph | None"
    native_hidden_path_aliases: dict[str, str]


class CoreLike(Protocol):
    RUNTIME: RuntimeLike

    def _resolve_module_by_path_with_fallback(self, model: torch.nn.Module, path: str) -> tuple[torch.nn.Module, str]:
        ...

    def _infer_hidden_dtype(self, layer: torch.nn.Module) -> torch.dtype:
        ...

    def _runner_uses_compilation_or_cudagraph(self, runner: object) -> bool:
        ...


def resolve_runtime_setup(*, core: CoreLike, runner: object) -> ResidualRuntimeSetup:
    from tllm.backends.protocol import runner_model_module

    model = runner_model_module(runner)
    cfg = core.RUNTIME.config
    plan = getattr(core.RUNTIME, "dispatch_plan", None)
    required = set(plan.required_residual_layers()) if plan is not None and hasattr(plan, "required_residual_layers") else set()
    if plan is not None and hasattr(plan, "required_residual_writes"):
        # Written layers must be tapped too: the staged write applies inside
        # the tapped forward, before capture.
        required |= set(plan.required_residual_writes())

    raw_paths_by_locator = _residual_bindings.raw_paths_from_runtime(core.RUNTIME)
    native_graph = core.RUNTIME.native_schedule_graph
    if not required and native_graph is not None:
        # Native resources declare the exact logical locators they can bind.
        # The empty legacy plan must not expand that contract to the historical
        # ESamp source/target pair before the backend timeline is compiled.
        required = {
            (locator.layer, locator.site, locator.phase)
            for locator in raw_paths_by_locator
        }
    tap_paths = _residual_bindings.build_raw_tap_paths(
        raw_paths_by_locator=raw_paths_by_locator,
        required=required,
    )
    native_paths = (
        []
        if native_graph is None
        else list(dict.fromkeys(raw_paths_by_locator.values()))
    )
    for path in native_paths:
        if path not in tap_paths:
            tap_paths.append(path)
    if not tap_paths:
        fallback: list[str] = []
        for path in cfg.tap_layer_paths:
            if path not in fallback:
                fallback.append(path)
        tap_paths = fallback

    resolved_layers: Dict[str, torch.nn.Module] = {}
    resolve_alias: Dict[str, str] = {}
    for raw_path in tap_paths:
        layer, resolved = core._resolve_module_by_path_with_fallback(model, raw_path)
        resolved_layers[resolved] = layer
        resolve_alias[raw_path] = resolved

    runtime_bindings = _residual_bindings.build_resolved_bindings(
        raw_paths_by_locator=raw_paths_by_locator,
        resolve_alias=resolve_alias,
        required=required,
    )
    source_resolved = _residual_bindings.resolved_path_for_locator(
        runtime_bindings,
        _residual_bindings.default_source_locator(),
    ) or ""
    target_resolved = _residual_bindings.resolved_path_for_locator(
        runtime_bindings,
        _residual_bindings.default_target_locator(),
    ) or ""

    hook_spec = (
        tuple(sorted(resolved_layers.keys())),
        source_resolved,
        target_resolved,
    )

    rows = int(cfg.graph_scratch_rows or 0)
    if rows <= 0:
        rows = int(getattr(runner, "max_num_reqs", 0) or 0)
    if rows <= 0:
        raise RuntimeError("Please set graph_scratch_rows > 0 (or ensure runner.max_num_reqs > 0)")
    compact_rows = 0
    compact_lane_enabled = bool(getattr(cfg, "compact_capture_lane", False))
    if compact_lane_enabled and plan is not None and hasattr(plan, "has_residual_row_compaction") and plan.has_residual_row_compaction("first_per_prompt"):
        compact_rows = int(plan.max_residual_compact_rows("first_per_prompt")) if hasattr(plan, "max_residual_compact_rows") else 0
        if compact_rows <= 0:
            compact_rows = rows
    capture_full_rows = True
    if plan is not None and hasattr(plan, "requires_full_residual_capture"):
        capture_full_rows = bool(plan.requires_full_residual_capture())
    if bool(getattr(cfg, "enable_distiller_intervention", False)):
        capture_full_rows = True
    hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise RuntimeError("Cannot infer hidden_size from model.config.hidden_size")

    any_layer = next(iter(resolved_layers.values()))
    hidden_dtype = core._infer_hidden_dtype(any_layer)

    return ResidualRuntimeSetup(
        resolved_layers=resolved_layers,
        runtime_bindings=runtime_bindings,
        hook_spec=hook_spec,
        rows=rows,
        compact_rows=compact_rows,
        capture_full_rows=bool(capture_full_rows),
        hidden_size=hidden_size,
        hidden_dtype=hidden_dtype,
        source_resolved=source_resolved,
        target_resolved=target_resolved,
        native_hidden_path_aliases={
            raw_path: resolve_alias[raw_path] for raw_path in native_paths
        },
        capture_native_hidden=bool(native_paths),
    )


def apply_runtime_setup(*, core: CoreLike, runner: object, setup: ResidualRuntimeSetup) -> None:
    from tllm.backends.protocol import runner_device

    device = runner_device(runner)
    core.RUNTIME.decode_row_idx = torch.zeros((setup.rows,), device=device, dtype=torch.long)
    core.RUNTIME.decode_valid_mask = torch.zeros((setup.rows, 1), device=device, dtype=setup.hidden_dtype)
    core.RUNTIME.decode_prompt_idx_buf = torch.full(
        (setup.rows,),
        fill_value=-1,
        device=device,
        dtype=torch.long,
    )
    core.RUNTIME.decode_sample_idx_buf = torch.full(
        (setup.rows,),
        fill_value=-1,
        device=device,
        dtype=torch.long,
    )
    from tllm.runtime.torch_adapter import CudaLongHostStaging

    core.RUNTIME.decode_identity_staging = CudaLongHostStaging(
        setup.rows, device, channels=2
    )
    core.RUNTIME.decode_logit_row_selector_buf = torch.empty(
        (setup.rows,), device=device, dtype=torch.long
    )
    core.RUNTIME.decode_logit_row_selector_staging = CudaLongHostStaging(
        setup.rows, device
    )
    core.RUNTIME.decode_count = 0
    if int(setup.compact_rows) > 0:
        core.RUNTIME.decode_compact_row_idx = torch.zeros((int(setup.compact_rows),), device=device, dtype=torch.long)
    else:
        core.RUNTIME.decode_compact_row_idx = None
    core.RUNTIME.decode_compact_count = 0
    core.RUNTIME.decode_compact_row_ids = ()
    core.RUNTIME.capture_full_residual_rows = bool(setup.compact_rows <= 0 or setup.capture_full_rows)

    _residual_capture_buffers.initialize_runtime_tap_buffers(
        runtime=core.RUNTIME,
        resolved_layers=setup.resolved_layers,
        device=device,
        rows=setup.rows,
        hidden_size=setup.hidden_size,
        hidden_dtype=setup.hidden_dtype,
        compact_rows=int(setup.compact_rows),
    )
    core.RUNTIME.residual_bindings = setup.runtime_bindings
    core.RUNTIME.source_resolved_path = setup.source_resolved
    core.RUNTIME.target_resolved_path = setup.target_resolved
    core.RUNTIME.native_hidden_path_aliases = dict(
        setup.native_hidden_path_aliases
    )
    core.RUNTIME.launch_consumer_from_hooks = not core._runner_uses_compilation_or_cudagraph(runner)

    handles = _residual_capture_hooks.install_layer_forward_taps(
        core=cast(_residual_capture_hooks.ResidualCaptureCoreLike, core),
        runner=cast(RunnerLike, runner),
        resolved_layers=setup.resolved_layers,
        capture_native_hidden=setup.capture_native_hidden,
    )
    core.RUNTIME.tap_handles = dict(handles)

    plan = getattr(core.RUNTIME, "dispatch_plan", None)
    write_specs = []
    if plan is not None and hasattr(plan, "residual_writes"):
        for consumer_id, _flow, write in plan.residual_writes():
            locator = getattr(write, "locator")
            if not isinstance(locator, ResidualLocator):
                continue
            write_specs.append(
                (
                    consumer_id,
                    locator,
                    WriteMode(str(getattr(write, "mode", "replace") or "replace")),
                    WriteRows(str(getattr(write, "rows", "decode_only"))),
                    int(getattr(write, "order", 0)),
                )
            )
    core.RUNTIME.write_leases = _residual_write.build_write_leases(
        writes=write_specs,
        resolve_path=lambda locator: _residual_bindings.resolved_path_for_locator(
            setup.runtime_bindings, locator
        ),
        device=device,
        max_rows=int(setup.rows),
        hidden_size=int(setup.hidden_size),
        dtype=setup.hidden_dtype,
    )
