#!/usr/bin/env python3
"""Internal helper for installing residual capture forward wrappers."""

from __future__ import annotations

import hashlib
from types import MethodType
from typing import Dict, Protocol, cast

import torch

from tllm.contracts.runtime_context import RunnerLike
from tllm.runtime import active_targets
from tllm.runtime import hidden_event_bridge as _hidden_bridge
from tllm.runtime.ports import residual_write as _residual_write
from tllm.runtime.ports.residual_write import ResidualWriteLease, WriteLeaseKey
from tllm.runtime.ports.tap_handle import TapHandle
from tllm.runtime.vllm_patch import sampler_patch as _sampler_patch
from tllm.runtime.vllm_patch import compilation as _compilation
from tllm.runtime.stats import PlacementStats


class ProducerExecutorLike(Protocol):
    capture_resident_producer_ids: tuple[str, ...]

    def run_layer_post(self, ctx: object) -> None: ...


class ResidualCaptureRuntimeLike(Protocol):
    decode_row_idx: torch.Tensor | None
    decode_valid_mask: torch.Tensor | None
    decode_compact_row_idx: torch.Tensor | None
    decode_compact_count: int
    tap_decode_hidden: dict[str, torch.Tensor]
    tap_decode_hidden_compact: dict[str, torch.Tensor]
    write_leases: Dict[WriteLeaseKey, ResidualWriteLease]
    launch_consumer_from_hooks: bool
    dispatch_plan: object | None
    consumer: object | None
    source_resolved_path: str
    target_resolved_path: str
    producer_executor: ProducerExecutorLike | None
    capture_full_residual_rows: bool
    placement_stats: PlacementStats


class ResidualCaptureCoreLike(Protocol):
    RUNTIME: ResidualCaptureRuntimeLike


_CAPTURE_SCHEMA_VERSION = 2
_CAPTURE_ROW_INDEX_BUFFER = "_tllm_decode_row_idx"
_CAPTURE_OUTPUT_BUFFER = "_tllm_decode_residual"


def materialize_decode_residual(
    *,
    runtime: object,
    runner: RunnerLike,
    layer_path: str,
    hidden: torch.Tensor,
) -> None:
    """Materialize the compiler-selected decode residual producer.

    The fixed-shape gathers run while vLLM captures the wrapped model and are
    replayed from stable runtime buffers. Host-only lifecycle work is kept in
    the same selected materializer so an unselected producer has no runtime
    side effects.
    """
    if not active_targets.runtime_has_active_targets(
        cast(active_targets.RuntimeWithTargets, runtime)
    ):
        return

    typed_runtime = cast(ResidualCaptureRuntimeLike, runtime)
    decode_row_idx = typed_runtime.decode_row_idx
    decode_valid_mask = typed_runtime.decode_valid_mask
    if not isinstance(decode_row_idx, torch.Tensor) or not isinstance(
        decode_valid_mask, torch.Tensor
    ):
        return

    capture_full = bool(typed_runtime.capture_full_residual_rows)
    if capture_full:
        decode_buf = typed_runtime.tap_decode_hidden[layer_path]
        if int(hidden.shape[-1]) != int(decode_buf.shape[1]):
            raise RuntimeError(
                "tap hidden width mismatch: "
                f"tensor_hidden={int(hidden.shape[-1])} "
                f"decode_buf_hidden={int(decode_buf.shape[1])}"
            )
        torch.index_select(hidden, 0, decode_row_idx, out=decode_buf)

    compact_row_idx = typed_runtime.decode_compact_row_idx
    compact_buf = typed_runtime.tap_decode_hidden_compact.get(layer_path)
    if isinstance(compact_row_idx, torch.Tensor) and isinstance(compact_buf, torch.Tensor):
        compact_count = max(0, int(typed_runtime.decode_compact_count))
        if compact_count > 0:
            if int(hidden.shape[-1]) != int(compact_buf.shape[1]):
                raise RuntimeError(
                    "compact tap hidden width mismatch: "
                    f"tensor_hidden={int(hidden.shape[-1])} "
                    f"compact_buf_hidden={int(compact_buf.shape[1])}"
                )
            if compact_count > int(compact_row_idx.numel()) or compact_count > int(
                compact_buf.shape[0]
            ):
                raise RuntimeError(
                    "compact tap active row count exceeds compact capture capacity: "
                    f"active={compact_count} row_idx_capacity={int(compact_row_idx.numel())} "
                    f"compact_buf_capacity={int(compact_buf.shape[0])}"
                )
            torch.index_select(
                hidden,
                0,
                compact_row_idx[:compact_count],
                out=compact_buf[:compact_count],
            )

    if capture_full:
        _sampler_patch.maybe_capture_source_precompute(
            runtime=runtime,
            runner=runner,
            layer_path=layer_path,
        )


def install_layer_forward_taps(
    *,
    core: ResidualCaptureCoreLike,
    runner: RunnerLike,
    resolved_layers: dict[str, torch.nn.Module],
) -> dict[str, TapHandle]:
    handles: dict[str, TapHandle] = {}
    from tllm.runtime.graph_policy import GraphMode, resolve_graph_mode

    executor = core.RUNTIME.producer_executor
    capture_ids = () if executor is None else executor.capture_resident_producer_ids
    capture_resident = (
        "tllm.residual.decode" in capture_ids
        and resolve_graph_mode(runner) != GraphMode.EAGER
    )
    capture_full = bool(core.RUNTIME.capture_full_residual_rows)
    if capture_resident and not capture_full:
        raise RuntimeError(
            "capture-resident residual producer currently requires the static full-row lane"
        )
    for resolved_path, layer in resolved_layers.items():
        orig_forward = layer.forward
        static_row_idx = core.RUNTIME.decode_row_idx
        static_decode_buf = core.RUNTIME.tap_decode_hidden.get(resolved_path)
        if capture_resident and (
            not isinstance(static_row_idx, torch.Tensor)
            or not isinstance(static_decode_buf, torch.Tensor)
        ):
            raise RuntimeError(
                f"capture-resident producer buffers are unavailable for layer {resolved_path!r}"
            )
        if capture_resident:
            assert isinstance(static_row_idx, torch.Tensor)
            assert isinstance(static_decode_buf, torch.Tensor)
            _register_capture_buffer(
                layer=layer,
                name=_CAPTURE_ROW_INDEX_BUFFER,
                tensor=static_row_idx,
            )
            _register_capture_buffer(
                layer=layer,
                name=_CAPTURE_OUTPUT_BUFFER,
                tensor=static_decode_buf,
            )

        def _forward_with_tap(
            __layer,
            *args,
            __orig=orig_forward,
            __path=resolved_path,
            __capture_resident=capture_resident,
            **kwargs,
        ):
            out = __orig(*args, **kwargs)
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(tensor, torch.Tensor):
                return out
            if __capture_resident:
                torch.index_select(
                    tensor,
                    0,
                    __layer._buffers[_CAPTURE_ROW_INDEX_BUFFER],
                    out=__layer._buffers[_CAPTURE_OUTPUT_BUFFER],
                )
                return out
            write_leases = core.RUNTIME.write_leases
            if write_leases:
                _residual_write.apply_pending_writes(write_leases, __path, tensor)
            executor = core.RUNTIME.producer_executor
            if executor is not None:
                from tllm.runtime.producer_executor import LayerPostContext

                executor.run_layer_post(
                    LayerPostContext(
                        runtime=core.RUNTIME,
                        runner=runner,
                        layer_path=__path,
                        hidden=tensor,
                    )
                )
            if not active_targets.runtime_has_active_targets(
                cast(active_targets.RuntimeWithTargets, core.RUNTIME)
            ):
                return out

            _hidden_bridge.dispatch_layer_lifecycle_events(
                core=cast(_hidden_bridge.EventCoreLike, core),
                runner=runner,
                layer_path=__path,
                capture_enabled=bool(core.RUNTIME.launch_consumer_from_hooks),
            )
            return out

        layer.forward = MethodType(_forward_with_tap, layer)  # type: ignore[method-assign]
        handles[resolved_path] = TapHandle(
            layer_path=resolved_path,
            layer=layer,
            orig_forward=orig_forward,
        )
    if capture_resident:
        model = runner.model
        if model is None:
            raise RuntimeError("capture-resident producer requires runner.model")
        resolved_layer_items = tuple(resolved_layers.items())
        _compilation.install_capture_compile_identity(
            model=model,
            layer_modules=tuple(layer for _path, layer in resolved_layer_items),
            fingerprint=_capture_plan_fingerprint(resolved_layer_items),
        )
        core.RUNTIME.placement_stats.capture_resident_stage_install_count += len(handles)
    return handles


def _register_capture_buffer(
    *,
    layer: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
) -> None:
    existing = layer._buffers.get(name)
    if existing is not None:
        if existing is tensor:
            return
        raise RuntimeError(f"capture-resident buffer {name!r} is already registered")
    layer.register_buffer(name, tensor, persistent=False)


def _capture_plan_fingerprint(
    resolved_layers: tuple[tuple[str, torch.nn.Module], ...],
) -> str:
    rows: list[str] = [f"schema={_CAPTURE_SCHEMA_VERSION}", "op=aten.index_select.out"]
    for layer_path, layer in sorted(resolved_layers, key=lambda item: item[0]):
        row_idx = layer._buffers[_CAPTURE_ROW_INDEX_BUFFER]
        output = layer._buffers[_CAPTURE_OUTPUT_BUFFER]
        if not isinstance(row_idx, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise RuntimeError(
                f"capture-resident producer buffers are unavailable for layer {layer_path!r}"
            )
        rows.append(
            "|".join(
                (
                    layer_path,
                    str(tuple(row_idx.shape)),
                    str(row_idx.dtype),
                    str(tuple(output.shape)),
                    str(output.dtype),
                )
            )
        )
    return hashlib.sha256("\n".join(rows).encode("ascii")).hexdigest()[:16]


