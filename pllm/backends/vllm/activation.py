"""ActivationPatch's masked residual edits and pin-authorized device field slots.

ModelSites calls the same tensor edit in eager execution, model graphs and custom
forward waves. PTTS owns the fields; each step uploads a mask and exact field slots only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F

from pllm.contracts.backend_attachment import ActivationPatchBinding, DecodeRefresh, DecodeWindow
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import StateResource
from pllm.contracts.state_ref import StateRef
from pllm.execution.cuda_stream import PinnedStaging
from pllm.execution.layer_route import (
    RouteSlotGather,
    masked_patch_add,
    masked_patch_add_into,
    masked_patch_affine,
    masked_patch_affine_into,
    requires_step_publish,
)
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.state_execution import StateTaskContext


def activation_delta(hidden: torch.Tensor, fields: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Add a direction or the low-rank affine edit R (W h + b - R^T h)."""
    if len(fields) == 1:
        return fields[0].to(hidden.dtype)
    weight, bias, rotation = fields
    if weight.ndim == 2:
        return (F.linear(hidden, weight, bias) - hidden @ rotation) @ rotation.T
    rows = hidden.unsqueeze(-2)
    coefficients = rows @ weight.transpose(-1, -2) + bias.unsqueeze(-2) - rows @ rotation
    return (coefficients @ rotation.transpose(-1, -2)).squeeze(-2)


@dataclass
class _Patch:
    """A binding's stable slot table and mask; field pools stay owned by PTTS."""

    binding: ActivationPatchBinding
    pools: tuple[torch.Tensor, ...]
    table: torch.Tensor
    coeff: torch.Tensor | None = None

    def apply(self, rows: torch.Tensor | None, hidden: torch.Tensor) -> torch.Tensor:
        table = self.table[:, : len(hidden)] if rows is None else self.table.index_select(1, rows)
        if len(self.pools) == 1:
            return masked_patch_add(hidden, table, self.pools[0])
        if self.coeff is not None:
            out = hidden.contiguous().clone()
            masked_patch_affine_into(out, table, *self.pools, self.coeff[: len(out)])
            return out
        return masked_patch_affine(hidden, table, *self.pools)

    def apply_into(self, rows: torch.Tensor | None, target: torch.Tensor) -> None:
        table = self.table[:, : len(target)] if rows is None else self.table.index_select(1, rows)
        if len(self.pools) == 1:
            masked_patch_add_into(target, table, self.pools[0])
            return
        if self.coeff is not None:
            masked_patch_affine_into(target, table, *self.pools, self.coeff[: len(target)])
            return
        mask, slots = table[0].bool(), table[1:]
        if not target.is_cuda and not bool(mask.any()):
            return
        fields = tuple(pool.index_select(0, indices) for pool, indices in zip(self.pools, slots, strict=True))
        target.copy_(torch.where(mask[:, None], target + activation_delta(target, fields), target))


class VllmActivationPatch:
    """ActivationPatch's device masks and field slots, shared by all residual-site callers."""

    def __init__(self, bindings: tuple[ActivationPatchBinding, ...], rows: int, arena: PageBundleArena) -> None:
        self.bindings = bindings
        self.patches: dict[tuple[StateResource, int], _Patch] = {}
        for binding in bindings:
            pools = tuple(
                view.read(arena.field_pool(binding.resource, view.field), pooled=True) for view in binding.fields
            )
            coeff = (
                torch.empty((rows, pools[0].shape[1]), dtype=torch.float32, device=arena.device)
                if len(pools) == 3 and arena.device.type == "cuda" and pools[0].shape[1] <= 256
                else None
            )
            self.patches[binding.resource, binding.layer] = _Patch(
                binding,
                pools,
                torch.zeros((1 + len(binding.fields), rows), dtype=torch.long, device=arena.device),
                coeff,
            )
        self._slot_staging = PinnedStaging()
        self._presence = np.zeros((rows, len(bindings)), dtype=np.bool_)
        self._route_gather = (
            RouteSlotGather(rows, arena.device)
            if any(binding.route_resource is not None for binding in bindings)
            else None
        )
        self._route_buffers: dict[tuple[StateResource, int, bool], torch.Tensor] = {}

    @property
    def read_buffers(self) -> list[torch.Tensor]:
        """Stable read ports of the captured residual hook."""
        return [buffer for patch in self.patches.values() for buffer in (patch.table, *patch.pools)]

    @property
    def write_buffers(self) -> list[torch.Tensor]:
        """Stable scratch buffers mutated by low-rank affine residual hooks."""
        return [patch.coeff for patch in self.patches.values() if patch.coeff is not None]

    def prepare(self, rows: PackedPhaseRowBatch, context: StateTaskContext, directory: RequestStateDirectory) -> None:
        """Upload row masks and the slots of the exact versions the invocation holds."""
        held = context.held_input
        assert held is None or (
            held.step_id == rows.step_id and {identity for identity, _, _ in held.successful} == set(rows.identities)
        )
        context.input_step_id = rows.step_id
        self._presence.fill(False)
        # Windows are relative to each request's rows in this invocation, matching
        # chunked prefill and retained replay as well as the ordinary full prompt.
        windows = self._row_windows(rows)
        versions: dict[tuple[StateResource, RequestIdentity, bool, bool, bool], tuple[PagePin, StateRef]] = {}
        routed_layers: dict[tuple[StateResource, int, bool], torch.Tensor] = {}
        packed_range = np.arange(rows.total_rows, dtype=np.int64)
        for ordinal, patch in enumerate(self.patches.values()):
            binding = patch.binding
            selected = (windows[2] == 1) & (
                (windows[0] < binding.first_tokens) | (windows[0] >= windows[1] - binding.last_tokens)
            )
            selected |= (windows[2] == 0) & (binding.decode is DecodeWindow.ALL)
            self._presence[: rows.total_rows, ordinal] = selected
            routed = binding.route_resource is not None
            with self._slot_staging.fill(patch.table) as host:
                host[0, : rows.total_rows] = selected
                for phase in (1, 0):
                    indices = np.flatnonzero(selected & (windows[2] == phase))
                    if not len(indices):
                        continue
                    ids, inverse = np.unique(rows.identity_indices[indices], return_inverse=True)
                    identities = tuple(rows.request_identities[int(i)] for i in ids)
                    refresh = (
                        binding.decode_refresh is DecodeRefresh.STEP
                        if routed
                        else (phase == 0 and binding.decode_refresh is DecodeRefresh.STEP)
                    )
                    held_versions = tuple(
                        self._version(
                            binding.resource,
                            identity,
                            refresh,
                            context,
                            directory,
                            versions,
                            require_publish=(phase == 0),
                            allow_import_persist=routed,
                        )
                        for identity in identities
                    )
                    pins, refs = zip(*held_versions, strict=True)
                    _, slots = context.arena.slots(pins, np.arange(len(pins), dtype=np.int64), StateRefs.of(refs))
                    host[1:, indices] = slots[inverse][:, [view.field for view in binding.fields]].T
            if binding.route_resource is not None and binding.route_field is not None and rows.total_rows > 0:
                assert self._route_gather is not None
                route_latest = binding.route_refresh is DecodeRefresh.STEP
                route_key = (binding.route_resource, int(binding.route_field.field), route_latest)
                if route_key not in routed_layers:
                    if route_key not in self._route_buffers:
                        self._route_buffers[route_key] = torch.full(
                            (patch.table.shape[1],), -1, dtype=torch.long, device=patch.table.device
                        )
                    buf = self._route_buffers[route_key]
                    decode_selected = np.flatnonzero(selected & (windows[2] == 0))
                    self._route_gather.gather(
                        rows,
                        packed_range,
                        binding.route_resource,
                        binding.route_field,
                        context,
                        directory,
                        latest=route_latest,
                        out=buf,
                        require_publish=bool(len(decode_selected)),
                    )
                    routed_layers[route_key] = buf
                patch.table[0, : rows.total_rows].mul_(routed_layers[route_key][: rows.total_rows] == binding.layer)

    @staticmethod
    def _row_windows(rows: PackedPhaseRowBatch) -> np.ndarray[tuple[int, int], np.dtype[np.int64]]:
        windows = np.zeros((3, rows.total_rows), dtype=np.int64)
        for phase in (1, 0):
            indices = np.flatnonzero(rows.phases == phase)
            if not len(indices):
                continue
            order = np.argsort(rows.identity_indices[indices], kind="stable")
            indices = indices[order]
            identities = rows.identity_indices[indices]
            starts = np.r_[0, np.flatnonzero(identities[1:] != identities[:-1]) + 1]
            lengths = np.diff(np.r_[starts, len(indices)])
            windows[0, indices] = np.arange(len(indices)) - np.repeat(starts, lengths)
            windows[1, indices] = np.repeat(lengths, lengths)
            windows[2, indices] = phase
        return windows

    @staticmethod
    def _version(
        resource: StateResource,
        identity: RequestIdentity,
        refresh: bool,
        context: StateTaskContext,
        directory: RequestStateDirectory,
        versions: dict[tuple[StateResource, RequestIdentity, bool, bool, bool], tuple[PagePin, StateRef]],
        *,
        require_publish: bool = True,
        allow_import_persist: bool = False,
    ) -> tuple[PagePin, StateRef]:
        key = resource, identity, refresh, require_publish, allow_import_persist
        if key not in versions:
            ((pin, ref),) = context.read_versions((identity,), resource, directory, latest=refresh)
            if (
                refresh
                and require_publish
                and ref in context.selection.for_resource(resource).refs
                and context.held_input is None
                and requires_step_publish(context.arena.plan, resource, allow_import_persist=allow_import_persist)
            ):
                raise RuntimeError(
                    f'ActivationPatch(refresh="step") found no new version of {resource!r} at this decode step: the '
                    "method that writes it before the forward did not publish one. Publish every decode step or use "
                    'refresh="admission"'
                )
            # Keep the exact admission or publication version alive through this
            # invocation, including custom-forward graph replays on other streams.
            pin = context.pin_for(StateRefs.of((ref,)))
            held = context.held_input
            assert held is None or any(
                identity == held_identity and ref in refs for held_identity, refs, _ in held.successful
            )
            context.injected_refs.setdefault(identity, set()).add(ref)
            versions[key] = pin, ref
        return versions[key]

    def presence(self, row: int) -> tuple[bool, ...]:
        """Which bindings patch this packed row when custom-forward waves are grouped."""
        return tuple(bool(value) for value in self._presence[row])

    def patch(
        self, resource: StateResource, layer: int, rows: torch.Tensor | None
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """The same masked edit for a whole model call or device-indexed wave rows."""
        return partial(self.patches[resource, layer].apply, rows)

    def patch_into(self, resource: StateResource, layer: int, rows: torch.Tensor | None, target: torch.Tensor) -> None:
        """Apply the masked edit directly into an already-populated residual port."""
        self.patches[resource, layer].apply_into(rows, target)

    def clear(self) -> None:
        """Forget host wave signatures; device read ports are replaced by the next prepare."""
        self._presence.fill(False)
