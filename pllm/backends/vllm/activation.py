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
from pllm.contracts.semantic_graph import StateEffectKind, StateResource
from pllm.contracts.state_ref import StateRef
from pllm.execution.cuda_stream import PinnedStaging
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

    def apply(self, rows: torch.Tensor | None, hidden: torch.Tensor) -> torch.Tensor:
        table = self.table[:, : len(hidden)] if rows is None else self.table.index_select(1, rows)
        mask, slots = table[0].bool(), table[1:]
        fields = tuple(pool.index_select(0, indices) for pool, indices in zip(self.pools, slots, strict=True))
        return torch.where(mask[:, None], hidden + activation_delta(hidden, fields), hidden)


class VllmActivationPatch:
    """ActivationPatch's device masks and field slots, shared by all residual-site callers."""

    def __init__(self, bindings: tuple[ActivationPatchBinding, ...], rows: int, arena: PageBundleArena) -> None:
        self.bindings = bindings
        self.patches = {
            (binding.resource, binding.layer): _Patch(
                binding,
                tuple(
                    view.read(arena.field_pool(binding.resource, view.field), pooled=True) for view in binding.fields
                ),
                torch.zeros((1 + len(binding.fields), rows), dtype=torch.long, device=arena.device),
            )
            for binding in bindings
        }
        self._slot_staging = PinnedStaging()
        self._presence = np.zeros((rows, len(bindings)), dtype=np.bool_)

    @property
    def buffers(self) -> list[torch.Tensor]:
        """Stable read ports of the captured residual hook."""
        return [buffer for patch in self.patches.values() for buffer in (patch.table, *patch.pools)]

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
        versions: dict[tuple[StateResource, RequestIdentity, bool], tuple[PagePin, StateRef]] = {}
        for ordinal, patch in enumerate(self.patches.values()):
            binding = patch.binding
            selected = (windows[2] == 1) & (
                (windows[0] < binding.first_tokens) | (windows[0] >= windows[1] - binding.last_tokens)
            )
            selected |= (windows[2] == 0) & (binding.decode is DecodeWindow.ALL)
            self._presence[: rows.total_rows, ordinal] = selected
            with self._slot_staging.fill(patch.table) as host:
                host[0, : rows.total_rows] = selected
                for phase in (1, 0):
                    indices = np.flatnonzero(selected & (windows[2] == phase))
                    if not len(indices):
                        continue
                    ids, inverse = np.unique(rows.identity_indices[indices], return_inverse=True)
                    identities = tuple(rows.request_identities[int(i)] for i in ids)
                    refresh = phase == 0 and binding.decode_refresh is DecodeRefresh.STEP
                    held_versions = tuple(
                        self._version(binding.resource, identity, refresh, context, directory, versions)
                        for identity in identities
                    )
                    pins, refs = zip(*held_versions, strict=True)
                    _, slots = context.arena.slots(pins, np.arange(len(pins), dtype=np.int64), StateRefs.of(refs))
                    host[1:, indices] = slots[inverse][:, [view.field for view in binding.fields]].T

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
        versions: dict[tuple[StateResource, RequestIdentity, bool], tuple[PagePin, StateRef]],
    ) -> tuple[PagePin, StateRef]:
        key = resource, identity, refresh
        if key not in versions:
            ((pin, ref),) = context.read_versions((identity,), resource, directory, latest=refresh)
            if (
                refresh
                and ref in context.selection.for_resource(resource).refs
                and context.held_input is None
                and any(
                    task.timing.after == "request.prepare"
                    and any(
                        effect.resource == resource and effect.kind is not StateEffectKind.READ
                        for effect in task.effects
                    )
                    for task in context.arena.plan.semantic_graph.tasks
                )
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

    def clear(self) -> None:
        """Forget host wave signatures; device read ports are replaced by the next prepare."""
        self._presence.fill(False)
