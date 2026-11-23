"""Device-side PTTS route-field gathering and kernel re-exports for pllm.model.LayerRoute.

Owned by the execution layer and used by vLLM's HiddenTaps, VllmActivationPatch and ParameterOverlay. Slot indices are
resolved from host PTTS metadata and uploaded through pinned memory; the int64 layer indices themselves stay on the
device so @pllm.write updates require zero host synchronization.
"""

from __future__ import annotations

import numpy as np
import torch

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.host_batch import StateImport
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import FieldView
from pllm.contracts.semantic_graph import HostServiceInputSource, StateEffectKind, StateResource
from pllm.execution.cuda_stream import PinnedStaging
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.ptts.slot_pools import Slots
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.state_execution import StateTaskContext
from pllm.kernels.layer_route import (
    gate_owners_by_layer,
    gated_dense_add,
    gated_low_rank_add,
    masked_patch_add,
    masked_patch_add_into,
    masked_patch_affine,
    masked_patch_affine_into,
    routed_tap_copy,
)


def requires_step_publish(plan: PlanBundle, resource: StateResource, *, allow_import_persist: bool = True) -> bool:
    """Whether `resource` has a transition before `model.execute` that must publish on every decode step."""
    forward_ordinal = next((a.ordinal for a in plan.timeline.anchors if a.symbol == "model.execute"), None)
    if forward_ordinal is None:
        return False
    return any(
        plan.scheduled(task.symbol).anchor_id < forward_ordinal
        and (
            not allow_import_persist
            or not any(
                isinstance(port.source, HostServiceInputSource) and port.source.schema == StateImport.schema
                for port in task.inputs
            )
        )
        and any(
            effect.resource == resource and effect.kind not in (StateEffectKind.READ, StateEffectKind.PUBLISH)
            for effect in task.effects
        )
        for task in plan.semantic_graph.tasks
    )


class RouteSlotGather:
    """Persistent device slot buffer and pinned staging to read scalar int64 route fields without host syncs."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.slots = torch.zeros(capacity, dtype=torch.long, device=device)
        self.staging = PinnedStaging()

    def gather(
        self,
        rows: PackedPhaseRowBatch,
        selected_rows: Slots,
        resource: StateResource,
        view: FieldView,
        context: StateTaskContext,
        directory: RequestStateDirectory,
        *,
        latest: bool,
        out: torch.Tensor,
        require_publish: bool = False,
    ) -> None:
        """Populate `out[:len(selected_rows)]` on device from PTTS field `(resource, view)` and fill the rest with -1."""
        out.fill_(-1)
        count = int(selected_rows.size)
        if count == 0:
            return
        ids, inverse = np.unique(rows.identity_indices[selected_rows], return_inverse=True)
        identities = tuple(rows.request_identities[int(i)] for i in ids)
        versions = context.read_versions(identities, resource, directory, latest=latest)
        _, refs = zip(*versions, strict=True)
        if (
            latest
            and require_publish
            and context.held_input is None
            and requires_step_publish(context.arena.plan, resource)
        ):
            selected_refs = context.selection.for_resource(resource).refs
            if any(ref in selected_refs for ref in refs):
                raise RuntimeError(
                    f'ActivationPatch(refresh="step") found no new version of {resource!r} at this decode step: the '
                    "method that writes it before the forward did not publish one. Publish every decode step or use "
                    'refresh="admission"'
                )
        state_refs = StateRefs.of(refs)
        pin = context.pin_for(state_refs)
        for identity, ref in zip(identities, refs, strict=True):
            context.injected_refs.setdefault(identity, set()).add(ref)
        _, matrix = context.arena.slots((pin,) * len(refs), np.arange(len(refs), dtype=np.int64), state_refs)
        field_slots = matrix[inverse, view.field]
        target = self.slots[:count]
        if target.is_cuda:
            host = self.staging.take(count)
            host[:] = field_slots
            self.staging.copy_to(target)
        else:
            target.numpy()[:] = field_slots
        pool = view.read(context.arena.field_pool(str(resource), view.field), pooled=True).reshape(-1)
        torch.index_select(pool, 0, target, out=out[:count])


__all__ = [
    "RouteSlotGather",
    "gate_owners_by_layer",
    "gated_dense_add",
    "gated_low_rank_add",
    "masked_patch_add",
    "masked_patch_add_into",
    "masked_patch_affine",
    "masked_patch_affine_into",
    "requires_step_publish",
    "routed_tap_copy",
]
