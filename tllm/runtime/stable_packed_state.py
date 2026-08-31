"""Address-stable packed tensor views over pinned immutable state roots."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import NewType

import torch

from tllm.contracts.request_state import (
    RequestStateSelectionBatch,
    RequestStateVersion,
)
from tllm.contracts.state_ref import StateRef
from tllm.runtime.paged_state import (
    FrozenTensorLayout,
    ImmutableStateVersionRoot,
    PagedStateVersionArena,
    StateResourceId,
    StateVersionRootPin,
    TensorFieldId,
)
from tllm.runtime.request_slot_adapter import (
    CompletionFence,
    CudaCompletionFence,
    NoCompletionFence,
)


PackedLayoutId = NewType("PackedLayoutId", int)


@dataclass(frozen=True, slots=True)
class FrozenPackedTensorField:
    field_id: TensorFieldId
    trailing_shape: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if int(self.field_id) < 0:
            raise ValueError("packed tensor field id must be non-negative")
        if any(size < 1 for size in self.trailing_shape):
            raise ValueError("packed tensor trailing dimensions must be positive")


@dataclass(frozen=True, slots=True)
class FrozenPackedTensorLayout:
    layout_id: PackedLayoutId
    row_capacity: int
    lease_capacity: int
    fields: tuple[FrozenPackedTensorField, ...]

    def __post_init__(self) -> None:
        if int(self.layout_id) < 0:
            raise ValueError("packed tensor layout id must be non-negative")
        if self.row_capacity < 1 or self.lease_capacity < 1:
            raise ValueError("packed tensor capacities must be positive")
        if self.lease_capacity > self.row_capacity:
            raise ValueError("packed lease capacity cannot exceed row capacity")
        if not self.fields:
            raise ValueError("packed tensor layout requires fields")
        if tuple(int(field.field_id) for field in self.fields) != tuple(
            range(len(self.fields))
        ):
            raise ValueError("packed tensor field ids must be dense and ordered")


@dataclass(frozen=True, slots=True)
class StablePackedTensorSet:
    layout: FrozenPackedTensorLayout
    tensors: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class StablePackedValidityMasks:
    rows: torch.Tensor
    leases: torch.Tensor


@dataclass(frozen=True, slots=True)
class StateStabilityEpoch:
    resource: StateResourceId
    state_epoch: int
    bundle_epoch: int
    view_epoch: int
    buffer_id: int
    state_layout: FrozenTensorLayout
    packed_layout_id: PackedLayoutId


@dataclass(frozen=True, slots=True)
class PinnedStateVersionEpoch:
    selection_id: int
    refs: tuple[StateRef, ...]
    roots: tuple[ImmutableStateVersionRoot, ...]
    root_pin: StateVersionRootPin
    page_ids: torch.Tensor
    state_operands: tuple[torch.Tensor, ...]
    readiness: CompletionFence


@dataclass(frozen=True, slots=True)
class StateViewCompletionFence:
    view_id: int
    buffer_id: int


@dataclass(frozen=True, slots=True)
class StablePackedProjectionSignature:
    resource: StateResourceId
    packed_layout: FrozenPackedTensorLayout
    state_layout: FrozenTensorLayout


@dataclass(frozen=True, slots=True)
class StablePackedStateProjection:
    signature: StablePackedProjectionSignature
    tensors: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class StablePackedStateView:
    packed_tensors: StablePackedTensorSet
    row_slot_ids: torch.Tensor
    lease_slot_ids: torch.Tensor
    valid_masks: StablePackedValidityMasks
    first_row_ids: torch.Tensor
    stability_epoch: StateStabilityEpoch
    version_epoch: PinnedStateVersionEpoch
    completion_fence: StateViewCompletionFence

    def tensor_projection(self) -> StablePackedStateProjection:
        return StablePackedStateProjection(
            StablePackedProjectionSignature(
                self.stability_epoch.resource,
                self.packed_tensors.layout,
                self.stability_epoch.state_layout,
            ),
            (
                *self.packed_tensors.tensors,
                self.row_slot_ids,
                self.lease_slot_ids,
                self.valid_masks.rows,
                self.valid_masks.leases,
                self.first_row_ids,
                self.version_epoch.page_ids,
                *self.version_epoch.state_operands,
            ),
        )


@dataclass(frozen=True, slots=True)
class _ViewBuffer:
    packed_tensors: tuple[torch.Tensor, ...]
    row_slot_ids: torch.Tensor
    lease_slot_ids: torch.Tensor
    row_valid_mask: torch.Tensor
    lease_valid_mask: torch.Tensor
    first_row_ids: torch.Tensor
    page_ids: torch.Tensor
    stability_epoch: StateStabilityEpoch


@dataclass(frozen=True, slots=True)
class _FreeViewBuffer:
    buffer: _ViewBuffer


@dataclass(frozen=True, slots=True)
class _ReservedViewBuffer:
    buffer: _ViewBuffer
    view_id: int


@dataclass(frozen=True, slots=True)
class _IssuedViewBuffer:
    buffer: _ViewBuffer
    completion: StateViewCompletionFence
    root_pin: StateVersionRootPin


@dataclass(frozen=True, slots=True)
class _PendingViewBuffer:
    buffer: _ViewBuffer
    completion: StateViewCompletionFence
    root_pin: StateVersionRootPin
    event: torch.cuda.Event


_ViewBufferState = (
    _FreeViewBuffer
    | _ReservedViewBuffer
    | _IssuedViewBuffer
    | _PendingViewBuffer
)


@dataclass(frozen=True, slots=True)
class _PreparedView:
    selection_id: int
    refs: tuple[StateRef, ...]
    canonical_row_ids: tuple[int, ...]
    row_slot_ids: tuple[int, ...]
    lease_slot_ids: tuple[int, ...]
    first_row_ids: tuple[int, ...]
    tensors: tuple[torch.Tensor, ...]


class StablePackedStateViewArena:
    """Own fixed view buffers and completion-gated root pins."""

    __slots__ = (
        "_roots",
        "_resource",
        "_state_epoch",
        "_view_epoch",
        "_layout",
        "_device",
        "_buffers",
        "_next_view_id",
        "_poisoned",
        "_lock",
    )

    def __init__(
        self,
        roots: PagedStateVersionArena,
        *,
        resource: StateResourceId,
        state_epoch: int,
        view_epoch: int,
        layout: FrozenPackedTensorLayout,
        buffer_count: int,
    ) -> None:
        if state_epoch < 1 or view_epoch < 1:
            raise ValueError("stable view epochs must be positive")
        if buffer_count < 1:
            raise ValueError("stable view buffer count must be positive")
        roots.bundle.require_stable()
        self._roots = roots
        self._resource = resource
        self._state_epoch = state_epoch
        self._view_epoch = view_epoch
        self._layout = layout
        self._device = roots.bundle.device
        self._buffers: list[_ViewBufferState] = []
        for buffer_id in range(buffer_count):
            buffer = _ViewBuffer(
                packed_tensors=tuple(
                    torch.zeros(
                        (layout.row_capacity, *field.trailing_shape),
                        dtype=field.dtype,
                        device=self._device,
                    )
                    for field in layout.fields
                ),
                row_slot_ids=torch.zeros(
                    (layout.row_capacity,),
                    dtype=torch.long,
                    device=self._device,
                ),
                lease_slot_ids=torch.zeros(
                    (layout.lease_capacity,),
                    dtype=torch.long,
                    device=self._device,
                ),
                row_valid_mask=torch.zeros(
                    (layout.row_capacity,),
                    dtype=torch.bool,
                    device=self._device,
                ),
                lease_valid_mask=torch.zeros(
                    (layout.lease_capacity,),
                    dtype=torch.bool,
                    device=self._device,
                ),
                first_row_ids=torch.zeros(
                    (layout.lease_capacity,),
                    dtype=torch.long,
                    device=self._device,
                ),
                page_ids=torch.zeros(
                    (
                        layout.lease_capacity,
                        roots.bundle.layout.root_page_count,
                    ),
                    dtype=torch.long,
                    device=self._device,
                ),
                stability_epoch=StateStabilityEpoch(
                    resource=resource,
                    state_epoch=state_epoch,
                    bundle_epoch=roots.bundle_epoch,
                    view_epoch=view_epoch,
                    buffer_id=buffer_id,
                    state_layout=roots.bundle.layout,
                    packed_layout_id=layout.layout_id,
                ),
            )
            self._buffers.append(_FreeViewBuffer(buffer))
        self._next_view_id = 1
        self._poisoned = False
        self._lock = RLock()

    @property
    def free_buffers(self) -> int:
        with self._lock:
            count = 0
            for state in self._buffers:
                match state:
                    case _FreeViewBuffer():
                        count += 1
                    case (
                        _ReservedViewBuffer()
                        | _IssuedViewBuffer()
                        | _PendingViewBuffer()
                    ):
                        pass
            return count

    @property
    def buffer_count(self) -> int:
        return len(self._buffers)

    @property
    def version_roots(self) -> PagedStateVersionArena:
        return self._roots

    @property
    def resource(self) -> StateResourceId:
        return self._resource

    @property
    def state_epoch(self) -> int:
        return self._state_epoch

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def poison(self) -> None:
        with self._lock:
            self._poisoned = True

    def create_synchronous(
        self,
        selection: RequestStateSelectionBatch,
        versions: tuple[RequestStateVersion, ...],
        tensors: tuple[torch.Tensor, ...],
    ) -> StablePackedStateView:
        with self._lock:
            self._require_cpu()
            prepared = self._prepare(selection, versions, tensors)
            buffer = self._reserve_buffer()
            try:
                root_pin = self._roots.pin_exact_synchronous(prepared.refs)
            except BaseException:
                self._release_reserved(buffer)
                raise
            try:
                with torch.no_grad():
                    self._fill(buffer.buffer, prepared, root_pin)
            except BaseException:
                self._poisoned = True
                raise
            return self._issue(
                buffer,
                prepared,
                root_pin,
                NoCompletionFence(),
            )

    def create_cuda(
        self,
        selection: RequestStateSelectionBatch,
        versions: tuple[RequestStateVersion, ...],
        tensors: tuple[torch.Tensor, ...],
        stream: torch.cuda.Stream,
    ) -> StablePackedStateView:
        with self._lock:
            self._require_cuda_stream(stream)
            prepared = self._prepare(selection, versions, tensors)
            buffer = self._reserve_buffer()
            try:
                root_pin = self._roots.pin_exact_cuda(prepared.refs, stream)
            except BaseException:
                self._release_reserved(buffer)
                raise
            try:
                with torch.no_grad(), torch.cuda.stream(stream):
                    self._fill(buffer.buffer, prepared, root_pin)
            except BaseException:
                self._poisoned = True
                raise
            readiness_event = torch.cuda.Event()
            readiness_event.record(stream)
            return self._issue(
                buffer,
                prepared,
                root_pin,
                CudaCompletionFence(readiness_event),
            )

    def order_cuda(
        self,
        view: StablePackedStateView,
        stream: torch.cuda.Stream,
    ) -> None:
        with self._lock:
            self._require_cuda_stream(stream)
            issued = self._issued_buffer(view.completion_fence)
            if (
                view.version_epoch.root_pin is not issued.root_pin
                or view.stability_epoch.buffer_id
                != view.completion_fence.buffer_id
            ):
                raise RuntimeError("stable view differs from its issued lifetime")
            match view.version_epoch.readiness:
                case CudaCompletionFence(event=event):
                    stream.wait_event(event)
                case NoCompletionFence():
                    raise RuntimeError("CUDA stable view has no readiness event")

    def complete_synchronous(
        self,
        completion: StateViewCompletionFence,
    ) -> None:
        with self._lock:
            self._require_cpu()
            issued = self._issued_buffer(completion)
            self._roots.release_pin(issued.root_pin, NoCompletionFence())
            self._buffers[completion.buffer_id] = _FreeViewBuffer(
                issued.buffer
            )

    def complete_cuda(
        self,
        completion: StateViewCompletionFence,
        stream: torch.cuda.Stream,
    ) -> None:
        with self._lock:
            self._require_cuda_stream(stream)
            issued = self._issued_buffer(completion)
            event = torch.cuda.Event()
            event.record(stream)
            self._buffers[completion.buffer_id] = _PendingViewBuffer(
                issued.buffer,
                completion,
                issued.root_pin,
                event,
            )

    def collect_completed(self) -> None:
        with self._lock:
            self._collect_completed_locked()

    def wait_completion(self, completion: StateViewCompletionFence) -> None:
        with self._lock:
            pending = self._pending_buffer(completion)
            pending.event.synchronize()
            self._roots.release_completed_pin(
                pending.root_pin,
                CudaCompletionFence(pending.event),
            )
            self._buffers[completion.buffer_id] = _FreeViewBuffer(
                pending.buffer
            )

    def _prepare(
        self,
        selection: RequestStateSelectionBatch,
        versions: tuple[RequestStateVersion, ...],
        tensors: tuple[torch.Tensor, ...],
    ) -> _PreparedView:
        self._require_live()
        if (
            selection.resource != str(self._resource)
            or selection.epoch != self._state_epoch
        ):
            raise RuntimeError("stable view selection belongs to another state epoch")
        row_count = len(selection.rows)
        lease_count = len(selection.leases)
        if row_count < 1 or row_count > self._layout.row_capacity:
            raise RuntimeError("stable view row capacity is exhausted")
        if lease_count < 1 or lease_count > self._layout.lease_capacity:
            raise RuntimeError("stable view lease capacity is exhausted")
        if len(versions) != lease_count or any(
            version.key != lease.key
            or version.slot != lease.slot
            or version.lineage_id != lease.lineage_id
            for version, lease in zip(
                versions,
                selection.leases,
                strict=True,
            )
        ):
            raise RuntimeError("stable view versions differ from selection leases")
        if len(tensors) != len(self._layout.fields):
            raise ValueError("stable view tensors differ from the packed layout")
        canonical_row_ids = tuple(
            row.canonical_row_index for row in selection.rows
        )
        required_rows = max(canonical_row_ids) + 1
        for tensor, field in zip(tensors, self._layout.fields, strict=True):
            if (
                tensor.device != self._device
                or tensor.dtype is not field.dtype
                or tensor.ndim != len(field.trailing_shape) + 1
                or tuple(tensor.shape[1:]) != field.trailing_shape
                or int(tensor.shape[0]) < required_rows
            ):
                raise RuntimeError(
                    "stable view tensor signature differs from its layout"
                )
        first_position_by_lease = {
            row.lease: position
            for position, row in reversed(tuple(enumerate(selection.rows)))
        }
        refs = tuple(
            StateRef(
                self._roots.bundle.layout.layout_id,
                version.lineage_id,
                version.version,
            )
            for version in versions
        )
        return _PreparedView(
            selection_id=selection.selection_id,
            refs=refs,
            canonical_row_ids=canonical_row_ids,
            row_slot_ids=tuple(
                row.lease.slot.slot_id for row in selection.rows
            ),
            lease_slot_ids=tuple(
                lease.slot.slot_id for lease in selection.leases
            ),
            first_row_ids=tuple(
                first_position_by_lease[lease]
                for lease in selection.leases
            ),
            tensors=tensors,
        )

    def _reserve_buffer(self) -> _ReservedViewBuffer:
        self._collect_completed_locked()
        for buffer_id, state in enumerate(self._buffers):
            match state:
                case _FreeViewBuffer(buffer=buffer):
                    reserved = _ReservedViewBuffer(
                        buffer,
                        self._next_view_id,
                    )
                    self._buffers[buffer_id] = reserved
                    self._next_view_id += 1
                    return reserved
                case (
                    _ReservedViewBuffer()
                    | _IssuedViewBuffer()
                    | _PendingViewBuffer()
                ):
                    pass
        raise RuntimeError("stable packed view capacity is exhausted")

    def _release_reserved(self, reserved: _ReservedViewBuffer) -> None:
        buffer_id = reserved.buffer.stability_epoch.buffer_id
        if self._buffers[buffer_id] is not reserved:
            raise RuntimeError("stable view reservation is no longer active")
        self._buffers[buffer_id] = _FreeViewBuffer(reserved.buffer)

    def _fill(
        self,
        buffer: _ViewBuffer,
        prepared: _PreparedView,
        root_pin: StateVersionRootPin,
    ) -> None:
        row_count = len(prepared.canonical_row_ids)
        lease_count = len(prepared.refs)
        canonical_row_ids = torch.tensor(
            prepared.canonical_row_ids,
            dtype=torch.long,
            device=self._device,
        )
        for destination, source in zip(
            buffer.packed_tensors,
            prepared.tensors,
            strict=True,
        ):
            destination.zero_()
            destination[:row_count].copy_(
                source.index_select(0, canonical_row_ids)
            )
        buffer.row_slot_ids.zero_()
        buffer.row_slot_ids[:row_count].copy_(
            torch.tensor(
                prepared.row_slot_ids,
                dtype=torch.long,
                device=self._device,
            )
        )
        buffer.lease_slot_ids.zero_()
        buffer.lease_slot_ids[:lease_count].copy_(
            torch.tensor(
                prepared.lease_slot_ids,
                dtype=torch.long,
                device=self._device,
            )
        )
        buffer.row_valid_mask.zero_()
        buffer.row_valid_mask[:row_count].fill_(True)
        buffer.lease_valid_mask.zero_()
        buffer.lease_valid_mask[:lease_count].fill_(True)
        buffer.first_row_ids.zero_()
        buffer.first_row_ids[:lease_count].copy_(
            torch.tensor(
                prepared.first_row_ids,
                dtype=torch.long,
                device=self._device,
            )
        )
        buffer.page_ids.zero_()
        buffer.page_ids[:lease_count].copy_(
            torch.tensor(
                tuple(root.page_ids for root in root_pin.roots),
                dtype=torch.long,
                device=self._device,
            )
        )

    def _issue(
        self,
        reserved: _ReservedViewBuffer,
        prepared: _PreparedView,
        root_pin: StateVersionRootPin,
        readiness: CompletionFence,
    ) -> StablePackedStateView:
        buffer = reserved.buffer
        buffer_id = buffer.stability_epoch.buffer_id
        if self._buffers[buffer_id] is not reserved:
            self._poisoned = True
            raise RuntimeError("stable view reservation changed before issue")
        completion = StateViewCompletionFence(
            reserved.view_id,
            buffer_id,
        )
        issued = _IssuedViewBuffer(buffer, completion, root_pin)
        self._buffers[buffer_id] = issued
        version_epoch = PinnedStateVersionEpoch(
            selection_id=prepared.selection_id,
            refs=prepared.refs,
            roots=root_pin.roots,
            root_pin=root_pin,
            page_ids=buffer.page_ids,
            state_operands=self._roots.bundle.capture_operands,
            readiness=readiness,
        )
        return StablePackedStateView(
            packed_tensors=StablePackedTensorSet(
                self._layout,
                buffer.packed_tensors,
            ),
            row_slot_ids=buffer.row_slot_ids,
            lease_slot_ids=buffer.lease_slot_ids,
            valid_masks=StablePackedValidityMasks(
                buffer.row_valid_mask,
                buffer.lease_valid_mask,
            ),
            first_row_ids=buffer.first_row_ids,
            stability_epoch=buffer.stability_epoch,
            version_epoch=version_epoch,
            completion_fence=completion,
        )

    def _issued_buffer(
        self,
        completion: StateViewCompletionFence,
    ) -> _IssuedViewBuffer:
        self._require_live()
        if completion.buffer_id < 0 or completion.buffer_id >= len(self._buffers):
            raise RuntimeError("stable view completion names an unknown buffer")
        match self._buffers[completion.buffer_id]:
            case _IssuedViewBuffer(completion=issued) as state if (
                issued is completion
            ):
                return state
            case _:
                raise RuntimeError("stable view completion is not issued")

    def _pending_buffer(
        self,
        completion: StateViewCompletionFence,
    ) -> _PendingViewBuffer:
        self._require_live()
        if completion.buffer_id < 0 or completion.buffer_id >= len(self._buffers):
            raise RuntimeError("stable view completion names an unknown buffer")
        match self._buffers[completion.buffer_id]:
            case _PendingViewBuffer(completion=issued) as state if (
                issued is completion
            ):
                return state
            case _:
                raise RuntimeError("stable view completion is not pending")

    def _collect_completed_locked(self) -> None:
        self._require_live()
        for buffer_id, state in enumerate(self._buffers):
            match state:
                case _PendingViewBuffer(event=event) as pending if event.query():
                    self._roots.release_completed_pin(
                        pending.root_pin,
                        CudaCompletionFence(event),
                    )
                    self._buffers[buffer_id] = _FreeViewBuffer(pending.buffer)
                case (
                    _FreeViewBuffer()
                    | _ReservedViewBuffer()
                    | _IssuedViewBuffer()
                    | _PendingViewBuffer()
                ):
                    pass
        self._roots.collect_completed()

    def _require_live(self) -> None:
        if self._poisoned:
            raise RuntimeError("stable packed view arena is poisoned")

    def _require_cpu(self) -> None:
        self._require_live()
        if self._device.type != "cpu":
            raise RuntimeError("synchronous stable view requires CPU storage")

    def _require_cuda_stream(self, stream: torch.cuda.Stream) -> None:
        self._require_live()
        if self._device.type != "cuda":
            raise RuntimeError("CUDA stable view requires CUDA storage")
        if stream.device != self._device:
            raise RuntimeError("stable view stream belongs to another device")


__all__ = [
    "FrozenPackedTensorField",
    "FrozenPackedTensorLayout",
    "PackedLayoutId",
    "PinnedStateVersionEpoch",
    "StablePackedStateView",
    "StablePackedStateProjection",
    "StablePackedProjectionSignature",
    "StablePackedStateViewArena",
    "StablePackedTensorSet",
    "StablePackedValidityMasks",
    "StateStabilityEpoch",
    "StateViewCompletionFence",
]
