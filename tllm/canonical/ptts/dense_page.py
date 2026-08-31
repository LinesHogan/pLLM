"""Strict CPU-only dense-page provider for canonical PTTS state."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch
from torch.optim.adamw import adamw

from tllm.canonical.state_representation import (
    validate_state_physical_plan,
    validate_state_representation,
    validate_state_representation_reconstruction,
)
from tllm.canonical.tensor_layout import validate_tensor_layouts
from tllm.contracts.state_ref import LineageId, StateRef, VersionId
from tllm.contracts.state_representation import (
    StateLayoutRequirement,
    StatePhysicalPlan,
    StateProviderPlan,
    StateRepresentation,
    StateRepresentationDecision,
    StateRepresentationRequest,
)
from tllm.contracts.tensor_layout import (
    LayoutFieldId,
    StateTensorLayout,
    TensorDType,
    TensorLayoutDecision,
    TensorLayoutRequest,
    TensorLayoutSpec,
)


@dataclass(frozen=True, slots=True)
class DensePageRoot:
    """One immutable logical root backed by a CPU page row."""

    ref: StateRef
    page_id: int


@dataclass(frozen=True, slots=True)
class DensePageAdmission:
    """One exact V0 root and its typed CPU initial field tensors."""

    ref: StateRef
    initial_tensors: tuple[torch.Tensor, ...]


@dataclass(slots=True)
class _StoredDensePageRoot:
    root: DensePageRoot
    pins: int = 0


@dataclass(frozen=True, slots=True)
class DensePageVersionPin:
    """Exact CPU pin over immutable StateRef roots."""

    pin_id: int
    refs: tuple[StateRef, ...]


@dataclass(frozen=True, slots=True)
class DensePageTransition:
    """Opaque logical member of a provider-owned COW transaction."""

    source: StateRef
    target: StateRef


@dataclass(frozen=True, slots=True)
class DensePageTransitionBatch:
    """Opaque token for one pending, all-or-nothing CPU COW publication."""

    transaction_id: int
    transitions: tuple[DensePageTransition, ...]


@dataclass(frozen=True, slots=True)
class _NoPendingTransition:
    pass


@dataclass(frozen=True, slots=True)
class _ReservedTransitionBatch:
    batch: DensePageTransitionBatch
    source_page_ids: tuple[int, ...]
    target_page_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ComputedTransitionBatch:
    reserved: _ReservedTransitionBatch


_PendingTransition = (
    _NoPendingTransition
    | _ReservedTransitionBatch
    | _ComputedTransitionBatch
)
_NO_PENDING_TRANSITION = _NoPendingTransition()


@dataclass(frozen=True, slots=True)
class DensePageAdamWFieldBinding:
    """Typed layout-field binding for one reusable AdamW transition."""

    parameter: LayoutFieldId
    gradient: LayoutFieldId
    step: LayoutFieldId
    exp_avg: LayoutFieldId
    exp_avg_sq: LayoutFieldId

    def __post_init__(self) -> None:
        field_ids = (
            int(self.parameter),
            int(self.gradient),
            int(self.step),
            int(self.exp_avg),
            int(self.exp_avg_sq),
        )
        if any(field_id < 0 for field_id in field_ids) or len(
            set(field_ids)
        ) != len(field_ids):
            raise ValueError("dense page AdamW fields must be distinct non-negative IDs")


@dataclass(frozen=True, slots=True)
class DensePageAdamWConfig:
    """Closed numeric configuration for a provider-owned CPU AdamW step."""

    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float

    def __post_init__(self) -> None:
        beta_one, beta_two = self.betas
        if (
            self.lr <= 0.0
            or not 0.0 <= beta_one < 1.0
            or not 0.0 <= beta_two < 1.0
            or self.eps <= 0.0
            or self.weight_decay < 0.0
        ):
            raise ValueError("dense page AdamW configuration is invalid")


def _cpu_dtype(dtype: TensorDType) -> torch.dtype:
    match dtype:
        case TensorDType.BOOL:
            return torch.bool
        case TensorDType.UINT8:
            return torch.uint8
        case TensorDType.INT32:
            return torch.int32
        case TensorDType.INT64:
            return torch.int64
        case TensorDType.FLOAT16:
            return torch.float16
        case TensorDType.BFLOAT16:
            return torch.bfloat16
        case TensorDType.FLOAT32:
            return torch.float32
        case TensorDType.FLOAT64:
            return torch.float64


class DensePageStateProvider:
    """A CPU-only dense-page StateRef provider with no representation fallback.

    CUDA allocation, CUDA fences, capture, and worker transport are deliberately
    unsupported here.  This class is a clean-room CPU producer for the signed
    tensor-layout and state-representation contracts.
    """

    __slots__ = (
        "_resource",
        "_layout",
        "_selected",
        "_device",
        "_pools",
        "_pool_bytes",
        "_free_page_ids",
        "_roots",
        "_latest_by_lineage",
        "_admitted_lineages",
        "_pins",
        "_pending_transition",
        "_next_pin_id",
        "_next_transaction_id",
        "_lock",
    )

    def __init__(
        self,
        *,
        tensor_layout_request: TensorLayoutRequest,
        tensor_layout_decision: TensorLayoutDecision,
        representation_request: StateRepresentationRequest,
        representation_decision: StateRepresentationDecision,
        physical_plan: StatePhysicalPlan,
        local_candidates: tuple[StateProviderPlan, ...],
        resource: str,
    ) -> None:
        validate_tensor_layouts(tensor_layout_request, tensor_layout_decision)
        validate_state_representation(
            representation_request,
            representation_decision,
        )
        validate_state_representation_reconstruction(
            representation_request,
            representation_decision,
            local_candidates,
        )
        validate_state_physical_plan(
            representation_request,
            representation_decision,
            physical_plan,
        )
        if (
            physical_plan.request_digest != representation_decision.request_digest
            or physical_plan.representation_signature
            != representation_decision.signature
            or physical_plan.representations != representation_decision.plans
            or physical_plan.reserved_bytes != representation_decision.reserved_bytes
        ):
            raise RuntimeError("physical state plan differs from its signed decision")
        if not resource:
            raise ValueError("dense page provider requires a resource")
        layout_items = tuple(
            item
            for item in tensor_layout_request.layouts
            if item.resource == resource
        )
        layout_bindings = tuple(
            binding
            for binding in tensor_layout_decision.bindings
            if binding.resource == resource
        )
        selected_plans = tuple(
            plan
            for plan in representation_decision.plans
            if plan.resource == resource
        )
        requirements = tuple(
            requirement
            for requirement in representation_request.states
            if requirement.resource == resource
        )
        if (
            len(layout_items) != 1
            or len(layout_bindings) != 1
            or len(selected_plans) != 1
            or len(requirements) != 1
        ):
            raise RuntimeError("signed plans do not define one selected state resource")
        layout_item: StateTensorLayout = layout_items[0]
        layout: TensorLayoutSpec = layout_item.layout
        layout_binding = layout_bindings[0]
        selected = selected_plans[0]
        requirement: StateLayoutRequirement = requirements[0]
        if selected.representation is not StateRepresentation.DENSE_PAGE_ARENA:
            raise RuntimeError("signed state representation is not dense_page_arena")
        if (
            selected.layout_id != layout_binding.layout_id
            or selected.layout_namespace != layout_binding.layout_namespace
            or selected.layout_digest != layout_binding.layout_digest
            or selected.max_admitted_lineages
            != requirement.max_admitted_lineages
            or selected.root_page_count != 1
        ):
            raise RuntimeError("signed dense provider differs from tensor layout binding")
        physical_rows = (
            selected.max_admitted_lineages
            * selected.max_live_versions_per_lineage
        )
        device = torch.device("cpu")
        pools = tuple(
            torch.zeros(
                (physical_rows, *field.shape),
                dtype=_cpu_dtype(field.dtype),
                device=device,
            )
            for field in layout.fields
        )
        pool_bytes = sum(
            int(pool.numel()) * int(pool.element_size()) for pool in pools
        )
        if pool_bytes != selected.reserved_bytes:
            raise RuntimeError(
                "signed reserved bytes differ from dense CPU pool allocation"
            )
        self._resource = resource
        self._layout = layout
        self._selected = selected
        self._device = device
        self._pools = pools
        self._pool_bytes = pool_bytes
        self._free_page_ids = list(range(physical_rows))
        self._roots: dict[StateRef, _StoredDensePageRoot] = {}
        self._latest_by_lineage: dict[LineageId, StateRef] = {}
        self._admitted_lineages: set[LineageId] = set()
        self._pins: dict[int, DensePageVersionPin] = {}
        self._pending_transition: _PendingTransition = _NO_PENDING_TRANSITION
        self._next_pin_id = 1
        self._next_transaction_id = 1
        self._lock = RLock()

    @property
    def resource(self) -> str:
        return self._resource

    @property
    def device(self) -> torch.device:
        """The only supported allocation domain: CPU."""

        return self._device

    @property
    def selected_plan(self) -> StateProviderPlan:
        return self._selected

    @property
    def pool_bytes(self) -> int:
        return self._pool_bytes

    @property
    def free_pages(self) -> int:
        with self._lock:
            return len(self._free_page_ids)

    def admit_batch(
        self,
        admissions: tuple[DensePageAdmission, ...],
    ) -> tuple[DensePageRoot, ...]:
        """Atomically admit distinct V0 lineages into fixed CPU page rows."""

        with self._lock:
            self._require_no_pending_transition()
            refs = tuple(admission.ref for admission in admissions)
            if not refs or len(refs) != len(set(refs)):
                raise ValueError("dense page admission requires unique references")
            lineages = tuple(ref.lineage_id for ref in refs)
            if len(lineages) != len(set(lineages)):
                raise ValueError("dense page admission requires unique lineages")
            if (
                len(self._admitted_lineages) + len(lineages)
                > self._selected.max_admitted_lineages
            ):
                raise RuntimeError("dense page admission entitlement is exhausted")
            for admission in admissions:
                self._require_admission_ref(admission.ref)
                self._require_initial_tensors(admission.initial_tensors)
            page_ids = self._reserve_pages(len(refs))
            try:
                with torch.no_grad():
                    for admission, page_id in zip(
                        admissions,
                        page_ids,
                        strict=True,
                    ):
                        for pool, initial in zip(
                            self._pools,
                            admission.initial_tensors,
                            strict=True,
                        ):
                            pool[page_id].copy_(initial)
            except Exception:
                self._return_pages(page_ids)
                raise
            roots = tuple(
                DensePageRoot(ref, page_id)
                for ref, page_id in zip(refs, page_ids, strict=True)
            )
            for root in roots:
                self._roots[root.ref] = _StoredDensePageRoot(root)
                self._latest_by_lineage[root.ref.lineage_id] = root.ref
                self._admitted_lineages.add(root.ref.lineage_id)
            return roots

    def pin_exact(self, refs: tuple[StateRef, ...]) -> DensePageVersionPin:
        """Pin exact immutable roots; later versions do not alter this pin."""

        with self._lock:
            self._require_no_pending_transition()
            if not refs or len(refs) != len(set(refs)):
                raise ValueError("dense page pin requires unique references")
            for ref in refs:
                self._stored_root(ref)
            pin = DensePageVersionPin(self._next_pin_id, refs)
            for ref in refs:
                self._stored_root(ref).pins += 1
            self._pins[pin.pin_id] = pin
            self._next_pin_id += 1
            return pin

    def release_pin(self, pin: DensePageVersionPin) -> None:
        with self._lock:
            self._require_no_pending_transition()
            try:
                issued = self._pins[pin.pin_id]
            except KeyError as exc:
                raise RuntimeError("dense page pin is unknown or released") from exc
            if issued is not pin:
                raise RuntimeError("dense page pin differs from the issued pin")
            for ref in pin.refs:
                self._stored_root(ref).pins -= 1
            self._pins.pop(pin.pin_id)
            self._reclaim_unpinned_roots()

    def prepare_transition_batch(
        self,
        sources: tuple[StateRef, ...],
        targets: tuple[StateRef, ...],
    ) -> DensePageTransitionBatch:
        """Reserve all COW targets before any target root becomes visible."""

        with self._lock:
            self._require_no_pending_transition()
            if (
                not sources
                or len(sources) != len(targets)
                or len(sources) != len(set(sources))
                or len(targets) != len(set(targets))
            ):
                raise ValueError(
                    "dense page transition requires aligned unique references"
                )
            lineages = tuple(source.lineage_id for source in sources)
            if len(lineages) != len(set(lineages)):
                raise ValueError("dense page transition requires unique lineages")
            for source, target in zip(sources, targets, strict=True):
                self._require_transition_refs(source, target)
            page_ids = self._reserve_pages(len(targets))
            try:
                with torch.no_grad():
                    for page_id in page_ids:
                        for pool in self._pools:
                            pool[page_id].zero_()
                transitions = tuple(
                    DensePageTransition(source, target)
                    for source, target in zip(sources, targets, strict=True)
                )
                source_page_ids = tuple(
                    self._stored_root(source).root.page_id for source in sources
                )
            except Exception:
                self._return_pages(page_ids)
                raise
            for source in sources:
                self._stored_root(source).pins += 1
            batch = DensePageTransitionBatch(
                self._next_transaction_id,
                transitions,
            )
            self._pending_transition = _ReservedTransitionBatch(
                batch,
                source_page_ids,
                page_ids,
            )
            self._next_transaction_id += 1
            return batch

    def compute_adamw_transition(
        self,
        batch: DensePageTransitionBatch,
        fields: DensePageAdamWFieldBinding,
        config: DensePageAdamWConfig,
    ) -> None:
        """Copy every source root and compute the batch's private CPU targets."""

        with self._lock:
            reserved = self._reserved_transition(batch)
            try:
                self._require_adamw_fields(fields)
                with torch.no_grad():
                    for source_page_id, target_page_id in zip(
                        reserved.source_page_ids,
                        reserved.target_page_ids,
                        strict=True,
                    ):
                        for pool in self._pools:
                            pool[target_page_id].copy_(pool[source_page_id])
                    adamw(
                        params=[
                            self._pools[int(fields.parameter)][page_id]
                            for page_id in reserved.target_page_ids
                        ],
                        grads=[
                            self._pools[int(fields.gradient)][page_id]
                            for page_id in reserved.target_page_ids
                        ],
                        exp_avgs=[
                            self._pools[int(fields.exp_avg)][page_id]
                            for page_id in reserved.target_page_ids
                        ],
                        exp_avg_sqs=[
                            self._pools[int(fields.exp_avg_sq)][page_id]
                            for page_id in reserved.target_page_ids
                        ],
                        max_exp_avg_sqs=[],
                        state_steps=[
                            self._pools[int(fields.step)][page_id]
                            for page_id in reserved.target_page_ids
                        ],
                        foreach=False,
                        capturable=False,
                        differentiable=False,
                        fused=False,
                        amsgrad=False,
                        beta1=config.betas[0],
                        beta2=config.betas[1],
                        lr=config.lr,
                        weight_decay=config.weight_decay,
                        eps=config.eps,
                        maximize=False,
                    )
            except Exception:
                self._rollback_reserved_transition(reserved)
                raise
            self._pending_transition = _ComputedTransitionBatch(reserved)

    def publish_transition_batch(
        self,
        batch: DensePageTransitionBatch,
    ) -> tuple[DensePageRoot, ...]:
        """Atomically publish every reserved target root after CPU compute."""

        with self._lock:
            reserved = self._computed_transition(batch)
            roots = tuple(
                DensePageRoot(transition.target, target_page_id)
                for transition, target_page_id in zip(
                    batch.transitions,
                    reserved.target_page_ids,
                    strict=True,
                )
            )
            for root in roots:
                self._roots[root.ref] = _StoredDensePageRoot(root)
                self._latest_by_lineage[root.ref.lineage_id] = root.ref
            for transition in batch.transitions:
                self._stored_root(transition.source).pins -= 1
            self._pending_transition = _NO_PENDING_TRANSITION
            self._reclaim_unpinned_roots()
            return roots

    def abort_transition_batch(self, batch: DensePageTransitionBatch) -> None:
        """Discard all reserved CPU targets before they are published."""

        with self._lock:
            match self._pending_transition:
                case _ReservedTransitionBatch(batch=issued) as reserved if issued is batch:
                    self._rollback_reserved_transition(reserved)
                case _ComputedTransitionBatch(reserved=reserved) if reserved.batch is batch:
                    self._rollback_reserved_transition(reserved)
                case _:
                    raise RuntimeError("dense page transition is not the pending batch")

    def root(self, ref: StateRef) -> DensePageRoot:
        with self._lock:
            return self._stored_root(ref).root

    def snapshot(self, ref: StateRef) -> tuple[torch.Tensor, ...]:
        """Return immutable oracle copies for one exact logical version root."""

        with self._lock:
            root = self._stored_root(ref).root
            return tuple(pool[root.page_id].detach().clone() for pool in self._pools)

    def retire(self, ref: StateRef) -> None:
        """Retire an exact latest root only after all retained versions reclaim."""

        with self._lock:
            self._require_no_pending_transition()
            self._reclaim_unpinned_roots()
            stored = self._stored_root(ref)
            if (
                ref.lineage_id not in self._latest_by_lineage
                or self._latest_by_lineage[ref.lineage_id] != ref
            ):
                raise RuntimeError("dense page retirement requires the latest root")
            if stored.pins:
                raise RuntimeError("cannot retire a pinned dense page root")
            if any(
                candidate.lineage_id == ref.lineage_id and candidate != ref
                for candidate in self._roots
            ):
                raise RuntimeError(
                    "cannot retire a lineage with retained dense page roots"
                )
            self._roots.pop(ref)
            self._latest_by_lineage.pop(ref.lineage_id)
            self._admitted_lineages.remove(ref.lineage_id)
            self._return_pages((stored.root.page_id,))

    def _require_admission_ref(self, ref: StateRef) -> None:
        if (
            ref.layout_id != self._selected.layout_id
            or ref.version_id != VersionId(0)
        ):
            raise ValueError("dense page admission requires this layout at V0")
        if ref.lineage_id in self._admitted_lineages or ref in self._roots:
            raise RuntimeError("dense page lineage is already admitted")

    def _require_transition_refs(self, source: StateRef, target: StateRef) -> None:
        if (
            target.layout_id != source.layout_id
            or target.lineage_id != source.lineage_id
            or int(target.version_id) != int(source.version_id) + 1
        ):
            raise ValueError("dense page targets must be next lineage versions")
        if source.layout_id != self._selected.layout_id:
            raise RuntimeError("dense page transition has another layout")
        if (
            source.lineage_id not in self._latest_by_lineage
            or self._latest_by_lineage[source.lineage_id] != source
        ):
            raise RuntimeError("dense page transition source is not latest")
        if target in self._roots:
            raise RuntimeError("dense page transition target already exists")
        live_versions = sum(
            root_ref.lineage_id == source.lineage_id for root_ref in self._roots
        )
        if live_versions + 1 > self._selected.max_live_versions_per_lineage:
            raise RuntimeError("dense page version entitlement is exhausted")

    def _stored_root(self, ref: StateRef) -> _StoredDensePageRoot:
        try:
            return self._roots[ref]
        except KeyError as exc:
            raise RuntimeError("dense page root is unknown or reclaimed") from exc

    def _reserved_transition(
        self,
        batch: DensePageTransitionBatch,
    ) -> _ReservedTransitionBatch:
        match self._pending_transition:
            case _ReservedTransitionBatch(batch=issued) as reserved if issued is batch:
                return reserved
            case _:
                raise RuntimeError("dense page transition is not reserved")

    def _computed_transition(
        self,
        batch: DensePageTransitionBatch,
    ) -> _ReservedTransitionBatch:
        match self._pending_transition:
            case _ComputedTransitionBatch(reserved=reserved) if reserved.batch is batch:
                return reserved
            case _:
                raise RuntimeError("dense page transition has not completed AdamW")

    def _rollback_reserved_transition(
        self,
        reserved: _ReservedTransitionBatch,
    ) -> None:
        for transition in reserved.batch.transitions:
            self._stored_root(transition.source).pins -= 1
        self._return_pages(reserved.target_page_ids)
        self._pending_transition = _NO_PENDING_TRANSITION
        self._reclaim_unpinned_roots()

    def _require_adamw_fields(self, fields: DensePageAdamWFieldBinding) -> None:
        field_ids = (
            int(fields.parameter),
            int(fields.gradient),
            int(fields.step),
            int(fields.exp_avg),
            int(fields.exp_avg_sq),
        )
        if any(field_id >= len(self._layout.fields) for field_id in field_ids):
            raise ValueError("dense page AdamW field ID is outside the signed layout")
        parameter = self._layout.fields[int(fields.parameter)]
        gradient = self._layout.fields[int(fields.gradient)]
        exp_avg = self._layout.fields[int(fields.exp_avg)]
        exp_avg_sq = self._layout.fields[int(fields.exp_avg_sq)]
        step = self._layout.fields[int(fields.step)]
        if (
            parameter.shape != gradient.shape
            or parameter.shape != exp_avg.shape
            or parameter.shape != exp_avg_sq.shape
            or parameter.dtype != gradient.dtype
            or parameter.dtype != exp_avg.dtype
            or parameter.dtype != exp_avg_sq.dtype
            or step.shape != ()
        ):
            raise ValueError("dense page AdamW fields differ from the signed layout")

    def _require_initial_tensors(
        self,
        initial_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        if len(initial_tensors) != len(self._layout.fields):
            raise ValueError("dense page admission field count differs from layout")
        for initial, field, pool in zip(
            initial_tensors,
            self._layout.fields,
            self._pools,
            strict=True,
        ):
            if (
                tuple(initial.shape) != field.shape
                or initial.dtype != pool.dtype
                or initial.device != self._device
            ):
                raise ValueError(
                    "dense page admission tensor differs from signed CPU layout"
                )

    def _reserve_pages(self, count: int) -> tuple[int, ...]:
        if len(self._free_page_ids) < count:
            raise AssertionError("dense page entitlement lost its physical pages")
        pages = tuple(self._free_page_ids[:count])
        del self._free_page_ids[:count]
        return pages

    def _return_pages(self, page_ids: tuple[int, ...]) -> None:
        self._free_page_ids.extend(page_ids)
        self._free_page_ids.sort()

    def _reclaim_unpinned_roots(self) -> None:
        latest = frozenset(self._latest_by_lineage.values())
        reclaimable = tuple(
            ref
            for ref, stored in self._roots.items()
            if ref not in latest and stored.pins == 0
        )
        pages = tuple(self._roots.pop(ref).root.page_id for ref in reclaimable)
        self._return_pages(pages)

    def _require_no_pending_transition(self) -> None:
        match self._pending_transition:
            case _NoPendingTransition():
                return
            case _ReservedTransitionBatch() | _ComputedTransitionBatch():
                raise RuntimeError(
                    "dense page provider already has a pending transition"
                )


__all__ = [
    "DensePageAdmission",
    "DensePageAdamWConfig",
    "DensePageAdamWFieldBinding",
    "DensePageRoot",
    "DensePageStateProvider",
    "DensePageTransition",
    "DensePageTransitionBatch",
    "DensePageVersionPin",
]
