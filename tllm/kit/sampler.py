"""``SamplerModifier``: the framework-owned sampler bridge.

A workload that reweights candidates declares a ``sink(...)`` output whose
publication lands in the kit :class:`SinkStore` (row identity attached by
the runtime, never by the author) and implements one method on the
workload::

    def modify_candidates(self, value: torch.Tensor, view: SamplerStepView)
        -> CandidateModifierState: ...

Everything else is owned here once: the enabled check (derived from the
signed ``enabled_field`` plus the optional ``modifier_active`` predicate),
the per-step store lookup, and the row/identity/phase/device alignment
checks — the signed timing plus the pre-draw publication fence already
guarantee them, so they are verified at this single internal point instead
of in every workload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tllm.kit.lm_head import resolve_lm_head
from tllm.kit.resources import RegionResourcesBase
from tllm.ports.sampler import CandidateModifierState, SamplerBackend
from tllm.runtime.sampler_bridge.types import SamplerStepView

if TYPE_CHECKING:
    from tllm.kit.algorithm import RegionAlgorithm


class SamplerModifier:
    """Bridge from one step's sink publication to the sampler's draw."""

    def __init__(self, workload: "RegionAlgorithm", sink: str) -> None:  # type: ignore[type-arg]
        self._workload = workload
        self._sink = sink

    def is_active(self) -> bool:
        workload = self._workload
        enabled_field = type(workload).enabled_field
        enabled = enabled_field is None or bool(getattr(workload.config, enabled_field))
        return enabled and workload.modifier_active()

    def prepare_step(self, view: SamplerStepView) -> CandidateModifierState | None:
        if not self.is_active():
            return None
        resources = self._workload.resources
        if not isinstance(resources, RegionResourcesBase):
            raise RuntimeError("sampler modifier requires RegionResourcesBase resources")
        record = resources.sinks.for_step(self._sink, view.engine_step_id)
        if record is None:
            return None
        identity = record.identity
        rows = int(view.decode_count)
        if identity.rows != rows:
            raise RuntimeError(
                f"sink {self._sink!r} published {identity.rows} rows but the "
                f"sampler step has {rows} decode rows"
            )
        observed = (
            tuple(view.request_ids[:rows]),
            tuple(view.prompt_idxs[:rows]),
            tuple(view.sample_idxs[:rows]),
        )
        expected = (identity.request_ids, identity.prompt_idxs, identity.sample_idxs)
        if observed != expected:
            raise RuntimeError(
                f"sink {self._sink!r} publication rows do not match the sampler step"
            )
        if identity.phase != view.phase:
            raise RuntimeError(
                f"sink {self._sink!r} phase {identity.phase!r} != sampler phase "
                f"{view.phase!r}"
            )
        if record.value.device != view.device:
            raise RuntimeError(
                f"sink {self._sink!r} publication device does not match the sampler step"
            )
        return self._workload.modify_candidates(record.value, view)


def candidate_modifier_state(
    *,
    beta: float,
    backend: SamplerBackend,
    pred_hidden: torch.Tensor,
    view: SamplerStepView,
) -> CandidateModifierState:
    """Assemble the modifier payload: lm_head resolution and row ids included."""
    head = resolve_lm_head(view.model)
    rows = int(pred_hidden.shape[0])
    return CandidateModifierState(
        beta=float(beta),
        backend=backend,
        affected_row_ids=torch.arange(rows, device=view.device, dtype=torch.long),
        pred_hidden=pred_hidden,
        lm_head_weight=head.weight,
        lm_head_bias=head.bias,
    )


__all__ = ["SamplerModifier", "candidate_modifier_state"]
