#!/usr/bin/env python3
"""Static dispatch plan for runtime event -> consumer routing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional

from tllm.ports.consumer import BaseConsumer
from tllm.ports.base import ConsumerFlow
from tllm.ports.base import PortKind
from tllm.contracts.subscription import ConsumerSubscription
from tllm.runtime.consumer_compat import consumer_subscriptions
from tllm.runtime.graph_policy import GraphMode
from tllm.runtime.placement import CompiledPlacement


@dataclass(frozen=True)
class DispatchTarget:
    consumer: BaseConsumer
    subscription: ConsumerSubscription


@dataclass(frozen=True)
class FlowDispatchTarget:
    consumer: BaseConsumer
    flow: ConsumerFlow
    flow_id: str = ""
    placement_hook: str = "execute_model.post"
    flow_ordinal: int = 0

    @property
    def consumer_id(self) -> str:
        return str(self.consumer.consumer_id)


class DispatchPlan:
    def __init__(self, by_event: Dict[str, List[DispatchTarget]], flow_targets: List[FlowDispatchTarget]) -> None:
        self._by_event = by_event
        self._flow_targets = flow_targets
        self._graph_mode = "eager"
        self._fusion_groups: tuple[tuple[str, ...], ...] = ()
        self._fusion_rejections: tuple[tuple[str, str], ...] = ()
        self._placement_metadata_pending = True
        self._rebuild_flow_hook_table()

    def _rebuild_flow_hook_table(self) -> None:
        by_hook: dict[str, list[FlowDispatchTarget]] = {}
        for target in self._flow_targets:
            by_hook.setdefault(target.placement_hook, []).append(target)
        self._by_flow_hook = {hook: tuple(targets) for hook, targets in by_hook.items()}

    def _install_compiled_placement(self, compiled: CompiledPlacement) -> None:
        order = {flow_id: index for index, flow_id in enumerate(compiled.execution_order)}
        targets = [
            replace(
                target,
                placement_hook=compiled.by_flow_id.get(target.flow_id, target.placement_hook),
            )
            for target in self._flow_targets
        ]
        # Python's sort is stable, so post-generate flows absent from the Rust
        # schedule retain declaration order after all dependency-scheduled flows.
        targets.sort(key=lambda target: order.get(target.flow_id, len(order)))
        self._flow_targets = targets
        self._fusion_groups = compiled.fusion_groups
        self._fusion_rejections = compiled.fusion_rejections
        self._rebuild_flow_hook_table()

    @classmethod
    def build(cls, consumers: Iterable[BaseConsumer]) -> "DispatchPlan":
        by_event: Dict[str, List[DispatchTarget]] = {}
        flow_records: List[tuple[str, BaseConsumer, ConsumerFlow]] = []
        for consumer in consumers:
            flows = list(consumer.flows())
            for flow_i, flow in enumerate(flows):
                flow_records.append((f"{consumer.consumer_id}#{flow_i}", consumer, flow))
            if flows:
                continue
            for sub in consumer_subscriptions(consumer):
                by_event.setdefault(sub.event_name, []).append(DispatchTarget(consumer=consumer, subscription=sub))
        # Timing-contract validation is owned by the Rust placement compiler;
        # its result is cached so the per-step path does no graph traversal.
        from tllm.runtime.placement import compile_flow_placement
        from tllm.runtime.ports.post_generate_dispatch import is_post_generate_flow

        flow_targets = [
            FlowDispatchTarget(
                consumer=consumer,
                flow=flow,
                flow_id=flow_id,
                flow_ordinal=int(flow_id.rsplit("#", 1)[1]),
            )
            for flow_id, consumer, flow in flow_records
        ]
        plan = cls(by_event=by_event, flow_targets=flow_targets)
        # Preserve the public DispatchPlan validation contract before the Rust
        # compiler builds WAW edges from the same descriptors.
        plan.write_conflict_check()
        compiled = compile_flow_placement(
            [(flow_id, flow) for flow_id, _consumer, flow in flow_records if not is_post_generate_flow(flow)]
        )
        plan._install_compiled_placement(compiled)
        return plan

    def select(
        self,
        *,
        event_name: str,
        phase: Optional[str],
        layer_path: Optional[str],
        capture_enabled: bool,
    ) -> List[DispatchTarget]:
        targets = self._by_event.get(event_name, [])
        out: List[DispatchTarget] = []
        for target in targets:
            sub = target.subscription
            if sub.phase_filter is not None and phase is not None and sub.phase_filter != phase:
                continue
            if sub.layer_filter is not None and layer_path is not None and sub.layer_filter != layer_path:
                continue
            if sub.capture_policy == "required" and (not capture_enabled):
                continue
            if sub.capture_policy == "never" and capture_enabled:
                continue
            out.append(target)
        return out

    def flow_targets(self) -> List[FlowDispatchTarget]:
        return list(self._flow_targets)

    def has_movable_flow_targets(self) -> bool:
        return any(target.placement_hook != "execute_model.post" for target in self._flow_targets)

    def requires_graph_mode_resolution(self) -> bool:
        return any(
            str(target.flow.fusion) != "none"
            or any(write.fusion_candidate for write in target.flow.writes)
            for target in self._flow_targets
        )

    def requires_graph_mode_refresh(self, graph_mode: GraphMode) -> bool:
        """Whether resolving ``graph_mode`` will still change placement state.

        This is a cold-path query for startup/control-plane observers.  The
        dispatch loop continues to own the actual ``ensure_graph_mode`` call.
        """

        return self.requires_graph_mode_resolution() and (
            self._placement_metadata_pending or graph_mode.name.lower() != self._graph_mode
        )

    def flow_targets_for_hook(self, hook: str) -> tuple[FlowDispatchTarget, ...]:
        return self._by_flow_hook.get(str(hook), ())

    @property
    def fusion_groups(self) -> tuple[tuple[str, ...], ...]:
        return self._fusion_groups

    @property
    def fusion_rejections(self) -> tuple[tuple[str, str], ...]:
        return self._fusion_rejections

    def ensure_graph_mode(self, graph_mode: GraphMode) -> bool:
        from tllm.runtime.placement import compile_flow_placement
        from tllm.runtime.ports.post_generate_dispatch import is_post_generate_flow

        mode_name = graph_mode.name.lower()
        if mode_name == self._graph_mode:
            pending = self._placement_metadata_pending
            self._placement_metadata_pending = False
            return pending
        compiled = compile_flow_placement(
            [
                (target.flow_id, target.flow)
                for target in self._flow_targets
                if not is_post_generate_flow(target.flow)
            ],
            graph_mode=graph_mode,
        )
        self._graph_mode = mode_name
        self._placement_metadata_pending = False
        self._install_compiled_placement(compiled)
        return True

    def has_active_targets(self) -> bool:
        return not self.is_empty()

    def is_empty(self) -> bool:
        return (not self._flow_targets) and all((not targets) for targets in self._by_event.values())

    def requires_device_decode_metadata(self) -> bool:
        return any(bool(targets) for targets in self._by_event.values())

    def requires_full_residual_capture(self) -> bool:
        if any(targets for targets in self._by_event.values()):
            return True
        saw_residual_flow = False
        for target in self._flow_targets:
            has_residual_read = any(read.kind == PortKind.RESIDUAL_STREAM for read in target.flow.reads)
            if not has_residual_read:
                continue
            saw_residual_flow = True
            if str(getattr(target.flow, "row_compaction", "none")) != "first_per_prompt":
                return True
        return not saw_residual_flow

    def required_residual_layers(self) -> set[tuple[int, str, str]]:
        required: set[tuple[int, str, str]] = set()
        for target in self._flow_targets:
            for read in target.flow.reads:
                if read.kind != PortKind.RESIDUAL_STREAM or read.locator is None:
                    continue
                layer = int(getattr(read.locator, "layer"))
                site = str(getattr(read.locator, "site"))
                phase = str(getattr(read.locator, "phase"))
                required.add((layer, site, phase))
        return required

    def residual_writes(self) -> List[tuple[str, ConsumerFlow, object]]:
        """All declared residual-stream writes as (consumer_id, flow, write)."""
        writes: List[tuple[str, ConsumerFlow, object]] = []
        for target in self._flow_targets:
            for write in target.flow.writes:
                if getattr(write, "kind", None) != PortKind.RESIDUAL_STREAM:
                    continue
                if getattr(write, "locator", None) is None:
                    continue
                writes.append((str(target.consumer.consumer_id), target.flow, write))
        return writes

    def required_residual_writes(self) -> set[tuple[int, str, str]]:
        required: set[tuple[int, str, str]] = set()
        for _consumer_id, _flow, write in self.residual_writes():
            locator = getattr(write, "locator")
            required.add(
                (
                    int(getattr(locator, "layer")),
                    str(getattr(locator, "site")),
                    str(getattr(locator, "phase")),
                )
            )
        return required

    def write_conflict_check(self) -> None:
        """Writers on one residual locator must be disambiguated by ``order``.

        ``append`` writes are post-generate sinks and never conflict; every
        other pair of writers on the same (layer, site, phase) must carry
        distinct ``order`` values so the runtime applies them deterministically.
        """
        by_locator: Dict[tuple[int, str, str], List[tuple[str, int, str]]] = {}
        for consumer_id, _flow, write in self.residual_writes():
            mode = str(getattr(write, "mode", "") or "")
            if mode == "append":
                continue
            locator = getattr(write, "locator")
            key = (
                int(getattr(locator, "layer")),
                str(getattr(locator, "site")),
                str(getattr(locator, "phase")),
            )
            by_locator.setdefault(key, []).append((consumer_id, int(getattr(write, "order", 0)), mode))
        for locator, writers in by_locator.items():
            if len(writers) < 2:
                continue
            orders = [order for _cid, order, _mode in writers]
            if len(set(orders)) != len(orders):
                detail = ", ".join(f"{cid}(order={order}, mode={mode})" for cid, order, mode in writers)
                raise ValueError(
                    f"conflicting residual writes on locator {locator}: {detail}; "
                    "assign distinct `order` values to same-site writers"
                )

    def has_residual_row_compaction(self, mode: str) -> bool:
        expected = str(mode)
        for target in self._flow_targets:
            has_residual_read = any(read.kind == PortKind.RESIDUAL_STREAM for read in target.flow.reads)
            if not has_residual_read:
                continue
            if str(getattr(target.flow, "row_compaction", "none")) == expected:
                return True
        return False

    def max_residual_compact_rows(self, mode: str = "first_per_prompt") -> int:
        expected = str(mode)
        caps: list[int] = []
        has_compact_residual_flow = False
        has_uncapped_compact_flow = False
        for target in self._flow_targets:
            has_residual_read = any(read.kind == PortKind.RESIDUAL_STREAM for read in target.flow.reads)
            if not has_residual_read:
                continue
            if str(getattr(target.flow, "row_compaction", "none")) != expected:
                continue
            has_compact_residual_flow = True
            cap = int(getattr(target.flow, "max_bundle_rows", 0) or 0)
            if cap <= 0:
                has_uncapped_compact_flow = True
            else:
                caps.append(cap)
        if not has_compact_residual_flow:
            return 0
        if has_uncapped_compact_flow or not caps:
            return 0
        return max(1, max(caps))
