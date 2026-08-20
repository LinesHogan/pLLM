"""RED contracts for the Phase 4 runtime audit.

These tests intentionally stay at the Python control-plane boundary.  The
communicator, adapter, lifecycle, and rank-agreement objects are fakes, so a
test run never initializes CUDA or ``torch.distributed``.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

from tllm.runtime import active_targets, residual_runtime
from tllm.runtime.scheduling import phase4_collective as phase4_runtime
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.vllm_patch import common_hooks
from tllm.runtime.vllm_patch import port_runtime_hooks
from tllm.workgraph import phase3
from tllm.workgraph.phase4 import (
    CollectiveBackend,
    CollectiveKey,
    CollectiveKind,
    CollectiveRequestDesc,
    CollectiveWindow,
    CommunicatorKey,
    DistributedTopology,
    ParallelGroupKind,
    TopologyRankDesc,
)


class _FakeFence:
    def query(self) -> bool:
        return True


class _FakeCollectiveAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object, object]] = []

    def launch(
        self,
        collective: object,
        communicator_desc: object,
        payload: object,
        communicator: object,
    ) -> _FakeFence:
        self.calls.append((collective, communicator_desc, payload, communicator))
        return _FakeFence()


class _FakeAgreement:
    """A typed-agreement stand-in that records the frozen rank snapshot."""

    def __init__(self) -> None:
        self.calls: list[object] = []
        self.error: BaseException | None = None
        self.divergent_field: str | None = None

    def exchange(self, metadata: object) -> tuple[object, ...]:
        if self.error is not None:
            raise self.error
        self.calls.append(metadata)
        if self.divergent_field is not None and is_dataclass(metadata):
            value = getattr(metadata, self.divergent_field)
            if isinstance(value, int):
                metadata = replace(metadata, **{self.divergent_field: value + 1})
            elif isinstance(value, str):
                metadata = replace(metadata, **{self.divergent_field: "f" * len(value)})
        return (metadata,)


def _topology() -> DistributedTopology:
    return DistributedTopology(
        ranks=(
            TopologyRankDesc(
                rank=0,
                node_id="cpu-node",
                local_rank=0,
                placement=phase3.Placement("cpu"),
                tp_rank=0,
                pp_rank=0,
                dp_rank=0,
            ),
        ),
        tp_size=1,
        pp_size=1,
        dp_size=1,
    )


def _two_rank_topology() -> DistributedTopology:
    return DistributedTopology(
        ranks=tuple(
            TopologyRankDesc(
                rank=rank,
                node_id=f"cpu-node-{rank}",
                local_rank=0,
                placement=phase3.Placement("cpu"),
                tp_rank=rank,
                pp_rank=0,
                dp_rank=0,
            )
            for rank in range(2)
        ),
        tp_size=2,
        pp_size=1,
        dp_size=1,
    )


def _request(
    *,
    name: str = "audit",
    window: CollectiveWindow = CollectiveWindow.AFTER_COMPUTE,
    depends_on: tuple[CollectiveKey, ...] = (),
) -> CollectiveRequestDesc:
    return CollectiveRequestDesc(
        key=CollectiveKey("phase4.audit", name, 0),
        kind=CollectiveKind.ALL_REDUCE,
        group=ParallelGroupKind.GLOBAL,
        backend=CollectiveBackend.GLOO,
        window=window,
        depends_on=depends_on,
    )


def _binding(
    test_case: unittest.TestCase,
    *,
    request: CollectiveRequestDesc | None = None,
    agreement: _FakeAgreement | None = None,
) -> object:
    """Build a CPU binding and make the new agreement input an explicit RED.

    Keeping the compatibility check here makes a missing production field a
    normal assertion failure instead of an unhelpful constructor traceback.
    """

    kwargs: dict[str, object] = {
        "topology": _topology(),
        "rank": 0,
        "communicators": (
            (CommunicatorKey(ParallelGroupKind.GLOBAL, 0), SimpleNamespace(owner_rank=0)),
        ),
        "adapter": _FakeCollectiveAdapter(),
        "collective_requests": (request or _request(),),
    }
    if agreement is not None:
        try:
            binding = phase4_runtime.DistributedRuntimeBinding(
                **kwargs,
                agreement=agreement,
            )
        except TypeError as exc:
            test_case.fail(
                "DistributedRuntimeBinding must accept a typed cross-rank "
                f"agreement object: {exc}"
            )
        else:
            return binding
    return phase4_runtime.DistributedRuntimeBinding(**kwargs)


class Phase4CollectiveLifecycleAuditRedTest(unittest.TestCase):
    def test_collective_launch_uses_runtime_current_window_not_a_caller_forged_window(self) -> None:
        first = _request(name="before", window=CollectiveWindow.BEFORE_COMPUTE)
        second = _request(
            name="after",
            window=CollectiveWindow.AFTER_COMPUTE,
            depends_on=(first.key,),
        )
        plan = phase4_runtime.compile_distributed_flow_plan(
            plan_epoch=1,
            topology=_topology(),
            collective_requests=(first, second),
        )
        adapter = _FakeCollectiveAdapter()
        executor = phase4_runtime.DistributedPlanExecutor(
            plan=plan,
            rank=0,
            communicators=((CommunicatorKey(ParallelGroupKind.GLOBAL, 0), SimpleNamespace()),),
            adapter=adapter,
        )
        resources = phase4_runtime.DistributedPlanResources(plan, executor)
        store = phase4_runtime.DistributedPlanSnapshotStore(resources)
        state = ResidualRuntimeState()
        state.distributed_plan_store = store
        state.distributed_prepared_step_id = 4
        store.prepare_step(4)

        self.assertNotIn(
            "window",
            inspect.signature(executor.launch).parameters,
            "launch must not trust a caller-provided CollectiveWindow",
        )
        key = first.key
        with self.assertRaises(RuntimeError):
            executor.launch(4, collective=key, payload=object())

        runner = SimpleNamespace()
        common_hooks.dispatch_runtime_event(
            runtime=state,
            runner=runner,
            event_name="prepare_inputs.pre",
        )
        with self.assertRaises(RuntimeError):
            common_hooks.dispatch_runtime_event(
                runtime=state,
                runner=runner,
                event_name="execute_model.post",
            )
        common_hooks.dispatch_runtime_event(
            runtime=state,
            runner=runner,
            event_name="execute_model.pre",
        )
        executor.launch(4, collective=key, payload=object())

        common_hooks.dispatch_runtime_event(
            runtime=state,
            runner=runner,
            event_name="execute_model.post",
        )
        executor.launch(4, collective=second.key, payload=object())

        common_hooks.dispatch_runtime_event(
            runtime=state,
            runner=runner,
            event_name="sample_tokens.post",
        )
        executor.finish_step(4)


class Phase4CrossRankAgreementAuditRedTest(unittest.TestCase):
    def test_multi_rank_agreement_requires_exact_rank_projections(self) -> None:
        plan = phase4_runtime.compile_distributed_flow_plan(
            plan_epoch=8,
            topology=_two_rank_topology(),
            collective_requests=(_request(),),
        )
        agreements = tuple(
            phase4_runtime.distributed_plan_agreement_desc(plan, rank)
            for rank in range(2)
        )
        phase4_runtime.validate_distributed_plan_agreement(plan, agreements)

        with self.assertRaises(RuntimeError):
            phase4_runtime.validate_distributed_plan_agreement(
                plan,
                (agreements[0], replace(agreements[1], plan_signature="f" * 64)),
            )
        with self.assertRaises(ValueError):
            phase4_runtime.DistributedRuntimeBinding(
                topology=plan.topology,
                rank=0,
                communicators=((plan.communicators[0].key, SimpleNamespace()),),
                adapter=_FakeCollectiveAdapter(),
                collective_requests=plan.collective_requests,
            )

    def test_publish_agrees_on_typed_epoch_signature_topology_communicators_and_order(self) -> None:
        agreement = _FakeAgreement()
        binding = _binding(self, agreement=agreement)
        state = ResidualRuntimeState()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.configure_workgraph_distributed(binding)
            residual_runtime.publish_workgraph_distributed(plan_epoch=9)

        self.assertEqual(len(agreement.calls), 1)
        metadata = agreement.calls[0]
        self.assertTrue(
            is_dataclass(metadata),
            "rank agreement must exchange a typed metadata record, not a dict",
        )
        names = {field.name for field in fields(metadata)}
        self.assertTrue(
            {
                "plan_epoch",
                "plan_signature",
                "topology_signature",
                "communicators",
                "collectives",
            }
            <= names,
            f"agreement metadata is missing required fields: {names}",
        )

    def test_publish_fails_fast_on_rank_agreement_divergence_before_staging(self) -> None:
        agreement = _FakeAgreement()
        agreement.error = RuntimeError("rank agreement diverged: topology")
        binding = _binding(self, agreement=agreement)
        state = ResidualRuntimeState()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.configure_workgraph_distributed(binding)
            with self.assertRaises(RuntimeError):
                residual_runtime.publish_workgraph_distributed(plan_epoch=10)
            self.assertIsNone(state.distributed_plan_store)

    def test_publish_rejects_a_mismatched_typed_rank_descriptor(self) -> None:
        agreement = _FakeAgreement()
        agreement.divergent_field = "plan_epoch"
        binding = _binding(self, agreement=agreement)
        state = ResidualRuntimeState()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.configure_workgraph_distributed(binding)
            with self.assertRaises(RuntimeError):
                residual_runtime.publish_workgraph_distributed(plan_epoch=10)
            self.assertIsNone(state.distributed_plan_store)

    def test_prepare_activates_the_already_agreed_staged_epoch_without_reexchange(self) -> None:
        agreement = _FakeAgreement()
        binding = _binding(self, agreement=agreement)
        state = ResidualRuntimeState()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.configure_workgraph_distributed(binding)
            residual_runtime.publish_workgraph_distributed(plan_epoch=11)
            self.assertEqual(len(agreement.calls), 1)
            residual_runtime.prepare_workgraph_step(1)
            self.assertEqual(state.distributed_plan_store.active(1).plan_epoch, 11)
            self.assertEqual(len(agreement.calls), 1)


class Phase4DistributedBindingActivationAuditRedTest(unittest.TestCase):
    def test_distributed_binding_alone_counts_as_an_active_runtime_target(self) -> None:
        binding = _binding(self)
        state = ResidualRuntimeState()
        state.distributed_runtime_binding = binding
        self.assertTrue(
            active_targets.runtime_has_active_targets(state),
            "a configured distributed binding must activate production targets",
        )

    def test_distributed_binding_alone_enters_hook_prepare_path(self) -> None:
        binding = _binding(self)
        state = ResidualRuntimeState()
        state.distributed_runtime_binding = binding
        model = SimpleNamespace()
        runner = SimpleNamespace(model=model)
        core = SimpleNamespace(
            RUNTIME=state,
            MODEL_HOOK_FLAG="_audit_hooks_installed",
            MODEL_HOOK_SPEC_ATTR="_audit_hook_spec",
            _runner_uses_compilation_or_cudagraph=lambda _runner: False,
        )
        setup = SimpleNamespace(
            resolved_layers={},
            target_resolved="target",
            hook_spec=(),
        )
        with (
            mock.patch.object(
                port_runtime_hooks._residual_runtime_setup,
                "resolve_runtime_setup",
                return_value=setup,
            ) as resolve_setup,
            mock.patch.object(
                port_runtime_hooks._residual_runtime_setup,
                "apply_runtime_setup",
            ) as apply_setup,
            mock.patch.object(port_runtime_hooks._sampler_patch, "bind_runner_sampler"),
            mock.patch.object(
                port_runtime_hooks._sampler_patch,
                "ensure_sampler_precompute_buffers",
            ),
        ):
            port_runtime_hooks.setup_runtime_hooks_if_active(core=core, runner=runner)

        resolve_setup.assert_called_once_with(core=core, runner=runner)
        apply_setup.assert_called_once_with(core=core, runner=runner, setup=setup)
        self.assertTrue(getattr(model, "_audit_hooks_installed", False))


if __name__ == "__main__":
    unittest.main()
