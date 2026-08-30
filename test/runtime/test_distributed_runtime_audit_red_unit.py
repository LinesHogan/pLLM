"""RED contracts for the distributed runtime audit.

These tests intentionally stay at the Python control-plane boundary.  The
communicator, adapter, lifecycle, and rank-agreement objects are fakes, so a
test run never initializes CUDA or ``torch.distributed``.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import inspect
from types import SimpleNamespace
from typing import cast
import unittest
from unittest import mock

import torch

from tllm import _core
from tllm.backends.vllm import v027_mrv1, v027_mrv2
from tllm.runtime.step_session import StepSession, set_step_session
from tllm.runtime import active_targets, residual_runtime
from tllm.runtime.scheduling import collective as collective_runtime
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.vllm_patch import common_hooks
from tllm.schedule.contracts import BackendObservableEvent
from tllm.runtime.vllm_patch import port_runtime_hooks
from tllm.workgraph import placement
from tllm.workgraph.distributed import (
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
                placement=placement.Placement("cpu"),
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
                placement=placement.Placement("cpu"),
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
        key=CollectiveKey("distributed.audit", name, 0),
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
            binding = collective_runtime.DistributedRuntimeBinding(
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
    return collective_runtime.DistributedRuntimeBinding(**kwargs)


class Phase4CollectiveLifecycleAuditRedTest(unittest.TestCase):
    def test_collective_launch_uses_runtime_current_window_not_a_caller_forged_window(self) -> None:
        first = _request(name="before", window=CollectiveWindow.BEFORE_COMPUTE)
        second = _request(
            name="after",
            window=CollectiveWindow.AFTER_COMPUTE,
            depends_on=(first.key,),
        )
        plan = collective_runtime.compile_distributed_flow_plan(
            plan_epoch=1,
            topology=_topology(),
            collective_requests=(first, second),
        )
        adapter = _FakeCollectiveAdapter()
        executor = collective_runtime.DistributedPlanExecutor(
            plan=plan,
            rank=0,
            communicators=((CommunicatorKey(ParallelGroupKind.GLOBAL, 0), SimpleNamespace()),),
            adapter=adapter,
        )
        resources = collective_runtime.DistributedPlanResources(plan, executor)
        store = collective_runtime.DistributedPlanSnapshotStore(resources)
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
            event=BackendObservableEvent.PREPARE_INPUTS_PRE,
        )
        with self.assertRaises(RuntimeError):
            common_hooks.dispatch_runtime_event(
                runtime=state,
                runner=runner,
                event=BackendObservableEvent.EXECUTE_MODEL_POST,
            )
        common_hooks.dispatch_runtime_event(
            runtime=state,
            runner=runner,
            event=BackendObservableEvent.EXECUTE_MODEL_PRE,
        )
        executor.launch(4, collective=key, payload=object())

        common_hooks.dispatch_runtime_event(
            runtime=state,
            runner=runner,
            event=BackendObservableEvent.EXECUTE_MODEL_POST,
        )
        executor.launch(4, collective=second.key, payload=object())

        common_hooks.dispatch_runtime_event(
            runtime=state,
            runner=runner,
            event=BackendObservableEvent.SAMPLE_TOKENS_POST,
        )
        executor.finish_step(4)


class Phase4CrossRankAgreementAuditRedTest(unittest.TestCase):
    def test_multi_rank_agreement_requires_exact_rank_projections(self) -> None:
        plan = collective_runtime.compile_distributed_flow_plan(
            plan_epoch=8,
            topology=_two_rank_topology(),
            collective_requests=(_request(),),
        )
        agreements = tuple(
            collective_runtime.distributed_plan_agreement_desc(plan, rank)
            for rank in range(2)
        )
        collective_runtime.validate_distributed_plan_agreement(plan, agreements)

        with self.assertRaises(RuntimeError):
            collective_runtime.validate_distributed_plan_agreement(
                plan,
                (agreements[0], replace(agreements[1], plan_signature="f" * 64)),
            )
        with self.assertRaises(ValueError):
            collective_runtime.DistributedRuntimeBinding(
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
        model = torch.nn.Module()
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


class Phase4VllmLifecycleAuditRedTest(unittest.TestCase):
    """vLLM 0.27 drivers must enter PREPARE after acquiring the step."""

    def _prepare_binding(self) -> object:
        return _binding(
            self,
            request=_request(name="prepare", window=CollectiveWindow.PREPARE),
        )

    def test_mrv1_dispatches_prepare_lifecycle_after_distributed_step_prepare(self) -> None:
        binding = self._prepare_binding()
        state = ResidualRuntimeState()
        runner = SimpleNamespace(
            model=None,
            input_batch=SimpleNamespace(req_ids=(), num_reqs=0),
        )
        view = SimpleNamespace(
            logits_indices=None,
            spec_decode_metadata=None,
            num_scheduled_tokens_np=(),
        )
        events: list[tuple[str, int]] = []
        order: list[str] = []
        real_dispatch = common_hooks.dispatch_runtime_event

        def dispatch(**kwargs: object) -> int:
            event = kwargs["event"]
            self.assertIsInstance(event, BackendObservableEvent)
            event_name = event.value
            events.append((event_name, state.distributed_prepared_step_id))
            order.append(event_name)
            result = real_dispatch(**kwargs)
            if event_name == "prepare_inputs.pre":
                step = state.distributed_prepared_step_id
                self.assertGreaterEqual(step, 0)
                key = state.distributed_plan_store.active(step).local_schedules[0].collectives[0]
                residual_runtime.launch_workgraph_collective(
                    step,
                    collective=key,
                    payload=object(),
                )
            return result

        def run_prepare_post_producers(**_kwargs: object) -> None:
            order.append("prepare_producers")

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(v027_mrv1._common_hooks, "dispatch_runtime_event", side_effect=dispatch),
            mock.patch.object(v027_mrv1.active_targets, "runtime_has_active_targets", return_value=True),
            mock.patch.object(v027_mrv1._hooks, "setup_runtime_hooks_if_active"),
            mock.patch.object(
                v027_mrv1._hooks,
                "run_prepare_inputs_post_producers",
                side_effect=run_prepare_post_producers,
            ),
            mock.patch.object(v027_mrv1._sampler_patch, "maybe_prepare_sampler_decode_step"),
            mock.patch.object(v027_mrv1._ADAPTER, "unpack_prepare_inputs_output", return_value=view),
        ):
            residual_runtime.configure_workgraph_distributed(binding)
            residual_runtime.publish_workgraph_distributed(plan_epoch=1)
            v027_mrv1._wrapped_prepare_inputs(
                core=residual_runtime,
                runner=runner,
                scheduler_output=object(),
                num_scheduled_tokens=object(),
                orig=mock.Mock(return_value=(None, None)),
            )

        self.assertEqual(
            events,
            [
                ("prepare_inputs.pre", 1),
                ("prepare_inputs.post", 1),
                ("execute_model.pre", 1),
            ],
        )
        self.assertEqual(
            order,
            [
                "prepare_inputs.pre",
                "prepare_producers",
                "prepare_inputs.post",
                "execute_model.pre",
            ],
        )
        self.assertEqual(binding.adapter.calls[0][0].key.name, "prepare")

    def test_mrv2_dispatches_prepare_lifecycle_before_execute_model_compute_window(self) -> None:
        binding = self._prepare_binding()
        state = ResidualRuntimeState()
        runner = SimpleNamespace(model=None)
        events: list[tuple[str, int]] = []
        order: list[str] = []
        real_dispatch = common_hooks.dispatch_runtime_event

        def dispatch(**kwargs: object) -> int:
            event = kwargs["event"]
            self.assertIsInstance(event, BackendObservableEvent)
            event_name = event.value
            events.append((event_name, state.distributed_prepared_step_id))
            order.append(event_name)
            result = real_dispatch(**kwargs)
            if event_name == "prepare_inputs.pre":
                step = state.distributed_prepared_step_id
                self.assertGreaterEqual(step, 0)
                key = state.distributed_plan_store.active(step).local_schedules[0].collectives[0]
                residual_runtime.launch_workgraph_collective(
                    step,
                    collective=key,
                    payload=object(),
                )
            return result

        input_batch = SimpleNamespace(
            req_ids=(),
            num_reqs=0,
            prefill_len_np=(),
            num_computed_tokens_np=(),
            num_scheduled_tokens=(),
            logits_indices=None,
        )

        def run_prepare_post_producers(**kwargs: object) -> None:
            order.append("prepare_producers")
            localize_decode = kwargs["localize_decode"]
            self.assertTrue(callable(localize_decode))

        def execute_with_prepared_inputs(_runner: object) -> str:
            result = v027_mrv2._wrapped_prepare_inputs(
                core=residual_runtime,
                runner=runner,
                args=(),
                kwargs={},
                orig=mock.Mock(return_value=input_batch),
            )
            self.assertIs(result, input_batch)
            order.append("forward")
            return "executed"

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(v027_mrv2._common_hooks, "dispatch_runtime_event", side_effect=dispatch),
            mock.patch.object(v027_mrv2.active_targets, "runtime_has_active_targets", return_value=True),
            mock.patch.object(v027_mrv2._hooks, "setup_runtime_hooks_if_active"),
            mock.patch.object(
                v027_mrv2._hooks,
                "run_prepare_inputs_post_producers",
                side_effect=run_prepare_post_producers,
            ),
            mock.patch.object(v027_mrv2._hooks, "run_execute_model_post_producers"),
            mock.patch.object(v027_mrv2._model_region, "prepare_model_region_step"),
            mock.patch.object(v027_mrv2._sampler_patch, "maybe_prepare_sampler_decode_step"),
            mock.patch.object(v027_mrv2, "_decode_localization_from_mrv2", return_value=()),
        ):
            residual_runtime.configure_workgraph_distributed(binding)
            residual_runtime.publish_workgraph_distributed(plan_epoch=1)
            v027_mrv2._wrapped_execute_model(
                core=residual_runtime,
                runner=runner,
                args=(),
                kwargs={},
                orig=execute_with_prepared_inputs,
            )

        self.assertEqual(
            events,
            [
                ("prepare_inputs.pre", 1),
                ("prepare_inputs.post", 1),
                ("execute_model.pre", 1),
                ("execute_model.post", 1),
            ],
        )
        self.assertEqual(
            order,
            [
                "prepare_inputs.pre",
                "prepare_producers",
                "prepare_inputs.post",
                "execute_model.pre",
                "forward",
                "execute_model.post",
            ],
        )
        self.assertEqual(binding.adapter.calls[0][0].key.name, "prepare")


class Phase4LegacyGenericLifecycleAuditRedTest(unittest.TestCase):
    def test_legacy_generic_execute_emits_terminal_sample_event_for_step_end(self) -> None:
        """Legacy execute_model owns sampling and must close the distributed step."""

        binding = _binding(
            self,
            request=_request(name="step-end", window=CollectiveWindow.STEP_END),
        )
        state = ResidualRuntimeState()
        runner = SimpleNamespace(model=SimpleNamespace())
        events: list[str] = []

        def orig_execute(_runner: object, *_args: object, **_kwargs: object) -> str:
            # Legacy MRv1 performs sampling before execute_model returns.
            events.append("legacy_sampling_complete")
            return "executed"

        core = SimpleNamespace(
            RUNTIME=state,
            MODEL_HOOK_FLAG="_legacy_hooks_installed",
            _ORIG_EXECUTE_MODEL=orig_execute,
            _runner_uses_compilation_or_cudagraph=lambda _runner: False,
        )
        real_dispatch = common_hooks.dispatch_runtime_event

        def dispatch(**kwargs: object) -> int:
            event = kwargs["event"]
            self.assertIsInstance(event, BackendObservableEvent)
            event_name = event.value
            events.append(event_name)
            result = real_dispatch(**kwargs)
            if event_name == "sample_tokens.post":
                step = state.distributed_prepared_step_id
                self.assertEqual(step, 1)
                key = state.distributed_plan_store.active(step).local_schedules[0].collectives[0]
                residual_runtime.launch_workgraph_collective(
                    step,
                    collective=key,
                    payload=object(),
                )
            return result

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(
                port_runtime_hooks._common_hooks,
                "dispatch_runtime_event",
                side_effect=dispatch,
            ),
            mock.patch.object(port_runtime_hooks, "setup_runtime_hooks_if_active"),
        ):
            residual_runtime.configure_workgraph_distributed(binding)
            residual_runtime.publish_workgraph_distributed(plan_epoch=1)
            residual_runtime.prepare_workgraph_step(1)
            result = port_runtime_hooks.wrapped_execute_model(
                core=core,
                runner=runner,
                args=(),
                kwargs={},
            )

        self.assertEqual(result, "executed")
        self.assertEqual(
            events,
            [
                "execute_model.pre",
                "legacy_sampling_complete",
                "execute_model.post",
                "sample_tokens.post",
            ],
        )
        self.assertEqual(binding.adapter.calls[0][0].key.name, "step-end")

    def test_split_execute_path_leaves_terminal_event_to_sample_wrapper(self) -> None:
        state = ResidualRuntimeState()
        runner = SimpleNamespace(model=SimpleNamespace())
        core = SimpleNamespace(
            RUNTIME=state,
            MODEL_HOOK_FLAG="_split_hooks_installed",
            _ORIG_EXECUTE_MODEL=lambda _runner: "executed",
            _runner_uses_compilation_or_cudagraph=lambda _runner: False,
        )
        events: list[str] = []
        with (
            mock.patch.object(
                port_runtime_hooks.active_targets,
                "runtime_has_active_targets",
                return_value=True,
            ),
            mock.patch.object(port_runtime_hooks, "setup_runtime_hooks_if_active"),
            mock.patch.object(
                port_runtime_hooks._common_hooks,
                "dispatch_runtime_event",
                side_effect=lambda **kwargs: (
                    events.append(cast(BackendObservableEvent, kwargs["event"]).value)
                    or 0
                ),
            ),
        ):
            result = port_runtime_hooks.wrapped_execute_model(
                core=core,
                runner=runner,
                args=(),
                kwargs={},
                split_execute_and_sample=True,
            )

        self.assertEqual(result, "executed")
        self.assertEqual(events, ["execute_model.pre", "execute_model.post"])


class Phase4DistributedTerminalCloseAuditRedTest(unittest.TestCase):
    def _assert_terminal_closes_step(
        self,
        wrapper: object,
        *,
        sample_result: object = "sampled",
    ) -> None:
        binding = _binding(
            self,
            request=_request(name="terminal-close", window=CollectiveWindow.STEP_END),
        )
        state = ResidualRuntimeState()
        state.event_step_id = 1
        runner = SimpleNamespace(model=SimpleNamespace())
        set_step_session(
            runner,
            StepSession(
                step_id=1,
                backend_id="test",
                batch_phase=_core.SchedulerBatchPhase.Decode,
            ),
        )
        real_dispatch = common_hooks.dispatch_runtime_event

        def dispatch(**kwargs: object) -> int:
            result = real_dispatch(**kwargs)
            if kwargs["event"] is BackendObservableEvent.SAMPLE_TOKENS_POST:
                step = state.distributed_prepared_step_id
                self.assertEqual(step, 1)
                key = state.distributed_plan_store.active(step).local_schedules[0].collectives[0]
                residual_runtime.launch_workgraph_collective(
                    step,
                    collective=key,
                    payload=object(),
                )
            return result

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(common_hooks, "dispatch_runtime_event", side_effect=dispatch),
        ):
            residual_runtime.configure_workgraph_distributed(binding)
            residual_runtime.publish_workgraph_distributed(plan_epoch=1)
            residual_runtime.prepare_workgraph_step(1)
            result = wrapper(
                core=residual_runtime,
                runner=runner,
                grammar_output=object(),
                orig=mock.Mock(return_value=sample_result),
            )

            self.assertEqual(result, sample_result)
            self.assertEqual(state.distributed_prepared_step_id, -1)
            self.assertNotIn(1, state.distributed_plan_store._active)
            store = state.distributed_plan_store
            self.assertTrue(store.closeable)
            self.assertEqual(store.poll(), 0)

    def test_legacy_generic_terminal_closes_prepared_step_and_retires_fence(self) -> None:
        binding = _binding(
            self,
            request=_request(name="terminal-close", window=CollectiveWindow.STEP_END),
        )
        state = ResidualRuntimeState()
        state.event_step_id = 1
        runner = SimpleNamespace(model=SimpleNamespace())
        real_dispatch = common_hooks.dispatch_runtime_event

        def orig_execute(_runner: object, *_args: object, **_kwargs: object) -> str:
            return "executed"

        def dispatch(**kwargs: object) -> int:
            result = real_dispatch(**kwargs)
            if kwargs["event"] is BackendObservableEvent.SAMPLE_TOKENS_POST:
                step = state.distributed_prepared_step_id
                self.assertEqual(step, 1)
                key = state.distributed_plan_store.active(step).local_schedules[0].collectives[0]
                residual_runtime.launch_workgraph_collective(
                    step,
                    collective=key,
                    payload=object(),
                )
            return result

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(residual_runtime, "_ORIG_EXECUTE_MODEL", orig_execute),
            mock.patch.object(port_runtime_hooks._common_hooks, "dispatch_runtime_event", side_effect=dispatch),
            mock.patch.object(port_runtime_hooks, "setup_runtime_hooks_if_active"),
        ):
            residual_runtime.configure_workgraph_distributed(binding)
            residual_runtime.publish_workgraph_distributed(plan_epoch=1)
            residual_runtime.prepare_workgraph_step(1)
            result = port_runtime_hooks.wrapped_execute_model(
                core=residual_runtime,
                runner=runner,
                args=(),
                kwargs={},
            )

            self.assertEqual(result, "executed")
            self.assertEqual(state.distributed_prepared_step_id, -1)
            self.assertNotIn(1, state.distributed_plan_store._active)
            store = state.distributed_plan_store
            self.assertTrue(store.closeable)
            self.assertEqual(store.poll(), 0)

    def test_mrv1_sample_terminal_closes_prepared_step_and_retires_fence(self) -> None:
        self._assert_terminal_closes_step(v027_mrv1._wrapped_sample_tokens)

    def test_mrv2_sample_terminal_closes_prepared_step_and_retires_fence(self) -> None:
        self._assert_terminal_closes_step(v027_mrv2._wrapped_sample_tokens)


if __name__ == "__main__":
    unittest.main()
