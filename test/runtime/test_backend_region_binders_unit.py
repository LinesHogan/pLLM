from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.runtime.backend_region_binders import (
    BackendRegionBinderSlot,
    BackendRegionStageContext,
    BackendRegionStagerRegistry,
)
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_execution import (
    WorkloadExecutionBindings,
    build_native_workload_execution,
)
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    WorkloadWorkerSpec,
    reconstruct_worker_bindings,
    static_backend_binding_manifest_wire,
)
from tllm.runtime.workload_bindings import BackendRuntimeBinding, WorkloadRuntimeBinding
from tllm.schedule import (
    BackendObservableEvent,
    CompileOptions,
    WorkloadCompilerCoordinator,
)
from tllm.workload import Between, ExternImplementation, task, workload, declare_workload


def _timeline():
    return export_main_workload_timeline(
        BackendCapabilities(
            backend_id="backend-region-binder-test",
            runner_family="mrv2",
            sampler_stages=frozenset(),
            native_min_p=False,
            tuple_sampler_return=False,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.EAGER_ONLY,
        ),
        graph_mode="eager",
        static_headroom_bytes=1 << 20,
    )


class BackendRegionBinderSlotTest(unittest.TestCase):
    def test_slot_owned_backend_stager_is_identity_bound_and_fails_closed(self) -> None:
        implementation_id = "test.backend:slot-owned@1"
        observed: list[BackendRegionStageContext] = []
        slot = BackendRegionBinderSlot(implementation_id, backend_stager=observed.append)
        context = BackendRegionStageContext(
            implementation_id=implementation_id,
            backend_event="prepare_inputs.post",
            region_id="region-0000-slot-owned",
            slot=slot,
            runtime=object(),
            runner=object(),
            step_id=2,
            num_scheduled_tokens=None,
            input_batch=None,
        )
        slot.stage_from_backend(context)
        self.assertEqual(observed, [context])

        unbound = BackendRegionBinderSlot(implementation_id)
        unbound_context = BackendRegionStageContext(
            implementation_id=implementation_id,
            backend_event="prepare_inputs.post",
            region_id="region-0000-slot-owned",
            slot=unbound,
            runtime=object(),
            runner=object(),
            step_id=2,
            num_scheduled_tokens=None,
            input_batch=None,
        )
        with self.assertRaisesRegex(RuntimeError, "no worker-local backend stager"):
            unbound.stage_from_backend(unbound_context)

    def test_backend_stager_registry_is_identity_bound_and_fails_closed(self) -> None:
        implementation_id = "test.backend:extern@1"
        slot = BackendRegionBinderSlot(implementation_id)
        observed: list[BackendRegionStageContext] = []
        registry = BackendRegionStagerRegistry(
            {implementation_id: observed.append}
        )
        context = BackendRegionStageContext(
            implementation_id=implementation_id,
            backend_event="prepare_inputs.post",
            region_id="region-0000-test",
            slot=slot,
            runtime=object(),
            runner=object(),
            step_id=3,
            num_scheduled_tokens=None,
            input_batch=None,
        )

        registry.stage(context)
        self.assertEqual(observed, [context])
        with self.assertRaisesRegex(RuntimeError, "no worker-local backend stager"):
            BackendRegionStagerRegistry({}).stage(context)
        with self.assertRaisesRegex(ValueError, "slot does not match"):
            BackendRegionStageContext(
                implementation_id=implementation_id,
                backend_event="prepare_inputs.post",
                region_id="region-0000-test",
                slot=BackendRegionBinderSlot("test.backend:other@1"),
                runtime=object(),
                runner=object(),
                step_id=3,
                num_scheduled_tokens=None,
                input_batch=None,
            )

    def test_stage_retrieves_provided_inputs_and_invokes_exactly_once(self) -> None:
        slot = BackendRegionBinderSlot(
            "test.backend:extern@1",
            provided_inputs={"request": lambda step_id: {"step": step_id}},
        )
        calls: list[tuple[object, object, int]] = []

        slot.stage(
            {"kind": "opaque-plan"},
            lambda plan, inputs, step_id: calls.append(
                (plan, inputs["request"], step_id)
            ),
            7,
        )
        self.assertEqual(slot.plan_for_step(7), {"kind": "opaque-plan"})
        self.assertEqual(slot.provided_input("request", 7), {"step": 7})
        self.assertEqual(slot.invoke(7), None)
        self.assertEqual(calls, [({"kind": "opaque-plan"}, {"step": 7}, 7)])
        with self.assertRaisesRegex(RuntimeError, "already completed"):
            slot.invoke(7)
        with self.assertRaisesRegex(RuntimeError, "already completed"):
            slot.stage(object(), lambda *_args: None, 7)

    def test_missing_stale_and_duplicate_stages_fail_closed(self) -> None:
        slot = BackendRegionBinderSlot("test.backend:extern@1")
        with self.assertRaisesRegex(RuntimeError, "missing staged backend Region"):
            slot.clear(3)
        slot.stage(object(), lambda *_args: None, 3)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            slot.stage(object(), lambda *_args: None, 3)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            slot.stage(object(), lambda *_args: None, 4)
        with self.assertRaisesRegex(RuntimeError, "staged for step 3"):
            slot.provided_inputs(4)
        slot.clear(3)
        with self.assertRaisesRegex(RuntimeError, "already completed"):
            slot.stage(object(), lambda *_args: None, 3)

    def test_failed_enclosing_step_aborts_an_uninvoked_stage(self) -> None:
        slot = BackendRegionBinderSlot("test.backend:extern@1")
        slot.stage(object(), lambda *_args: None, 4)

        self.assertTrue(slot.abort(4))
        self.assertFalse(slot.abort(4))
        slot.stage(object(), lambda *_args: None, 5)
        with self.assertRaisesRegex(RuntimeError, "staged for step 5"):
            slot.abort(6)

    def test_signed_extern_regions_require_exact_local_binder_coverage(self) -> None:
        implementation = ExternImplementation(
            namespace="test.backend",
            name="opaque",
            input_types=(),
            output_types=("None",),
            capabilities=("backend/region_binder",),
        )

        @workload(id="backend-region-binder.coverage")
        class ExternWorkload:
            @task(
                timing=Between("model.execute", "step.end"),
                implementation=implementation,
            )
            def run(self) -> None:
                return None

        declaration = declare_workload(ExternWorkload())
        timeline = _timeline()
        compilation = WorkloadCompilerCoordinator(
            (declaration,), options=CompileOptions(memory_budget_bytes=1 << 20)
        ).compile(
            replace(
                timeline,
                capabilities=tuple(
                    sorted((*timeline.capabilities, "backend/region_binder"))
                ),
            )
        )
        implementation_id = compilation.decision.execution_regions[0].implementation_ids[0]
        slot = BackendRegionBinderSlot(implementation_id)
        execution = build_native_workload_execution(
            compilation,
            runtime=ResidualRuntimeState(),
            consumers=(),
            bindings=WorkloadExecutionBindings(
                declarations=(declaration,),
                backend_region_binders={implementation_id: slot},
            ),
        )
        assert execution is not None
        self.assertIs(execution.backend_region_binders[implementation_id], slot)
        signed_stages = tuple(
            stage
            for advance in compilation.decision.runtime_advance_plan
            if advance.backend_event is BackendObservableEvent.SAMPLE_TOKENS_POST
            for stage in advance.backend_region_binder_stages
        )
        self.assertEqual(
            execution.backend_region_binder_stages(
                BackendObservableEvent.SAMPLE_TOKENS_POST
            ),
            signed_stages,
        )
        self.assertEqual(
            tuple(stage.implementation_id for stage in signed_stages),
            (implementation_id,),
        )
        execution.close()

        with self.assertRaisesRegex(RuntimeError, "missing=.*test.backend:opaque@1"):
            build_native_workload_execution(
                compilation,
                runtime=ResidualRuntimeState(),
                consumers=(),
                bindings=WorkloadExecutionBindings(declarations=(declaration,)),
            )

    def test_duplicate_signed_binder_identity_at_one_event_is_rejected_by_rust(self) -> None:
        implementation = ExternImplementation(
            namespace="test.backend",
            name="duplicate",
            input_types=(),
            output_types=("None",),
            capabilities=("backend/region_binder",),
        )

        @workload(id="backend-region-binder.duplicate")
        class DuplicateExternWorkload:
            @task(
                timing=Between("model.execute", "step.end"),
                implementation=implementation,
            )
            def first(self) -> None:
                return None

            @task(
                timing=Between("model.execute", "step.end"),
                implementation=implementation,
            )
            def second(self) -> None:
                return None

        timeline = _timeline()
        with self.assertRaisesRegex(RuntimeError, "binder implementation.*duplicated"):
            WorkloadCompilerCoordinator(
                (declare_workload(DuplicateExternWorkload()),),
                options=CompileOptions(memory_budget_bytes=1 << 20),
            ).compile(
                replace(
                    timeline,
                    capabilities=tuple(
                        sorted((*timeline.capabilities, "backend/region_binder"))
                    ),
                )
            )

    def test_worker_reconstruction_rejects_duplicate_binder_keys(self) -> None:
        @workload(id="backend-region-binder.reconstruct.alpha")
        class Alpha(NativeWorkload):
            def __init__(self) -> None:
                self.resources = WorkloadRuntimeBinding(owner=self)

            def native_workload_resources(self) -> WorkloadRuntimeBinding:
                return self.resources

            @task(timing=Between("model.execute", "step.end"))
            def run(self) -> None:
                return None

        @workload(id="backend-region-binder.reconstruct.beta")
        class Beta(NativeWorkload):
            def __init__(self) -> None:
                self.resources = WorkloadRuntimeBinding(owner=self)

            def native_workload_resources(self) -> WorkloadRuntimeBinding:
                return self.resources

            @task(timing=Between("model.execute", "step.end"))
            def run(self) -> None:
                return None

        alpha = Alpha()
        beta = Beta()
        shared_key = "test.backend:extern@1"
        alpha.resources = WorkloadRuntimeBinding(
            owner=alpha,
            backend=BackendRuntimeBinding(
                binders={shared_key: BackendRegionBinderSlot(shared_key)}
            ),
        )
        beta.resources = WorkloadRuntimeBinding(
            owner=beta,
            backend=BackendRuntimeBinding(
                binders={shared_key: BackendRegionBinderSlot(shared_key)}
            ),
        )
        specs = (
            WorkloadWorkerSpec("test.backend:alpha@1", {}),
            WorkloadWorkerSpec("test.backend:beta@1", {}),
        )
        compilation = WorkloadCompilerCoordinator(
            (declare_workload(alpha), declare_workload(beta)),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                static_binding_manifest=static_backend_binding_manifest_wire(
                    (
                        ("backend-region-binder.reconstruct.alpha", alpha.resources, specs[0]),
                        ("backend-region-binder.reconstruct.beta", beta.resources, specs[1]),
                    )
                ),
            ),
        ).compile(_timeline())
        with mock.patch(
            "tllm.runtime.workload_plugins._resolve_factory",
            side_effect=(lambda _config: alpha.resources, lambda _config: beta.resources),
        ):
            with self.assertRaisesRegex(RuntimeError, "duplicate backend Region binder"):
                reconstruct_worker_bindings(compilation, specs)


if __name__ == "__main__":
    unittest.main()
