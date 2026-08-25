from __future__ import annotations

import unittest
from unittest import mock

from tllm.workload import (
    AdamWSpec,
    AffineAccess,
    ApproximationPolicy,
    Before,
    Between,
    Domain,
    ExternImplementation,
    ExecutorRegistry,
    Index,
    RegionArtifactRegistry,
    RegionCallableArtifact,
    State,
    StateScope,
    declare_workload,
    hidden,
    read,
    request_key,
    result,
    sink,
    step_id,
    task,
    temporary,
    transition,
    workload,
    lower_program,
    lower_schedule_graph,
    materialize_region_artifacts,
    provided,
)
from tllm.schedule import (
    CompileOptions,
    MainWorkloadTimeline,
    TimelineAnchor,
    WorkloadCompilerCoordinator,
    PlanEpochAllocator,
)


class WorkloadDslUnitTest(unittest.TestCase):
    def test_eager_region_artifact_owns_grad_context(self) -> None:
        import torch

        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        value = torch.tensor([3.0])
        artifact = RegionCallableArtifact(
            region_id="region-grad",
            artifact_key="0" * 64,
            input_keys=("value",),
            output_keys=("loss",),
            callable=lambda tensor: ((tensor * parameter).sum(),),
            compiled=True,
        )

        with torch.inference_mode():
            outputs = artifact.execute({"value": value})

        loss = outputs["loss"]
        self.assertIsInstance(loss, torch.Tensor)
        assert isinstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertEqual(parameter.grad, torch.tensor([3.0]))

    def test_class_discovery_builds_one_stable_task_ir(self) -> None:
        token = Index("token")

        @workload(id="esamp.test")
        class ESamp:
            distiller = State(
                scope=StateScope.REQUEST,
                key=request_key("request_id", "sample_idx"),
                parameters="model",
                optimizer=AdamWSpec(lr=1e-3),
            )

            @task(
                inputs={"source": hidden(2, "post", "decode")},
                outputs={"prediction": sink("guidance"), "loss": temporary()},
                effects=(read(distiller),),
                timing=Between("model.execute", "sampler.pre_draw"),
                domain=Domain("tokens", indices=(token,), bounds=(("token", 0, "T"),)),
            )
            def predict_and_loss(self, source: object) -> tuple[object, object]:
                return source, source

            @task(
                inputs={"loss": result("predict_and_loss", "loss")},
                effects=(transition(distiller),),
                timing=Before("next.prepare"),
            )
            def update(self, loss: object) -> None:
                _ = loss

        declaration = declare_workload(ESamp())
        ir = declaration.ir

        self.assertEqual(ir.workload_id, "esamp.test")
        self.assertEqual(tuple(task.name for task in ir.tasks), ("predict_and_loss", "update"))
        self.assertEqual(ir.schedule, ("predict_and_loss", "update"))
        self.assertEqual(ir.states[0].name, "distiller")
        self.assertEqual(ir.states[0].request_key, ("request_id", "sample_idx"))
        self.assertEqual(ir.tasks[0].domain.name, "tokens")
        self.assertEqual(ir.edges[0].kind.value, "data")
        self.assertEqual(len(ir.semantic_key), 64)
        self.assertEqual(ir.semantic_key, declare_workload(ESamp()).ir.semantic_key)
        self.assertNotIn("<object at", ir.to_json())
        graph = lower_schedule_graph((ir,))
        self.assertEqual(graph.schema_version, 2)
        self.assertEqual(graph.states[0].resource, "distiller")
        self.assertEqual(graph.states[0].scope, "request")
        self.assertEqual(graph.states[0].request_key, ("request_id", "sample_idx"))
        self.assertEqual(graph.tasks[0].effects[0].resource, "distiller")
        self.assertEqual(
            graph.tasks[0].implementation_capabilities,
            (("torch.export",),),
        )

    def test_jacobi_is_explicit_and_changes_semantic_key(self) -> None:
        def build(policy: ApproximationPolicy) -> str:
            @workload(id="recur", approximation=policy)
            class Recurrence:
                @task(timing=Before("step.end"))
                def advance(self) -> None:
                    return None

            return declare_workload(Recurrence()).ir.semantic_key

        exact = build(ApproximationPolicy.EXACT)
        jacobi = build(ApproximationPolicy.JACOBI)
        self.assertNotEqual(exact, jacobi)

    def test_schedule_lowering_preserves_affine_domain_and_accesses(self) -> None:
        token = Index("token")
        layer = Index("layer")

        @workload(id="affine.recurrence")
        class Recurrence:
            @task(
                timing=Before("step.end"),
                domain=Domain(
                    "token_layer",
                    indices=(token, layer),
                    bounds=(("token", 0, "T"), ("layer", 0, "L")),
                    parameters=("T", "L"),
                ),
                accesses=(
                    AffineAccess("state", (token - 1, layer), "read"),
                    AffineAccess("state", (token, layer), "write"),
                ),
            )
            def advance(self) -> None:
                return None

        graph = lower_schedule_graph((declare_workload(Recurrence()).ir,))

        self.assertEqual(graph.tasks[0].domain.indices, ("token", "layer"))
        self.assertEqual(graph.tasks[0].domain.parameters, ("T", "L"))
        self.assertEqual(
            tuple(access.mode for access in graph.tasks[0].accesses),
            ("read", "write"),
        )
        self.assertEqual(
            graph.tasks[0].accesses[0].indices[0].coefficients,
            (("token", 1),),
        )
        self.assertEqual(graph.tasks[0].accesses[0].indices[0].constant, -1)

    def test_request_state_requires_a_key_and_transition_requires_optimizer(self) -> None:
        with self.assertRaisesRegex(ValueError, "REQUEST.*request_key"):
            State(scope=StateScope.REQUEST, parameters="model")

        state = State(
            scope=StateScope.REQUEST,
            key=request_key("request_id"),
            parameters="model",
        )
        with self.assertRaisesRegex(ValueError, "transition.*optimizer"):
            transition(state)

    def test_typed_extern_registry_is_separate_from_serialized_ir(self) -> None:
        implementation = ExternImplementation(
            namespace="test",
            name="write_loss",
            version="2",
            input_types=("Tensor",),
            output_types=("None",),
            capabilities=("eager",),
        )
        registry = ExecutorRegistry()

        def executor(value: object) -> None:
            _ = value

        registry.register(implementation, executor)
        self.assertIs(registry.resolve(implementation.key), executor)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(implementation, executor)

    def test_legacy_program_adapter_preserves_data_and_state_effects(self) -> None:
        import torch

        from tllm.program import StateSpec
        from tllm.workgraph.frontend import (
            StateScope as LegacyStateScope,
            extern,
            flow,
            hidden as legacy_hidden,
            local,
            request_sample_key,
            result as legacy_result,
        )

        class Legacy:
            @flow(
                inputs={"source": legacy_hidden(2, "post", "decode")},
                outputs={"loss": local()},
                state="distiller",
            )
            def predict(self, source: object) -> object:
                return source

            @extern(
                inputs={"loss": legacy_result("predict", "loss")},
                state="distiller",
            )
            def update(self, loss: object) -> None:
                _ = loss

        ir = lower_program(
            Legacy(),
            states={
                "distiller": StateSpec(
                    scope=LegacyStateScope.REQUEST,
                    key=request_sample_key(),
                    optimizer=torch.optim.AdamW,
                )
            },
            workload_id="legacy.esamp",
        )
        graph = lower_schedule_graph((ir,))

        self.assertEqual(ir.schedule, ("predict", "update"))
        self.assertEqual(tuple(effect.kind.value for task in ir.tasks for effect in task.effects), ("read", "transition"))
        self.assertEqual(tuple(edge.kind.value for edge in ir.edges), ("data", "effect"))
        self.assertEqual(tuple(task.task_id for task in graph.tasks), ("predict", "update"))

    def test_builtin_dummy_is_native_while_legacy_esamp_keeps_its_adapter(self) -> None:
        from tllm.consumers.dummy import DummyWorkload
        from tllm.consumers.esamp import ESampConsumerConfig, build_esamp_workload
        from tllm.runtime.runtime import TllmRuntime

        dummy = DummyWorkload()
        esamp = build_esamp_workload(ESampConsumerConfig(implementation="legacy"))
        dummy_runtime = TllmRuntime(workloads=(dummy,))
        esamp_runtime = TllmRuntime(workloads=(esamp,))

        self.assertFalse(hasattr(dummy, "legacy_consumer"))
        self.assertEqual(dummy_runtime.consumers, ())
        self.assertIsNotNone(dummy_runtime.workload_execution_bindings)
        self.assertEqual(len(esamp_runtime.consumers), 1)
        self.assertEqual(declare_workload(dummy).ir.workload_id, "dummy")
        self.assertEqual(
            tuple(item.name for item in declare_workload(dummy).ir.tasks[0].inputs),
            ("hidden_rows", "step_id"),
        )
        self.assertEqual(
            tuple(item.kind.value for item in declare_workload(dummy).ir.tasks[0].inputs),
            ("hidden", "step_id"),
        )
        esamp_ir = declare_workload(esamp).ir
        self.assertEqual(esamp_ir.workload_id, "esamp")
        self.assertEqual(esamp_ir.schedule, ("predict_and_loss", "update"))

    def test_step_id_is_a_typed_input_not_a_reserved_provider_name(self) -> None:
        @workload(id="typed.step")
        class TypedStep:
            @task(
                inputs={"builtin_step": step_id(), "user_value": provided("step_id")},
                timing=Before("step.end"),
            )
            def run(self, builtin_step: int, user_value: object) -> None:
                _ = builtin_step, user_value

        graph = lower_schedule_graph((declare_workload(TypedStep()).ir,))

        self.assertEqual(
            tuple(input_.kind.value for input_ in graph.tasks[0].inputs),
            ("step_id", "provided"),
        )

    def test_instance_scoped_coordinator_caches_immutable_compilation(self) -> None:
        @workload(id="coordinator.test")
        class Auxiliary:
            @task(timing=Between("model.execute", "step.end"))
            def run(self) -> None:
                return None

        declaration = declare_workload(Auxiliary())
        timeline = MainWorkloadTimeline(
            backend_id="vllm",
            backend_version="0.27.1",
            runner_family="mrv2",
            graph_mode="eager",
            anchors=(
                TimelineAnchor("model.execute", 0),
                TimelineAnchor("step.end", 1),
            ),
            regions=(),
            static_headroom_bytes=64,
        )
        coordinator = WorkloadCompilerCoordinator(
            (declaration,), options=CompileOptions(memory_budget_bytes=64)
        )

        first = coordinator.compile(timeline)
        second = coordinator.compile(timeline)

        self.assertIs(first, second)
        self.assertEqual(first.cache_key, second.cache_key)
        self.assertIsNotNone(first.physical)
        assert first.physical is not None
        self.assertEqual(first.decision.signature, first.physical.schedule_signature)
        self.assertEqual(first.physical.plan_epoch, 1)

        from tllm.runtime.runtime import TllmRuntime

        left = TllmRuntime(
            workloads=(Auxiliary(),), workload_memory_budget_bytes=64
        )
        right = TllmRuntime(
            workloads=(Auxiliary(),), workload_memory_budget_bytes=64
        )
        left_compilation = left.compile_workloads(timeline)

        self.assertIs(left.workload_compilation, left_compilation)
        self.assertIsNone(right.workload_compilation)

    def test_plan_epoch_allocator_survives_coordinator_rebuild(self) -> None:
        @workload(id="coordinator.epoch")
        class Auxiliary:
            @task(timing=Between("model.execute", "step.end"))
            def run(self) -> None:
                return None

        declaration = declare_workload(Auxiliary())
        timeline = MainWorkloadTimeline(
            backend_id="vllm",
            backend_version="0.27.1",
            runner_family="mrv2",
            graph_mode="eager",
            anchors=(
                TimelineAnchor("model.execute", 0),
                TimelineAnchor("step.end", 1),
            ),
            regions=(),
            static_headroom_bytes=64,
        )
        allocator = PlanEpochAllocator()
        first = WorkloadCompilerCoordinator(
            (declaration,), options=CompileOptions(memory_budget_bytes=64),
            epoch_allocator=allocator,
        ).compile(timeline)
        rebuilt = WorkloadCompilerCoordinator(
            (declaration,), options=CompileOptions(memory_budget_bytes=64),
            epoch_allocator=allocator,
        ).compile(timeline)

        assert first.physical is not None and rebuilt.physical is not None
        self.assertEqual((first.physical.plan_epoch, rebuilt.physical.plan_epoch), (1, 2))

    def test_region_artifact_compiles_once_and_keeps_intermediate_results_local(self) -> None:
        import torch

        @workload(id="region.callable")
        class Fused:
            @task(
                inputs={"value": provided("value")},
                outputs={"hidden_value": temporary()},
                timing=Between("model.execute", "step.end"),
            )
            def produce(self, value: object) -> object:
                assert isinstance(value, torch.Tensor)
                return value + 1

            @task(
                inputs={"hidden_value": result("produce", "hidden_value")},
                outputs={"result_value": sink("result")},
                timing=Between("model.execute", "step.end"),
            )
            def consume(self, hidden_value: object) -> object:
                assert isinstance(hidden_value, torch.Tensor)
                return hidden_value * 2

        declaration = declare_workload(Fused())
        timeline = MainWorkloadTimeline(
            backend_id="vllm",
            backend_version="0.27.1",
            runner_family="mrv2",
            graph_mode="eager",
            anchors=(
                TimelineAnchor("model.execute", 0),
                TimelineAnchor("step.end", 1),
            ),
            regions=(),
            static_headroom_bytes=64,
        )
        options = CompileOptions(
            memory_budget_bytes=64,
            compiler_fingerprint="1" * 64,
        )
        compilation = WorkloadCompilerCoordinator(
            (declaration,), options=options
        ).compile(timeline)

        real_compile = torch.compile

        def compile_region(function: object, **kwargs: object) -> object:
            return real_compile(function, backend="eager", **kwargs)  # type: ignore[arg-type]

        with mock.patch("torch.compile", side_effect=compile_region) as compile_mock:
            artifacts = materialize_region_artifacts(
                (declaration,), compilation.decision, compilation.options
            )

        self.assertEqual(len(artifacts), 1)
        artifact = artifacts[0]
        self.assertTrue(artifact.compiled)
        assert artifact.compile_command is not None
        self.assertEqual(artifact.compile_command.adapter.value, "torch_compile")
        self.assertTrue(artifact.compile_command.fullgraph)
        self.assertFalse(artifact.compile_command.dynamic)
        self.assertEqual(artifact.compile_command.autograd_policy.value, "enable_grad")
        self.assertEqual(artifact.compile_command.fallback_policy.value, "reject")
        self.assertEqual(compile_mock.call_count, 1)
        self.assertEqual(
            tuple((binding.kind, binding.source) for binding in artifact.input_bindings),
            (("provided", "value"),),
        )
        self.assertEqual(
            tuple((binding.kind, binding.target) for binding in artifact.output_bindings),
            (("sink", "result"),),
        )
        output = artifact.execute(
            {artifact.input_bindings[0].binding_id: torch.tensor([3.0])}
        )
        torch.testing.assert_close(
            output[artifact.output_bindings[0].binding_id], torch.tensor([8.0])
        )
        self.assertNotIn(
            "hidden_value",
            tuple(binding.target for binding in artifact.output_bindings),
        )

    def test_extern_region_artifact_is_singleton_and_not_torch_compiled(self) -> None:
        @workload(id="region.extern")
        class External:
            @task(
                inputs={"value": provided("value")},
                outputs={"result_value": sink("result")},
                timing=Between("model.execute", "step.end"),
                implementation=ExternImplementation(
                    namespace="test",
                    name="identity",
                    input_types=("object",),
                    output_types=("object",),
                ),
            )
            def identity(self, value: object) -> object:
                return value

        declaration = declare_workload(External())
        timeline = MainWorkloadTimeline(
            backend_id="vllm",
            backend_version="0.27.1",
            runner_family="mrv2",
            graph_mode="eager",
            anchors=(
                TimelineAnchor("model.execute", 0),
                TimelineAnchor("step.end", 1),
            ),
            regions=(),
            static_headroom_bytes=64,
        )
        options = CompileOptions(
            memory_budget_bytes=64,
            compiler_fingerprint="1" * 64,
        )
        compilation = WorkloadCompilerCoordinator(
            (declaration,), options=options
        ).compile(timeline)

        with mock.patch("torch.compile") as compile_mock:
            artifact = materialize_region_artifacts(
                (declaration,), compilation.decision, compilation.options
            )[0]

        self.assertFalse(artifact.compiled)
        self.assertIsNone(artifact.compile_command)
        compile_mock.assert_not_called()
        self.assertEqual(
            artifact.execute({artifact.input_bindings[0].binding_id: 7}),
            {artifact.output_bindings[0].binding_id: 7},
        )
        with self.assertRaisesRegex(RuntimeError, "artifact coverage mismatch"):
            RegionArtifactRegistry(compilation.decision, ())


if __name__ == "__main__":
    unittest.main()
