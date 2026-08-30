from __future__ import annotations

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from tllm.contracts.hidden_locator import HiddenLocatorCapability
from tllm.schedule import (
    BackendObservableEvent,
    CompileOptions,
    ExecutionRegion,
    HiddenPhase,
    HiddenLayerKind,
    HiddenSite,
    MainWorkloadRegion,
    MainWorkloadTimeline,
    RegionCaptureMode,
    RegionFallbackPolicy,
    RegionImplementationKind,
    RegionStreamClass,
    ResidualResourceBinding,
    ResourceReservation,
    ScheduleAffineAccess,
    ScheduleAffineExpr,
    ScheduleDomain,
    ScheduleEdge,
    ScheduleEdgeKind,
    ScheduleEffect,
    ScheduleEffectKind,
    ScheduleGraph,
    ScheduleInput,
    ScheduleInputKind,
    ScheduleOutput,
    ScheduleOutputKind,
    ScheduleTask,
    ScheduleState,
    ScheduleValue,
    TaskTiming,
    TimelineAnchor,
    compile_schedule,
    install_workload_compilation,
    lower_physical_flow_plan,
    validate_schedule,
    workload_compilation_cache_key,
)
from tllm.workgraph.physical import CapturePolicy, FallbackPolicy, StreamClass


def _timeline() -> MainWorkloadTimeline:
    return MainWorkloadTimeline(
        backend_id="vllm",
        backend_version="0.27.1",
        runner_family="mrv2",
        graph_mode="eager",
        anchors=(
            TimelineAnchor("model.execute", 0),
            TimelineAnchor("sampler.pre_draw", 1),
            TimelineAnchor("step.end", 2),
            TimelineAnchor("next.prepare", 3),
        ),
        regions=(
            MainWorkloadRegion(
                "model",
                start="model.execute",
                end="sampler.pre_draw",
                reservation=ResourceReservation(memory_bytes=32),
            ),
            MainWorkloadRegion(
                "sampler",
                start="sampler.pre_draw",
                end="step.end",
                reservation=ResourceReservation(memory_bytes=8),
            ),
        ),
        static_headroom_bytes=64,
        capabilities=tuple(
            capability.value
            for capability in (
                HiddenLocatorCapability.BLOCK_OUTPUT_DECODE,
                HiddenLocatorCapability.POST_DECODE,
            )
        ),
    )


def _indexed_hidden_manifest(
    layer: int,
    site: HiddenSite | str,
) -> tuple[ResidualResourceBinding, ...]:
    return (
        ResidualResourceBinding(
            layer_index=layer,
            site=HiddenSite(site),
            phase=HiddenPhase.DECODE,
            module_path=f"model.layers.{layer}",
        ),
    )


class ScheduleContractUnitTest(unittest.TestCase):
    def test_rust_selects_implementation_candidate_from_timeline(self) -> None:
        timing = TaskTiming(after="model.execute", before="step.end")
        graph = ScheduleGraph(
            workload_key="candidate-selection",
            tasks=(
                ScheduleTask(
                    "select",
                    timing=timing,
                    implementation_ids=("backend:extern@1", "python:compiled@1"),
                    implementation_kinds=("extern", "captured"),
                    implementation_capabilities=(
                        ("backend/alternate",),
                        ("backend/default", "torch.export"),
                    ),
                    required_capabilities=(),
                    capture_safe=False,
                ),
            ),
            schema_version=2,
        )

        # The first candidate is unavailable, so Rust selects the declared
        # compiled fallback and signs only that implementation into the plan.
        compiled_timeline = replace(
            _timeline(),
            capabilities=tuple(sorted((*_timeline().capabilities, "backend/default"))),
        )
        compiled = compile_schedule(
            graph,
            compiled_timeline,
            CompileOptions(memory_budget_bytes=64),
        )
        region = compiled.execution_regions[0]
        self.assertEqual(region.implementation_ids, ("python:compiled@1",))
        self.assertIs(region.implementation_kind, RegionImplementationKind.COMPILED)
        self.assertTrue(region.compile_safe)
        self.assertEqual(region.required_capabilities, ("backend/default",))
        self.assertEqual(compiled.placements[0].implementation_id, "python:compiled@1")

        # Once both are legal, declaration order is the deterministic tie
        # breaker and Rust selects the extern implementation instead.
        extern_timeline = replace(
            compiled_timeline,
            capabilities=tuple(
                sorted((*compiled_timeline.capabilities, "backend/alternate"))
            ),
        )
        extern = compile_schedule(
            graph,
            extern_timeline,
            CompileOptions(memory_budget_bytes=64),
        )
        extern_region = extern.execution_regions[0]
        self.assertEqual(extern_region.implementation_ids, ("backend:extern@1",))
        self.assertIs(extern_region.implementation_kind, RegionImplementationKind.EXTERN)
        self.assertFalse(extern_region.compile_safe)
        self.assertEqual(extern_region.required_capabilities, ("backend/alternate",))

        unavailable = replace(_timeline(), capabilities=())
        with self.assertRaisesRegex(RuntimeError, "no implementation candidate"):
            compile_schedule(graph, unavailable, CompileOptions(memory_budget_bytes=64))

    def test_graph_v2_rust_cross_checks_python_derived_task_semantics(self) -> None:
        """V2 keeps transition fields only as Rust-verified wire evidence."""

        timing = TaskTiming(after="model.execute", before="step.end")
        graph = ScheduleGraph(
            workload_key="derived-semantics",
            tasks=(
                ScheduleTask(
                    "captured",
                    timing=timing,
                    implementation_capabilities=(("backend/test", "torch.export"),),
                    required_capabilities=("backend/test",),
                    capture_safe=True,
                ),
                ScheduleTask(
                    "read",
                    timing=timing,
                    implementation_capabilities=(("torch.export",),),
                    capture_safe=False,
                    effects=(ScheduleEffect(ScheduleEffectKind.STATE_READ, "state"),),
                ),
                ScheduleTask(
                    "transition",
                    timing=timing,
                    implementation_capabilities=(("torch.export",),),
                    capture_safe=False,
                    effects=(
                        ScheduleEffect(ScheduleEffectKind.STATE_TRANSITION, "state"),
                    ),
                ),
                ScheduleTask(
                    "after",
                    timing=timing,
                    implementation_capabilities=(("torch.export",),),
                    capture_safe=False,
                    effects=(ScheduleEffect(ScheduleEffectKind.STATE_READ, "state"),),
                ),
            ),
            edges=(
                ScheduleEdge("read", "transition", ScheduleEdgeKind.EFFECT),
                ScheduleEdge("transition", "after", ScheduleEdgeKind.EFFECT),
            ),
            states=(ScheduleState("state", "global"),),
            schema_version=2,
        )
        timeline = replace(
            _timeline(),
            capabilities=tuple(sorted((*_timeline().capabilities, "backend/test"))),
        )
        options = CompileOptions(memory_budget_bytes=64)

        decision = compile_schedule(graph, timeline, options)
        validate_schedule(graph, timeline, decision, options)

        bad_capabilities = replace(
            graph,
            tasks=(
                replace(graph.tasks[0], required_capabilities=()),
                *graph.tasks[1:],
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "derived required_capabilities"):
            compile_schedule(bad_capabilities, timeline, options)

        bad_capture = replace(
            graph,
            tasks=(
                replace(graph.tasks[0], capture_safe=False),
                *graph.tasks[1:],
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "derived capture_safe"):
            compile_schedule(bad_capture, timeline, options)

        bad_effect_edges = replace(graph, edges=graph.edges[:-1])
        with self.assertRaisesRegex(RuntimeError, "derived effect edges"):
            compile_schedule(bad_effect_edges, timeline, options)

    def _wire_fixture(
        self,
    ) -> tuple[ScheduleGraph, MainWorkloadTimeline, CompileOptions, object]:
        graph = ScheduleGraph(
            workload_key="raw-wire-validation",
            tasks=(
                ScheduleTask(
                    "task",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    inputs=(
                        ScheduleInput(
                            "step",
                            ScheduleInputKind.STEP_ID,
                            "step_id",
                        ),
                    ),
                ),
            ),
        )
        timeline = _timeline()
        from tllm.schedule.compiler import effective_compile_options

        options = effective_compile_options(CompileOptions(memory_budget_bytes=64))
        return graph, timeline, options, compile_schedule(graph, timeline, options)

    def test_install_rejects_unknown_raw_wire_fields_before_python_parsing(self) -> None:
        graph, timeline, options, decision = self._wire_fixture()
        payloads = {
            "graph_json": graph.to_json(),
            "timeline_json": timeline.to_json(),
            "options_json": options.to_json(),
            "decision_json": decision.to_json(),
        }
        cache_key = workload_compilation_cache_key(graph, timeline, options)

        for field, payload in payloads.items():
            with self.subTest(field=field):
                mutated = json.loads(payload)
                mutated["unexpected"] = True
                raw = dict(payloads)
                raw[field] = json.dumps(mutated, sort_keys=True, separators=(",", ":"))
                with self.assertRaisesRegex(ValueError, "unknown field"):
                    install_workload_compilation(
                        cache_key=cache_key,
                        plan_epoch=1,
                        **raw,
                    )

    def test_install_rejects_duplicate_raw_wire_fields(self) -> None:
        graph, timeline, options, decision = self._wire_fixture()
        graph_json = graph.to_json()
        duplicate_graph_json = '{"schema_version":1,"schema_version":1,' + graph_json[1:]

        with self.assertRaisesRegex(ValueError, "duplicate field"):
            install_workload_compilation(
                cache_key=workload_compilation_cache_key(graph, timeline, options),
                graph_json=duplicate_graph_json,
                timeline_json=timeline.to_json(),
                options_json=options.to_json(),
                decision_json=decision.to_json(),
                plan_epoch=1,
            )

    def test_install_raw_wire_rejects_cache_and_epoch_and_keeps_valid_plan(self) -> None:
        graph, timeline, options, decision = self._wire_fixture()
        wire = {
            "cache_key": workload_compilation_cache_key(graph, timeline, options),
            "graph_json": graph.to_json(),
            "timeline_json": timeline.to_json(),
            "options_json": options.to_json(),
            "decision_json": decision.to_json(),
        }

        with self.assertRaisesRegex(ValueError, "cache key"):
            install_workload_compilation(
                **{**wire, "cache_key": "0" * 64}, plan_epoch=1
            )
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            install_workload_compilation(
                **{**wire, "cache_key": "A" * 64}, plan_epoch=1
            )
        with self.assertRaisesRegex(ValueError, "plan_epoch"):
            install_workload_compilation(**wire, plan_epoch=0)
        with self.assertRaisesRegex(ValueError, "plan_epoch"):
            install_workload_compilation(**wire, plan_epoch=True)

        invalid_signature = json.loads(wire["decision_json"])
        invalid_signature["signature"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "signature is invalid"):
            install_workload_compilation(
                **{
                    **wire,
                    "decision_json": json.dumps(
                        invalid_signature, sort_keys=True, separators=(",", ":")
                    ),
                },
                plan_epoch=1,
            )

        installed = install_workload_compilation(**wire, plan_epoch=7)
        self.assertEqual(installed.cache_key, wire["cache_key"])
        self.assertEqual(installed.decision.signature, decision.signature)
        self.assertIsNotNone(installed.physical)
        assert installed.physical is not None
        self.assertEqual(installed.physical.plan_epoch, 7)

    def test_schema_v2_rejects_unknown_fields_before_install(self) -> None:
        graph = ScheduleGraph(
            workload_key="strict-region-unknown-fields",
            tasks=(
                ScheduleTask(
                    "task",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    inputs=(
                        ScheduleInput(
                            "step",
                            ScheduleInputKind.STEP_ID,
                            "step_id",
                        ),
                    ),
                ),
            ),
        )
        timeline = _timeline()
        from tllm.schedule.compiler import effective_compile_options

        options = effective_compile_options(CompileOptions(memory_budget_bytes=64))
        decision = compile_schedule(graph, timeline, options)

        unknown_top_level = json.loads(decision.to_json())
        unknown_top_level["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields.*unexpected"):
            type(decision).from_json(json.dumps(unknown_top_level))

        unknown_nested = json.loads(decision.to_json())
        unknown_nested["execution_regions"][0]["input_bindings"][0][
            "unexpected"
        ] = True
        with self.assertRaisesRegex(ValueError, "unknown fields.*unexpected"):
            type(decision).from_json(json.dumps(unknown_nested))
        with self.assertRaisesRegex(ValueError, "unknown field[s]?.*unexpected"):
            install_workload_compilation(
                cache_key=workload_compilation_cache_key(graph, timeline, options),
                graph_json=graph.to_json(),
                timeline_json=timeline.to_json(),
                options_json=options.to_json(),
                decision_json=json.dumps(unknown_nested),
            )

    def test_rust_validator_rejects_unknown_nested_decision_records(self) -> None:
        graph = ScheduleGraph(
            workload_key="strict-decision-records",
            tasks=(
                ScheduleTask(
                    "produce",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    temporary_bytes=1,
                    outputs=(
                        ScheduleOutput("hidden", ScheduleOutputKind.TEMPORARY, "hidden"),
                    ),
                ),
                ScheduleTask(
                    "consume",
                    timing=TaskTiming(after="step.end", before="next.prepare"),
                    temporary_bytes=1,
                    inputs=(
                        ScheduleInput("hidden", ScheduleInputKind.RESULT, "produce", "hidden"),
                    ),
                ),
            ),
            edges=(ScheduleEdge("produce", "consume", ScheduleEdgeKind.DATA),),
            values=(
                ScheduleValue(
                    "hidden",
                    producer="produce",
                    producer_output="hidden",
                    consumers=("consume",),
                    size_bytes=16,
                ),
            ),
        )
        timeline = _timeline()
        from tllm.schedule.compiler import effective_compile_options

        options = effective_compile_options(CompileOptions(memory_budget_bytes=64))
        decision = compile_schedule(graph, timeline, options)
        from tllm import _core

        def placement(payload: dict[str, object]) -> dict[str, object]:
            return payload["placements"][0]  # type: ignore[index, return-value]

        def tree(payload: dict[str, object]) -> dict[str, object]:
            return payload["schedule_tree"]  # type: ignore[return-value]

        def affine_region(payload: dict[str, object]) -> dict[str, object]:
            affine_regions = payload["schedule_tree"]["affine_regions"]  # type: ignore[index]
            affine_regions.append(  # type: ignore[union-attr]
                {
                    "tasks": ["produce"],
                    "schedule": "{ produce[] }",
                    "dependences": "{ }",
                }
            )
            return affine_regions[-1]  # type: ignore[index, return-value]

        def memory(payload: dict[str, object]) -> dict[str, object]:
            return payload["memory_plan"]  # type: ignore[return-value]

        def allocation(payload: dict[str, object]) -> dict[str, object]:
            return payload["memory_plan"]["allocations"][0]  # type: ignore[index, return-value]

        def fence(payload: dict[str, object]) -> dict[str, object]:
            return payload["fences"][0]  # type: ignore[index, return-value]

        def cost(payload: dict[str, object]) -> dict[str, object]:
            return payload["cost"]  # type: ignore[return-value]

        def materialization(payload: dict[str, object]) -> dict[str, object]:
            return payload["materializations"][0]  # type: ignore[index, return-value]

        for label, record in (
            ("placement", placement),
            ("tree", tree),
            ("affine_region", affine_region),
            ("memory", memory),
            ("allocation", allocation),
            ("fence", fence),
            ("cost", cost),
            ("materialization", materialization),
        ):
            with self.subTest(record=label):
                payload = json.loads(decision.to_json())
                record(payload)["unexpected"] = True
                payload_json = json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                )
                with self.assertRaisesRegex(ValueError, "unknown fields.*unexpected"):
                    type(decision).from_json(payload_json)
                with self.assertRaisesRegex(ValueError, "unknown field.*unexpected"):
                    _core.validate_schedule(
                        graph.to_json(),
                        timeline.to_json(),
                        options.to_json(),
                        payload_json,
                    )

    def test_schema_v2_requires_complete_region_wire_metadata(self) -> None:
        graph = ScheduleGraph(
            workload_key="strict-region-wire",
            tasks=(
                ScheduleTask(
                    "task",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    inputs=(
                        ScheduleInput(
                            "step",
                            ScheduleInputKind.STEP_ID,
                            "step_id",
                        ),
                    ),
                ),
            ),
        )
        decision = compile_schedule(
            graph,
            _timeline(),
            CompileOptions(memory_budget_bytes=64),
        )

        missing_region_field = json.loads(decision.to_json())
        del missing_region_field["execution_regions"][0]["effects"]
        with self.assertRaisesRegex(
            ValueError, r"ExecutionRegion\[0\].*effects"
        ):
            type(decision).from_json(json.dumps(missing_region_field))

        missing_capture_mode = json.loads(decision.to_json())
        del missing_capture_mode["execution_regions"][0]["capture_mode"]
        with self.assertRaisesRegex(
            ValueError, r"ExecutionRegion\[0\].*capture_mode"
        ):
            type(decision).from_json(json.dumps(missing_capture_mode))

        missing_runtime_plan = json.loads(decision.to_json())
        del missing_runtime_plan["runtime_advance_plan"]
        with self.assertRaisesRegex(ValueError, "runtime_advance_plan"):
            type(decision).from_json(json.dumps(missing_runtime_plan))

        missing_binder_stages = json.loads(decision.to_json())
        missing_binder_stages["runtime_advance_plan"].append(
            {
                "backend_event": "prepare_inputs.post",
                "anchor_id": "model.execute",
                "region_ids": [],
                "backend_region_binder_stages": [],
            }
        )
        del missing_binder_stages["runtime_advance_plan"][0][
            "backend_region_binder_stages"
        ]
        with self.assertRaisesRegex(
            ValueError, r"RuntimeAdvancePlan\[0\].*backend_region_binder_stages"
        ):
            type(decision).from_json(json.dumps(missing_binder_stages))

        missing_completion_anchor = json.loads(decision.to_json())
        del missing_completion_anchor["completion_anchor"]
        with self.assertRaisesRegex(ValueError, "completion_anchor"):
            type(decision).from_json(json.dumps(missing_completion_anchor))

        missing_stream = json.loads(decision.to_json())
        del missing_stream["execution_regions"][0]["stream"]
        with self.assertRaisesRegex(ValueError, r"ExecutionRegion\[0\].*stream"):
            type(decision).from_json(json.dumps(missing_stream))

        missing_fallback = json.loads(decision.to_json())
        del missing_fallback["execution_regions"][0]["fallback_policy"]
        with self.assertRaisesRegex(
            ValueError, r"ExecutionRegion\[0\].*fallback_policy"
        ):
            type(decision).from_json(json.dumps(missing_fallback))

        missing_compile_command = json.loads(decision.to_json())
        del missing_compile_command["execution_regions"][0]["compile_command"]
        with self.assertRaisesRegex(
            ValueError, r"ExecutionRegion\[0\].*compile_command"
        ):
            type(decision).from_json(json.dumps(missing_compile_command))

        side_stream = json.loads(decision.to_json())
        side_stream["execution_regions"][0]["stream"] = "side_forward"
        parsed_side = type(decision).from_json(json.dumps(side_stream))
        self.assertEqual(parsed_side.execution_regions[0].stream.value, "side_forward")

        unknown_stream = json.loads(decision.to_json())
        unknown_stream["execution_regions"][0]["stream"] = "side_train"
        with self.assertRaisesRegex(ValueError, "stream"):
            type(decision).from_json(json.dumps(unknown_stream))

        unknown_fallback = json.loads(decision.to_json())
        unknown_fallback["execution_regions"][0]["fallback_policy"] = "eager"
        with self.assertRaisesRegex(ValueError, "fallback_policy"):
            type(decision).from_json(json.dumps(unknown_fallback))

        missing_nested_field = json.loads(decision.to_json())
        del missing_nested_field["execution_regions"][0]["input_bindings"][0][
            "layer_kind"
        ]
        with self.assertRaisesRegex(
            ValueError, r"input_bindings\[0\].*layer_kind"
        ):
            type(decision).from_json(json.dumps(missing_nested_field))

        missing_version = json.loads(decision.to_json())
        del missing_version["schema_version"]
        with self.assertRaisesRegex(ValueError, "explicit schema_version"):
            type(decision).from_json(json.dumps(missing_version))

        stale_v2 = json.loads(decision.to_json())
        stale_v2["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema v2 is stale.*recompile"):
            type(decision).from_json(json.dumps(stale_v2))

    def test_schema_v2_rejects_malformed_region_json_shapes(self) -> None:
        graph = ScheduleGraph(
            workload_key="strict-region-json-shapes",
            tasks=(
                ScheduleTask(
                    "task",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    inputs=(
                        ScheduleInput(
                            "hidden",
                            ScheduleInputKind.HIDDEN,
                            "hidden:0:post:decode",
                            layer="0",
                            layer_kind=HiddenLayerKind.INDEX,
                            site="post",
                            phase="decode",
                        ),
                    ),
                ),
            ),
        )
        decision = compile_schedule(
            graph,
            _timeline(),
            CompileOptions(
                memory_budget_bytes=64,
                residual_resource_manifest=_indexed_hidden_manifest(0, "post"),
            ),
        )

        malformed_array = json.loads(decision.to_json())
        malformed_array["execution_regions"][0]["input_bindings"] = {}
        with self.assertRaisesRegex(TypeError, "input_bindings must be a JSON array"):
            type(decision).from_json(json.dumps(malformed_array))

        malformed_boolean = json.loads(decision.to_json())
        malformed_boolean["execution_regions"][0]["capture_safe"] = "true"
        with self.assertRaisesRegex(TypeError, "capture_safe must be a JSON boolean"):
            type(decision).from_json(json.dumps(malformed_boolean))

        malformed_integer = json.loads(decision.to_json())
        malformed_integer["execution_regions"][0]["input_bindings"][0][
            "layer_index"
        ] = True
        with self.assertRaisesRegex(
            TypeError, "layer_index must be a JSON integer or null"
        ):
            type(decision).from_json(json.dumps(malformed_integer))

        unknown_site = json.loads(decision.to_json())
        unknown_site["execution_regions"][0]["input_bindings"][0]["site"] = "middle"
        with self.assertRaisesRegex(ValueError, "HiddenSite|site"):
            type(decision).from_json(json.dumps(unknown_site))

        unknown_phase = json.loads(decision.to_json())
        unknown_phase["execution_regions"][0]["input_bindings"][0]["phase"] = "train"
        with self.assertRaisesRegex(ValueError, "HiddenPhase|phase"):
            type(decision).from_json(json.dumps(unknown_phase))

    def test_hidden_locator_requires_a_signed_timeline_capability(self) -> None:
        graph = ScheduleGraph(
            workload_key="hidden-capability",
            tasks=(
                ScheduleTask(
                    "hidden",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    inputs=(
                        ScheduleInput(
                            "value",
                            ScheduleInputKind.HIDDEN,
                            "hidden:0:block_output:decode",
                            layer="0",
                            layer_kind=HiddenLayerKind.INDEX,
                            site="block_output",
                            phase="decode",
                        ),
                    ),
                ),
            ),
        )
        options = CompileOptions(
            memory_budget_bytes=64,
            residual_resource_manifest=_indexed_hidden_manifest(0, "block_output"),
        )
        unsupported = replace(_timeline(), capabilities=())
        with self.assertRaisesRegex(RuntimeError, "hidden/block_output/decode"):
            compile_schedule(graph, unsupported, options)

        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver

        vllm_timeline = V027MRv2Driver().export_main_workload_timeline(
            graph_mode="eager", static_headroom_bytes=64
        )
        self.assertIn(
            HiddenLocatorCapability.BLOCK_OUTPUT_DECODE.value,
            vllm_timeline.capabilities,
        )
        decision = compile_schedule(graph, vllm_timeline, options)
        binding = decision.execution_regions[0].input_bindings[0]
        self.assertEqual(binding.site.value, "block_output")
        self.assertEqual(binding.phase.value, "decode")

    def test_negative_hidden_layer_is_preserved_as_typed_relative_index(self) -> None:
        graph = ScheduleGraph(
            workload_key="relative-hidden",
            tasks=(
                ScheduleTask(
                    "read-last",
                    timing=TaskTiming(
                        after="model.execute", before="sampler.pre_draw"
                    ),
                    inputs=(
                        ScheduleInput(
                            "hidden",
                            ScheduleInputKind.HIDDEN,
                            "hidden:-1:block_output:decode",
                            layer="-1",
                            layer_kind=HiddenLayerKind.INDEX,
                            site="block_output",
                            phase="decode",
                        ),
                    ),
                ),
            ),
        )

        decision = compile_schedule(
            graph,
            _timeline(),
            CompileOptions(
                memory_budget_bytes=64,
                residual_resource_manifest=_indexed_hidden_manifest(
                    -1, "block_output"
                ),
            ),
        )

        binding = decision.execution_regions[0].input_bindings[0]
        self.assertIs(binding.layer_kind, HiddenLayerKind.INDEX)
        self.assertEqual(binding.layer_index, -1)
        self.assertEqual(
            type(decision).from_json(decision.to_json())
            .execution_regions[0]
            .input_bindings[0]
            .layer_index,
            -1,
        )

    def test_compiler_emits_deterministic_maximal_execution_region(self) -> None:
        graph = ScheduleGraph(
            workload_key="abc",
            tasks=(
                ScheduleTask(
                    "predict",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    temporary_bytes=8,
                ),
                ScheduleTask(
                    "update",
                    timing=TaskTiming(after="sampler.pre_draw", before="next.prepare"),
                    temporary_bytes=8,
                ),
            ),
            edges=(ScheduleEdge("predict", "update", ScheduleEdgeKind.DATA),),
        )
        options = CompileOptions(memory_budget_bytes=64)

        first = compile_schedule(graph, _timeline(), options)
        second = compile_schedule(graph, _timeline(), options)

        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.schema_version, 3)
        self.assertEqual(first.execution_order, ("predict", "update"))
        self.assertEqual(len(first.execution_regions), 1)
        self.assertEqual(first.execution_regions[0].tasks, ("predict", "update"))
        self.assertEqual(first.execution_regions[0].dependencies, ())
        self.assertEqual(
            tuple((placement.task_id, placement.after_anchor) for placement in first.placements),
            (("predict", "sampler.pre_draw"), ("update", "sampler.pre_draw")),
        )
        self.assertLessEqual(first.memory_plan.peak_bytes, 64)
        self.assertEqual(first.fences, ())
        self.assertEqual(first.cost.materialization_bytes, 0)
        self.assertEqual(first.cost.launch_group_count, 1)
        self.assertEqual(first.cost.cross_region_fence_count, 0)
        self.assertTrue(first.signature)
        validate_schedule(graph, _timeline(), first, options)
        self.assertEqual(ScheduleGraph.from_json(graph.to_json()), graph)
        self.assertEqual(
            MainWorkloadTimeline.from_json(_timeline().to_json()), _timeline()
        )
        self.assertEqual(CompileOptions.from_json(options.to_json()), options)
        self.assertEqual(type(first).from_json(first.to_json()), first)

        physical = lower_physical_flow_plan(
            graph, _timeline(), first, options, plan_epoch=3
        )
        self.assertEqual(physical.plan_epoch, 3)
        self.assertEqual(physical.schedule_signature, first.signature)
        self.assertEqual(len(physical.nodes), 1)
        self.assertEqual(
            tuple(placement.after_anchor for placement in physical.anchor_placements),
            ("sampler.pre_draw",),
        )
        self.assertEqual(physical.nodes[0].fences, ())
        self.assertEqual(physical.nodes[0].fallback_policy.value, "reject")

    def test_cross_region_value_is_materialized_with_one_fence(self) -> None:
        graph = ScheduleGraph(
            workload_key="split-value",
            tasks=(
                ScheduleTask(
                    "produce",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    outputs=(
                        ScheduleOutput("hidden", ScheduleOutputKind.TEMPORARY, "hidden"),
                    ),
                ),
                ScheduleTask(
                    "consume",
                    timing=TaskTiming(after="step.end", before="next.prepare"),
                    inputs=(
                        ScheduleInput("hidden", ScheduleInputKind.RESULT, "produce", "hidden"),
                    ),
                ),
            ),
            edges=(ScheduleEdge("produce", "consume", ScheduleEdgeKind.DATA),),
            values=(
                ScheduleValue(
                    "hidden",
                    producer="produce",
                    producer_output="hidden",
                    consumers=("consume",),
                    size_bytes=16,
                ),
            ),
        )

        decision = compile_schedule(
            graph, _timeline(), CompileOptions(memory_budget_bytes=64)
        )

        self.assertEqual(len(decision.execution_regions), 2)
        source, target = decision.execution_regions
        self.assertEqual(target.dependencies, (source.region_id,))
        self.assertEqual(len(decision.materializations), 1)
        materialization = decision.materializations[0]
        self.assertEqual(materialization.value_id, "hidden")
        self.assertEqual(materialization.producer_region, source.region_id)
        self.assertEqual(materialization.consumer_region, target.region_id)
        self.assertEqual(materialization.size_bytes, 16)
        self.assertEqual(materialization.live_start, source.placement_anchor)
        self.assertEqual(materialization.live_end, target.placement_anchor)
        self.assertEqual(len(decision.fences), 1)
        self.assertEqual(
            (decision.fences[0].source, decision.fences[0].target),
            (source.region_id, target.region_id),
        )
        self.assertEqual(decision.cost.materialization_bytes, 16)
        self.assertEqual(decision.cost.launch_group_count, 2)
        self.assertEqual(decision.cost.cross_region_fence_count, 1)
        validate_schedule(graph, _timeline(), decision, CompileOptions(memory_budget_bytes=64))

        physical = lower_physical_flow_plan(
            graph,
            _timeline(),
            decision,
            CompileOptions(memory_budget_bytes=64),
        )
        self.assertEqual(len(physical.nodes), 2)
        self.assertEqual(len(physical.nodes[1].fences), 1)

    def test_materialization_identity_keeps_value_versions_in_one_consumer_region(self) -> None:
        graph = ScheduleGraph(
            workload_key="versioned-cross-region-values",
            tasks=(
                ScheduleTask(
                    "produce",
                    timing=TaskTiming(after="model.execute", before="sampler.pre_draw"),
                    outputs=(
                        ScheduleOutput("v1", ScheduleOutputKind.TEMPORARY, "v1"),
                        ScheduleOutput("v2", ScheduleOutputKind.TEMPORARY, "v2"),
                    ),
                ),
                ScheduleTask(
                    "consume_v1",
                    timing=TaskTiming(after="sampler.pre_draw", before="step.end"),
                    inputs=(
                        ScheduleInput("value", ScheduleInputKind.RESULT, "produce", "v1"),
                    ),
                ),
                ScheduleTask(
                    "consume_v2",
                    timing=TaskTiming(after="sampler.pre_draw", before="step.end"),
                    inputs=(
                        ScheduleInput("value", ScheduleInputKind.RESULT, "produce", "v2"),
                    ),
                ),
            ),
            edges=(
                ScheduleEdge("produce", "consume_v1", ScheduleEdgeKind.DATA),
                ScheduleEdge("produce", "consume_v2", ScheduleEdgeKind.DATA),
            ),
            values=(
                ScheduleValue(
                    "hidden",
                    version=1,
                    producer="produce",
                    producer_output="v1",
                    consumers=("consume_v1",),
                    size_bytes=16,
                ),
                ScheduleValue(
                    "hidden",
                    version=2,
                    producer="produce",
                    producer_output="v2",
                    consumers=("consume_v2",),
                    size_bytes=16,
                ),
            ),
        )

        decision = compile_schedule(
            graph, _timeline(), CompileOptions(memory_budget_bytes=64)
        )
        self.assertEqual(len(decision.materializations), 2)
        self.assertEqual(
            {item.version for item in decision.materializations}, {1, 2}
        )
        self.assertEqual(
            len({item.materialization_id for item in decision.materializations}), 2
        )
        self.assertEqual(
            len({item.consumer_region for item in decision.materializations}), 1
        )
        target = next(
            region
            for region in decision.execution_regions
            if {"consume_v1", "consume_v2"}.issubset(region.tasks)
        )
        self.assertEqual(
            {binding.binding_id for binding in target.input_bindings},
            {item.materialization_id for item in decision.materializations},
        )
        validate_schedule(graph, _timeline(), decision, CompileOptions(memory_budget_bytes=64))

    def test_ordered_effect_edge_stays_inside_compiled_region_and_extern_splits(self) -> None:
        common_timing = TaskTiming(after="model.execute", before="step.end")
        effect_graph = ScheduleGraph(
            workload_key="effect",
            tasks=(
                ScheduleTask("read", timing=common_timing),
                ScheduleTask("publish", timing=common_timing),
            ),
            edges=(ScheduleEdge("read", "publish", ScheduleEdgeKind.EFFECT),),
        )
        effect_decision = compile_schedule(
            effect_graph, _timeline(), CompileOptions(memory_budget_bytes=64)
        )
        self.assertEqual(
            tuple(region.tasks for region in effect_decision.execution_regions),
            (("read", "publish"),),
        )

        extern_graph = ScheduleGraph(
            workload_key="extern",
            tasks=(
                ScheduleTask("compute", timing=common_timing),
                ScheduleTask(
                    "io",
                    timing=common_timing,
                    implementation_ids=("typed:extern@1",),
                    implementation_kinds=("extern",),
                ),
            ),
            edges=(ScheduleEdge("compute", "io", ScheduleEdgeKind.DATA),),
        )
        extern_decision = compile_schedule(
            extern_graph, _timeline(), CompileOptions(memory_budget_bytes=64)
        )
        self.assertEqual(
            tuple(region.tasks for region in extern_decision.execution_regions),
            (("compute",), ("io",)),
        )
        self.assertEqual(
            tuple(region.implementation_kind for region in extern_decision.execution_regions),
            ("compiled", "extern"),
        )
        self.assertIs(
            extern_decision.execution_regions[0].implementation_kind,
            RegionImplementationKind.COMPILED,
        )
        self.assertIs(
            extern_decision.execution_regions[1].implementation_kind,
            RegionImplementationKind.EXTERN,
        )

    def test_specialization_and_compiler_environment_key_region_artifacts(self) -> None:
        graph = ScheduleGraph(
            workload_key="specialized",
            tasks=(
                ScheduleTask(
                    "compute",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    capture_safe=True,
                ),
            ),
        )
        base = CompileOptions(
            memory_budget_bytes=64,
            shape_specialization=(("batch", "8"),),
        )
        changed_shape = replace(
            base, shape_specialization=(("batch", "16"),)
        )
        changed_environment = replace(base, compiler_fingerprint="2" * 64)
        eager_policy = replace(base, compile_policy="eager")

        first_two = tuple(
            compile_schedule(graph, _timeline(), options)
            for options in (base, changed_shape)
        )
        with mock.patch(
            "tllm.schedule.compiler.compiler_environment_fingerprint",
            return_value="2" * 64,
        ):
            changed_environment_decision = compile_schedule(
                graph, _timeline(), changed_environment
            )
        decisions = (*first_two, changed_environment_decision)

        self.assertEqual(len({decision.signature for decision in decisions}), 3)
        self.assertEqual(
            len(
                {
                    decision.execution_regions[0].artifact_key
                    for decision in decisions
                }
            ),
            3,
        )
        for decision, options in zip(
            decisions, (base, changed_shape, changed_environment), strict=True
        ):
            command = decision.execution_regions[0].compile_command
            assert command is not None
            self.assertEqual(command.adapter.value, "torch_compile")
            self.assertTrue(command.fullgraph)
            self.assertFalse(command.dynamic)
            self.assertEqual(command.shape_specialization, options.shape_specialization)
            self.assertEqual(command.autograd_policy.value, "enable_grad")
            self.assertEqual(command.fallback_policy.value, "reject")

        eager_decision = compile_schedule(graph, _timeline(), eager_policy)
        compiled_region = decisions[0].execution_regions[0]
        eager_region = eager_decision.execution_regions[0]
        self.assertEqual(eager_region.compile_safe, compiled_region.compile_safe)
        assert eager_region.compile_command is not None
        self.assertEqual(eager_region.compile_command.adapter.value, "eager")
        self.assertFalse(eager_region.compile_command.fullgraph)
        self.assertNotEqual(eager_region.artifact_key, compiled_region.artifact_key)
        self.assertNotEqual(eager_decision.signature, decisions[0].signature)

        disabled = compile_schedule(
            graph,
            _timeline(),
            replace(base, capture_policy="disabled"),
        )
        self.assertFalse(disabled.execution_regions[0].capture_safe)
        self.assertIs(
            disabled.execution_regions[0].capture_mode,
            RegionCaptureMode.DISABLED,
        )

        with self.assertRaisesRegex(RuntimeError, "required Region capture"):
            compile_schedule(
                graph,
                _timeline(),
                replace(base, capture_policy="require"),
            )

    def test_rust_owns_closed_region_capture_decision(self) -> None:
        graph = ScheduleGraph(
            workload_key="region-capture-mode",
            tasks=(
                ScheduleTask(
                    "captured",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    capture_safe=True,
                    inputs=(
                        ScheduleInput(
                            "hidden",
                            ScheduleInputKind.HIDDEN,
                            "hidden:0:post:decode",
                            layer="0",
                            layer_kind=HiddenLayerKind.INDEX,
                            site="post",
                            phase="decode",
                        ),
                    ),
                ),
            ),
        )
        eager = _timeline()
        piecewise = replace(eager, graph_mode="piecewise")
        auto = CompileOptions(
            memory_budget_bytes=64,
            residual_resource_manifest=_indexed_hidden_manifest(0, "post"),
        )

        first = compile_schedule(graph, piecewise, auto)
        second = compile_schedule(graph, piecewise, auto)
        region = first.execution_regions[0]
        self.assertTrue(region.capture_safe)
        self.assertIs(region.capture_mode, RegionCaptureMode.REQUIRE)
        self.assertEqual(first.cost.capture_boundary_count, 1)
        self.assertEqual(first.signature, second.signature)
        self.assertEqual(region.artifact_key, second.execution_regions[0].artifact_key)
        self.assertEqual(
            lower_physical_flow_plan(graph, piecewise, first, auto).nodes[0].capture_policy,
            CapturePolicy.REQUIRE,
        )

        disabled = compile_schedule(
            graph, piecewise, replace(auto, capture_policy="disabled")
        )
        self.assertTrue(disabled.execution_regions[0].capture_safe)
        self.assertIs(
            disabled.execution_regions[0].capture_mode,
            RegionCaptureMode.DISABLED,
        )
        self.assertEqual(disabled.cost.capture_boundary_count, 0)
        self.assertEqual(
            lower_physical_flow_plan(
                graph, piecewise, disabled, replace(auto, capture_policy="disabled")
            ).nodes[0].capture_policy,
            CapturePolicy.DISABLED,
        )

        required = compile_schedule(
            graph, piecewise, replace(auto, capture_policy="require")
        )
        self.assertIs(
            required.execution_regions[0].capture_mode,
            RegionCaptureMode.REQUIRE,
        )
        eager_auto = compile_schedule(graph, eager, auto)
        self.assertTrue(eager_auto.execution_regions[0].capture_safe)
        self.assertIs(
            eager_auto.execution_regions[0].capture_mode,
            RegionCaptureMode.DISABLED,
        )
        self.assertEqual(eager_auto.cost.capture_boundary_count, 0)
        with self.assertRaisesRegex(RuntimeError, "required Region capture"):
            compile_schedule(graph, eager, replace(auto, capture_policy="require"))

        invalid_mode = json.loads(first.to_json())
        invalid_mode["execution_regions"][0]["capture_mode"] = "prefer"
        with self.assertRaisesRegex(ValueError, "capture_mode"):
            type(first).from_json(json.dumps(invalid_mode))
        tampered = replace(
            first,
            execution_regions=(
                replace(region, capture_mode=RegionCaptureMode.DISABLED),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "signature is invalid"):
            validate_schedule(graph, piecewise, tampered, auto)

    def test_schema_v3_region_signature_golden(self) -> None:
        graph = ScheduleGraph(
            workload_key="golden",
            tasks=(
                ScheduleTask(
                    "a",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                ),
                ScheduleTask(
                    "b",
                    timing=TaskTiming(
                        after="sampler.pre_draw", before="next.prepare"
                    ),
                ),
            ),
            edges=(ScheduleEdge("a", "b", ScheduleEdgeKind.DATA),),
        )
        timeline = MainWorkloadTimeline(
            backend_id="vllm",
            backend_version="0.27.1",
            runner_family="mrv2",
            graph_mode="eager",
            anchors=(
                TimelineAnchor("model.execute", 0),
                TimelineAnchor("sampler.pre_draw", 1),
                TimelineAnchor("step.end", 2),
                TimelineAnchor("next.prepare", 3),
            ),
            regions=(),
            static_headroom_bytes=64,
        )
        with mock.patch(
            "tllm.schedule.compiler.compiler_environment_fingerprint",
            return_value="1" * 64,
        ):
            decision = compile_schedule(
                graph,
                timeline,
                CompileOptions(
                    memory_budget_bytes=64,
                    compiler_fingerprint="1" * 64,
                ),
            )

        golden = (
            Path(__file__).with_name("golden") / "schedule_v3_region.json"
        )
        if os.environ.get("TLLM_REGEN_GOLDEN") == "1":
            golden.write_text(decision.to_json() + "\n", encoding="utf-8")
            print(f"signature={decision.signature}")
            print(f"artifact_key={decision.execution_regions[0].artifact_key}")
        golden_text = golden.read_text(encoding="utf-8")
        self.assertEqual(decision.to_json() + "\n", golden_text)
        parsed_golden = type(decision).from_json(golden_text)
        with mock.patch(
            "tllm.schedule.compiler.compiler_environment_fingerprint",
            return_value="1" * 64,
        ):
            validate_schedule(graph, timeline, parsed_golden, CompileOptions(
                memory_budget_bytes=64,
                compiler_fingerprint="1" * 64,
            ))
        self.assertEqual(
            decision.signature,
            "bb5a377306f518e4970ade2cf6554c7837dffb3aea2b71a3ba5e11a786d6f744",
        )
        self.assertEqual(
            decision.execution_regions[0].artifact_key,
            "de6234b02e0b6c932500e562095cae8bbf9b2c75ed2ba77c2043f4d3552cafc1",
        )

    def test_schema_v3_complex_region_signature_golden(self) -> None:
        """Freeze Rust-owned binder, state, and cross-Region wire records."""

        graph = ScheduleGraph(
            workload_key="complex-golden",
            tasks=(
                ScheduleTask(
                    "produce",
                    timing=TaskTiming(
                        after="model.execute", before="sampler.pre_draw"
                    ),
                    implementation_ids=("python:produce@1",),
                    implementation_capabilities=(("torch.export",),),
                    required_capabilities=(),
                    effects=(
                        ScheduleEffect(
                            ScheduleEffectKind.STATE_TRANSITION, "optimizer"
                        ),
                    ),
                    outputs=(
                        ScheduleOutput(
                            "handoff", ScheduleOutputKind.TEMPORARY, "handoff"
                        ),
                    ),
                ),
                ScheduleTask(
                    "bind",
                    timing=TaskTiming(
                        after="sampler.pre_draw", before="step.end"
                    ),
                    implementation_ids=("test.backend:bind@1",),
                    implementation_kinds=("extern",),
                    implementation_capabilities=(("backend/region_binder",),),
                    required_capabilities=("backend/region_binder",),
                    inputs=(
                        ScheduleInput(
                            "handoff",
                            ScheduleInputKind.RESULT,
                            "produce",
                            "handoff",
                        ),
                    ),
                ),
            ),
            edges=(ScheduleEdge("produce", "bind", ScheduleEdgeKind.DATA),),
            values=(
                ScheduleValue(
                    "handoff",
                    producer="produce",
                    producer_output="handoff",
                    consumers=("bind",),
                    size_bytes=16,
                ),
            ),
            states=(ScheduleState("optimizer", "replica"),),
            schema_version=2,
        )
        timeline = MainWorkloadTimeline(
            backend_id="complex-golden",
            backend_version="1",
            runner_family="mrv2",
            graph_mode="eager",
            anchors=(
                TimelineAnchor(
                    "model.execute",
                    0,
                    backend_events=(BackendObservableEvent.PREPARE_INPUTS_POST,),
                ),
                TimelineAnchor(
                    "sampler.pre_draw",
                    1,
                    backend_events=(BackendObservableEvent.SAMPLER_PRE_DRAW,),
                ),
                TimelineAnchor(
                    "step.end",
                    2,
                    backend_events=(BackendObservableEvent.SAMPLE_TOKENS_POST,),
                ),
            ),
            regions=(),
            static_headroom_bytes=64,
            capabilities=("backend/region_binder", "torch.export"),
            schema_version=2,
        )
        options = CompileOptions(
            memory_budget_bytes=64,
            compiler_fingerprint="2" * 64,
        )
        with mock.patch(
            "tllm.schedule.compiler.compiler_environment_fingerprint",
            return_value="2" * 64,
        ):
            decision = compile_schedule(graph, timeline, options)
            repeated = compile_schedule(graph, timeline, options)

        self.assertEqual(decision.to_json(), repeated.to_json())
        self.assertEqual(len(decision.execution_regions), 2)
        self.assertTrue(
            any(
                region.implementation_kind is RegionImplementationKind.EXTERN
                and "backend/region_binder" in region.required_capabilities
                for region in decision.execution_regions
            )
        )
        self.assertTrue(
            any(
                effect.kind is ScheduleEffectKind.STATE_TRANSITION
                and effect.resource == "optimizer"
                for region in decision.execution_regions
                for effect in region.effects
            )
        )
        self.assertEqual(len(decision.materializations), 1)
        materialization = decision.materializations[0]
        self.assertEqual(materialization.value_id, "handoff")
        self.assertEqual(materialization.size_bytes, 16)
        self.assertEqual(
            (
                materialization.producer_region,
                materialization.consumer_region,
            ),
            (
                decision.execution_regions[0].region_id,
                decision.execution_regions[1].region_id,
            ),
        )
        self.assertEqual(len(decision.fences), 1)
        self.assertEqual(
            (decision.fences[0].source, decision.fences[0].target),
            (
                decision.execution_regions[0].region_id,
                decision.execution_regions[1].region_id,
            ),
        )
        binder_stages = tuple(
            stage
            for advance in decision.runtime_advance_plan
            for stage in advance.backend_region_binder_stages
        )
        self.assertEqual(len(binder_stages), 1)
        self.assertEqual(
            binder_stages[0].implementation_id, "test.backend:bind@1"
        )
        self.assertIs(
            binder_stages[0].backend_event, BackendObservableEvent.SAMPLER_PRE_DRAW
        )

        golden = (
            Path(__file__).with_name("golden") / "schedule_v3_complex_region.json"
        )
        if os.environ.get("TLLM_REGEN_GOLDEN") == "1":
            golden.write_text(decision.to_json() + "\n", encoding="utf-8")
        golden_text = golden.read_text(encoding="utf-8")
        self.assertEqual(decision.to_json() + "\n", golden_text)
        parsed_golden = type(decision).from_json(golden_text)
        self.assertEqual(parsed_golden, decision)
        with mock.patch(
            "tllm.schedule.compiler.compiler_environment_fingerprint",
            return_value="2" * 64,
        ):
            validate_schedule(graph, timeline, parsed_golden, options)

    def test_memory_budget_is_a_hard_constraint(self) -> None:
        graph = ScheduleGraph(
            workload_key="abc",
            tasks=(
                ScheduleTask(
                    "oversized",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    temporary_bytes=65,
                ),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "memory budget"):
            compile_schedule(graph, _timeline(), CompileOptions(memory_budget_bytes=64))

    def test_main_region_reservation_is_part_of_peak_memory(self) -> None:
        graph = ScheduleGraph(
            workload_key="abc",
            tasks=(
                ScheduleTask(
                    "model_adjacent",
                    timing=TaskTiming(
                        after="model.execute", before="sampler.pre_draw"
                    ),
                    temporary_bytes=40,
                ),
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "memory budget"):
            compile_schedule(graph, _timeline(), CompileOptions(memory_budget_bytes=64))

    def test_memory_split_keeps_the_largest_feasible_regions(self) -> None:
        timeline = MainWorkloadTimeline(
            backend_id="vllm",
            backend_version="0.27.1",
            runner_family="mrv2",
            graph_mode="eager",
            anchors=(
                TimelineAnchor("begin", 0),
                TimelineAnchor("hot", 1),
                TimelineAnchor("cool", 2),
                TimelineAnchor("end", 3),
            ),
            regions=(
                MainWorkloadRegion(
                    "reserved",
                    start="hot",
                    end="cool",
                    reservation=ResourceReservation(memory_bytes=50),
                ),
            ),
            static_headroom_bytes=64,
        )
        graph = ScheduleGraph(
            workload_key="memory-split",
            tasks=(
                ScheduleTask(
                    "wide_a",
                    timing=TaskTiming(after="begin", before="end"),
                    temporary_bytes=20,
                ),
                ScheduleTask(
                    "wide_b",
                    timing=TaskTiming(after="begin", before="end"),
                    temporary_bytes=20,
                ),
                ScheduleTask(
                    "narrow",
                    timing=TaskTiming(after="hot", before="cool"),
                    temporary_bytes=1,
                ),
            ),
        )

        decision = compile_schedule(
            graph, timeline, CompileOptions(memory_budget_bytes=64)
        )

        self.assertEqual(
            tuple(region.tasks for region in decision.execution_regions),
            (("wide_a", "wide_b"), ("narrow",)),
        )
        self.assertEqual(decision.memory_plan.peak_bytes, 51)
        self.assertEqual(decision.cost.launch_group_count, 2)
        self.assertEqual(
            tuple(region.serial_fallback for region in decision.execution_regions),
            (False, False),
        )
        no_fallback = compile_schedule(
            graph,
            timeline,
            CompileOptions(
                memory_budget_bytes=64,
                allow_serial_fallback=False,
            ),
        )
        self.assertEqual(
            tuple(region.tasks for region in no_fallback.execution_regions),
            (("wide_a", "wide_b"), ("narrow",)),
        )
        self.assertTrue(
            all(not region.serial_fallback for region in no_fallback.execution_regions)
        )

    def test_affine_region_uses_deterministic_isl_schedule(self) -> None:
        token = ScheduleAffineExpr((("token", 1),))
        previous_token = ScheduleAffineExpr((("token", 1),), -1)
        layer = ScheduleAffineExpr((("layer", 1),))
        graph = ScheduleGraph(
            workload_key="affine",
            tasks=(
                ScheduleTask(
                    "advance",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    domain=ScheduleDomain(
                        name="token_layer",
                        indices=("token", "layer"),
                        bounds=(("token", "0", "T"), ("layer", "0", "L")),
                        parameters=("T", "L"),
                    ),
                    accesses=(
                        ScheduleAffineAccess(
                            "state", (previous_token, layer), "read"
                        ),
                        ScheduleAffineAccess("state", (token, layer), "write"),
                    ),
                ),
            ),
        )
        options = CompileOptions(memory_budget_bytes=64)

        first = compile_schedule(graph, _timeline(), options)
        second = compile_schedule(graph, _timeline(), options)

        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.schedule_tree.kind, "hybrid")
        self.assertEqual(len(first.schedule_tree.affine_regions), 1)
        region = first.schedule_tree.affine_regions[0]
        self.assertEqual(region.tasks, ("advance",))
        self.assertIn("T0", region.schedule)
        self.assertIn("T0", region.dependences)
        validate_schedule(graph, _timeline(), first, options)

    def test_capture_is_a_region_artifact_not_a_task_partition(self) -> None:
        graph = ScheduleGraph(
            workload_key="capture",
            tasks=(
                ScheduleTask(
                    "captured",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                    capture_safe=True,
                    inputs=(
                        ScheduleInput(
                            "hidden",
                            ScheduleInputKind.HIDDEN,
                            "hidden:0:post:decode",
                            layer="0",
                            layer_kind=HiddenLayerKind.INDEX,
                            site="post",
                            phase="decode",
                        ),
                    ),
                ),
            ),
        )
        eager = _timeline()
        piecewise = MainWorkloadTimeline(
            backend_id=eager.backend_id,
            backend_version=eager.backend_version,
            runner_family=eager.runner_family,
            graph_mode="piecewise",
            anchors=eager.anchors,
            regions=eager.regions,
            static_headroom_bytes=eager.static_headroom_bytes,
            capabilities=eager.capabilities,
        )
        options = CompileOptions(
            memory_budget_bytes=64,
            residual_resource_manifest=_indexed_hidden_manifest(0, "post"),
        )
        decision = compile_schedule(graph, piecewise, options)

        self.assertEqual(decision.capture_partitions, ())
        self.assertEqual(len(decision.execution_regions), 1)
        region = decision.execution_regions[0]
        self.assertTrue(region.capture_safe)
        self.assertIs(region.capture_mode, RegionCaptureMode.REQUIRE)
        self.assertIs(region.stream, RegionStreamClass.DEFAULT)
        self.assertIs(region.fallback_policy, RegionFallbackPolicy.REJECT)
        self.assertEqual(len(region.artifact_key), 64)
        physical = lower_physical_flow_plan(graph, piecewise, decision, options)
        self.assertEqual(len(physical.nodes), 1)
        self.assertEqual(physical.nodes[0].capture_policy, CapturePolicy.REQUIRE)
        self.assertEqual(
            physical.nodes[0].stream,
            StreamClass(region.stream.value),
        )
        self.assertEqual(
            physical.nodes[0].fallback_policy,
            FallbackPolicy(region.fallback_policy.value),
        )
        self.assertIsNotNone(physical.nodes[0].artifact_key)

    def test_legacy_decision_is_parseable_but_rejected_for_execution(self) -> None:
        graph = ScheduleGraph(
            workload_key="legacy",
            tasks=(
                ScheduleTask(
                    "task",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                ),
            ),
        )
        options = CompileOptions(memory_budget_bytes=64)
        current = compile_schedule(graph, _timeline(), options)
        legacy = replace(
            current,
            execution_regions=(),
            runtime_advance_plan=(),
            completion_anchor="",
            materializations=(),
            schema_version=1,
        )
        parsed = type(current).from_json(legacy.to_json())
        self.assertEqual(parsed.schema_version, 1)
        self.assertEqual(parsed.execution_regions, ())
        with self.assertRaisesRegex(RuntimeError, "identity or schema"):
            validate_schedule(graph, _timeline(), parsed, options)

    def test_validator_rejects_a_tampered_decision(self) -> None:
        graph = ScheduleGraph(
            workload_key="abc",
            tasks=(
                ScheduleTask(
                    "predict",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                ),
            ),
        )
        options = CompileOptions(memory_budget_bytes=64)
        decision = compile_schedule(graph, _timeline(), options)
        tampered = replace(
            decision,
            placements=(replace(decision.placements[0], after_anchor="model.execute"),),
        )
        with self.assertRaisesRegex(RuntimeError, "signature is invalid"):
            validate_schedule(graph, _timeline(), tampered, options)

    def test_validator_rejects_a_tampered_physical_region_command_signature(self) -> None:
        graph = ScheduleGraph(
            workload_key="physical-command-signature",
            tasks=(
                ScheduleTask(
                    "predict",
                    timing=TaskTiming(after="model.execute", before="step.end"),
                ),
            ),
        )
        options = CompileOptions(memory_budget_bytes=64)
        decision = compile_schedule(graph, _timeline(), options)
        tampered = json.loads(decision.to_json())
        command = tampered["execution_regions"][0]["compile_command"]
        assert isinstance(command, dict)
        command["adapter"] = "eager"
        command["fullgraph"] = False
        parsed = type(decision).from_json(json.dumps(tampered))
        with self.assertRaisesRegex(RuntimeError, "signature is invalid"):
            validate_schedule(graph, _timeline(), parsed, options)

    def test_backend_timeline_never_exposes_fake_layer_anchors(self) -> None:
        from tllm.backends.vllm.v027_mrv1 import V027MRv1Driver
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver

        for driver in (V027MRv1Driver(), V027MRv2Driver()):
            with self.subTest(driver=driver.backend_id):
                timeline = driver.export_main_workload_timeline(
                    graph_mode="piecewise",
                    static_headroom_bytes=64,
                )
                self.assertNotIn("layer.post", tuple(anchor.anchor_id for anchor in timeline.anchors))
                self.assertTrue(all(region.pinned for region in timeline.regions))

    def test_timeline_rejects_unpinned_or_hidden_graph_boundaries(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned"):
            MainWorkloadRegion(
                "model",
                start="a",
                end="b",
                pinned=False,
            )
        with self.assertRaisesRegex(ValueError, "observable"):
            TimelineAnchor("layer.post", 0, observable=False, graph_boundary=True)


if __name__ == "__main__":
    unittest.main()
