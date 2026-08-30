# pyright: strict

from __future__ import annotations

import unittest
from collections.abc import Hashable
from dataclasses import dataclass, field
from threading import Event, Thread, current_thread
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.contracts.hidden_locator import (
    HiddenLocatorCapability,
)
from tllm.runtime import residual_runtime
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_execution import (
    NativeWorkloadExecution,
    StateVersionBinding,
    WorkloadExecutionBindings,
)
from tllm.schedule import (
    BackendObservableEvent,
    CompileOptions,
    HostCommandKind,
    MainWorkloadTimeline,
    RegionHostSubmission,
    RuntimeHostCommand,
    WorkloadCompilation,
    WorkloadCompilerCoordinator,
)
from tllm.workload import (
    Between,
    State,
    StateScope,
    WorkloadDeclaration,
    declare_workload,
    extern,
    read,
    result,
    step_id,
    task,
    temporary,
    transition,
    workload,
)


_JOIN_SIDE_STREAMS = (
    NativeWorkloadExecution._join_side_streams_locked  # pyright: ignore[reportPrivateUsage]
)


@dataclass(slots=True)
class _TailControl:
    update_entered: Event = field(default_factory=Event)
    release_update: Event = field(default_factory=Event)
    trace: list[str] = field(default_factory=list[str])
    versions: dict[Hashable, int] = field(default_factory=lambda: {None: 0})
    fail_update: bool = False
    worker_thread: Thread | None = None


@dataclass(slots=True)
class _CapacityControl:
    first_entered: Event = field(default_factory=Event)
    release_first: Event = field(default_factory=Event)
    trace: list[str] = field(default_factory=list[str])
    versions: dict[Hashable, int] = field(default_factory=lambda: {None: 0})


@workload(id="runtime.async-host-submission-test")  # pyright: ignore[reportUntypedClassDecorator]
class _AsyncHostSubmissionWorkload:
    state: State[torch.nn.Module] = State[torch.nn.Module](
        scope=StateScope.REPLICA,
        parameters="model",
        transition="manual",
    )

    def __init__(self, control: _TailControl) -> None:
        self._control = control
        self._weight = torch.nn.Parameter(torch.tensor(2.0))

    @task(
        inputs={"current_step": step_id()},
        outputs={"prediction": temporary()},
        effects=(read(state),),
        timing=Between("model.execute", "sampler.pre_draw"),
    )
    def predict(self, current_step: int) -> torch.Tensor:
        self._control.trace.append(f"predict:{current_step}")
        return self._weight * (current_step + 1)

    @task(
        inputs={"prediction": result("predict", "prediction")},
        effects=(transition(state),),
        timing=Between("token.commit", "next.prepare"),
        implementation=extern(
            "test.async-host:update@1",
            capabilities=(
                "runtime.side_stream_safe",
                "runtime.async_host_safe",
            ),
        ),
    )
    def update(self, prediction: torch.Tensor) -> None:
        self._control.worker_thread = current_thread()
        self._control.trace.append("update:entered")
        self._control.update_entered.set()
        if not self._control.release_update.wait(2.0):
            raise TimeoutError("test did not release the async tail")
        if self._control.fail_update:
            raise RuntimeError("async tail failed")
        _ = prediction.square()
        self._control.versions[None] += 1
        self._control.trace.append("update:finished")


@workload(id="runtime.async-host-capacity-test")  # pyright: ignore[reportUntypedClassDecorator]
class _AsyncHostCapacityWorkload:
    state: State[torch.nn.Module] = State[torch.nn.Module](
        scope=StateScope.REPLICA,
        parameters="model",
        transition="manual",
    )

    def __init__(self, control: _CapacityControl) -> None:
        self._control = control

    @task(
        outputs={"first_input": temporary()},
        effects=(read(state),),
        timing=Between("prepare.post", "model.execute"),
    )
    def produce(self) -> torch.Tensor:
        return torch.ones(1)

    @task(
        inputs={"first_input": result("produce", "first_input")},
        outputs={"second_input": temporary()},
        effects=(read(state),),
        timing=Between("model.execute", "sampler.pre_draw"),
        implementation=extern(
            "test.async-host:first@1",
            capabilities=(
                "runtime.side_stream_safe",
                "runtime.async_host_safe",
            ),
        ),
    )
    def first(self, first_input: torch.Tensor) -> torch.Tensor:
        self._control.trace.append("first:entered")
        self._control.first_entered.set()
        if not self._control.release_first.wait(2.0):
            raise TimeoutError("test did not release the first async submission")
        self._control.trace.append("first:finished")
        return first_input

    @task(
        inputs={"second_input": result("first", "second_input")},
        effects=(read(state),),
        timing=Between("sampler.pre_draw", "token.commit"),
        implementation=extern(
            "test.async-host:second@1",
            capabilities=(
                "runtime.side_stream_safe",
                "runtime.async_host_safe",
            ),
        ),
    )
    def second(self, second_input: torch.Tensor) -> None:
        _ = second_input
        self._control.trace.append("second:executed")


def _timeline() -> MainWorkloadTimeline:
    capabilities = BackendCapabilities(
        backend_id="vllm-0.27-mrv2",
        runner_family="mrv2",
        sampler_stages=frozenset(),
        native_min_p=False,
        tuple_sampler_return=False,
        split_execute_and_sample=True,
        graph_safety=GraphSafety.EAGER_ONLY,
        hidden_locators=frozenset(
            {HiddenLocatorCapability.BLOCK_OUTPUT_DECODE}
        ),
    )
    return export_main_workload_timeline(
        capabilities,
        graph_mode="eager",
        static_headroom_bytes=1 << 20,
    )


class AsyncHostSubmissionUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        cuda_availability = mock.patch.object(
            torch.cuda,
            "is_available",
            return_value=False,
        )
        _ = cuda_availability.start()
        self.addCleanup(cuda_availability.stop)

    def _compile(
        self, control: _TailControl
    ) -> tuple[WorkloadDeclaration, WorkloadCompilation]:
        declaration = declare_workload(_AsyncHostSubmissionWorkload(control))
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
                allow_side_stream=True,
            ),
        ).compile(_timeline())
        return declaration, compilation

    def _build_execution(
        self, control: _TailControl
    ) -> NativeWorkloadExecution:
        declaration, compilation = self._compile(control)
        schedule_state = compilation.graph.states[0]
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            state_versions={
                schedule_state.resource: StateVersionBinding(
                    schedule_state,
                    lambda _step_id: dict(control.versions),
                )
            },
        )
        state = ResidualRuntimeState()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=bindings,
            )
        execution = state.workload_execution
        self.assertIsNotNone(execution)
        assert execution is not None
        return execution

    def _build_capacity_execution(
        self, control: _CapacityControl
    ) -> NativeWorkloadExecution:
        declaration = declare_workload(_AsyncHostCapacityWorkload(control))
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
                allow_side_stream=True,
            ),
        ).compile(_timeline())
        schedule_state = compilation.graph.states[0]
        state = ResidualRuntimeState()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=WorkloadExecutionBindings(
                    declarations=(declaration,),
                    state_versions={
                        schedule_state.resource: StateVersionBinding(
                            schedule_state,
                            lambda _step_id: dict(control.versions),
                        )
                    },
                ),
            )
        execution = state.workload_execution
        self.assertIsNotNone(execution)
        assert execution is not None
        return execution

    @staticmethod
    def _record_main_cuda_wait(
        execution: NativeWorkloadExecution,
        async_started: Event,
        trace: list[str],
    ) -> None:
        _JOIN_SIDE_STREAMS(execution)
        if async_started.is_set() and "main-cuda-wait" not in trace:
            trace.append("main-cuda-wait")

    def test_rust_v4_signs_closed_host_commands_in_region_order(self) -> None:
        _, compilation = self._compile(_TailControl())
        regions = {
            task_name: region.region_id
            for region in compilation.decision.execution_regions
            for task_name in region.tasks
        }
        execute_advance = next(
            advance
            for advance in compilation.decision.runtime_advance_plan
            if advance.backend_event
            is BackendObservableEvent.EXECUTE_MODEL_POST
        )
        submit_advance = next(
            advance
            for advance in compilation.decision.runtime_advance_plan
            if regions["update"] in advance.region_ids
        )

        self.assertEqual(compilation.decision.schema_version, 4)
        self.assertEqual(execute_advance.region_ids, (regions["predict"],))
        self.assertEqual(
            execute_advance.host_commands,
            (
                RuntimeHostCommand(
                    HostCommandKind.WAIT_ASYNC,
                    "async-serial-0",
                    (),
                ),
                RuntimeHostCommand(
                    HostCommandKind.EXECUTE_INLINE,
                    "",
                    (regions["predict"],),
                ),
            ),
        )
        self.assertEqual(submit_advance.region_ids, (regions["update"],))
        self.assertEqual(
            submit_advance.host_commands,
            (
                RuntimeHostCommand(
                    HostCommandKind.WAIT_ASYNC,
                    "async-serial-0",
                    (),
                ),
                RuntimeHostCommand(
                    HostCommandKind.SUBMIT_ASYNC,
                    "async-serial-0",
                    (regions["update"],),
                ),
            ),
        )
        for advance in compilation.decision.runtime_advance_plan:
            flattened = tuple(
                region_id
                for command in advance.host_commands
                if command.kind
                in {
                    HostCommandKind.EXECUTE_INLINE,
                    HostCommandKind.SUBMIT_ASYNC,
                }
                for region_id in command.region_ids
            )
            self.assertEqual(flattened, advance.region_ids)

    def test_submit_returns_and_next_prepare_preserves_pending_step_frame(
        self,
    ) -> None:
        control = _TailControl()
        execution = self._build_execution(control)
        submit_returned = Event()
        submit_results: list[bool] = []
        submit_errors: list[Exception] = []
        prepare_returned = Event()
        prepare_errors: list[Exception] = []

        execution.prepare_step(1)
        self.assertTrue(
            execution.advance_backend_event(
                BackendObservableEvent.EXECUTE_MODEL_POST
            )
        )

        def submit_tail() -> None:
            try:
                submit_results.append(
                    execution.advance_backend_event(
                        BackendObservableEvent.SAMPLE_TOKENS_POST
                    )
                )
            except Exception as exc:
                submit_errors.append(exc)
            finally:
                submit_returned.set()

        submitter = Thread(target=submit_tail)
        submitter.start()
        try:
            self.assertTrue(control.update_entered.wait(1.0))
            self.assertTrue(submit_returned.wait(0.2))
            self.assertFalse(submitter.is_alive())
            self.assertEqual(submit_errors, [])
            self.assertEqual(submit_results, [True])

            execution.finish_step(1)
            self.assertEqual(execution.retained_value_count, 1)
            self.assertEqual(execution.executor.stats.finish_count, 0)

            def prepare_next_step() -> None:
                try:
                    execution.prepare_step(2)
                except Exception as exc:
                    prepare_errors.append(exc)
                finally:
                    prepare_returned.set()

            preparer = Thread(target=prepare_next_step)
            preparer.start()
            try:
                self.assertTrue(prepare_returned.wait(0.2))
            finally:
                if not prepare_returned.is_set():
                    control.release_update.set()
                preparer.join(1.0)
            self.assertFalse(preparer.is_alive())
            self.assertEqual(prepare_errors, [])
            self.assertEqual(execution.retained_value_count, 1)
            self.assertEqual(execution.executor.stats.finish_count, 0)
        finally:
            control.release_update.set()
            submitter.join(1.0)

        self.assertTrue(
            execution.advance_backend_event(
                BackendObservableEvent.EXECUTE_MODEL_POST
            )
        )
        self.assertEqual(
            control.trace,
            [
                "predict:1",
                "update:entered",
                "update:finished",
                "predict:2",
            ],
        )
        self.assertEqual(execution.retained_value_count, 1)
        self.assertEqual(execution.executor.stats.finish_count, 1)
        execution.finish_step(2)
        execution.close()
        self.assertEqual(execution.retained_value_count, 0)

    def test_wait_completes_future_then_cuda_wait_then_default_reader(
        self,
    ) -> None:
        control = _TailControl()
        execution = self._build_execution(control)

        def record_main_cuda_wait(
            _execution: NativeWorkloadExecution,
        ) -> None:
            self._record_main_cuda_wait(
                _execution,
                control.update_entered,
                control.trace,
            )

        with mock.patch.object(
            NativeWorkloadExecution,
            "_join_side_streams_locked",
            autospec=True,
            side_effect=record_main_cuda_wait,
        ):
            execution.prepare_step(1)
            execution.advance_backend_event(
                BackendObservableEvent.EXECUTE_MODEL_POST
            )
            execution.advance_backend_event(
                BackendObservableEvent.SAMPLE_TOKENS_POST
            )
            self.assertTrue(control.update_entered.wait(1.0))
            execution.finish_step(1)
            execution.prepare_step(2)
            control.release_update.set()

            self.assertTrue(
                execution.advance_backend_event(
                    BackendObservableEvent.EXECUTE_MODEL_POST
                )
            )
        self.assertEqual(
            control.trace[-3:],
            [
                "update:finished",
                "main-cuda-wait",
                "predict:2",
            ],
        )
        execution.finish_step(2)
        execution.close()

    def test_worker_failure_propagates_at_wait_without_cuda_wait(self) -> None:
        control = _TailControl(fail_update=True)
        execution = self._build_execution(control)

        def record_main_cuda_wait(
            _execution: NativeWorkloadExecution,
        ) -> None:
            self._record_main_cuda_wait(
                _execution,
                control.update_entered,
                control.trace,
            )

        with mock.patch.object(
            NativeWorkloadExecution,
            "_join_side_streams_locked",
            autospec=True,
            side_effect=record_main_cuda_wait,
        ):
            execution.prepare_step(1)
            execution.advance_backend_event(
                BackendObservableEvent.EXECUTE_MODEL_POST
            )
            execution.advance_backend_event(
                BackendObservableEvent.SAMPLE_TOKENS_POST
            )
            self.assertTrue(control.update_entered.wait(1.0))
            execution.finish_step(1)
            execution.prepare_step(2)
            control.release_update.set()

            with self.assertRaisesRegex(RuntimeError, "async tail failed"):
                execution.advance_backend_event(
                    BackendObservableEvent.EXECUTE_MODEL_POST
                )
        self.assertNotIn("main-cuda-wait", control.trace)
        self.assertNotIn("predict:2", control.trace)
        self.assertEqual(execution.retained_value_count, 0)
        execution.close()

    def test_close_first_failure_still_retires_and_shuts_down_lane(self) -> None:
        control = _TailControl(fail_update=True)
        execution = self._build_execution(control)
        execution.prepare_step(1)
        execution.advance_backend_event(
            BackendObservableEvent.EXECUTE_MODEL_POST
        )
        execution.advance_backend_event(
            BackendObservableEvent.SAMPLE_TOKENS_POST
        )
        self.assertTrue(control.update_entered.wait(1.0))
        worker_thread = control.worker_thread
        self.assertIsNotNone(worker_thread)
        assert worker_thread is not None
        control.release_update.set()

        with self.assertRaisesRegex(RuntimeError, "async tail failed"):
            execution.close()

        self.assertEqual(execution.retained_value_count, 0)
        self.assertEqual(execution.executor.stats.finish_count, 1)
        self.assertFalse(worker_thread.is_alive())

    def test_synchronize_and_close_drain_pending_submission(self) -> None:
        for operation in ("synchronize", "close"):
            with self.subTest(operation=operation):
                control = _TailControl()
                execution = self._build_execution(control)
                drained = Event()
                drain_errors: list[Exception] = []

                def record_main_cuda_wait(
                    _execution: NativeWorkloadExecution,
                ) -> None:
                    self._record_main_cuda_wait(
                        _execution,
                        control.update_entered,
                        control.trace,
                    )

                with mock.patch.object(
                    NativeWorkloadExecution,
                    "_join_side_streams_locked",
                    autospec=True,
                    side_effect=record_main_cuda_wait,
                ):
                    execution.prepare_step(1)
                    execution.advance_backend_event(
                        BackendObservableEvent.EXECUTE_MODEL_POST
                    )
                    execution.advance_backend_event(
                        BackendObservableEvent.SAMPLE_TOKENS_POST
                    )
                    self.assertTrue(control.update_entered.wait(1.0))

                    def drain() -> None:
                        try:
                            if operation == "synchronize":
                                execution.synchronize()
                            else:
                                execution.close()
                        except Exception as exc:
                            drain_errors.append(exc)
                        finally:
                            drained.set()

                    drainer = Thread(target=drain)
                    drainer.start()
                    try:
                        self.assertFalse(drained.wait(0.05))
                        control.release_update.set()
                        self.assertTrue(drained.wait(1.0))
                        self.assertFalse(drainer.is_alive())
                        self.assertEqual(drain_errors, [])
                    finally:
                        control.release_update.set()
                        drainer.join(1.0)
                self.assertLess(
                    control.trace.index("update:finished"),
                    control.trace.index("main-cuda-wait"),
                )
                if operation == "synchronize":
                    execution.finish_step(1)
                    execution.close()

    def test_serial_lane_capacity_blocks_a_second_signed_submission(
        self,
    ) -> None:
        control = _CapacityControl()
        execution = self._build_capacity_execution(control)
        second_returned = Event()
        second_results: list[bool] = []
        second_errors: list[Exception] = []
        regions = {
            task_name: region
            for region in execution.compilation.decision.execution_regions
            for task_name in region.tasks
        }
        self.assertIs(
            regions["first"].host_submission,
            RegionHostSubmission.ASYNC_SERIAL,
        )
        self.assertIs(
            regions["second"].host_submission,
            RegionHostSubmission.ASYNC_SERIAL,
        )
        for task_name in ("first", "second"):
            advance = next(
                candidate
                for candidate in execution.compilation.decision.runtime_advance_plan
                if regions[task_name].region_id in candidate.region_ids
            )
            self.assertEqual(
                tuple(command.kind for command in advance.host_commands),
                (
                    HostCommandKind.WAIT_ASYNC,
                    HostCommandKind.SUBMIT_ASYNC,
                ),
            )

        execution.prepare_step(1)
        self.assertTrue(
            execution.advance_backend_event(
                BackendObservableEvent.PREPARE_INPUTS_POST
            )
        )
        self.assertTrue(
            execution.advance_backend_event(
                BackendObservableEvent.EXECUTE_MODEL_POST
            )
        )
        self.assertTrue(control.first_entered.wait(1.0))

        def submit_second() -> None:
            try:
                second_results.append(
                    execution.advance_backend_event(
                        BackendObservableEvent.SAMPLE_TOKENS_PRE
                    )
                )
            except Exception as exc:
                second_errors.append(exc)
            finally:
                second_returned.set()

        submitter = Thread(target=submit_second)
        submitter.start()
        try:
            self.assertFalse(second_returned.wait(0.05))
            self.assertTrue(submitter.is_alive())
            self.assertNotIn("second:executed", control.trace)
            control.release_first.set()
            self.assertTrue(second_returned.wait(1.0))
            self.assertEqual(second_errors, [])
            self.assertEqual(second_results, [True])
        finally:
            control.release_first.set()
            submitter.join(1.0)
        self.assertFalse(submitter.is_alive())

        execution.synchronize()
        self.assertEqual(
            control.trace,
            ["first:entered", "first:finished", "second:executed"],
        )
        execution.finish_step(1)
        execution.close()


if __name__ == "__main__":
    unittest.main()
