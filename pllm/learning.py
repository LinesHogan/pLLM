"""Structured learning surfaces: host transactions, completion learning and causal adapters.

These classes compose the ordinary declaration/binding machinery for learning-shaped
workloads whose mathematics arrives as frozen user functions with a signed optimizer
transition. They are ordinary clients of the DSL, not a second authoring stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import ClassVar, Generic, TypeVar, cast

import torch

from pllm.authoring import BackendRequirements, BackendWorkload, CausalModel, CausalTrainBatch
from pllm.contracts.backend_attachment import BackendAttachmentSpec
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import ParameterOverlayFieldBinding, ParameterOverlayOperator
from pllm.contracts.region_program import TensorType
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import (
    AnchorSymbol,
    MathContract,
    SchedulerTimeline,
    SemanticGraph,
    StateEffectKind,
    TaskResourceFact,
    TaskSymbol,
    ValueResourceFact,
)
from pllm.contracts.tensor_layout import TensorDType, TensorLayoutRequest
from pllm.execution.binding_base import Buffer
from pllm.execution.codec import boundary_digest
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.declaration import Builder, Dimensions, Field, Shape, When, native
from pllm.execution.freeze import fingerprint, freeze, owner_snapshot, owner_values
from pllm.execution.host_memory import transaction_resources
from pllm.execution.host_runtime import (
    CompletionInput,
    HostRuntimeFacts,
    HostTaskState,
    HostTransactionService,
    HostWorkload,
    TensorHostBinding,
    TransactionHostBinding,
)
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.region_adamw import AdamWRegion
from pllm.execution.shared_tensor import tensor_dtype
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.step_input import TensorStepInput
from pllm.execution.tensor_frame import Tensors
from pllm.execution.tensor_program import FunctionMath, InvocationOutput, Roots, TensorBatch, autograd_loss
from pllm.kernels.adamw import AdamWConfig
from pllm.usage import Activation, InstanceState, Overlay, Phase, RequestState

ContextT = TypeVar("ContextT")
ResultT = TypeVar("ResultT")
InputT = TypeVar("InputT")
TransactionT = TypeVar("TransactionT")
SnapshotT = TypeVar("SnapshotT")
RequestT = TypeVar("RequestT")

_MARKERS = (InstanceState, Activation, Overlay)


def transaction(
    name: str,
    facts: HostRuntimeFacts,
    *,
    config: object,
    state_factory: Callable[[], HostTaskState[TransactionT, ContextT, SnapshotT, ResultT]],
    service: HostTransactionService[RequestT, TransactionT, ResultT],
    update: Callable[[ContextT], None],
    query: Callable[[ContextT], None],
    publish: Callable[[ContextT], None],
    timeline: SchedulerTimeline | None = None,
    example: Callable[[HostRuntimeFacts], tuple[ContextT, object]] | None = None,
) -> TransactionHostBinding[RequestT, TransactionT, ContextT, SnapshotT, ResultT]:
    """Bind native structured math; retain the original state, service and context."""
    b = Builder[ContextT](name, boundary_digest(config, facts))
    resource = b.state("state")
    changed = b.compute(
        "update",
        host_function=update,
        inputs={"context": b.input(name, "transaction", f"{name}.transaction@1")},
        outputs=("context",),
        changes=resource,
        when=When.HOST_COMPUTE,
        implementations=(native(f"{name}:update@1", f"compute/{facts.device.type}@1"),),
    )
    queried = b.compute(
        "query",
        host_function=query,
        inputs={"context": changed["context"]},
        outputs=("context",),
        reads=(resource.version(changed),),
        when=When.HOST_OUTPUT,
        implementations=(native(f"{name}:query@1", f"compute/{facts.device.type}@1"),),
    )
    b.compute(
        "publish",
        host_function=publish,
        inputs={"context": queried["context"]},
        outputs=("result",),
        result_schema=f"{name}.result@1",
        reads=(resource.version(changed),),
        when=When.HOST_COMMIT,
        implementations=(native(f"{name}:publish@1", "host/service@1", host=True),),
    )
    if example is not None:
        timeline = transaction_resources(facts, example, tuple(b.host_operations.values()))
    return TransactionHostBinding(b.graph, state_factory, b.bind_host(), service, facts, timeline)


class _LearningMath(ABC, Generic[InputT, ContextT, ResultT]):
    @abstractmethod
    def prepare(self, value: InputT) -> ContextT: ...

    @abstractmethod
    def loss(self, data: ContextT, parameters: Tensors) -> torch.Tensor | None: ...

    @abstractmethod
    def trained_rows(self, data: ContextT) -> int: ...

    @abstractmethod
    def output(self, data: ContextT, result: InvocationOutput) -> ResultT: ...


class CompletionWorkload(
    _LearningMath[CompletionInput, ContextT, ResultT],
    HostWorkload[StateExecution[CompletionInput, ContextT, ResultT]],
):
    """Structured completion math with named parameters and one original instance root."""

    name: ClassVar[str] = "completion.learning"

    @property
    @abstractmethod
    def parameters(self) -> InstanceState: ...

    def training_example(self, facts: HostRuntimeFacts) -> ContextT:
        """Maximum-shape host input; override when host metadata selects training rows."""
        return self.prepare(CompletionInput.example(facts))

    def declare_host(self, facts: HostRuntimeFacts) -> TensorHostBinding[CompletionInput, ContextT, ResultT]:
        if facts.hidden_size < 1:
            raise ValueError("completion learning requires a positive hidden size")
        owner = owner_snapshot(self, _MARKERS)
        definition = owner.parameters
        if isinstance(definition, RequestState) or not isinstance(definition.optimizer, AdamWConfig):
            raise ValueError("completion learning requires instance parameters with an optimizer")
        if any(isinstance(shape, Buffer) for shape in definition.fields.values()):
            raise ValueError("completion parameters require tensor shapes, not overlay buffers")
        shapes = cast(Mapping[str, Shape], definition.fields)
        identity = fingerprint(owner.loss)
        b = Builder[ContextT](
            self.name,
            boundary_digest(
                facts,
                fingerprint(owner_values(owner, _MARKERS)),
                fingerprint(definition),
                identity,
                fingerprint(owner.trained_rows),
            ),
            Dimensions(facts.hidden_size, 0),
        )
        state = b.state(
            "parameters",
            tuple(
                Field(name, shape, TensorDType(str(definition.dtype).removeprefix("torch.")), definition.alignment)
                for name, shape in shapes.items()
            ),
            namespace=definition.namespace,
            optimizer=definition.optimizer,
        )
        (layout,) = b.layouts.layouts
        device = "cpu" if facts.device.type == "cpu" else f"cuda:{facts.device.index or 0}"
        optimizer = AdamWRegion(
            tuple(TensorType(field.shape, "torch." + field.dtype.value, device) for field in layout.layout.fields[::5]),
            definition.optimizer,
        )
        update = b.compute(
            "train",
            function=autograd_loss(freeze(owner.loss), tuple(shapes), freeze(owner.trained_rows)),
            inputs={"batch": b.input("input", "completions", f"{self.name}.training@1"), "state": state.read()},
            outputs=("control",),
            changes=state,
            effect=StateEffectKind.TRANSITION_MAYBE,
            when=When.HOST_COMPUTE,
            implementations=(
                native(
                    "region.completion_adamw@1:"
                    + boundary_digest(identity, optimizer.program, optimizer.capacity, optimizer.copy_workspace_bytes)
                ),
            ),
            optimizer=optimizer,
            math=MathContract(
                parameter_gradients=(state.declaration.resource,),
                detached_inputs=("batch",),
                forward_rows="batch",
                parameter_gradient_rows="batch",
                update_states="state",
                reduction="custom",
            ),
        )
        b.publish(
            update,
            state,
            output="control",
            schema=f"{self.name}.result@1",
            when=When.HOST_COMMIT,
            implementations=(native("canonical.state:publish@1"),),
        )
        generator = torch.Generator(device=facts.device).manual_seed(facts.initialization_seed)
        initialize_parameters = None if definition.initialize is None else freeze(definition.initialize)
        prepare_data, output_data = freeze(owner.prepare), freeze(owner.output)

        def initialize(roots: Roots, keys: tuple[RequestStateKey | None, ...]) -> None:
            if initialize_parameters is not None:
                (root,) = roots
                initialize_parameters({name: root[5 * index] for index, name in enumerate(shapes)}, generator)

        def prepare(value: CompletionInput, batch: TensorBatch) -> ContextT:
            if value.hidden.shape[1] != facts.hidden_size or value.hidden.device != facts.device:
                raise ValueError("completion hidden rows differ from specialization")
            if facts.max_input_rows is not None and len(value.hidden) > facts.max_input_rows:
                raise ValueError("completion rows exceed the signed input capacity")
            if value.readiness is not None:
                if batch.stream is None:
                    raise ValueError("CUDA completion input requires a CUDA invocation")
                value.readiness.order_on(batch.stream)
                value.hidden.record_stream(batch.stream)
            return prepare_data(value)

        timeline = facts.timeline()
        if facts.memory_budget_bytes is not None:
            if facts.max_input_rows is None:
                raise ValueError("finite completion costs require max_input_rows")
            data = owner.training_example(facts)
            root = tuple(
                torch.empty(field.shape, dtype=tensor_dtype(field.dtype), device="meta")
                for field in layout.layout.fields
            )
            batch = TensorBatch(RequestStateSelection((), ()), (root,), (), None)
            usage = b.operations[update.symbol].memory(data, batch)
            timeline = replace(
                timeline,
                reserved_bytes=(facts.max_input_rows * facts.hidden_size * 4,) * len(timeline.anchors),
                task_resources=(
                    TaskResourceFact(TaskSymbol("publish"), 0),
                    TaskResourceFact(update.symbol, usage.workspace),
                ),
                value_resources=(
                    ValueResourceFact(TaskSymbol("publish"), "control", 0),
                    ValueResourceFact(update.symbol, "control", usage.output),
                ),
            )
        return TensorHostBinding(
            b.graph,
            b.layouts,
            MathBinding(
                FunctionMath({str(state.declaration.resource): initialize}, lambda value: (), prepare, output_data),
                b.bind_all(),
            ),
            facts,
            AnchorSymbol("host.compute.end"),
            timeline,
        )


class CausalAdapterWorkload(
    _LearningMath[CausalTrainBatch[torch.Tensor] | None, ContextT, ResultT], BackendWorkload[ContextT, ResultT]
):
    """Instance low-rank overlays with an explicit frozen causal-model dependency."""

    name: ClassVar[str] = "causal.adapter"
    namespace: ClassVar[str] = ""
    model: CausalModel

    @property
    @abstractmethod
    def optimizer(self) -> AdamWConfig: ...

    @property
    @abstractmethod
    def phase(self) -> Phase: ...

    def backend_requirements(self) -> BackendRequirements:
        return BackendRequirements(
            parameter_targets=tuple(target.target for target in self.model.targets),
            model_source=self.model.model_source,
            resident_bytes=self.model.resident_bytes,
        )

    def _binding(
        self, facts: BackendCompileFacts
    ) -> tuple[Builder[ContextT], FunctionMath[CausalTrainBatch[torch.Tensor] | TensorStepInput, ContextT, ResultT]]:
        owner = owner_snapshot(self, _MARKERS, (self.model,))
        model = owner.model
        if (
            model.resident_bytes < 0
            or model.resident_bytes
            and (
                len(facts.timeline.reserved_bytes) != len(facts.timeline.anchors)
                or any(reserved < model.resident_bytes for reserved in facts.timeline.reserved_bytes)
            )
        ):
            raise ValueError("causal model residency exceeds the signed timeline reservation")
        if (
            facts.targets != model.targets
            or facts.probe.tensor_dtype != model.tensor_dtype
            or model.device.type != "meta"
            and facts.device != model.device
        ):
            raise ValueError("causal model and backend projection facts differ")
        if owner.phase not in (Phase.BEFORE_MODEL, Phase.AFTER_SAMPLE):
            raise ValueError("causal adapter updates require BEFORE_MODEL or AFTER_SAMPLE")
        targets = tuple(item.target for item in model.targets)
        if not targets or targets != tuple(sorted(set(targets))) or len(model.initial_factors) != 2 * len(targets):
            raise ValueError("causal adapters require sorted unique targets and one A/B pair per target")
        for index, target in enumerate(model.targets):
            down, up = model.initial_factors[2 * index : 2 * index + 2]
            if (
                down.ndim != 2
                or up.ndim != 2
                or down.shape[0] < 1
                or down.shape[0] != up.shape[1]
                or down.shape[1] != target.input_size
                or up.shape[0] != target.output_size
                or down.device != model.device
                or up.device != model.device
                or down.dtype != tensor_dtype(model.tensor_dtype)
                or up.dtype != down.dtype
            ):
                raise ValueError("causal adapter factors differ from their target shape, device or dtype")
        identity = fingerprint(owner.loss)
        values = {key: value for key, value in owner_values(owner, _MARKERS).items() if key != "model"}
        b = Builder[ContextT](
            owner.name,
            boundary_digest(
                fingerprint(values),
                model.fingerprint,
                model.scale,
                owner.optimizer,
                identity,
                fingerprint(owner.trained_rows),
            ),
        )
        names = tuple(f"factor_{index}" for index in range(len(model.initial_factors)))
        state = b.state(
            "factors",
            tuple(
                Field(name, tuple(factor.shape), model.tensor_dtype)
                for name, factor in zip(names, model.initial_factors, strict=True)
            ),
            namespace=owner.namespace,
            optimizer=owner.optimizer,
        )
        update = b.compute(
            "update",
            function=autograd_loss(freeze(owner.loss), names, freeze(owner.trained_rows)),
            inputs={"batch": b.input("training", "batch", CausalTrainBatch.schema), "state": state.read()},
            outputs=("updated",),
            changes=state,
            effect=StateEffectKind.TRANSITION_MAYBE,
            when=owner.phase.value,
            implementations=(
                native(
                    f"python:{identity}",
                    f"compute/{facts.device.type}@1",
                    "optimizer/adamw@1",
                    "transition/full_root@1",
                ),
            ),
            math=MathContract(
                parameter_gradients=(state.declaration.resource,),
                detached_inputs=("batch",),
                forward_rows="batch",
                parameter_gradient_rows="batch",
                update_states="state",
                reduction="custom",
            ),
        )
        publication = b.publish(update, state, when=owner.phase.value, schema=f"{owner.name}.result@1")
        b.overlay(
            state,
            update,
            publication,
            tuple(
                ParameterOverlayFieldBinding(
                    target.target,
                    ParameterOverlayOperator.LOW_RANK_ADD,
                    (state.field(f"factor_{2 * index}.parameters"), state.field(f"factor_{2 * index + 1}.parameters")),
                    str(state.declaration.resource),
                    model.scale,
                )
                for index, target in enumerate(model.targets)
            ),
        )

        def initialize(roots: Roots, keys: tuple[RequestStateKey | None, ...]) -> None:
            (root,) = roots
            for index, factor in enumerate(model.initial_factors):
                root[5 * index].copy_(factor)

        prepare_data, output_data = freeze(owner.prepare), freeze(owner.output)
        limit = next((item for item in facts.timeline.task_resources if item.task == update.symbol), None)
        if limit is not None and (
            len(limit.input_shapes) != 1
            or len(limit.input_shapes[0]) != 2
            or limit.input_shapes[0][0] < 1
            or limit.input_shapes[0][1] < 2
        ):
            raise ValueError("causal training costs require signed [sequences, tokens] input bounds")

        def prepare(value: CausalTrainBatch[torch.Tensor] | TensorStepInput, batch: TensorBatch) -> ContextT:
            if isinstance(value, TensorStepInput):
                if value.rows.plan_epoch != batch.epoch:
                    raise ValueError("causal input differs from the signed epoch")
                if value.auxiliary is None:
                    return prepare_data(None)
                if not isinstance(value.auxiliary, CausalTrainBatch):
                    raise ValueError("causal adapter requires causal training tensors")
                value = value.auxiliary
            if (
                value.input_ids.ndim != 2
                or value.input_ids.shape != value.labels.shape
                or value.input_ids.shape[0] < 1
                or value.input_ids.shape[1] < 2
                or value.input_ids.dtype != torch.int64
                or value.labels.dtype != torch.int64
                or value.input_ids.device != model.device
                or value.labels.device != model.device
                or value.attention_mask is not None
                and (value.attention_mask.shape != value.input_ids.shape or value.attention_mask.device != model.device)
            ):
                raise ValueError("causal batch must contain aligned token/label tensors")
            if limit is not None and any(
                size > bound for size, bound in zip(value.input_ids.shape, limit.input_shapes[0], strict=True)
            ):
                raise ValueError("causal auxiliary input exceeds its signed input bounds")
            return prepare_data(value)

        def rows(value: CausalTrainBatch[torch.Tensor] | TensorStepInput) -> tuple[PackedPhaseRow, ...]:
            if isinstance(value, TensorStepInput) and value.rows.total_rows > facts.probe.max_packed_rows:
                raise ValueError("packed input exceeds the signed backend row capacity")
            return ()

        return b, FunctionMath({str(state.declaration.resource): initialize}, rows, prepare, output_data)

    def declare(self, facts: BackendCompileFacts) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        builder, _ = self._binding(facts)
        return builder.finish()

    def bind_math(
        self, facts: BackendCompileFacts, plan: PlanBundle
    ) -> MathBinding[TensorStepInput, ContextT, ResultT]:
        if self.model.device != facts.device:
            raise ValueError("causal mathematics require the materialized model on the execution device")
        builder, functions = self._binding(facts)
        if builder.graph != plan.semantic_graph or builder.layouts != plan.tensor_layout_request:
            raise ValueError("causal mathematics differ from the original signed declaration")
        return MathBinding(functions, builder.bind_all(plan))
