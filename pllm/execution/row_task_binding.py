"""Row tasks: per-row training, gradients, sampler guidance and tensor exports.

Module functions declare packed-row tasks on the TensorBinding supplied by workload authoring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from math import prod
from typing import TYPE_CHECKING

import torch

from pllm.contracts.auxiliary import WeightedCEBatch
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import FieldView
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    MathContract,
    PackedSamplerRowsInputSource,
    SemanticEffect,
    SemanticInput,
    SemanticInputSource,
    SemanticOutputKind,
    StateEffectKind,
    StateResource,
    StateScope,
    StateSelectionInputSource,
    TaskResultInputSource,
    TaskSymbol,
)
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType
from pllm.execution.binding_base import SGD
from pllm.execution.declaration import Call, Dimensions, Field, Shape, State, When, native
from pllm.execution.ptts.page_borrow import borrowed_root
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.tensor_frame import (
    TensorFrame,
    TensorMath,
    TensorPort,
    Tensors,
    feedback_port,
    field_rows,
    input_port,
    row_index,
)
from pllm.execution.tensor_program import (
    BatchOperation,
    BatchPreparation,
    Observation,
    PoolSGD,
    TaskOutput,
    TensorBatch,
)
from pllm.execution.tensor_task import TensorExport, TensorTask
from pllm.kernels.adamw import AdamWConfig

if TYPE_CHECKING:
    from pllm.execution.tensor_binding import TensorBinding


@torch.no_grad()
def weighted_training_output(
    data: TensorFrame,
    batch: TensorBatch,
    losses: torch.Tensor,
    gradients: tuple[torch.Tensor, ...],
    fields: tuple[str, ...],
    steps: int,
    parameters: tuple[torch.Tensor, ...] = (),
) -> TaskOutput:
    for index, state in enumerate(data.selected):
        if parameters:
            for target, value in zip(batch.roots[state], parameters, strict=True):
                target.copy_(value[index])
        data.parameter_gradient_fields[batch.selection.refs[state]] = tuple(value[index] for value in gradients)
    observations = (
        tuple(
            Observation(
                name,
                value,
                ref=ref,
                iteration=steps - 1,
                identities=tuple(
                    row.identity
                    for row, position in zip(batch.selection.rows, batch.selection.state_indices, strict=True)
                    if position == state
                ),
                gradient=name.endswith(".grad"),
            )
            for index, state in enumerate(data.selected)
            for ref in (batch.selection.refs[state],)
            for name, value in (
                ("loss", losses[index]),
                *zip((f"{field}.grad" for field in fields), data.parameter_gradient_fields[ref], strict=True),
            )
        )
        if batch.observing
        else ()
    )
    data.batch_inputs = data.parameters = ()
    return TaskOutput(
        loss=losses,
        loss_count=len(data.selected) * steps,
        update_count=len(data.selected),
        trained_rows=data.trained_rows,
        changed=data.selected,
        observations=observations,
    )


def weighted_rows(valid: torch.Tensor) -> tuple[int, bool]:
    """Count training rows for accounting; meta traces the partial-batch masking path."""
    counts = valid.sum(-1)
    if valid.is_meta:
        return valid.numel(), True
    count = int(counts.sum().item())
    return count, count != valid.numel()


def weighted_selection(batch: WeightedCEBatch[torch.Tensor], selection: RequestStateSelection) -> tuple[int, ...]:
    """Resolve imported prompt keys to distinct co-resident state roots."""
    shared = selection.keys == (None,)
    fields = batch.keys[0].fields if batch.keys else ()
    row_states: dict[RequestStateKey, int] = {}
    if not shared and batch.keys:
        for row, index in zip(selection.rows, selection.state_indices, strict=True):
            row_states.setdefault(RequestStateKey.project(row.identity, fields), index)
    selected = (0,) if shared else tuple(row_states[key] for key in batch.keys if key in row_states)
    if (shared and batch.keys) or (
        not shared
        and (
            not batch.keys
            or any(key.fields != fields for key in batch.keys)
            or len(selected) != len(batch.keys)
            or len(set(selected)) != len(selected)
        )
    ):
        raise ValueError("use distinct co-resident request keys for weighted CE, or no keys for shared state")
    return selected


def prepare_weighted_batch(
    dimensions: Dimensions,
    shape: tuple[int, ...] | None,
    fields: tuple[str, ...],
    reset: Callable[[Tensors, torch.Generator], None],
    data: TensorFrame,
    view: TensorBatch,
) -> TaskOutput:
    """Validate imported weighted-CE metadata and initialize the selected parameters."""
    match data.step.auxiliary:
        case WeightedCEBatch() as batch:
            pass
        case _:
            return TaskOutput()
    selected = weighted_selection(batch, view.selection)
    tensors = (batch.hidden, batch.base_logits, batch.labels, batch.weights)
    hidden, logits, labels, weights = (
        tensor.detach().unsqueeze(0) if view.selection.keys == (None,) else tensor.detach() for tensor in tensors
    )
    root = view.roots[selected[0]]
    if (
        hidden.ndim != 3
        or hidden.shape[0] != len(selected)
        or hidden.shape[2] != dimensions.hidden_size
        or logits.shape != (*hidden.shape[:2], dimensions.vocab_size)
        or labels.shape != hidden.shape[:2]
        or weights.shape != labels.shape
        or labels.dtype != torch.long
        or any(tensor.device != root[0].device for tensor in tensors)
        or any(tensor.dtype != root[0].dtype for tensor in (hidden, logits, weights))
        or shape is not None
        and any(actual > bound for actual, bound in zip(hidden.shape, shape, strict=True))
    ):
        raise ValueError("align weighted-CE tensors with the parameter shape, dtype, device and input bounds")
    valid = (labels >= 0) & (labels < dimensions.vocab_size) & (weights > 0)
    data.selected = selected
    data.trained_rows, incomplete = weighted_rows(valid)
    if incomplete:
        hidden, logits = (tensor.masked_fill(~valid.unsqueeze(-1), 0) for tensor in (hidden, logits))
        labels, weights = (tensor.masked_fill(~valid, 0) for tensor in (labels, weights))
    data.batch_inputs = tuple(
        tensor.clone() if torch.is_inference(tensor) else tensor for tensor in (hidden, logits, labels, weights)
    )
    data.parameters = tuple(tensor.new_empty((len(selected), *tensor.shape)) for tensor in root)
    with torch.no_grad():
        reset(
            dict(zip(fields, data.parameters, strict=True)),
            torch.Generator(device="cpu" if root[0].is_meta else root[0].device),
        )
    return TaskOutput()


def sampled_token_port() -> TensorPort:
    """Read sampled token feedback for a tensor task."""
    return feedback_port("sampled_tokens")


def declare_parameters(
    b: TensorBinding,
    name: str,
    shapes: Mapping[str, Shape],
    *,
    optimizer: AdamWConfig | SGD | None,
    namespace: str = "",
    initialize: Callable[[Tensors, torch.Generator], None] | None = None,
    scope: StateScope = StateScope.INSTANCE,
    dtype: TensorDType = TensorDType.FLOAT32,
    alignment: int = 16,
    keys: tuple[RequestKeyField, ...] = (RequestKeyField.REQUEST_ID,),
    memory: Mapping[str, Shape] = {},
) -> tuple[State, dict[str, TensorPort]]:
    """Allocate parameter state and expose its named tensor ports."""

    def sizes(model: Dimensions) -> tuple[tuple[int, ...], ...]:
        return tuple(shape(model) if callable(shape) else shape for shape in shapes.values())

    # Without an optimizer the fields are the layout directly: readable, importable, never updated.
    raw = not isinstance(optimizer, AdamWConfig)
    memory_fields = tuple(Field(key, shape, dtype, alignment) for key, shape in memory.items())
    state = b.state(
        name,
        tuple(Field(key, shape, dtype, alignment) for key, shape in shapes.items())
        if raw
        else (Field("parameters", lambda m: (sum(prod(shape) for shape in sizes(m)),), dtype, alignment),),
        namespace=namespace,
        optimizer=optimizer if isinstance(optimizer, AdamWConfig) else None,
        scope=scope,
        keys=keys if scope is StateScope.REQUEST else (),
        memory=memory_fields,
    )
    return state, parameter_ports(b, state, shapes, initialize=initialize, memory=memory)


def parameter_ports(
    b: TensorBinding,
    state: State,
    shapes: Mapping[str, Shape],
    *,
    initialize: Callable[[Tensors, torch.Generator], None] | None = None,
    memory: Mapping[str, Shape] = {},
) -> dict[str, TensorPort]:
    """Bind named views of an allocated state without creating another root."""
    dimensions = b.dimensions
    raw = state.optimizer is None
    sizes = tuple(shape(dimensions) if callable(shape) else shape for shape in shapes.values())
    views: dict[str, FieldView] = {}
    offset = 0
    for index, (key, shape) in enumerate(zip(shapes, sizes, strict=True)):
        views[key] = FieldView(LayoutFieldId(index if raw else 0), 0 if raw else offset, shape)
        offset += prod(shape)
    base = len(shapes) if raw else 5
    for index, (key, shape) in enumerate(memory.items()):
        views[key] = FieldView(LayoutFieldId(base + index), 0, shape(dimensions) if callable(shape) else shape)
    b.field_views.update({(str(state.declaration.resource), key): view for key, view in views.items()})

    if initialize is not None:

        def init(root: tuple[torch.Tensor, ...], generator: torch.Generator) -> None:
            initialize({key: view.read(root[view.field]) for key, view in views.items()}, generator)

        b.initializers[str(state.declaration.resource)] = init

    def port(key: str) -> TensorPort:
        view = views[key]
        return TensorPort(
            state.read(),
            lambda data, batch: view.read(batch.parameter_root(str(state.declaration.resource))[view.field]),
            rowwise=False,
            field=view.field,
        )

    if state.declaration.scope is StateScope.REQUEST:

        def request_port(key: str) -> TensorPort:
            view = views[key]
            field_index, start, shape = view.field, view.offset, view.shape

            def read(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
                selected = batch.for_resource(state.declaration.resource)
                data = data.for_resource(state.declaration.resource, batch)
                batch = selected
                size = prod(shape)
                if field_index == 0 and data.leaf is not None:
                    leaf = data.leaf.narrow(1, start or 0, size)
                    rows = row_index(data, batch, leaf.device).states.take(leaf)
                else:
                    rows = field_rows(data, batch, field_index, start or 0, size)
                return rows.view(len(rows), *shape)

            return TensorPort(state.read(), read, field=field_index)

        return {key: request_port(key) for key in views}

    return {key: port(key) for key in views}


def declare_weighted_train(
    b: TensorBinding,
    name: str,
    state: State,
    *,
    identity: str,
    optimizer: SGD,
    steps: int,
    reset: Callable[[Tensors, torch.Generator], None],
    function: Callable[[Tensors], torch.Tensor],
    arguments: Mapping[str, tuple[str, str]],
) -> Call:
    """Declare weighted batch preparation and its optimizer transition."""
    if steps < 1 or b.weighted_training or b.host:
        raise ValueError("weighted training requires one model training sink and positive steps")
    b.weighted_training = True
    fields = tuple(field.name for field in state.fields)
    application = PoolSGD(optimizer.lr, len(fields))
    dimensions = b.dimensions
    limit = next((fact for fact in b.resources if fact.task == name), None)
    # Resource derivation builds the graph before the batch bound exists; execution always has it.
    shape = None if limit is None or not limit.input_shapes else limit.input_shapes[0]
    prepare_function = partial(prepare_weighted_batch, dimensions, shape, fields, reset)
    prepared = b.compute(
        f"__{name}_prepare",
        inputs={"state": state.read(), "batch": b.input("training", "batch", WeightedCEBatch.schema)},
        outputs=("batch",),
        when=When.BEFORE_SAMPLE,
        function=prepare_function,
        operation_type=BatchPreparation,
        implementations=(native("canonical.weighted_batch:prepare@1"),),
    )

    def execute(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        if not data.selected:
            return TaskOutput()
        with torch.enable_grad():
            parameters = {
                key: tensor.detach().requires_grad_() for key, tensor in zip(fields, data.parameters, strict=True)
            }
            external: dict[str, torch.Tensor] = dict(
                zip(("hidden", "base_logits", "labels", "weights"), data.batch_inputs, strict=True)
            )
            inputs = {
                key: (parameters if kind == "parameter" else external)[field]
                for key, (kind, field) in arguments.items()
            }

            def step() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
                losses = function(inputs)
                gradients = (
                    torch.autograd.grad(losses.mean(), tuple(parameters.values()), materialize_grads=True)
                    if losses.requires_grad
                    else tuple(torch.zeros_like(parameter) for parameter in parameters.values())
                )
                application.update((tuple(parameters.values()),), batch.field_slots, gradients)
                return losses.detach(), gradients

            losses, gradients = step()
            for _ in range(steps - 1):
                losses, gradients = step()
        return weighted_training_output(data, batch, losses, gradients, fields, steps, data.parameters)

    transition = b.compute(
        name,
        inputs={"state": state.read(), "batch": TaskResultInputSource(prepared.symbol, "batch")},
        outputs=("publication",),
        result_schema="canonical.page_publication@1",
        changes=state,
        when=When.AFTER_SAMPLE,
        math=MathContract(
            parameter_gradients=(state.declaration.resource,),
            detached_inputs=("batch",),
            forward_rows="batch",
            parameter_gradient_rows="batch",
            update_states="state",
            reduction="weighted_state_mean_then_mean",
        ),
        function=execute,
        operation_type=BatchOperation,
        select=lambda data, selection: data.selected,
        implementations=(native(identity),),
    )
    b.operations[transition.symbol] = replace(
        b.operations[transition.symbol], mode="transition", copy_fields=(), publish=True
    )
    return transition


def parameter_rows(source: SemanticInputSource | TensorPort) -> TensorPort:
    """Expand shared parameters across the invocation rows."""
    port = input_port(source)
    if port.rowwise:
        return port

    def read(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
        value = port.read(data, batch)
        return value.unsqueeze(0).expand(len(data.rows), *value.shape)

    return TensorPort(port.source, read, field=port.field)


def state_metadata(state: State, *, version: bool) -> TensorPort:
    """Read state indices or versions, including borrowed instance roots."""

    def read(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
        # Remote instance metadata never indexes local refs; test_video covers colliding lineage IDs.
        borrowed = borrowed_root(batch.borrowed_roots, str(state.declaration.resource))
        if borrowed is not None:
            tensors = batch.parameter_root(str(state.declaration.resource))
            return torch.full(
                (len(data.rows),),
                int(borrowed.ref.version_id) if version else 0,
                device=tensors[0].device,
                dtype=torch.long,
            )
        selected = batch.for_resource(state.declaration.resource)
        frame = data.for_resource(state.declaration.resource, batch)
        index = row_index(frame, selected, selected.roots[0][0].device)
        return index.versions.device if version else index.states.device

    return TensorPort(state.read(), read)


def declare_tensor(
    b: TensorBinding,
    name: str,
    *,
    inputs: Mapping[str, SemanticInputSource | TensorPort],
    when: When = When.BEFORE_SAMPLE,
    identity: str = "",
    update: State | None = None,
    math: MathContract | None = None,
    outputs: tuple[str, ...] = ("value",),
    output_kind: SemanticOutputKind = SemanticOutputKind.TEMPORARY,
    result_schema: str = "",
    effect: StateEffectKind = StateEffectKind.UPDATE_MAYBE,
    captures: Mapping[str, LayoutFieldId] = {},
    export: TensorExport | None = None,
    sampler_stage: SamplerStage | None = None,
    sampler_beta: float = 0.0,
) -> Callable[[TensorMath], Call]:
    """Declare a tensor task with its final inputs, effects and output adapter."""
    sources = {key: value.source if isinstance(value, TensorPort) else value for key, value in inputs.items()}

    def phases(source: SemanticInputSource) -> set[HiddenPhase]:
        if isinstance(source, HiddenInputSource):
            return {source.phase}
        return set[HiddenPhase]().union(
            *(
                phases(port.source)
                for task in b.graph.tasks
                if isinstance(source, TaskResultInputSource) and task.symbol == source.task
                for port in task.inputs
            )
        )

    domains = set[HiddenPhase]().union(*(phases(source) for source in sources.values()))
    if len(domains) > 1 or domains and when is When.BEFORE_MODEL:
        raise ValueError("tensor inputs need one phase; hidden inputs become available after model forward")
    phase = next(iter(domains), None)
    sources = {
        key: replace(source, phase=phase or HiddenPhase.DECODE)
        if isinstance(source, StateSelectionInputSource)
        else source
        for key, source in sources.items()
    }
    if captures and update is None:
        raise ValueError("memory captures require the task's own training transition")
    if any(key not in sources for key in captures):
        raise ValueError("memory captures must name a declared tensor input")
    if update is not None and not any(
        isinstance(source, StateSelectionInputSource) and source.resource == update.declaration.resource
        for source in sources.values()
    ):
        if "__update_state" in sources:
            raise ValueError("__update_state is reserved for the optimizer's state selection")
        sources["__update_state"] = update.read()
    if update is not None and update.optimizer is None:
        raise ValueError("training requires an optimizer on its declared state")
    policy = MathContract() if math is None else math
    reads = tuple(
        SemanticEffect(resource, StateEffectKind.READ)
        for resource in dict.fromkeys(
            source.resource
            for source in sources.values()
            if isinstance(source, StateSelectionInputSource)
            and (update is None or source.resource != update.declaration.resource)
        )
    )
    sampler_inputs = (
        (
            SemanticInput("packed_rows", b.rows()),
            *(
                (SemanticInput("__sampler_rows", b.sampler_rows()),)
                if not any(isinstance(source, PackedSamplerRowsInputSource) for source in sources.values())
                else ()
            ),
        )
        if output_kind is SemanticOutputKind.SAMPLER
        else ()
    )

    def decorate(function: TensorMath) -> Call:
        symbol = TaskSymbol(name)
        ports = {key: input_port(value) for key, value in inputs.items()}
        if policy.parameter_gradient_rows and any(
            not port.rowwise
            for port in ports.values()
            if isinstance(port.source, StateSelectionInputSource) and port.source.resource in policy.parameter_gradients
        ):
            raise ValueError("parameter gradient rows require Parameter(..., per_row=True) for shared parameters")
        task = TensorTask(
            symbol,
            function,
            ports,
            sources,
            policy,
            outputs,
            captures,
            phase,
            update.declaration.resource
            if update is not None
            else next(
                (source.resource for source in sources.values() if isinstance(source, StateSelectionInputSource)),
                StateResource(""),
            ),
            export=export,
            sampler_stage=sampler_stage,
            sampler_beta=sampler_beta,
            training=update,
        )
        b.tensor_tasks[symbol] = task
        return b.compute(
            name,
            inputs=sources,
            extra_inputs=sampler_inputs,
            outputs=outputs,
            output_kind=output_kind,
            changes=update,
            effect=effect,
            math=math,
            result_schema="canonical.tensor_training_result@1" if update is not None else result_schema,
            when=When.HOST if b.host else when,
            reads=reads,
            implementations=(native(identity or f"{b.graph.workload_id}:{name}@1"),),
        )

    return decorate


def validate_gradients(b: TensorBinding, training: Call | None) -> None:
    """Check that one training sink consumes all declared differentiable paths."""
    tasks = {task.symbol: task for task in b.graph.tasks}
    retained = {
        task.symbol
        for task in tasks.values()
        if task.math is not None and (task.math.input_gradients or task.math.parameter_gradients)
    }
    if training is None:
        if retained:
            raise ValueError("differentiable operations require an explicit training sink")
        return
    visited: set[TaskSymbol] = set()
    resources: set[str] = set()

    def visit(symbol: TaskSymbol) -> None:
        if symbol in visited:
            return
        visited.add(symbol)
        task = tasks[symbol]
        if task.math is None:
            return
        resources.update(task.math.parameter_gradients)
        for port in task.inputs:
            if port.name in task.math.input_gradients and isinstance(port.source, TaskResultInputSource):
                visit(port.source.task)

    visit(training.symbol)
    if retained - visited or resources != {
        effect.resource for effect in tasks[training.symbol].effects if effect.kind is not StateEffectKind.READ
    }:
        raise ValueError("training must consume differentiable paths for its one optimizer state")


def declare_logits(
    b: TensorBinding,
    prediction: Call,
    training: Call,
    state: State,
    *,
    stage: SamplerStage = SamplerStage.PACKED_PRE_FILTER_ADD,
) -> None:
    """Attach prediction and training tasks to the sampler."""
    if stage is SamplerStage.PACKED_PRE_FILTER_ADD and b.dimensions.vocab_size < 1:
        raise ValueError("logits binding requires the backend vocabulary size")
    b.sampler(prediction, training, state, output=prediction.outputs[0], stage=stage)
