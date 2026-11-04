"""Row tasks: per-row training, gradients, sampler guidance and tensor exports.

Module functions declare packed-row tasks on the TensorBinding supplied by workload authoring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from math import prod
from typing import TYPE_CHECKING

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.host_batch import WeightedCEBatch
from pllm.contracts.parameter_overlay import FieldView
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    MathContract,
    PackedSamplerRowsInputSource,
    SamplerFeedbackInputSource,
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
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.binding_base import SGD
from pllm.execution.declaration import Call, Dimensions, LayoutField, Shape, State, When, native
from pllm.execution.ptts.page_borrow import borrowed_root
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.tensor_frame import (
    TensorFrame,
    TensorMath,
    TensorPort,
    Tensors,
    feedback_port,
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


@dataclass(frozen=True, slots=True)
class WeightedBatch:
    """One step's prepared weighted CE batch: the selected states, their reset parameters and the masked inputs.

    `valid_rows` stays on the device; reading it would sync the host.
    """

    selected: tuple[int, ...]
    parameters: tuple[torch.Tensor, ...]
    inputs: tuple[torch.Tensor, ...]
    valid_rows: torch.Tensor


@torch.no_grad()
def weighted_training_output(
    weighted: WeightedBatch,
    data: TensorFrame,
    batch: TensorBatch,
    losses: torch.Tensor,
    gradients: tuple[torch.Tensor, ...],
    fields: tuple[str, ...],
    steps: int,
) -> TaskOutput:
    for index, state in enumerate(weighted.selected):
        for target, value in zip(batch.roots[state], weighted.parameters, strict=True):
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
            for index, state in enumerate(weighted.selected)
            for ref in (batch.selection.refs[state],)
            for name, value in (
                ("loss", losses[index]),
                *zip((f"{field}.grad" for field in fields), data.parameter_gradient_fields[ref], strict=True),
            )
        )
        if batch.observing
        else ()
    )
    data.weighted = None
    return TaskOutput(
        loss=losses,
        loss_count=len(weighted.selected) * steps,
        update_count=len(weighted.selected),
        counters=torch.stack((torch.zeros_like(weighted.valid_rows), weighted.valid_rows)),
        changed=weighted.selected,
        observations=observations,
    )


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
    fields: tuple[str, ...],
    reset: Callable[[Tensors, torch.Generator], None],
    data: TensorFrame,
    view: TensorBatch,
) -> TaskOutput:
    """Mask invalid rows of the generate-validated batch and initialize the selected parameters."""
    match data.step.batch:
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
    valid = (labels >= 0) & (labels < dimensions.vocab_size) & (weights > 0)
    hidden, logits = (tensor.masked_fill(~valid.unsqueeze(-1), 0) for tensor in (hidden, logits))
    labels, weights = (tensor.masked_fill(~valid, 0) for tensor in (labels, weights))
    hidden, logits, weights = (
        tensor.to(torch.promote_types(tensor.dtype, torch.float32)) for tensor in (hidden, logits, weights)
    )
    parameters = tuple(
        tensor.new_empty((len(selected), *tensor.shape), dtype=torch.promote_types(tensor.dtype, torch.float32))
        for tensor in root
    )
    data.weighted = WeightedBatch(
        selected,
        parameters,
        tuple(tensor.clone() if torch.is_inference(tensor) else tensor for tensor in (hidden, logits, labels, weights)),
        valid.sum(),
    )
    with torch.no_grad():
        reset(
            dict(zip(fields, parameters, strict=True)),
            torch.Generator(device="cpu" if root[0].is_meta else root[0].device),
        )
    return TaskOutput()


def sampled_token_port() -> TensorPort:
    """Read sampled token feedback for a tensor task."""
    return feedback_port("sampled_tokens")


def declare_parameters(
    b: TensorBinding,
    name: str,
    fields: tuple[LayoutField, ...],
    *,
    optimizer: AdamWConfig | SGD | None,
    initialize: Callable[[Tensors, torch.Generator], None] | None = None,
    scope: StateScope = StateScope.INSTANCE,
    keys: tuple[RequestKeyField, ...] = (RequestKeyField.REQUEST_ID,),
    carry: tuple[LayoutField, ...] = (),
) -> tuple[State, dict[str, TensorPort]]:
    """Allocate a state and expose a tensor port per field.

    Without AdamW the fields are the layout directly: readable, importable, updated only by their effects. AdamW
    packs them into one flat parameters field that its optimizer fields mirror.
    """
    shapes = {field.name: field.shape for field in fields}
    if isinstance(optimizer, AdamWConfig):
        first = fields[0]
        dtype = first.dtype(b.dimensions) if callable(first.dtype) else first.dtype

        def size(model: Dimensions) -> tuple[int]:
            return (sum(prod(shape(model) if callable(shape) else shape) for shape in shapes.values()),)

        fields = (LayoutField("parameters", size, dtype, first.alignment),)
    state = b.state(
        name,
        fields,
        optimizer=optimizer if isinstance(optimizer, AdamWConfig) else None,
        scope=scope,
        keys=keys if scope is StateScope.REQUEST else (),
        carry=carry,
    )
    return state, parameter_ports(
        b, state, shapes, initialize=initialize, carry={field.name: field.shape for field in carry}
    )


def parameter_ports(
    b: TensorBinding,
    state: State,
    shapes: Mapping[str, Shape],
    *,
    initialize: Callable[[Tensors, torch.Generator], None] | None = None,
    carry: Mapping[str, Shape] = {},
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
    # Carried fields follow the parameter (or AdamW) fields.
    base = len(state.fields) - len(carry)
    for index, (key, shape) in enumerate(carry.items()):
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

            def read(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
                table = data.table.for_resource(state.declaration.resource)
                field = table.leaf if view.field == 0 and table.leaf is not None else table.fields[view.field]
                return view.read(field, pooled=True).index_select(0, table.index.states)

            return TensorPort(state.read(), read, field=view.field)

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
    function: Callable[..., torch.Tensor],
    arguments: Mapping[str, tuple[str, str]],
) -> Call:
    """Declare weighted batch preparation and its optimizer transition."""
    if steps < 1:
        raise ValueError(f"@pllm.train(steps={steps}): use at least one step")
    if b.host:
        raise ValueError("SGD batch training needs a model backend; host workloads train a pllm.AdamW state")
    fields = tuple(field.name for field in state.fields)
    application = PoolSGD(optimizer.lr)
    prepare_function = partial(prepare_weighted_batch, b.dimensions, fields, reset)
    prepared = b.compute(
        f"{name}.prepare",
        inputs={"state": state.read(), "batch": b.input("training", "batch", WeightedCEBatch.schema)},
        outputs=("batch",),
        when=When.BEFORE_SAMPLE,
        function=prepare_function,
        operation_type=BatchPreparation,
        implementations=(native(f"{name}:weighted_batch:prepare@1"),),
    )

    def execute(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        weighted = data.weighted
        if weighted is None:
            return TaskOutput()
        with torch.enable_grad():
            parameters = {
                key: tensor.detach().requires_grad_() for key, tensor in zip(fields, weighted.parameters, strict=True)
            }
            external: dict[str, torch.Tensor] = dict(
                zip(("hidden", "base_logits", "labels", "weights"), weighted.inputs, strict=True)
            )
            inputs = {
                key: (parameters if kind == "parameter" else external)[field]
                for key, (kind, field) in arguments.items()
            }

            def step() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
                losses = function(**inputs)
                gradients = (
                    torch.autograd.grad(
                        losses.to(torch.promote_types(losses.dtype, torch.float32)).mean(),
                        tuple(parameters.values()),
                        materialize_grads=True,
                    )
                    if losses.requires_grad
                    else tuple(torch.zeros_like(parameter) for parameter in parameters.values())
                )
                application.update((tuple(parameters.values()),), batch.field_slots, gradients)
                return losses.detach(), gradients

            losses, gradients = step()
            for _ in range(steps - 1):
                losses, gradients = step()
        return weighted_training_output(weighted, data, batch, losses, gradients, fields, steps)

    transition = b.compute(
        name,
        inputs={"state": state.read(), "batch": TaskResultInputSource(prepared.symbol, "batch")},
        outputs=("publication",),
        result_schema="page_publication@1",
        changes=state,
        when=When.AFTER_SAMPLE,
        math=MathContract(
            parameter_gradients=(state.declaration.resource,),
            detached_inputs=("batch",),
            forward_rows="batch",
            parameter_gradient_rows="batch",
            update_rows="state",
        ),
        function=execute,
        operation_type=BatchOperation,
        select=lambda data, selection: () if data.weighted is None else data.weighted.selected,
        implementations=(native(identity),),
    )
    b.operations[transition.symbol] = replace(
        b.operations[transition.symbol], mode="transition", copy_fields=(), publish=True
    )
    return transition


def parameter_rows(port: TensorPort) -> TensorPort:
    """Expand shared parameters across the invocation rows."""
    if port.rowwise:
        return port

    def read(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
        value = port.read(data, batch)
        return value.unsqueeze(0).expand(len(data.rows), *value.shape)

    return replace(port, read=read, rowwise=True)


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
        index = data.table.for_resource(state.declaration.resource).index
        return index.versions if version else index.states

    return TensorPort(replace(state.read(), metadata=True), read)


def declare_tensor(
    b: TensorBinding,
    name: str,
    *,
    inputs: Mapping[str, TensorPort],
    when: When = When.BEFORE_SAMPLE,
    identity: str = "",
    update: State | None = None,
    math: MathContract | None = None,
    outputs: tuple[str, ...] = ("value",),
    tuple_output: bool = False,
    output_kind: SemanticOutputKind = SemanticOutputKind.TEMPORARY,
    result_schema: str = "",
    effect: StateEffectKind = StateEffectKind.UPDATE_MAYBE,
    carries: Mapping[str, LayoutFieldId] = {},
    export: TensorExport | None = None,
    sampler_stage: SamplerStage | None = None,
    sampler_beta: float = 0.0,
) -> Callable[[TensorMath], Call]:
    """Declare a tensor task with its final inputs, effects and output adapter."""
    sources = {key: value.source for key, value in inputs.items()}

    def phases(source: SemanticInputSource) -> set[HiddenPhase]:
        if isinstance(source, HiddenInputSource):
            # Layer logits cover every row the sampler draws, of either phase.
            return set() if source.phase is HiddenPhase.NONE else {source.phase}
        return set[HiddenPhase]().union(
            *(
                phases(port.source)
                for task in b.graph.tasks
                if isinstance(source, TaskResultInputSource) and task.symbol == source.task
                for port in task.inputs
            )
        )

    domains = set[HiddenPhase]().union(*(phases(source) for source in sources.values()))
    if len(domains) > 1 or domains and when is When.BEFORE_FORWARD:
        raise ValueError("tensor inputs need one phase; hidden inputs become available after model forward")
    phase = next(iter(domains), None)
    sources = {
        key: replace(source, phase=phase or HiddenPhase.DECODE)
        if isinstance(source, StateSelectionInputSource)
        else source
        for key, source in sources.items()
    }
    if carries and update is None:
        raise ValueError("pllm.carry needs the method to train its own state")
    if any(key not in sources for key in carries):
        raise ValueError("pllm.carry must wrap a declared tensor input")
    if update is not None and not any(
        isinstance(source, StateSelectionInputSource) and source.resource == update.declaration.resource
        for source in sources.values()
    ):
        if "__update_state" in sources:
            raise ValueError("__update_state is reserved for the optimizer's state selection")
        sources["__update_state"] = update.read()
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
        if policy.parameter_gradient_rows and any(
            not port.rowwise
            for port in inputs.values()
            if isinstance(port.source, StateSelectionInputSource) and port.source.resource in policy.parameter_gradients
        ):
            raise ValueError("parameter gradient rows require Parameter(..., per_row=True) for shared parameters")
        task = TensorTask(
            symbol,
            function,
            inputs,
            sources,
            policy,
            outputs,
            carries,
            phase,
            update.declaration.resource
            if update is not None
            else next(
                (source.resource for source in sources.values() if isinstance(source, StateSelectionInputSource)),
                StateResource(""),
            ),
            tuple_output=tuple_output,
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
            result_schema="tensor_training_result@1" if update is not None else result_schema,
            when=When.HOST if b.host else when,
            reads=reads,
            implementations=(native(identity or f"{b.graph.workload_id}:{name}@1"),),
        )

    return decorate


def validate_gradients(b: TensorBinding, trainers: tuple[Call, ...]) -> None:
    """Check that every differentiable path ends in a @pllm.train loss and each loss differentiates only its state."""
    tasks = {task.symbol: task for task in b.graph.tasks}
    retained = {
        task.symbol
        for task in tasks.values()
        if task.math is not None and (task.math.input_gradients or task.math.parameter_gradients)
    }
    if retained and not trainers:
        raise ValueError("differentiable operations require an explicit training sink")
    reached: set[TaskSymbol] = set()
    for training in trainers:
        visited: set[TaskSymbol] = set()
        resources: set[str] = set()
        pending = [training.symbol]
        while pending:
            symbol = pending.pop()
            if symbol in visited:
                continue
            visited.add(symbol)
            math = tasks[symbol].math
            if math is None:
                continue
            resources.update(math.parameter_gradients)
            pending.extend(
                port.source.task
                for port in tasks[symbol].inputs
                if port.name in math.input_gradients and isinstance(port.source, TaskResultInputSource)
            )
        reached |= visited
        if resources != {
            effect.resource for effect in tasks[training.symbol].effects if effect.kind is not StateEffectKind.READ
        }:
            raise ValueError(f"the @pllm.train loss of {training.symbol} must differentiate only its own state")
    if retained - reached:
        raise ValueError("every @pllm.differentiate result must feed a @pllm.train loss")


def declare_logits(b: TensorBinding, prediction: Call, *, stage: SamplerStage) -> None:
    """Send the prediction to the sampler: added to the logits before filtering, or guiding the draw after it."""
    if stage is SamplerStage.PACKED_PRE_FILTER_ADD and b.dimensions.vocab_size < 1:
        raise ValueError("@pllm.add_logits needs the model's vocabulary size; compile this workload with make_llm")
    for task in b.tensor_tasks[prediction.symbol].dependencies(b.tensor_tasks):
        for name, port in task.ports.items():
            if isinstance(port.source, SamplerFeedbackInputSource):
                raise ValueError(
                    f"sampler method `{prediction.symbol}` reads pllm.{port.row_input}() through `{task.symbol}.{name}`, "
                    "which is available only in Phase.AFTER_SAMPLE. Read feedback in @pllm.train or a read-only "
                    "helper feeding it beside @pllm.add_logits; before sampling, use pllm.layer_logits((-1,)) "
                    "for logits or pllm.final_hidden() for hidden states."
                )
    b.sampler(prediction, stage)
