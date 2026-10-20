"""Capture original module calls and cut their FX graph at a resource boundary.

The backend supplies the resource semantics of leaf calls. This adapter does
not infer purity from a module name, execute leaf kernels during capture, or
reimplement their mathematics. Source modules must be trusted for FX tracing.
"""

import operator
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import _dynamo
from torch.export.graph_signature import InputKind, TensorArgument
from torch.fx import _symbolic_trace

from pllm.contracts.region_imports import KernelAllocation
from pllm.execution.codec import (
    boundary_digest,
    check_functional_aten,
    tensor_graph_outputs,
    tensor_graph_result_nodes,
    tensor_node_metadata,
)

_ASSERT_IS_NONE: object = getattr(_symbolic_trace, "_assert_is_none")


class _CaptureConfig(Protocol):
    def patch(self, changes: dict[str, object]) -> AbstractContextManager[None]: ...


class _Reparameterize(Protocol):
    def __call__(
        self, module: torch.nn.Module, parameters: dict[str, torch.Tensor]
    ) -> AbstractContextManager[None]: ...


@dataclass(frozen=True, slots=True)
class ModuleExport:
    """Original exported computation and its local Torch capture conditions."""

    program: torch.export.ExportedProgram
    _check: Callable[[], None]

    def validate(self) -> None:
        self._check()


def capture_module_export(
    module: torch.nn.Module, examples: tuple[torch.Tensor, ...], *, parameters: Mapping[str, torch.Tensor] | None = None
) -> ModuleExport:
    """Retain the graph and executable guards from one original Torch capture.

    Only private input samples and fake tensor execution are used. Lifted fields
    retain their original storage. Python effects outside the tensor graph are
    rejected by Torch. Parameter views use Torch's functional_call substitution
    scope outside tracing, so its bookkeeping is not mistaken for model effects.
    """
    from torch._dynamo import convert_frame
    from torch._dynamo.convert_frame import FrameInfo, fullgraph_capture
    from torch._dynamo.functional_export import ExportTracerOutput
    from torch._dynamo.utils import dynamo_timed, get_metrics_context
    from torch._dynamo.variables.torch_function import torch_function_mode_stack_state_mgr
    from torch._guards import Source
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.nn.utils import stateless
    from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

    bindings = tuple(ModuleCallBinding.bind(child) for child in module.modules())
    samples = tuple(value.detach().clone() for value in examples)
    state = None if parameters is None else dict(parameters)
    reparameterize = cast(_Reparameterize, getattr(stateless, "_reparametrize_module"))
    get_frame = cast(
        Callable[[Callable[..., object], tuple[torch.Tensor, ...]], FrameInfo], getattr(convert_frame, "_get_frame")
    )

    def scope() -> AbstractContextManager[None]:
        return nullcontext() if state is None else reparameterize(module, dict(state))

    def invoke(*values: torch.Tensor) -> object:
        return ExportTracerOutput(*tree_flatten(module(*values)))

    with (
        scope(),
        torch.no_grad(),
        get_metrics_context(),
        dynamo_timed("fullgraph_capture"),
        torch_function_mode_stack_state_mgr,
        cast(_CaptureConfig, _dynamo.config).patch(
            {
                "specialize_int": True,
                "specialize_float": True,
                "assume_static_by_default": True,
                "automatic_dynamic_shapes": False,
                "side_effect_replay_policy": "error",
                "log_graph_in_out_metadata": True,
            }
        ),
    ):
        captured = fullgraph_capture(invoke, samples)
        if captured.backend_input is None:
            raise ValueError("module source has no captured tensor computation")
        manager = captured.graph_capture_output.build_guards(invoke.__code__).guard_manager
        frame = get_frame(invoke, samples)
        metadata = captured.graph_capture_output.output_graph.export_metadata
        output_spec: object = getattr(metadata, "out_spec")
        root = captured.backend_input.graph_module
        graph = torch.fx.Graph()
        inputs = tuple(graph.placeholder(f"arg{i}") for i in range(len(samples)))
        mapped: dict[torch.fx.Node, torch.fx.Node] = {}
        fields: list[tuple[Source, torch.Tensor]] = []
        ordinal = 0
        for node in root.graph.nodes:
            if node.op == "placeholder":
                source = metadata.graph_input_idx_to_local_source[ordinal]
                if source is None:
                    raise ValueError("captured tensor input has no original source")
                value: object = source.get_value({"L": frame.locals, "G": frame.globals}, {}, {})
                user = next((i for i, sample in enumerate(samples) if value is sample), None)
                if user is not None:
                    mapped[node] = inputs[user]
                else:
                    if not isinstance(value, torch.Tensor):
                        raise ValueError("captured module requires static tensor input sources")
                    name = f"_pllm_lifted_{ordinal}"
                    if hasattr(root, name):
                        raise ValueError("captured graph already uses the reserved lifted field name")
                    root.register_buffer(name, value)
                    mapped[node] = graph.get_attr(name)
                    # Snapshot the view, not its contents. set_/data assignment
                    # can otherwise change both sides of a later identity check.
                    fields.append((source, value.detach()))
                ordinal += 1
            elif node.op == "output":
                outputs = torch.fx.map_arg(node.args[0], mapped.__getitem__)
                if not isinstance(outputs, (tuple, list)) or not isinstance(output_spec, TreeSpec):
                    raise ValueError("captured module requires an explicit tensor result structure")
                flat_outputs = cast(tuple[object, ...] | list[object], outputs)
                leaves: list[object] = []
                for i in range(output_spec.num_leaves):
                    info = metadata.output_return_type[i]
                    value: object = info.value
                    if info.kind == "graph_out" and type(value) is int:
                        leaves.append(flat_outputs[value])
                    elif info.kind == "input" and isinstance(value, Source):
                        original: object = value.get_value({"L": frame.locals, "G": frame.globals}, {}, {})
                        user = next((i for i, sample in enumerate(samples) if original is sample), None)
                        if user is None:
                            raise ValueError("captured output does not identify an original input")
                        leaves.append(inputs[user])
                    elif info.kind == "constant":
                        leaves.append(value)
                    else:
                        raise ValueError("captured module has an unsupported output source")
                graph.output(tree_unflatten(leaves, output_spec))
            else:
                mapped[node] = graph.node_copy(node, mapped.__getitem__)
        bound = torch.fx.GraphModule(root, graph)
        # Convert the captured body, rather than recapturing the source method.
        # make_fx turns the original .data access into an explicit ATen alias.
        aten = make_fx(bound, tracing_mode="fake", _allow_non_fake_inputs=True)(*samples)
        exported = torch.export.export(aten, samples, strict=False)

    def check() -> None:
        children = tuple(module.modules())
        if len(children) != len(bindings):
            raise ValueError("original module no longer satisfies its Torch capture guards: module tree changed")
        for binding, child in zip(bindings, children, strict=True):
            binding.validate(child)
        # A tensor layout guard alone permits replacement by a same-shaped
        # tensor. The exported field still refers to the original backing.
        for source, expected in fields:
            current: object = source.get_value({"L": frame.locals, "G": frame.globals}, {}, {})
            if not isinstance(current, torch.Tensor) or not _same_tensor_view(current, expected):
                raise ValueError("original module tensor field changed after capture")
        with scope(), torch.no_grad():
            if not manager.check(frame.locals):
                reason = manager.check_verbose(frame.locals)
                raise ValueError(f"original module no longer satisfies its Torch capture guards: {reason}")

    result = ModuleExport(exported, check)
    result.validate()
    return result


def _same_tensor_view(value: torch.Tensor, field: torch.Tensor) -> bool:
    return (
        value.device == field.device
        and value.dtype == field.dtype
        and value.untyped_storage() == field.untyped_storage()
        and value.storage_offset() == field.storage_offset()
        and tuple(value.shape) == tuple(field.shape)
        and tuple(value.stride()) == tuple(field.stride())
    )


def exported_parameter_fields(
    program: torch.export.ExportedProgram, fields: tuple[torch.Tensor, ...]
) -> dict[str, int]:
    """Resolve lifted targets, including detached aliases, to original fields.

    Only identical storage views qualify. Constants unrelated to the original
    owner cannot silently become parameters or new owned allocations.
    """
    result: dict[str, int] = {}
    for spec in program.graph_signature.input_specs:
        if spec.kind is InputKind.USER_INPUT:
            continue
        if not isinstance(spec.arg, TensorArgument) or spec.target is None:
            raise ValueError("export field binding requires lifted tensor targets")
        value: object = (
            program.state_dict[spec.target]
            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
            else program.constants[spec.target]
            if spec.kind is InputKind.CONSTANT_TENSOR
            else None
        )
        if not isinstance(value, torch.Tensor) or not value.is_contiguous():
            raise ValueError("export field binding requires an original contiguous tensor view")
        candidates = tuple(i for i, field in enumerate(fields) if _same_tensor_view(value, field))
        if not candidates:
            raise ValueError("lifted export tensor is not a field of the original owner")
        result[spec.target] = candidates[0]
    return result


@dataclass(frozen=True, slots=True)
class TensorGraphPartition:
    """An original FX fragment, with tensor frontiers and no bound parameters."""

    graph: torch.fx.GraphModule
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    opaque: bool


def partition_tensor_graph(graph: torch.fx.GraphModule, *, carry: int = 0) -> tuple[TensorGraphPartition, ...]:
    """Keep functional external calls and their tuple projections together.

    This preserves source order and copies FX nodes, not operator mathematics.
    The backend must authorize every opaque fragment before it can be bound.
    Carry dependence separates runs; storage-aliasing views are rebuilt inside
    consumers so only distinct storage roots cross fragment boundaries.
    """
    dependent = {n: i < carry for i, n in enumerate(n for n in graph.graph.nodes if n.op == "placeholder")}
    groups: list[tuple[tuple[bool, bool], list[torch.fx.Node]]] = []
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op != "call_function":
            raise ValueError("tensor source must lift attributes and contain functional calls")
        opaque = tensor_node_metadata(node) is None
        try:
            check_functional_aten(node.target)
        except ValueError:
            opaque = True
        dependent[node] = any(dependent[n] for n in node.all_input_nodes)
        key = opaque, dependent[node]
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        groups[-1][1].append(node)
    storages = {n: t.untyped_storage() for n in graph.graph.nodes if (t := tensor_node_metadata(n)) is not None}
    views = {
        n
        for (opaque, _), nodes in groups
        if not opaque
        for n in nodes
        if any(storages[n] == storages.get(arg) for arg in n.all_input_nodes)
    }

    def roots(node: torch.fx.Node) -> tuple[torch.fx.Node, ...]:
        return tuple(root for arg in node.all_input_nodes for root in roots(arg)) if node in views else (node,)

    # Opaque kernels retain their audited ABI; only tensor consumers rebuild views.
    frontiers = [
        tuple(
            dict.fromkeys(
                root
                for n in nodes
                for arg in n.all_input_nodes
                for root in ((arg,) if opaque else roots(arg))
                if root not in nodes
            )
        )
        for (opaque, _), nodes in groups
    ]
    escaping = {n for inputs in frontiers for n in inputs} | set(tensor_graph_result_nodes(graph))
    result: list[TensorGraphPartition] = []
    for ((opaque, _), nodes), inputs in zip(groups, frontiers, strict=True):
        outputs = tuple(node for node in nodes if node in escaping)
        if any(tensor_node_metadata(node) is None for node in (*inputs, *outputs)):
            raise ValueError("tensor partition must retain tuple-valued calls with their projections")
        if not outputs:
            if opaque:
                raise ValueError("effect-only tensor fragments require an explicit resource contract")
            continue
        fragment = torch.fx.Graph()
        mapped: dict[torch.fx.Node, torch.fx.Node] = {}
        for node in inputs:
            copied = fragment.placeholder(node.name)
            copied.meta = node.meta.copy()
            mapped[node] = copied

        def read(node: torch.fx.Node) -> torch.fx.Node:
            return mapped[node] if node in mapped else fragment.node_copy(node, read)

        for node in nodes:
            mapped[node] = fragment.node_copy(node, read)
        fragment.output(tuple(mapped[node] for node in outputs))
        result.append(
            TensorGraphPartition(
                torch.fx.GraphModule({}, fragment),
                tuple(node.name for node in inputs),
                tuple(node.name for node in outputs),
                opaque,
            )
        )
    return tuple(result)


def functionalized_kernel_operators(graph: torch.fx.GraphModule) -> tuple[str, ...]:
    """Read original kernel identities from Torch's functionalization wrapper."""
    from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2
    from torch._ops import OpOverload

    operators: list[str] = []
    wrapped: set[torch.fx.Node] = set()
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op == "call_function" and node.target is auto_functionalized_v2:
            # Inductor may later elide functionalization clones using its own
            # liveness proof. An imported fragment borrows its inputs and has
            # no such proof, even if fake execution reports fresh outputs.
            if len(node.args) != 1 or node.kwargs.get("_only_clone_these_bases") is not None:
                raise ValueError("functionalized kernel requires the original clone-all policy")
            target: object = node.args[0]
            if not isinstance(target, OpOverload) or torch.Tag.nondeterministic_seeded in target.tags:
                raise ValueError("functionalized kernel requires an original deterministic operator")
            operators.append(str(cast(object, target)))
            wrapped.add(node)
        elif (
            node.op == "call_function"
            and node.target is operator.getitem
            and len(node.args) == 2
            and isinstance(node.args[0], torch.fx.Node)
            and node.args[0] in wrapped
            and type(node.args[1]) is int
        ):
            continue
        else:
            raise ValueError("opaque fragment contains an unaudited operation")
    if not operators:
        raise ValueError("opaque fragment contains no functionalized kernel")
    return tuple(operators)


def tensor_fragment_allocations(graph: torch.fx.GraphModule) -> tuple[KernelAllocation, ...]:
    """Bound visible tensor backings; a backend must account for hidden scratch.

    Tuple results include Torch's functionalization copies. Original input
    storage and duplicate aliases are excluded from this local allocation sum.
    """

    def tensors(value: object) -> Iterator[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in cast(tuple[object, ...] | list[object], value):
                yield from tensors(item)
        elif value is not None:
            raise ValueError("kernel allocation metadata must contain tensors or tuples")

    seen = {
        tensor.untyped_storage()
        for node in graph.graph.nodes
        if node.op == "placeholder"
        for tensor in tensors(node.meta.get("val"))
    }
    result: list[KernelAllocation] = []
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        for tensor in tensors(node.meta.get("val")):
            storage = tensor.untyped_storage()
            if storage not in seen:
                seen.add(storage)
                size = storage.nbytes()
                if type(size) is not int:
                    raise ValueError("kernel allocations require static backing extents")
                result.append(KernelAllocation(f"{node.name}:{len(result)}", str(tensor.device), size))
    return tuple(result)


def module_has_execution_hooks(module: torch.nn.Module) -> bool:
    for owner, names in (
        (module, ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")),
        (
            torch.nn.modules.module,
            (
                "_global_forward_hooks",
                "_global_forward_pre_hooks",
                "_global_backward_hooks",
                "_global_backward_pre_hooks",
            ),
        ),
    ):
        for name in names:
            registry: object = getattr(owner, name)
            if not isinstance(registry, dict) or bool(cast(dict[int, object], registry)):
                return True
    return False


@dataclass(frozen=True, slots=True)
class ModuleCallBinding:
    """Local method and parameter-field binding; values may update in place."""

    module: torch.nn.Module
    forward: object
    parameters: tuple[tuple[str, torch.Tensor], ...]

    @classmethod
    def bind(cls, module: torch.nn.Module) -> "ModuleCallBinding":
        if module_has_execution_hooks(module):
            raise ValueError("model source requires explicit hook effects")
        return cls(module, module.forward, tuple(module.named_parameters(recurse=False)))

    def validate(self, module: torch.nn.Module) -> None:
        if module is not self.module or module.forward != self.forward or module_has_execution_hooks(module):
            raise ValueError("model call changed after binding its source definition")
        parameters = tuple(module.named_parameters(recurse=False))
        if len(parameters) != len(self.parameters) or any(
            name != original_name or value is not original
            for (name, value), (original_name, original) in zip(parameters, self.parameters, strict=True)
        ):
            raise ValueError("model parameter fields changed after binding its source definition")


class _ModuleCalls(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        # Keep child calls in their original representation, including custom
        # kernels and mutations. The backend, not the tracer, owns their effects.
        return bool(module_qualified_name)


@dataclass(frozen=True, slots=True)
class ModuleGraphPartition:
    prefix: torch.fx.GraphModule
    suffix: torch.fx.GraphModule
    digest: str
    frontier: tuple[str, ...]

    def run_prefix(self, inputs: tuple[torch.Tensor | None, ...]) -> tuple[torch.Tensor, ...]:
        result: object = self.prefix(*inputs)
        if not isinstance(result, tuple) or not all(
            isinstance(value, torch.Tensor) for value in cast(tuple[object, ...], result)
        ):
            raise ValueError("module partition frontier must contain tensors")
        return cast(tuple[torch.Tensor, ...], result)

    def run_suffix(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tensor_graph_outputs(self.suffix, inputs)


def capture_module_partition(
    module: torch.nn.Module, *, stateful: tuple[str, ...], readonly: tuple[str, ...], none_inputs: tuple[str, ...] = ()
) -> ModuleGraphPartition:
    """Cut after the last stateful leaf, preserving original call order.

    ``readonly`` describes external state, not tensor-input mutation: both
    pieces retain the native call's private-input ABI. Only module calls and
    tuple selection are supported here. General tensor graphs keep their own
    export/FX path; unknown effects cannot be moved into the readonly suffix.
    """
    if not stateful or set(stateful) & set(readonly):
        raise ValueError("module partition requires disjoint resource contracts")
    for child in module.modules():
        ModuleCallBinding.bind(child)
    graph = _ModuleCalls().trace(module, concrete_args={name: None for name in none_inputs} or None)
    nodes = tuple(graph.nodes)
    cuts: list[int] = []
    for index, node in enumerate(nodes):
        if node.op == "call_module":
            if node.target not in (*stateful, *readonly):
                raise ValueError(f"module call has no resource contract: {node.target}")
            if node.target in stateful:
                cuts.append(index)
        elif node.op == "call_function":
            if node.target is operator.getitem:
                if len(node.args) != 2 or not isinstance(node.args[0], torch.fx.Node) or type(node.args[1]) is not int:
                    raise ValueError("module graph requires constant tuple selection")
            elif node.target is _ASSERT_IS_NONE:
                # Retain FX's specialization assertion in the prefix, including
                # its input. The generated graph still rejects a non-None value.
                if not none_inputs:
                    raise ValueError("unexpected module input specialization")
            else:
                raise ValueError(f"module graph requires an explicit effect contract: {node.target}")
        elif node.op not in ("placeholder", "output"):
            raise ValueError(f"unsupported module graph node: {node.op}")
    if not cuts:
        raise ValueError("module graph has no stateful call to partition")
    cut = cuts[-1]
    before, after = nodes[: cut + 1], nodes[cut + 1 :]
    if not any(node.op == "call_module" for node in after):
        raise ValueError("module graph has no readonly suffix")
    before_set = set(before)
    # Frontier order follows the suffix's first uses, not model member names.
    frontier = tuple(dict.fromkeys(source for node in after for source in node.all_input_nodes if source in before_set))
    if not frontier:
        raise ValueError("module partition requires a tensor frontier")

    def copy_piece(
        source: tuple[torch.fx.Node, ...], inputs: tuple[torch.fx.Node, ...], outputs: tuple[torch.fx.Node, ...] | None
    ) -> torch.fx.GraphModule:
        piece = torch.fx.Graph()
        mapping = {node: piece.placeholder(f"arg{i}") for i, node in enumerate(inputs)}
        for node in source:
            if node.op == "placeholder":
                continue
            copied = piece.node_copy(node, mapping.__getitem__)
            copied.name = f"v{len(mapping)}"
            copied.type = None
            mapping[node] = copied
        if outputs is not None:
            piece.output(tuple(mapping[node] for node in outputs))
        piece.lint()
        return torch.fx.GraphModule(module, piece)

    prefix = copy_piece(before, tuple(node for node in nodes if node.op == "placeholder"), frontier)
    suffix = copy_piece(after, frontier, None)
    identity = boundary_digest(
        "module-fx-partition@1",
        str(prefix.graph),
        str(suffix.graph),
        stateful,
        readonly,
        none_inputs,
        torch.__version__,
    )
    return ModuleGraphPartition(prefix, suffix, identity, tuple(node.name for node in frontier))
