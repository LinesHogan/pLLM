"""Strict torch.export adapter and conservative flow effect analysis."""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
import operator
import textwrap
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Protocol, cast

import torch

from tllm.workgraph.frontend import (
    ArtifactKey,
    Capability,
    CapturedFlow,
    CapturedValueDesc,
    CaptureStatus,
    Device,
    DynamicDim,
    Effect,
    FlowCaptureCode,
    FlowCaptureDiagnostic,
    LeafArtifactDesc,
    LeafOperatorDesc,
    OperatorAttributeDesc,
    OperatorAttributeKind,
    OperatorKind,
    PortBindingDesc,
    PortBindingDirection,
    ShapeGuard,
    ShapeGuardExpr,
    ShapeGuardKind,
    SourceSpan,
    TensorDTypeDesc,
    TensorDeviceDesc,
    TensorLayout,
    UseDefDesc,
)
from tllm.workgraph.ir import GraphSafetyKind, NodeKind, TimingMobilityKind


@dataclass(frozen=True, slots=True)
class _Evidence:
    code: FlowCaptureCode
    message: str
    source_span: SourceSpan


class _AliasInfoLike(Protocol):
    before_set: Set[str]
    after_set: Set[str]
    is_write: bool


class _SchemaArgumentLike(Protocol):
    alias_info: _AliasInfoLike | None


class _SchemaLike(Protocol):
    name: str
    overload_name: str
    is_mutable: bool
    arguments: Sequence[_SchemaArgumentLike]
    returns: Sequence[_SchemaArgumentLike]


class _OperatorLike(Protocol):
    namespace: str
    _schema: _SchemaLike
    tags: Sequence[object]


class _RangeLike(Protocol):
    lower: object
    upper: object


class _PositionedNode(Protocol):
    lineno: int
    col_offset: int
    end_lineno: int | None
    end_col_offset: int | None


class _SliceLike(Protocol):
    start: object
    stop: object
    step: object


class _ExportModule(torch.nn.Module):
    def __init__(self, function: FunctionType) -> None:
        super().__init__()
        self._function = function

    def forward(self, *args: object, **kwargs: object) -> object:
        return self._function(*args, **kwargs)


class _BoundMethodExportModule(torch.nn.Module):
    """Export root for one Program flow method.

    The Program instance is registered as a submodule so torch.export lifts
    its parameters and buffers as typed placeholders (visible in
    ``graph_signature``) instead of baking them in as constants.
    """

    def __init__(self, instance: torch.nn.Module, function: FunctionType) -> None:
        super().__init__()
        self._program_instance = instance
        self._function = function

    def forward(self, *args: object, **kwargs: object) -> object:
        return self._function(self._program_instance, *args, **kwargs)


class _CaptureRejected(RuntimeError):
    def __init__(self, code: FlowCaptureCode, message: str, source_span: SourceSpan) -> None:
        self.code = code
        self.source_span = source_span
        super().__init__(message)


def _node_source_span(flow_span: SourceSpan, node: ast.AST) -> SourceSpan:
    positioned = cast(_PositionedNode, node)
    start_line = flow_span.start_line + positioned.lineno - 1
    end_line_offset = (
        positioned.end_lineno if positioned.end_lineno is not None else positioned.lineno
    )
    end_line = flow_span.start_line + end_line_offset - 1
    end_column = (
        positioned.end_col_offset
        if positioned.end_col_offset is not None
        else positioned.col_offset
    )
    return SourceSpan(
        file=flow_span.file,
        start_line=start_line,
        start_column=positioned.col_offset,
        end_line=end_line,
        end_column=end_column,
    )


def _root_name(node: ast.expr) -> str:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=value):
            return _root_name(value)
        case _:
            return ""


class _EffectVisitor(ast.NodeVisitor):
    def __init__(self, flow_span: SourceSpan) -> None:
        self.flow_span = flow_span
        self.evidence: list[_Evidence] = []

    def _record(self, code: FlowCaptureCode, message: str, node: ast.AST) -> None:
        evidence = _Evidence(code, message, _node_source_span(self.flow_span, node))
        if evidence not in self.evidence:
            self.evidence.append(evidence)

    def visit_Call(self, node: ast.Call) -> None:
        match node.func:
            case ast.Name(id=name) if name in {
                "__import__",
                "breakpoint",
                "compile",
                "eval",
                "exec",
                "input",
                "open",
                "print",
            }:
                self._record(FlowCaptureCode.HOST_IO, "host I/O is not capturable", node)
            case ast.Attribute(attr=attribute, value=value):
                root = _root_name(value)
                if root == "torch" and attribute in {"load", "save"}:
                    self._record(FlowCaptureCode.HOST_IO, "host I/O is not capturable", node)
                elif attribute in {"item", "numpy", "tolist"}:
                    self._record(
                        FlowCaptureCode.DATA_DEPENDENT_CONTROL_FLOW,
                        "Tensor data escapes to Python control flow",
                        node,
                    )
                elif attribute.endswith("_") and not attribute.startswith("__"):
                    self._record(
                        FlowCaptureCode.UNSUPPORTED_MUTATION,
                        "in-place mutation has no proven functional equivalent",
                        node,
                    )
                elif root in {
                    "datetime",
                    "io",
                    "os",
                    "pathlib",
                    "random",
                    "requests",
                    "secrets",
                    "shutil",
                    "socket",
                    "subprocess",
                    "tempfile",
                    "time",
                    "urllib",
                }:
                    self._record(FlowCaptureCode.HOST_IO, "host I/O is not capturable", node)
            case _:
                pass
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        root = _root_name(node)
        if (root == "os" and node.attr == "environ") or (
            root == "sys" and node.attr == "argv"
        ):
            self._record(FlowCaptureCode.HOST_IO, "host process state is not capturable", node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record(
            FlowCaptureCode.UNSUPPORTED_MUTATION,
            "Python mutation has no proven functional equivalent",
            node,
        )
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._record(
            FlowCaptureCode.UNSUPPORTED_MUTATION,
            "Python mutation has no proven functional equivalent",
            node,
        )
        self.generic_visit(node)


def _is_immutable_constant(value: object) -> bool:
    value_type = type(value)
    if value is None or value_type in {bool, bytes, complex, float, int, str}:
        return True
    if value_type is tuple:
        return all(_is_immutable_constant(item) for item in cast(tuple[object, ...], value))
    if value_type is frozenset:
        return all(_is_immutable_constant(item) for item in cast(frozenset[object], value))
    return False


def _is_supported_input(value: object) -> bool:
    if torch.is_tensor(value) or _is_immutable_constant(value):
        return True
    if type(value) is list:
        return all(_is_supported_input(item) for item in cast(list[object], value))
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        return all(
            _is_immutable_constant(key) and _is_supported_input(item)
            for key, item in mapping.items()
        )
    return False


def _is_tensor_input_tree(value: object) -> bool:
    if torch.is_tensor(value):
        return True
    if type(value) in {list, tuple}:
        return all(_is_tensor_input_tree(item) for item in cast(Sequence[object], value))
    if type(value) is dict:
        return all(
            _is_tensor_input_tree(item)
            for item in cast(dict[object, object], value).values()
        )
    return False


def _preflight(
    function: FunctionType,
    *,
    seen: set[FunctionType] | None = None,
) -> tuple[_Evidence, ...]:
    flow_span = SourceSpan.from_callable(function)
    try:
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source, filename=flow_span.file)
    except (OSError, SyntaxError, TypeError):
        return (
            _Evidence(
                FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                "Python source is unavailable for conservative effect analysis",
                flow_span,
            ),
        )
    visitor = _EffectVisitor(flow_span)
    visitor.visit(tree)
    if visitor.evidence:
        return tuple(visitor.evidence)

    visited: set[FunctionType] = set() if seen is None else seen
    if function in visited:
        return ()
    visited.add(function)
    closure = inspect.getclosurevars(function)
    referenced = tuple((*closure.nonlocals.items(), *closure.globals.items()))
    for _name, value in referenced:
        if _is_immutable_constant(value):
            continue
        if type(value) is ModuleType:
            module = value
            if module.__name__.split(".", maxsplit=1)[0] in {"math", "operator", "torch"}:
                continue
            return (
                _Evidence(
                    FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                    "flow references an unregistered Python module",
                    flow_span,
                ),
            )
        if type(value) is type:
            referenced_type = cast(type[object], value)
            if referenced_type.__module__.split(".", maxsplit=1)[0] in {"builtins", "torch"}:
                continue
            return (
                _Evidence(
                    FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                    "flow references an unregistered callable type",
                    flow_span,
                ),
            )
        if type(value) is FunctionType:
            helper = value
            if helper is torch.save or helper is torch.load:
                return (
                    _Evidence(
                        FlowCaptureCode.HOST_IO,
                        "host I/O is not capturable",
                        flow_span,
                    ),
                )
            if helper.__module__.startswith("torch."):
                continue
            helper_evidence = _preflight(helper, seen=visited)
            if helper_evidence:
                return helper_evidence
            continue
        if type(value) is BuiltinFunctionType:
            builtin = value
            if builtin in {
                builtins.__import__,
                builtins.breakpoint,
                builtins.compile,
                builtins.eval,
                builtins.exec,
                builtins.input,
                builtins.open,
                builtins.print,
            }:
                return (
                    _Evidence(
                        FlowCaptureCode.HOST_IO,
                        "host I/O is not capturable",
                        flow_span,
                    ),
                )
            module_name = builtin.__module__ or "builtins"
            if module_name.split(".", maxsplit=1)[0] in {"builtins", "math", "operator", "torch"}:
                continue
        return (
            _Evidence(
                FlowCaptureCode.UNREGISTERED_CLOSURE_STATE,
                "flow closes over unregistered mutable state",
                flow_span,
            ),
        )
    return ()


def _opaque_capture(
    function: FunctionType,
    bindings: tuple[PortBindingDesc, ...],
    evidence: Sequence[_Evidence],
) -> CapturedFlow:
    flow_span = SourceSpan.from_callable(function)
    diagnostics = tuple(
        FlowCaptureDiagnostic(
            code=item.code,
            message=item.message,
            source_span=item.source_span,
        )
        for item in evidence
    )
    known_effects: list[Effect] = []
    codes = {item.code for item in evidence}
    if FlowCaptureCode.HOST_IO in codes:
        known_effects.append(Effect.HOST_IO)
    if FlowCaptureCode.UNREGISTERED_CLOSURE_STATE in codes:
        known_effects.append(Effect.READ_STATE)
    if FlowCaptureCode.UNSUPPORTED_MUTATION in codes:
        known_effects.append(Effect.WRITE_STATE)
    known_effects.append(Effect.OPAQUE)
    digest_payload = {
        "status": CaptureStatus.OPAQUE_BARRIER.value,
        "diagnostics": [item.code.value for item in evidence],
        "effects": [effect.value for effect in known_effects],
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return CapturedFlow(
        artifact_key=ArtifactKey.from_callable(function, digest=digest),
        source_span=flow_span,
        inputs=tuple(
            binding for binding in bindings if binding.direction == PortBindingDirection.INPUT
        ),
        outputs=tuple(
            binding for binding in bindings if binding.direction == PortBindingDirection.OUTPUT
        ),
        effects=tuple(known_effects),
        diagnostics=diagnostics,
    )


def _dynamic_shapes(
    args: tuple[object, ...],
    dimensions: tuple[DynamicDim, ...],
    source_span: SourceSpan,
) -> dict[str, tuple[dict[int, object] | None, ...]] | None:
    if not dimensions:
        return None
    specs: list[dict[int, object] | None] = [None] * len(args)
    # Inputs with the same bounded spec (e.g. source/target batch rows) share
    # one Dim so export records their equality instead of rejecting it.
    shared: dict[tuple[int, int, int], object] = {}
    for dimension in dimensions:
        argument_index = dimension.value_path[0]
        if argument_index >= len(args) or not torch.is_tensor(args[argument_index]):
            raise _CaptureRejected(
                FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                "dynamic dimensions currently require positional Tensor inputs",
                source_span,
            )
        if dimension.upper_bound is None:
            raise _CaptureRejected(
                FlowCaptureCode.UNBOUNDED_DYNAMIC_SHAPE,
                "dynamic dimensions require a finite upper bound",
                source_span,
            )
        spec = specs[argument_index]
        if spec is None:
            spec = {}
            specs[argument_index] = spec
        if dimension.dimension in spec:
            raise _CaptureRejected(
                FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                "dynamic dimension is declared more than once",
                source_span,
            )
        shared_key = (dimension.dimension, dimension.lower_bound, dimension.upper_bound)
        if shared_key not in shared:
            shared[shared_key] = torch.export.Dim(
                f"d_{argument_index}_{dimension.dimension}",
                min=dimension.lower_bound,
                max=dimension.upper_bound,
            )
        spec[dimension.dimension] = shared[shared_key]
    return {"args": tuple(specs)}


def _layout(value: torch.Tensor, source_span: SourceSpan) -> TensorLayout:
    if value.layout == torch.strided:
        return TensorLayout.STRIDED
    if value.layout == torch.sparse_coo:
        return TensorLayout.SPARSE_COO
    if value.layout == torch.sparse_csr:
        return TensorLayout.SPARSE_CSR
    if value.layout == torch.sparse_csc:
        return TensorLayout.SPARSE_CSC
    if value.layout == torch.sparse_bsr:
        return TensorLayout.SPARSE_BSR
    if value.layout == torch.sparse_bsc:
        return TensorLayout.SPARSE_BSC
    raise _CaptureRejected(
        FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
        "export produced an unsupported Tensor layout",
        source_span,
    )


def _finite_bound(bound: object, source_span: SourceSpan) -> int:
    try:
        return int(bound)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise _CaptureRejected(
            FlowCaptureCode.UNBOUNDED_DYNAMIC_SHAPE,
            "export produced an unbounded dynamic shape",
            source_span,
        ) from exc


def _operator_module_path(metadata: Mapping[str, object]) -> tuple[str, ...]:
    raw_stack = metadata.get("nn_module_stack")
    if raw_stack is None:
        return ()
    if not isinstance(raw_stack, Mapping):
        return ()
    module_path = ""
    typed_stack = cast(Mapping[object, object], raw_stack)
    for entry in typed_stack.values():
        if isinstance(entry, tuple) and entry and isinstance(entry[0], str):
            module_path = entry[0]
    return tuple(segment for segment in module_path.split(".") if segment)


def _append_operator_attribute(
    out: list[OperatorAttributeDesc],
    value: object,
    path: tuple[str, ...],
    source_span: SourceSpan,
) -> None:
    if isinstance(value, torch.fx.Node):
        return
    if value is None:
        out.append(OperatorAttributeDesc(path, OperatorAttributeKind.NONE, ""))
        return
    if value is Ellipsis:
        out.append(OperatorAttributeDesc(path, OperatorAttributeKind.ELLIPSIS, ""))
        return
    if type(value) is bool:
        out.append(
            OperatorAttributeDesc(
                path,
                OperatorAttributeKind.BOOL,
                "true" if value else "false",
            )
        )
        return
    if type(value) is int:
        out.append(OperatorAttributeDesc(path, OperatorAttributeKind.INT, str(value)))
        return
    if type(value) is float:
        out.append(
            OperatorAttributeDesc(path, OperatorAttributeKind.FLOAT, value.hex())
        )
        return
    if type(value) is str:
        out.append(OperatorAttributeDesc(path, OperatorAttributeKind.STRING, value))
        return
    if isinstance(value, torch.dtype):
        out.append(OperatorAttributeDesc(path, OperatorAttributeKind.DTYPE, str(value)))
        return
    if isinstance(value, torch.device):
        out.append(OperatorAttributeDesc(path, OperatorAttributeKind.DEVICE, str(value)))
        return
    if isinstance(value, torch.layout):
        out.append(OperatorAttributeDesc(path, OperatorAttributeKind.LAYOUT, str(value)))
        return
    if isinstance(value, torch.memory_format):
        out.append(OperatorAttributeDesc(path, OperatorAttributeKind.MEMORY_FORMAT, str(value)))
        return
    if isinstance(value, slice):
        slice_value = cast(_SliceLike, value)
        out.append(OperatorAttributeDesc(path, OperatorAttributeKind.SLICE, ""))
        _append_operator_attribute(out, slice_value.start, (*path, "start"), source_span)
        _append_operator_attribute(out, slice_value.stop, (*path, "stop"), source_span)
        _append_operator_attribute(out, slice_value.step, (*path, "step"), source_span)
        return
    if isinstance(value, (tuple, list)):
        sequence = cast(Sequence[object], value)
        kind = (
            OperatorAttributeKind.TUPLE
            if isinstance(value, tuple)
            else OperatorAttributeKind.LIST
        )
        out.append(OperatorAttributeDesc(path, kind, str(len(sequence))))
        for index, item in enumerate(sequence):
            _append_operator_attribute(out, item, (*path, str(index)), source_span)
        return
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if any(not isinstance(key, str) for key in mapping):
            raise _CaptureRejected(
                FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                "operator attributes contain a non-string mapping key",
                source_span,
            )
        string_mapping = cast(Mapping[str, object], mapping)
        out.append(OperatorAttributeDesc(path, OperatorAttributeKind.MAPPING, str(len(mapping))))
        for key in sorted(string_mapping):
            _append_operator_attribute(out, string_mapping[key], (*path, key), source_span)
        return
    raise _CaptureRejected(
        FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
        "operator attributes contain an unsupported runtime value",
        source_span,
    )


def _operator_attributes(node: torch.fx.Node, source_span: SourceSpan) -> tuple[OperatorAttributeDesc, ...]:
    attributes: list[OperatorAttributeDesc] = []
    for index, value in enumerate(node.args):
        _append_operator_attribute(attributes, value, ("args", str(index)), source_span)
    for name in sorted(node.kwargs):
        _append_operator_attribute(attributes, node.kwargs[name], ("kwargs", name), source_span)
    return tuple(attributes)


def _operator_facts(target: _OperatorLike) -> tuple[tuple[Effect, ...], tuple[Capability, ...]]:
    tag_names = {
        tag.name
        for tag in target.tags
        if isinstance(tag, torch.Tag)
    }
    if "nondeterministic_seeded" in tag_names:
        return (Effect.RNG,), (Capability.REPLAY_SAFE,)
    return (Effect.PURE,), (Capability.CAPTURE_SAFE,)


def _captured_artifact_digest(
    operators: Sequence[LeafOperatorDesc],
    values: Sequence[CapturedValueDesc],
    shape_guard: ShapeGuard,
) -> str:
    payload = {
        "operators": [
            {
                "ordinal": operator.ordinal,
                "namespace": operator.namespace,
                "name": operator.name,
                "overload": operator.overload,
                "inputs": list(operator.inputs),
                "outputs": list(operator.outputs),
                "module_path": list(operator.module_path),
                "operator_kind": operator.operator_kind.value,
                "effects": [effect.value for effect in operator.effects],
                "capabilities": [capability.value for capability in operator.capabilities],
                "attributes": [
                    {
                        "path": list(attribute.path),
                        "kind": attribute.kind.value,
                        "value": attribute.value,
                    }
                    for attribute in operator.attributes
                ],
            }
            for operator in operators
        ],
        "values": [
            {
                "value_id": value.value_id,
                "version": value.version,
                "shape": list(value.shape),
                "dtype": [value.dtype.namespace, value.dtype.name],
                "device": [value.device.kind.value, value.device.index],
                "layout": value.layout.value,
                "stride": list(value.stride),
                "producer": value.use_def.producer,
                "inputs": list(value.use_def.inputs),
                "users": list(value.use_def.users),
                "alias_set": value.alias_set,
                "effect": value.effect.value,
            }
            for value in values
        ],
        "shape_guard": [
            {
                "value_path": list(expression.value_path),
                "dimension": expression.dimension,
                "kind": expression.kind.value,
                "lower_bound": expression.lower_bound,
                "upper_bound": expression.upper_bound,
            }
            for expression in shape_guard.expressions
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _lower_exported_program(
    function: FunctionType,
    exported_program: torch.export.ExportedProgram,
    bindings: tuple[PortBindingDesc, ...],
) -> CapturedFlow:
    source_span = SourceSpan.from_callable(function)
    nodes = tuple(exported_program.graph.nodes)
    ordinal_by_node = {node: ordinal for ordinal, node in enumerate(nodes)}
    tensor_by_node: dict[torch.fx.Node, torch.Tensor] = {}
    for node in nodes:
        meta_value = cast(object, node.meta.get("val"))
        if torch.is_tensor(meta_value):
            tensor_by_node[node] = meta_value

    parent = {ordinal_by_node[node]: ordinal_by_node[node] for node in tensor_by_node}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    operators: list[LeafOperatorDesc] = []
    effect_by_ordinal: dict[int, Effect] = {}
    for node in nodes:
        if node.op != "call_function":
            continue
        target = cast(_OperatorLike, node.target)
        try:
            schema = target._schema  # pyright: ignore[reportPrivateUsage]
            namespace = target.namespace
        except AttributeError as exc:
            if node.target is operator.getitem:
                # Pure pytree selector emitted by multi-return ATen ops; it has
                # no OpSchema but is a typed, pure, capture-safe projection.
                output_ordinal = ordinal_by_node[node]
                input_ordinals = tuple(ordinal_by_node[item] for item in node.all_input_nodes)
                output_ids = (
                    (f"leaf/{output_ordinal}@1",) if node in tensor_by_node else ()
                )
                effect_by_ordinal[output_ordinal] = Effect.PURE
                operators.append(
                    LeafOperatorDesc(
                        ordinal=output_ordinal,
                        namespace="operator",
                        name="getitem",
                        overload="",
                        inputs=input_ordinals,
                        outputs=output_ids,
                        module_path=_operator_module_path(node.meta),
                        operator_kind=OperatorKind.CUSTOM,
                        effects=(Effect.PURE,),
                        capabilities=(Capability.CAPTURE_SAFE,),
                        attributes=_operator_attributes(node, source_span),
                    )
                )
                continue
            raise _CaptureRejected(
                FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                "export produced an operator without a typed schema",
                source_span,
            ) from exc
        if namespace != "aten":
            raise _CaptureRejected(
                FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                "non-ATen operator requires a registered typed extern contract",
                source_span,
            )
        alias_arguments = tuple(
            argument.alias_info
            for argument in schema.arguments
            if argument.alias_info is not None
        )
        if schema.is_mutable or any(
            alias_info.is_write
            for alias_info in (*alias_arguments, *(item.alias_info for item in schema.returns))
            if alias_info is not None
        ):
            raise _CaptureRejected(
                FlowCaptureCode.UNSUPPORTED_MUTATION,
                "operator schema declares mutation without a proven equivalent lowering",
                source_span,
            )
        output_ordinal = ordinal_by_node[node]
        if node in tensor_by_node:
            return_aliases: set[str] = set()
            for return_item in schema.returns:
                if return_item.alias_info is not None:
                    return_aliases.update(return_item.alias_info.after_set)
            tensor_inputs = tuple(
                input_node for input_node in node.all_input_nodes if input_node in tensor_by_node
            )
            for alias_info, input_node in zip(alias_arguments, tensor_inputs, strict=False):
                if return_aliases.intersection(alias_info.before_set):
                    union(output_ordinal, ordinal_by_node[input_node])
        input_ordinals = tuple(ordinal_by_node[item] for item in node.all_input_nodes)
        output_ids = (
            (f"leaf/{output_ordinal}@1",) if node in tensor_by_node else ()
        )
        schema_name = schema.name.split("::", maxsplit=1)[-1]
        operator_effects, operator_capabilities = _operator_facts(target)
        effect_by_ordinal[output_ordinal] = operator_effects[0]
        operators.append(
            LeafOperatorDesc(
                ordinal=output_ordinal,
                namespace=namespace,
                name=schema_name,
                overload=schema.overload_name,
                inputs=input_ordinals,
                outputs=output_ids,
                module_path=_operator_module_path(node.meta),
                operator_kind=OperatorKind.ATEN,
                effects=operator_effects,
                capabilities=operator_capabilities,
                attributes=_operator_attributes(node, source_span),
            )
        )

    output_nodes = tuple(node for node in nodes if node.op == "output")
    if len(output_nodes) != 1:
        raise _CaptureRejected(
            FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
            "strict export did not produce exactly one output node",
            source_span,
        )

    def validate_output_tree(value: object) -> None:
        if isinstance(value, torch.fx.Node):
            if value not in tensor_by_node:
                raise _CaptureRejected(
                    FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                    "strict export produced a non-Tensor output without a typed SSA descriptor",
                    source_span,
                )
            return
        if isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            for item in mapping.values():
                validate_output_tree(item)
            return
        if isinstance(value, (tuple, list)):
            sequence = cast(Sequence[object], value)
            for item in sequence:
                validate_output_tree(item)
            return
        raise _CaptureRejected(
            FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
            "strict export produced a non-Tensor output without a typed SSA descriptor",
            source_span,
        )

    validate_output_tree(output_nodes[0].args)

    raw_range_constraints: object = exported_program.range_constraints  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    range_constraints = cast(Mapping[object, _RangeLike], raw_range_constraints)
    ranges = {
        str(symbol): (
            _finite_bound(value_range.lower, source_span),
            _finite_bound(value_range.upper, source_span),
        )
        for symbol, value_range in range_constraints.items()
    }
    guard_expressions: list[ShapeGuardExpr] = []
    captured_values: list[CapturedValueDesc] = []
    placeholder_index = 0
    for node, tensor in tensor_by_node.items():
        ordinal = ordinal_by_node[node]
        dtype_text = str(tensor.dtype)
        dtype_parts = dtype_text.split(".", maxsplit=1)
        dtype = TensorDTypeDesc(
            namespace=dtype_parts[0] if len(dtype_parts) == 2 else "torch",
            name=dtype_parts[-1],
        )
        device = TensorDeviceDesc(kind=Device(tensor.device.type), index=tensor.device.index)
        shape = tuple(str(dimension) for dimension in tensor.shape)
        stride = tuple(str(dimension) for dimension in tensor.stride())
        inputs = tuple(ordinal_by_node[item] for item in node.all_input_nodes)
        users = tuple(sorted(ordinal_by_node[item] for item in node.users))
        captured_values.append(
            CapturedValueDesc(
                value_id=f"leaf/{ordinal}",
                version=1,
                shape=shape,
                dtype=dtype,
                device=device,
                layout=_layout(tensor, source_span),
                stride=stride,
                use_def=UseDefDesc(producer=ordinal, inputs=inputs, users=users),
                alias_set=f"alias/{find(ordinal)}",
                effect=effect_by_ordinal.get(ordinal, Effect.PURE),
            )
        )
        if node.op == "placeholder":
            for dimension_index, dimension in enumerate(shape):
                if dimension in ranges:
                    lower, upper = ranges[dimension]
                    kind = ShapeGuardKind.RANGE
                else:
                    try:
                        lower = upper = int(dimension)
                    except ValueError as exc:
                        raise _CaptureRejected(
                            FlowCaptureCode.UNBOUNDED_DYNAMIC_SHAPE,
                            "symbolic Tensor dimension has no finite range constraint",
                            source_span,
                        ) from exc
                    kind = ShapeGuardKind.STATIC
                guard_expressions.append(
                    ShapeGuardExpr(
                        value_path=(placeholder_index,),
                        dimension=dimension_index,
                        kind=kind,
                        lower_bound=lower,
                        upper_bound=upper,
                    )
                )
            placeholder_index += 1

    if not captured_values:
        raise _CaptureRejected(
            FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
            "strict export did not produce a complete Tensor leaf graph",
            source_span,
        )
    input_values = tuple(
        f"leaf/{ordinal_by_node[node]}@1"
        for node in nodes
        if node.op == "placeholder" and node in tensor_by_node
    )
    output_values = tuple(
        f"leaf/{ordinal_by_node[node]}@1"
        for output_node in nodes
        if output_node.op == "output"
        for node in output_node.all_input_nodes
        if node in tensor_by_node
    )
    shape_guard = ShapeGuard(tuple(guard_expressions))
    artifact_key = ArtifactKey.from_callable(
        function,
        digest=_captured_artifact_digest(operators, captured_values, shape_guard),
    )
    leaf_artifact = LeafArtifactDesc(
        artifact_key=artifact_key,
        operators=tuple(operators),
        input_values=input_values,
        output_values=output_values,
        terminal_ordinal=ordinal_by_node[output_nodes[0]],
    )
    return CapturedFlow(
        artifact_key=artifact_key,
        source_span=source_span,
        shape_guard=shape_guard,
        inputs=tuple(
            binding for binding in bindings if binding.direction == PortBindingDirection.INPUT
        ),
        outputs=tuple(
            binding for binding in bindings if binding.direction == PortBindingDirection.OUTPUT
        ),
        values=tuple(captured_values),
        leaf_artifact=leaf_artifact,
        effects=tuple(
            dict.fromkeys(effect for operator in operators for effect in operator.effects)
        )
        or (Effect.PURE,),
        diagnostics=(),
        status=CaptureStatus.CAPTURED,
        node_kind=NodeKind.COMPUTE,
        mobility=TimingMobilityKind.PINNED,
        graph_safety=(
            GraphSafetyKind.REPLAY_STATEFUL
            if any(Effect.RNG in operator.effects for operator in operators)
            else GraphSafetyKind.CAPTURE_SAFE
        ),
    )


def capture_flow(
    function: FunctionType,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    bindings: tuple[PortBindingDesc, ...],
    dynamic_dims: tuple[DynamicDim, ...],
) -> tuple[CapturedFlow, torch.export.ExportedProgram | None]:
    """Capture one complete strict-export leaf, or one conservative barrier."""

    if any(not _is_supported_input(value) for value in (*args, *kwargs.values())):
        source_span = SourceSpan.from_callable(function)
        evidence = (
            _Evidence(
                FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                "capture inputs contain an unsupported runtime object",
                source_span,
            ),
        )
        return _opaque_capture(function, bindings, evidence), None
    if any(not _is_tensor_input_tree(value) for value in (*args, *kwargs.values())):
        source_span = SourceSpan.from_callable(function)
        evidence = (
            _Evidence(
                FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                "non-Tensor capture inputs require a typed scalar SSA descriptor",
                source_span,
            ),
        )
        return _opaque_capture(function, bindings, evidence), None
    evidence = _preflight(function)
    if evidence:
        return _opaque_capture(function, bindings, evidence), None
    source_span = SourceSpan.from_callable(function)
    try:
        dynamic_shapes = _dynamic_shapes(args, dynamic_dims, source_span)
        exported_program = torch.export.export(
            _ExportModule(function),
            args,
            kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
        captured = _lower_exported_program(function, exported_program, bindings)
    except _CaptureRejected as exc:
        evidence = (_Evidence(exc.code, str(exc), exc.source_span),)
        return _opaque_capture(function, bindings, evidence), None
    except Exception as exc:
        evidence = (
            _Evidence(
                FlowCaptureCode.STRICT_EXPORT_FAILURE,
                "strict torch.export capture failed",
                source_span,
            ),
        )
        _ = exc
        return _opaque_capture(function, bindings, evidence), None
    return captured, exported_program


def capture_module_flow(
    instance: torch.nn.Module,
    function: FunctionType,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    bindings: tuple[PortBindingDesc, ...] = (),
    dynamic_dims: tuple[DynamicDim, ...] = (),
) -> tuple[CapturedFlow, torch.export.ExportedProgram | None]:
    """Capture one module-bound Program flow with parameters lifted.

    Mirrors ``capture_flow`` but exports through a wrapper module that owns
    the Program instance, so ``ExportedProgram.graph_signature`` separates
    user inputs from parameters and buffers. ``function`` is the raw
    (unwrapped) method function; ``args`` exclude ``self``.
    """

    if not isinstance(cast(object, instance), torch.nn.Module):
        raise TypeError("capture_module_flow.instance must be an nn.Module")
    if not isinstance(cast(object, function), FunctionType):
        raise TypeError("capture_module_flow.function must be a Python function")
    if any(not _is_supported_input(value) for value in (*args, *kwargs.values())):
        source_span = SourceSpan.from_callable(function)
        return _opaque_capture(
            function,
            bindings,
            (
                _Evidence(
                    FlowCaptureCode.UNKNOWN_SIDE_EFFECT,
                    "capture inputs contain an unsupported runtime object",
                    source_span,
                ),
            ),
        ), None
    evidence = _preflight(function)
    if evidence:
        return _opaque_capture(function, bindings, evidence), None
    source_span = SourceSpan.from_callable(function)
    try:
        dynamic_shapes = _dynamic_shapes(args, dynamic_dims, source_span)
        exported_program = torch.export.export(
            _BoundMethodExportModule(instance, function),
            args,
            kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
        captured = _lower_exported_program(function, exported_program, bindings)
    except _CaptureRejected as exc:
        return _opaque_capture(
            function, bindings, (_Evidence(exc.code, str(exc), exc.source_span),)
        ), None
    except Exception as exc:
        evidence = (
            _Evidence(
                FlowCaptureCode.STRICT_EXPORT_FAILURE,
                "strict torch.export capture failed",
                source_span,
            ),
        )
        _ = exc
        return _opaque_capture(function, bindings, evidence), None
    return captured, exported_program


__all__ = ["capture_flow", "capture_module_flow"]
