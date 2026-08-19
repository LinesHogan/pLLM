"""Closed-world Python oracle for the Phase 1 capture hierarchy validator.

The oracle intentionally consumes attribute descriptors instead of requiring the
Python dataclasses.  Differential tests can therefore replace one field with a
malformed value and exercise the same boundary that PyO3 sees.
"""

from __future__ import annotations

import string
from collections.abc import Iterable


_EFFECTS = {"pure", "read_state", "write_state", "rng", "host_io", "opaque"}
_CAPABILITIES = {"eager_only", "capture_safe", "replay_safe"}
_OPERATOR_KINDS = {"aten", "custom"}
_ATTRIBUTE_KINDS = {
    "none",
    "bool",
    "int",
    "float",
    "string",
    "tuple",
    "list",
    "mapping",
    "slice",
    "ellipsis",
    "dtype",
    "device",
    "layout",
    "memory_format",
}
_BOUNDARIES = {"root", "module", "custom_op", "effect", "capability", "opaque"}
_ANCHOR_HOOKS = {
    "prepare_inputs.pre",
    "prepare_inputs.post",
    "execute_model.pre",
    "execute_model.post",
    "sample_tokens.pre",
    "sampler.pre_draw",
    "sample_tokens.post",
}


def _value(value: object) -> object:
    return getattr(value, "value", value)


def _text(value: object) -> str:
    result = _value(value)
    if not isinstance(result, str):
        raise ValueError("expected text")
    return result


def _attr(value: object, name: str) -> object:
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise ValueError(f"missing {name}") from exc


def _seq(value: object, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def _node_key(key: object) -> tuple[str, str, int]:
    namespace = _text(_attr(key, "namespace"))
    owner = _text(_attr(key, "owner"))
    ordinal = _nonnegative_int(_attr(key, "ordinal"), "node ordinal")
    if not namespace or not owner or "/" in namespace:
        raise ValueError("invalid node key")
    if namespace == "anchor":
        parts = owner.split("/")
        if len(parts) != 4 or any(not part or "\\" in part for part in parts):
            raise ValueError("invalid anchor key")
    elif "/" in owner or "\\" in owner:
        raise ValueError("invalid node owner")
    return namespace, owner, ordinal


def _node_text(key: object) -> str:
    namespace, owner, ordinal = _node_key(key)
    if namespace == "anchor":
        return f"{namespace}/{owner}"
    return f"{namespace}/{owner}/{ordinal}"


def _artifact_key(key: object) -> tuple[str, str, int, str]:
    namespace = _text(_attr(key, "namespace"))
    name = _text(_attr(key, "name"))
    version = _attr(key, "version")
    digest = _text(_attr(key, "digest"))
    if (
        not namespace
        or not name
        or not isinstance(version, int)
        or isinstance(version, bool)
        or version < 1
        or "/" in namespace
        or "\\" in namespace
        or "/" in name
        or "\\" in name
        or len(digest) != 64
        or any(char not in string.hexdigits[:16] for char in digest)
    ):
        raise ValueError("invalid artifact key")
    return namespace, name, version, digest


def _artifact_text(key: object) -> str:
    namespace, name, version, digest = _artifact_key(key)
    return f"{namespace}:{name}@{version}#{digest}"


def _region_key(key: object, expected_kind: str, artifact: object) -> tuple[str, str, int, str, int]:
    artifact_sig = _artifact_key(_attr(key, "artifact_key"))
    expected_artifact = _artifact_key(artifact)
    kind = _text(_attr(key, "kind"))
    ordinal = _nonnegative_int(_attr(key, "ordinal"), "region ordinal")
    if kind != expected_kind or artifact_sig != expected_artifact:
        raise ValueError("region identity mismatch")
    return (*artifact_sig, kind, ordinal)


def _port_ref(port: object) -> tuple[object, ...]:
    if hasattr(port, "namespace") and hasattr(port, "name") and hasattr(port, "version"):
        namespace = _text(_attr(port, "namespace"))
        name = _text(_attr(port, "name"))
        version = _attr(port, "version")
        if (
            not namespace
            or "." in namespace
            or ":" in namespace
            or not name
            or ":" in name
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version < 1
        ):
            raise ValueError("invalid PortId")
        return ("id", namespace, name, version)
    kind = _text(_attr(port, "kind"))
    locator_type = _text(_attr(port, "locator_type"))
    fields = []
    for field in _seq(_attr(port, "locator_fields"), "locator_fields"):
        path = tuple(_text(part) for part in _seq(_attr(field, "path"), "locator path"))
        value_type = _text(_attr(field, "value_type"))
        value = _text(_attr(field, "value"))
        if not path or any(not part for part in path) or not value_type:
            raise ValueError("invalid locator field")
        fields.append((path, value_type, value))
    if not kind or (not locator_type and fields):
        raise ValueError("invalid PortAddressDesc")
    return ("address", kind, locator_type, tuple(fields))


def _binding(binding: object) -> tuple[object, ...]:
    port = _port_ref(_attr(binding, "port"))
    direction = _text(_attr(binding, "direction"))
    path = tuple(
        _nonnegative_int(part, "binding value path")
        for part in _seq(_attr(binding, "value_path"), "binding value path")
    )
    if direction not in {"input", "output"}:
        raise ValueError("invalid binding direction")
    return (port, direction, path)


def _validate_operator(operator: object) -> None:
    ordinal = _nonnegative_int(_attr(operator, "ordinal"), "operator ordinal")
    del ordinal
    namespace = _text(_attr(operator, "namespace"))
    name = _text(_attr(operator, "name"))
    if not namespace or not name:
        raise ValueError("invalid operator identity")
    kind = _text(_attr(operator, "operator_kind"))
    if kind not in _OPERATOR_KINDS or (kind == "aten") != (namespace == "aten"):
        raise ValueError("invalid operator kind")
    effects = tuple(_text(effect) for effect in _seq(_attr(operator, "effects"), "effects"))
    capabilities = tuple(
        _text(capability)
        for capability in _seq(_attr(operator, "capabilities"), "capabilities")
    )
    if not effects or any(effect not in _EFFECTS for effect in effects):
        raise ValueError("invalid operator effects")
    if not capabilities or any(capability not in _CAPABILITIES for capability in capabilities):
        raise ValueError("invalid operator capabilities")
    for segment in _seq(_attr(operator, "module_path"), "module_path"):
        segment = _text(segment)
        if not segment or "/" in segment or "\\" in segment:
            raise ValueError("invalid operator module path")
    if any(_nonnegative_int(item, "operator input") < 0 for item in _seq(_attr(operator, "inputs"), "inputs")):
        raise ValueError("invalid operator inputs")
    if any(not _text(item) for item in _seq(_attr(operator, "outputs"), "outputs")):
        raise ValueError("invalid operator outputs")
    paths: set[tuple[str, ...]] = set()
    for attribute in _seq(_attr(operator, "attributes"), "attributes"):
        path = tuple(_text(part) for part in _seq(_attr(attribute, "path"), "attribute path"))
        kind = _text(_attr(attribute, "kind"))
        if not path or any(not part for part in path) or kind not in _ATTRIBUTE_KINDS or path in paths:
            raise ValueError("invalid operator attribute")
        paths.add(path)


def _leaf(leaf: object, artifact: object) -> tuple[set[int], object | None]:
    if _artifact_key(_attr(leaf, "artifact_key")) != _artifact_key(artifact):
        raise ValueError("leaf artifact mismatch")
    kind = _text(_attr(leaf, "kind"))
    _region_key(_attr(leaf, "key"), kind, artifact)
    ordinals = tuple(
        _nonnegative_int(item, "leaf operator ordinal")
        for item in _seq(_attr(leaf, "operator_ordinals"), "operator ordinals")
    )
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("duplicate leaf coverage")
    leaf_artifact = _attr(leaf, "leaf_artifact")
    if kind == "opaque":
        if leaf_artifact is not None or ordinals:
            raise ValueError("opaque leaf retained export graph")
        return set(), None
    if kind != "leaf" or leaf_artifact is None:
        raise ValueError("invalid leaf region")
    if _artifact_key(_attr(leaf_artifact, "artifact_key")) != _artifact_key(artifact):
        raise ValueError("leaf artifact identity mismatch")
    operators = _seq(_attr(leaf_artifact, "operators"), "leaf operators")
    known: dict[int, object] = {}
    for operator in operators:
        ordinal = _nonnegative_int(_attr(operator, "ordinal"), "operator ordinal")
        if ordinal in known:
            raise ValueError("duplicate leaf operator")
        known[ordinal] = operator
        _validate_operator(operator)
    if not set(ordinals).issubset(known) or (not ordinals and known):
        raise ValueError("invalid leaf coverage")
    for value in (*_seq(_attr(leaf_artifact, "input_values"), "leaf inputs"), *_seq(_attr(leaf_artifact, "output_values"), "leaf outputs")):
        if not _text(value):
            raise ValueError("empty leaf value")
    if _nonnegative_int(_attr(leaf_artifact, "terminal_ordinal"), "terminal ordinal") < 0:
        raise ValueError("invalid terminal ordinal")
    return set(ordinals), leaf_artifact


def _validate_value(value: object) -> tuple[str, int]:
    value_id = _text(_attr(value, "value_id"))
    version = _attr(value, "version")
    if not value_id or not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("invalid captured value identity")
    shape = tuple(_text(item) for item in _seq(_attr(value, "shape"), "shape"))
    stride = tuple(_text(item) for item in _seq(_attr(value, "stride"), "stride"))
    dtype = _attr(value, "dtype")
    device = _attr(value, "device")
    if (
        not shape
        or not stride
        or not _text(_attr(dtype, "namespace"))
        or not _text(_attr(dtype, "name"))
        or _text(_attr(device, "kind")) not in {"cpu", "cuda"}
        or (_attr(device, "index") is not None and _nonnegative_int(_attr(device, "index"), "device index") < 0)
        or not _text(_attr(value, "layout"))
        or not _text(_attr(value, "alias_set"))
        or _text(_attr(value, "effect")) not in _EFFECTS
    ):
        raise ValueError("invalid captured value facts")
    use_def = _attr(value, "use_def")
    _nonnegative_int(_attr(use_def, "producer"), "value producer")
    for field in ("inputs", "users"):
        for ordinal in _seq(_attr(use_def, field), field):
            _nonnegative_int(ordinal, f"value {field}")
    return value_id, version


def _boundary(previous: object | None, current: object | None) -> str:
    if current is None:
        return "root"
    if _text(_attr(current, "operator_kind")) == "custom":
        return "custom_op"
    if previous is None:
        return "root"
    if tuple(_attr(previous, "module_path")) != tuple(_attr(current, "module_path")):
        return "module"
    if tuple(_attr(previous, "effects")) != tuple(_attr(current, "effects")):
        return "effect"
    if tuple(_attr(previous, "capabilities")) != tuple(_attr(current, "capabilities")):
        return "capability"
    return "root"


def _semantic_index(semantic: object) -> tuple[dict[tuple[str, str, int], object], set[tuple[object, ...]], set[tuple[str, str, int]], tuple[object, ...]]:
    nodes = _seq(_attr(semantic, "nodes"), "semantic nodes")
    by_key: dict[tuple[str, str, int], object] = {}
    anchors: set[tuple[str, str, int]] = set()
    for node in nodes:
        key = _node_key(_attr(node, "node_key"))
        if key in by_key:
            raise ValueError("duplicate semantic node")
        by_key[key] = node
        if _text(_attr(node, "kind")) == "anchor":
            anchors.add(key)
    ports: set[tuple[object, ...]] = set()
    for value in _seq(_attr(semantic, "values"), "semantic values"):
        port = _attr(value, "port")
        if port is not None:
            ports.add(_port_ref(port))
    return by_key, ports, anchors, _seq(_attr(semantic, "edges"), "semantic edges")


def _validate_root(root: object) -> tuple[tuple[str, str, int], ...]:
    artifact = _attr(root, "artifact_key")
    _artifact_key(artifact)
    _region_key(_attr(root, "key"), "root", artifact)
    if _text(_attr(root, "kind")) != "root":
        raise ValueError("invalid root kind")
    span = _attr(root, "source_span")
    if (
        not _text(_attr(span, "file"))
        or _nonnegative_int(_attr(span, "start_line"), "start line") < 1
        or _nonnegative_int(_attr(span, "end_line"), "end line") < _attr(span, "start_line")
        or _nonnegative_int(_attr(span, "start_column"), "start column") < 0
        or _nonnegative_int(_attr(span, "end_column"), "end column") < 0
    ):
        raise ValueError("invalid source span")
    root_effects = tuple(_text(effect) for effect in _seq(_attr(root, "effects"), "root effects"))
    if not root_effects or any(effect not in _EFFECTS for effect in root_effects):
        raise ValueError("invalid root effects")
    for expression in _seq(_attr(_attr(root, "shape_guard"), "expressions"), "shape expressions"):
        kind = _text(_attr(expression, "kind"))
        lower = _nonnegative_int(_attr(expression, "lower_bound"), "shape lower bound")
        upper = _nonnegative_int(_attr(expression, "upper_bound"), "shape upper bound")
        if (
            any(_nonnegative_int(part, "shape path") < 0 for part in _seq(_attr(expression, "value_path"), "shape path"))
            or _nonnegative_int(_attr(expression, "dimension"), "shape dimension") < 0
            or upper < lower
            or kind not in {"static", "range"}
            or kind == "static" and lower != upper
        ):
            raise ValueError("invalid shape guard")
    root_bindings = []
    for collection, expected in (("inputs", "input"), ("outputs", "output")):
        for binding in _seq(_attr(root, collection), collection):
            desc = _binding(binding)
            if desc[1] != expected:
                raise ValueError("root binding direction mismatch")
            root_bindings.append(desc)
    if len(set(root_bindings)) != len(root_bindings):
        raise ValueError("duplicate root binding")
    values = _seq(_attr(root, "values"), "root values")
    value_keys = set()
    producer_ordinals = set()
    for value in values:
        key = _validate_value(value)
        if key in value_keys or _attr(_attr(value, "use_def"), "producer") in producer_ordinals:
            raise ValueError("duplicate value facts")
        value_keys.add(key)
        producer_ordinals.add(_attr(_attr(value, "use_def"), "producer"))
    for diagnostic in _seq(_attr(root, "diagnostics"), "diagnostics"):
        if not _text(_attr(diagnostic, "code")) or not _text(_attr(diagnostic, "message")):
            raise ValueError("invalid diagnostic")
        if not _text(_attr(_attr(diagnostic, "source_span"), "file")):
            raise ValueError("invalid diagnostic source")
    children = _seq(_attr(root, "children"), "root children")
    if not children:
        raise ValueError("root has no tasks")
    task_positions: dict[tuple[object, ...], int] = {}
    for position, task in enumerate(children):
        task_sig = _region_key(_attr(task, "key"), "task", artifact)
        if task_sig in task_positions:
            raise ValueError("duplicate task")
        task_positions[task_sig] = position
    covered: set[int] = set()
    expected: set[int] | None = None
    previous: object | None = None
    order = []
    for position, task in enumerate(children):
        if _text(_attr(task, "kind")) != "task" or len(_seq(_attr(task, "children"), "task children")) != 1:
            raise ValueError("invalid task shape")
        node_key = _attr(task, "node_key")
        key = _node_key(node_key)
        if key[0] != "capture_task":
            raise ValueError("invalid task namespace")
        executor = _attr(task, "executor")
        if any(not _text(_attr(executor, field)) for field in ("namespace", "name", "version")):
            raise ValueError("invalid task executor")
        boundary = _text(_attr(task, "boundary"))
        if boundary not in _BOUNDARIES:
            raise ValueError("invalid task boundary")
        task_effects = tuple(_text(effect) for effect in _seq(_attr(task, "effects"), "task effects"))
        capabilities = tuple(_text(capability) for capability in _seq(_attr(task, "capabilities"), "task capabilities"))
        if not task_effects or any(effect not in _EFFECTS for effect in task_effects) or not capabilities or any(cap not in _CAPABILITIES for cap in capabilities):
            raise ValueError("invalid task facts")
        for field in ("module_path", "input_values", "output_values"):
            for value in _seq(_attr(task, field), field):
                if not _text(value) or field == "module_path" and ("/" in _text(value) or "\\" in _text(value)):
                    raise ValueError("invalid task structural facts")
        for dependency in _seq(_attr(task, "dependencies"), "dependencies"):
            dependency_sig = _region_key(dependency, "task", artifact)
            if dependency_sig not in task_positions or task_positions[dependency_sig] >= position:
                raise ValueError("invalid task dependency")
        task_ordinals = tuple(_nonnegative_int(item, "task operator ordinal") for item in _seq(_attr(task, "operator_ordinals"), "task operator ordinals"))
        if len(set(task_ordinals)) != len(task_ordinals):
            raise ValueError("duplicate task coverage")
        leaf_ordinals, leaf_artifact = _leaf(_seq(_attr(task, "children"), "task children")[0], artifact)
        if leaf_ordinals != set(task_ordinals) or covered.intersection(task_ordinals):
            raise ValueError("task/leaf coverage mismatch")
        covered.update(task_ordinals)
        if leaf_artifact is not None:
            operators = _seq(_attr(leaf_artifact, "operators"), "leaf operators")
            by_ordinal = {_attr(operator, "ordinal"): operator for operator in operators}
            selected = [by_ordinal[ordinal] for ordinal in task_ordinals]
            if any(
                tuple(_attr(operator, "module_path")) != tuple(_attr(task, "module_path"))
                or tuple(_attr(operator, "effects")) != tuple(_attr(task, "effects"))
                or tuple(_attr(operator, "capabilities")) != tuple(_attr(task, "capabilities"))
                for operator in selected
            ):
                raise ValueError("task facts differ from leaf")
            if boundary != _boundary(previous, selected[0] if selected else None):
                raise ValueError("task boundary differs from leaf")
            if selected:
                previous = selected[-1]
            known = set(by_ordinal)
            if expected is not None and expected != known:
                raise ValueError("tasks disagree on leaf artifact")
            expected = known
        order.append(key)
    status = _text(_attr(root, "status"))
    if status == "captured":
        if not values or _seq(_attr(root, "diagnostics"), "diagnostics"):
            raise ValueError("captured root facts")
        if covered != (expected or set()):
            raise ValueError("incomplete operator coverage")
        first_task_leaf = _seq(_attr(children[0], "children"), "task children")[0]
        leaf_artifact_desc = _attr(first_task_leaf, "leaf_artifact")
        artifact_operators = _seq(_attr(leaf_artifact_desc, "operators"), "operators")
        operator_ordinals = {_attr(operator, "ordinal") for operator in artifact_operators}
        input_outputs = {
            _text(value)
            for value in (
                *_seq(_attr(leaf_artifact_desc, "input_values"), "leaf inputs"),
                *_seq(_attr(leaf_artifact_desc, "output_values"), "leaf outputs"),
            )
        }
        if not input_outputs.issubset({f"{value_id}@{version}" for value_id, version in value_keys}):
            raise ValueError("unknown leaf value")
        expected_effects = tuple(
            dict.fromkeys(
                _text(effect)
                for operator in artifact_operators
                for effect in _seq(_attr(operator, "effects"), "operator effects")
            )
        ) or ("pure",)
        if root_effects != expected_effects:
            raise ValueError("root effects differ from leaf operators")
        for value in values:
            use_def = _attr(value, "use_def")
            if not set(_seq(_attr(use_def, "inputs"), "value inputs")).issubset(producer_ordinals):
                raise ValueError("unknown value input")
            unknown_users = set(_seq(_attr(use_def, "users"), "value users")) - producer_ordinals - operator_ordinals
            if unknown_users and (
                unknown_users != {_attr(leaf_artifact_desc, "terminal_ordinal")}
                or f"{_text(_attr(value, 'value_id'))}@{_attr(value, 'version')}" not in input_outputs
            ):
                raise ValueError("unknown value user")
        for operator in artifact_operators:
            if not set(_seq(_attr(operator, "inputs"), "operator inputs")).issubset(producer_ordinals) or not {
                _text(value) for value in _seq(_attr(operator, "outputs"), "operator outputs")
            }.issubset({f"{_text(_attr(value, 'value_id'))}@{_attr(value, 'version')}" for value in values}):
                raise ValueError("dangling operator SSA")
    elif status == "opaque_barrier":
        if len(children) != 1 or _text(_attr(children[0], "boundary")) != "opaque" or _text(_attr(_seq(_attr(children[0], "children"), "task children")[0], "kind")) != "opaque":
            raise ValueError("invalid opaque root")
    else:
        raise ValueError("unsupported root status")
    return tuple(order)


def validate_capture_hierarchy(hierarchy: object, semantic: object) -> tuple[str, ...]:
    """Return task node texts when the hierarchy is valid; raise ValueError otherwise."""

    if _attr(hierarchy, "schema_version") != 2:
        raise ValueError("schema version")
    roots = _seq(_attr(hierarchy, "roots"), "roots")
    if not roots:
        raise ValueError("no roots")
    semantic_nodes, semantic_ports, semantic_anchors, semantic_edges = _semantic_index(semantic)
    if any(key[0] == "capture_operator" for key in semantic_nodes):
        raise ValueError("ATen operator entered semantic schedule")
    root_artifacts: dict[tuple[str, str, int, str], object] = {}
    order: list[tuple[str, str, int]] = []
    root_bindings: dict[tuple[str, str, int, str], set[tuple[object, ...]]] = {}
    for root in roots:
        artifact = _attr(root, "artifact_key")
        artifact_sig = _artifact_key(artifact)
        if artifact_sig in root_artifacts:
            raise ValueError("duplicate root artifact")
        root_artifacts[artifact_sig] = root
        root_bindings[artifact_sig] = {
            _binding(binding)
            for collection in ("inputs", "outputs")
            for binding in _seq(_attr(root, collection), collection)
        }
        order.extend(_validate_root(root))
    attachment_artifacts: set[tuple[str, str, int, str]] = set()
    for attachment in _seq(_attr(hierarchy, "attachments"), "attachments"):
        artifact_sig = _artifact_key(_attr(attachment, "artifact_key"))
        if artifact_sig not in root_artifacts or artifact_sig in attachment_artifacts:
            raise ValueError("unknown or duplicate attachment artifact")
        attachment_artifacts.add(artifact_sig)
        node_key_obj = _attr(attachment, "node_key")
        node_key = _node_key(node_key_obj)
        target = semantic_nodes.get(node_key)
        if target is None or node_key[0] != "consumer_flow" or _text(_attr(target, "kind")) != "compute":
            raise ValueError("invalid attachment endpoint")
        bindings = set()
        for binding in _seq(_attr(attachment, "bindings"), "attachment bindings"):
            desc = _binding(binding)
            if desc in bindings:
                raise ValueError("duplicate attachment binding")
            bindings.add(desc)
            port, direction, _path = desc
            if port[0] == "id":
                if port not in semantic_ports:
                    raise ValueError("unknown attachment PortId")
            else:
                accesses = _seq(_attr(target, "reads" if direction == "input" else "writes"), "target accesses")
                if not any(_attr(access, "port") is not None and _port_ref(_attr(access, "port")) == port for access in accesses):
                    raise ValueError("undeclared attachment address")
        if bindings != root_bindings[artifact_sig]:
            raise ValueError("attachment binding coverage mismatch")
        anchor = _attr(attachment, "anchor_node_key")
        if anchor is not None:
            anchor_sig = _node_key(anchor)
            if anchor_sig not in semantic_anchors or not any(
                _node_key(source) == anchor_sig and _node_key(target_node) == node_key and _text(kind) == "anchor"
                for source, target_node, kind in semantic_edges
            ):
                raise ValueError("mismatched attachment anchor")
    if set(root_artifacts) != attachment_artifacts:
        raise ValueError("root attachment parity")
    data = _attr(hierarchy, "data_centric")
    if _seq(_attr(data, "moves"), "moves"):
        raise ValueError("Move descriptors are unsupported")
    positions = {key: position for position, key in enumerate(order)}
    lease_values: set[str] = set()
    for lease in _seq(_attr(data, "leases"), "leases"):
        value_id = _text(_attr(lease, "value_id"))
        start = _node_key(_attr(lease, "live_start"))
        end = _node_key(_attr(lease, "live_end"))
        reader_count = _attr(lease, "reader_count")
        if (
            not value_id
            or not isinstance(reader_count, int)
            or isinstance(reader_count, bool)
            or reader_count < 0
            or bool(_attr(lease, "materialized"))
            or value_id in lease_values
            or start not in positions
            or end not in positions
            or positions[start] > positions[end]
        ):
            raise ValueError("invalid logical lease")
        lease_values.add(value_id)
    return tuple(f"{namespace}/{owner}/{ordinal}" if namespace != "anchor" else f"{namespace}/{owner}" for namespace, owner, ordinal in order)


__all__ = ["validate_capture_hierarchy"]
