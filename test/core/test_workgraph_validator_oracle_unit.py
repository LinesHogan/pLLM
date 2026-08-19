"""Differential contract tests for the Rust WorkGraph validator."""

from __future__ import annotations

from types import SimpleNamespace as Desc

import pytest

from tllm import _core
from tllm.workgraph.ir import (
    AccessDesc,
    AccessMode,
    AnchorDesc,
    BackendSnapshot,
    DataCentricIR,
    ExecutionNodeDesc,
    NodeKey,
    SemanticGraph,
    SemanticNodeDesc,
    LeaseDesc,
    ResidenceDesc,
    ValueDesc,
)
from tllm.ports.port_id import PortId
from test.core.oracles.workgraph_validator import validate_workgraph as oracle_validate


def _key(owner: str) -> Desc:
    return Desc(namespace="consumer_flow", owner=owner, ordinal=0)


def _timing(mobility: str = "pinned", observation: str = "at_declared_window") -> Desc:
    return Desc(
        window="same_step",
        mobility=mobility,
        deadline="step_end",
        max_defer_steps=0,
        observes_version=observation,
    )


def _node(
    owner: str,
    *,
    kind: str = "compute",
    mobility: str = "pinned",
    observation: str = "at_declared_window",
    anchor: object | None = None,
    legacy: bool = False,
) -> Desc:
    return Desc(
        node_key=_key(owner),
        kind=kind,
        executor=None,
        reads=[],
        writes=[],
        effects=[],
        timing=_timing(mobility, observation),
        anchor=anchor,
        graph_safety="eager_only",
        source=owner,
        legacy_boundary="legacy_boundary" if legacy else None,
        shape_guard=[],
    )


def _execution_node(node: Desc) -> Desc:
    return Desc(
        node_key=node.node_key,
        semantic_key=node.node_key,
        kind=node.kind,
        executor=None,
        accesses=[],
        residence_keys=[],
        move_keys=[],
        lease_keys=[],
        device="cuda",
        rank=0,
        anchor=node.anchor,
        graph_safety=node.graph_safety,
        fusion_group="",
    )


def _backend(
    *,
    backend_id: str = "vllm",
    runner_family: str = "mrv1",
    graph_mode: str = "eager",
    version: str = "0.27.1",
    capabilities: list[str] | None = None,
    device: str = "cuda",
    rank: int = 0,
    tp_size: int = 1,
    pp_size: int = 1,
    dp_size: int = 1,
    topology: list[str] | None = None,
) -> Desc:
    return Desc(
        backend_id=backend_id,
        runner_family=runner_family,
        graph_mode=graph_mode,
        backend_version=version,
        capabilities=[] if capabilities is None else capabilities,
        device=device,
        rank=rank,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        topology=[] if topology is None else topology,
    )


def _graph(
    nodes: list[Desc],
    *,
    edges: list[tuple[Desc, Desc, str]] | None = None,
    graph_mode: str = "eager",
    backend_version: str = "0.27.1",
    moves: list[Desc] | None = None,
    data_backend: Desc | None = None,
) -> tuple[Desc, Desc]:
    backend = _backend(graph_mode=graph_mode, version=backend_version)
    semantic = Desc(
        schema_version=1,
        nodes=nodes,
        edges=[] if edges is None else edges,
        values=[],
        backend=backend,
        graph_mode=graph_mode,
    )
    data = Desc(
        schema_version=1,
        nodes=[_execution_node(node) for node in nodes],
        values=[],
        accesses=[],
        residences=[],
        moves=[] if moves is None else moves,
        leases=[],
        edges=list(semantic.edges),
        backend=backend if data_backend is None else data_backend,
        graph_mode=graph_mode,
    )
    return semantic, data


def _value(
    value_id: str = "hidden",
    *,
    shape: list[str] | None = None,
    dtype: str = "bfloat16",
    layout: str = "row_major",
    alias_set: str = "hidden_state",
) -> Desc:
    return Desc(
        value_id=value_id,
        version=1,
        port=None,
        shape=[] if shape is None else shape,
        dtype=dtype,
        layout=layout,
        alias_set=alias_set,
    )


def _access(node_key: Desc, value: Desc, mode: str = "read") -> Desc:
    return Desc(
        node_key=node_key,
        value=value,
        mode=mode,
        port=None,
        subset=[],
        consistency="exact",
    )


def _graph_with_access() -> tuple[Desc, Desc]:
    node = _node("access")
    value = _value(shape=["batch", "hidden"])
    access = _access(node.node_key, value)
    node.reads = [access]
    semantic, data = _graph([node])
    semantic.values = [value]
    data.values = [value]
    data.nodes[0].accesses = [access]
    data.accesses = [access]
    return semantic, data


def _outcome(callable_: object, *args: object) -> tuple[str, object]:
    try:
        return "ok", callable_(*args)  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - differential outcome normalization
        return "err", str(exc)


@pytest.mark.parametrize(
    ("name", "graph"),
    [
        (
            "valid",
            _graph(
                [_node("b"), _node("a")],
                edges=[(_key("a"), _key("b"), "control")],
            ),
        ),
        (
            "legacy deferrable",
            _graph([_node("legacy", mobility="deferrable", legacy=True)]),
        ),
        (
            "legacy latest observation",
            _graph(
                [
                    _node(
                        "legacy-latest",
                        mobility="deferrable",
                        observation="latest_at_execution",
                        legacy=True,
                    )
                ]
            ),
        ),
        (
            "legacy stale observation",
            _graph(
                [
                    _node(
                        "legacy-stale",
                        mobility="free",
                        observation="stale_ok",
                        legacy=True,
                    )
                ]
            ),
        ),
        ("native deferrable", _graph([_node("native", mobility="deferrable")])),
        (
            "native latest observation",
            _graph([_node("native-latest", observation="latest_at_execution")]),
        ),
        (
            "cycle",
            _graph(
                [_node("a"), _node("b")],
                edges=[(_key("a"), _key("b"), "control"), (_key("b"), _key("a"), "control")],
            ),
        ),
        ("unsupported backend", _graph([_node("a")], backend_version="0.26.9")),
        ("unsupported graph mode", _graph([_node("a")], graph_mode="full")),
    ],
)
def test_rust_validator_matches_python_oracle(name: str, graph: tuple[Desc, Desc]) -> None:
    semantic, data = graph
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0], name
    if rust[0] == "ok":
        assert tuple(rust[1].topological_order) == oracle[1]


@pytest.mark.parametrize("mobility", ["pinned", "deferrable", "advanceable", "free"])
@pytest.mark.parametrize(
    "observation",
    ["at_declared_window", "latest_at_execution", "stale_ok"],
)
def test_legacy_timing_accepts_every_recognized_contract(
    mobility: str, observation: str
) -> None:
    semantic, data = _graph(
        [_node("legacy", mobility=mobility, observation=observation, legacy=True)]
    )
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "ok"


@pytest.mark.parametrize("mobility", ["deferrable", "advanceable", "free"])
def test_native_timing_rejects_non_pinned_mobility(mobility: str) -> None:
    semantic, data = _graph([_node("native", mobility=mobility)])
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"


@pytest.mark.parametrize("observation", ["latest_at_execution", "stale_ok"])
def test_native_timing_rejects_non_declared_observation(observation: str) -> None:
    semantic, data = _graph([_node("native", observation=observation)])
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"


def test_rust_validator_rejects_nonempty_moves_and_dangling_edges() -> None:
    move = Desc(value=Desc(value_id="v", version=1), source=Desc(), destination=Desc(), subset=[], layout="", engine="", asynchronous=False)
    semantic, data = _graph([_node("a")], moves=[move])
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"

    semantic, data = _graph([_node("a")], edges=[(_key("a"), _key("missing"), "control")])
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"


def test_rust_validator_rejects_duplicate_nodes_and_unknown_edge_kind() -> None:
    first = _node("a")
    semantic, data = _graph([first])
    semantic.nodes = [first, first]
    data.nodes = [_execution_node(first), _execution_node(first)]
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"

    semantic, data = _graph([_node("a"), _node("b")], edges=[(_key("a"), _key("b"), "not-an-edge")])
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"


@pytest.mark.parametrize(
    "data_backend",
    [
        _backend(backend_id="alternate"),
        _backend(runner_family="mrv2"),
        _backend(graph_mode="piecewise"),
        _backend(version="0.27.2"),
        _backend(capabilities=["flashinfer"]),
        _backend(device="cpu"),
        _backend(rank=1),
        _backend(tp_size=2),
        _backend(pp_size=2),
        _backend(dp_size=2),
        _backend(topology=["rank:0"]),
    ],
)
def test_rust_validator_compares_every_backend_snapshot_field(data_backend: Desc) -> None:
    semantic, data = _graph([_node("a")], data_backend=data_backend)
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"


def test_rust_validator_requires_execution_semantic_key_and_edge_parity() -> None:
    semantic, data = _graph([_node("a")])
    data.nodes[0].semantic_key = None
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"

    semantic, data = _graph(
        [_node("a"), _node("b")], edges=[(_key("a"), _key("b"), "control")]
    )
    data.edges = []
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"


def test_rust_validator_requires_explicit_anchor_nodes_and_anchor_parity() -> None:
    anchor = Desc(
        backend="vllm",
        runner="mrv1",
        graph_mode="eager",
        hook="execute_model.post",
        externally_owned=True,
        mobility="pinned",
        service_class="critical",
    )
    semantic, data = _graph([_node("compute", anchor=anchor)])
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"

    wrong_identity = _node("wrong", kind="anchor", anchor=anchor)
    semantic, data = _graph([wrong_identity])
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"

    anchor_node = _node("unused", kind="anchor", anchor=anchor)
    anchor_node.node_key = Desc(
        namespace="anchor",
        owner="vllm/mrv1/eager/execute_model.post",
        ordinal=0,
    )
    semantic, data = _graph([anchor_node])
    data.nodes[0].anchor = Desc(
        backend="vllm",
        runner="mrv1",
        graph_mode="eager",
        hook="execute_model.post",
        externally_owned=True,
        mobility="pinned",
        service_class="best_effort",
    )
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"


def test_rust_validator_requires_exact_value_metadata_and_access_projections() -> None:
    semantic, data = _graph_with_access()
    changed_value = _value(shape=["batch", "other"])
    data.values = [changed_value]
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"

    semantic, data = _graph_with_access()
    data.nodes[0].accesses = []
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"

    semantic, data = _graph_with_access()
    data.accesses = []
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"

    semantic, data = _graph_with_access()
    data.accesses.append(_access(data.nodes[0].node_key, data.values[0], "write"))
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda semantic, _data: setattr(semantic.nodes[0], "kind", "future"),
        lambda semantic, _data: setattr(semantic.nodes[0], "graph_safety", "future"),
        lambda semantic, _data: setattr(semantic.nodes[0], "legacy_boundary", "future"),
        lambda semantic, _data: setattr(
            semantic.nodes[0],
            "executor",
            Desc(namespace="", name="executor", version="1"),
        ),
        lambda semantic, _data: setattr(semantic.nodes[0].timing, "window", "future"),
        lambda semantic, _data: setattr(semantic.nodes[0].timing, "deadline", "future"),
        lambda semantic, _data: setattr(semantic.backend, "tp_size", 0),
        lambda _semantic, data: setattr(data.nodes[0], "device", ""),
        lambda _semantic, data: setattr(data.nodes[0], "rank", -1),
    ],
)
def test_rust_validator_rejects_invalid_descriptor_domains(mutation: object) -> None:
    semantic, data = _graph([_node("a")])
    mutation(semantic, data)  # type: ignore[operator]
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"


def test_rust_validator_rejects_invalid_access_anchor_and_anchor_edge_domains() -> None:
    semantic, data = _graph_with_access()
    semantic.nodes[0].reads[0].consistency = ""
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"

    anchor = Desc(
        backend="vllm",
        runner="mrv1",
        graph_mode="eager",
        hook="execute_model.post",
        externally_owned=True,
        mobility="pinned",
        service_class="",
    )
    anchor_key = Desc(
        namespace="anchor",
        owner="vllm/mrv1/eager/execute_model.post",
        ordinal=0,
    )
    anchor_node = _node("unused", kind="anchor", anchor=anchor)
    anchor_node.node_key = anchor_key
    semantic, data = _graph(
        [anchor_node, _node("compute")],
        edges=[(anchor_key, _key("compute"), "anchor")],
    )
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"

    semantic, data = _graph(
        [_node("source"), _node("target")],
        edges=[(_key("source"), _key("target"), "anchor")],
    )
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"


def test_rust_validator_accepts_canonical_python_descriptors() -> None:
    backend = BackendSnapshot(
        backend_id="vllm",
        runner_family="mrv2",
        graph_mode="piecewise",
        backend_version="0.27.1",
    )
    anchor = AnchorDesc("vllm", "mrv2", "piecewise", "execute_model.post")
    anchor_key = NodeKey.anchor("vllm", "mrv2", "piecewise", "execute_model.post")
    first = NodeKey.consumer_flow("canonical", 0)
    second = NodeKey.consumer_flow("canonical", 1)
    semantic = SemanticGraph(
        nodes=(
            SemanticNodeDesc(anchor_key, kind="anchor", anchor=anchor),
            SemanticNodeDesc(first),
            SemanticNodeDesc(second),
        ),
        edges=((anchor_key, first, "anchor"), (first, second, "control")),
        backend=backend,
        graph_mode="piecewise",
    )
    data = DataCentricIR(
        nodes=(
            ExecutionNodeDesc(anchor_key, semantic_key=anchor_key, kind="anchor", anchor=anchor),
            ExecutionNodeDesc(first, semantic_key=first),
            ExecutionNodeDesc(second, semantic_key=second),
        ),
        edges=((anchor_key, first, "anchor"), (first, second, "control")),
        backend=backend,
        graph_mode="piecewise",
    )
    result = _core.validate_workgraph(semantic, data)
    assert result.topological_order == [
        "anchor/vllm/mrv2/piecewise/execute_model.post",
        "consumer_flow/canonical/0",
        "consumer_flow/canonical/1",
    ]


def test_rust_validator_matches_oracle_for_anchor_and_ir_parity_failures() -> None:
    wrong_anchor = Desc(
        backend="other-backend",
        runner="mrv1",
        graph_mode="eager",
        hook="execute_model.post",
        externally_owned=True,
        mobility="pinned",
        service_class="critical",
    )
    semantic, data = _graph([_node("a", anchor=wrong_anchor)])
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"

    semantic, data = _graph([_node("a")])
    data.nodes = []
    rust = _outcome(_core.validate_workgraph, semantic, data)
    oracle = _outcome(oracle_validate, semantic, data)
    assert rust[0] == oracle[0] == "err"


def test_rust_validator_checks_structured_port_refs_and_lifetime_refs() -> None:
    backend = BackendSnapshot(
        backend_id="vllm",
        runner_family="mrv1",
        graph_mode="eager",
        backend_version="0.27.1",
    )
    key = NodeKey.consumer_flow("ports", 0)
    port = PortId("demo", "hidden")
    value = ValueDesc(
        "hidden",
        port=port,
        shape=("batch", "hidden"),
        dtype="bfloat16",
        layout="row_major",
        alias_set="hidden_state",
    )
    access = AccessDesc(key, value, AccessMode.READ, port=port)
    semantic = SemanticGraph(
        nodes=(SemanticNodeDesc(key, reads=(access,)),),
        values=(value,),
        backend=backend,
    )
    data = DataCentricIR(
        nodes=(
            ExecutionNodeDesc(
                key,
                semantic_key=key,
                accesses=(access,),
                residence_keys=(value.key,),
                lease_keys=(value.key,),
            ),
        ),
        values=(value,),
        accesses=(access,),
        residences=(ResidenceDesc(value, "cuda"),),
        leases=(LeaseDesc(value),),
        backend=backend,
    )
    assert _core.validate_workgraph(semantic, data).topological_order == [str(key)]

    with pytest.raises(ValueError, match="must match ValueDesc.port"):
        AccessDesc(key, value, AccessMode.READ, port=PortId("other", "hidden"))

    bad_data = Desc(
        schema_version=1,
        nodes=[
            Desc(
                node_key=key,
                semantic_key=key,
                kind="compute",
                executor=None,
                accesses=[access],
                residence_keys=["missing@1"],
                move_keys=[],
                lease_keys=[value.key],
                device="cuda",
                rank=0,
                anchor=None,
                graph_safety="eager_only",
                fusion_group="",
            )
        ],
        values=[value],
        accesses=[access],
        residences=[ResidenceDesc(value, "cuda")],
        moves=[],
        leases=[LeaseDesc(value)],
        edges=[],
        backend=backend,
        graph_mode="eager",
    )
    with pytest.raises(ValueError, match="unknown residence"):
        _core.validate_workgraph(semantic, bad_data)
