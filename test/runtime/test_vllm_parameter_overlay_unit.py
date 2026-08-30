"""Generic low-rank overlay translation into vLLM's adapter manager."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import pytest
import torch
from torch.nn import Parameter
from torch.nn import functional as F
from vllm.lora.lora_model import LoRAModel
from vllm.lora.request import LoRARequest

from tllm.backends.protocol import OverlayBackendState
from tllm.backends.vllm.versioned_parameter_overlay import (
    VLLMParameterOverlayController,
    maybe_bind_publisher,
)
from tllm.backends.vllm.runner_adapter import PackedRowPlanInputs
from tllm.contracts.request_identity import RequestFinishedBatch, RequestIdentity
from tllm.runtime.parameter_overlay import (
    DenseOperandCaptureBatch,
    OverlayChannel,
    OverlayCaptureRows,
    OverlayLeaseKey,
    OverlayLineage,
    OverlayPayload,
    OverlayPublication,
    OverlayRepresentation,
    OverlayRowAssignment,
    OverlayRowPhase,
    OverlayTarget,
    OverlayVersionRef,
    OverlayVersionStore,
)


class _AdapterManager:
    def __init__(self) -> None:
        self.models: dict[int, LoRAModel] = {}
        self.fail_after_remove: int | None = None

    def list_adapters(self) -> dict[int, LoRAModel]:
        return dict(self.models)

    def add_adapter(self, model: LoRAModel) -> bool:
        self.models[model.id] = model
        return True

    def remove_adapter(self, lora_id: int) -> bool:
        self.models.pop(lora_id)
        if lora_id == self.fail_after_remove:
            raise RuntimeError("injected remove failure")
        return True


@dataclass(slots=True)
class _LoRAManager:
    _adapter_manager: _AdapterManager


@dataclass(slots=True)
class _LoRAConfig:
    max_lora_rank: int


@dataclass(slots=True)
class _RequestStates:
    req_id_to_index: Mapping[str, int]


@dataclass(slots=True)
class _LoRAState:
    requests: dict[str, LoRARequest] = field(default_factory=dict[str, LoRARequest])

    def add_request(
        self, request_id: str, request_index: int, request: LoRARequest
    ) -> None:
        del request_index
        self.requests[request_id] = request


@dataclass(slots=True)
class _RequestLoRAMapping:
    values: dict[int, int] = field(default_factory=dict[int, int])

    def __setitem__(self, index: int, value: int) -> None:
        self.values[index] = value


@dataclass(slots=True)
class _InputBatch:
    req_id_to_index: Mapping[str, int] = field(default_factory=dict[str, int])
    request_lora_mapping: _RequestLoRAMapping = field(
        default_factory=_RequestLoRAMapping
    )
    lora_id_to_request_ids: dict[int, set[str]] = field(
        default_factory=dict[int, set[str]]
    )
    lora_id_to_lora_request: dict[int, LoRARequest] = field(
        default_factory=dict[int, LoRARequest]
    )


_DenseOutput = torch.Tensor | tuple[torch.Tensor, Parameter | None]


class _DenseLinear:
    def __init__(self) -> None:
        self.weight = Parameter(
            torch.tensor(
                [[0.2, -0.3, 0.5], [0.4, 0.1, -0.2]],
                dtype=torch.float64,
            ),
            requires_grad=False,
        )
        self.bias: Parameter | None = None
        self.tp_size = 1
        self.return_bias = True
        self.forward: Callable[[torch.Tensor], _DenseOutput] = self._base_forward

    def _base_forward(self, input_: torch.Tensor) -> _DenseOutput:
        return F.linear(input_, self.weight), None


class _MLP:
    def __init__(self, down_proj: _DenseLinear) -> None:
        self.down_proj = down_proj
        self.forward: Callable[[torch.Tensor], torch.Tensor] = (
            self._base_forward
        )

    def _base_forward(self, mlp_input: torch.Tensor) -> torch.Tensor:
        activated = torch.cat(
            (mlp_input, mlp_input.sum(dim=1, keepdim=True)), dim=1
        )
        return self.down_proj.forward(activated)[0]


@dataclass(slots=True)
class _Layer:
    mlp: _MLP


@dataclass(slots=True)
class _Decoder:
    layers: Sequence[_Layer]


@dataclass(slots=True)
class _CausalModel:
    model: _Decoder


@dataclass(slots=True)
class _ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1


@dataclass(slots=True)
class _ModelConfig:
    quantization: str | None = None


@dataclass(slots=True)
class _VllmConfig:
    model_config: _ModelConfig = field(default_factory=_ModelConfig)


@dataclass(slots=True)
class _Runner:
    lora_manager: _LoRAManager
    lora_config: _LoRAConfig
    req_states: _RequestStates
    lora_state: _LoRAState
    input_batch: _InputBatch
    model: _CausalModel
    parallel_config: _ParallelConfig
    vllm_config: _VllmConfig


@dataclass(slots=True)
class _Execution:
    parameter_overlays: Mapping[OverlayChannel, OverlayVersionStore]
    finished: list[RequestFinishedBatch] = field(
        default_factory=list[RequestFinishedBatch]
    )
    lifecycle_ports: int = 1
    captured: list[DenseOperandCaptureBatch] = field(
        default_factory=list[DenseOperandCaptureBatch]
    )

    def stage_request_finished(self, batch: RequestFinishedBatch) -> int:
        if not self.lifecycle_ports:
            return 0
        self.finished.append(batch)
        for store in set(self.parameter_overlays.values()):
            for request in batch.requests:
                store.retire_request(request)
        return self.lifecycle_ports

    def stage_dense_operand_captured(
        self, batch: DenseOperandCaptureBatch
    ) -> int:
        self.captured.append(batch)
        return 1


@dataclass(slots=True)
class _Runtime:
    reqid_to_promptidx: Mapping[str, int]
    reqid_to_sampleidx: Mapping[str, int]
    workload_execution: _Execution


@dataclass(slots=True)
class _OverlayHarness:
    manager: _AdapterManager
    runner: _Runner
    runtime: _Runtime
    state: OverlayBackendState
    dense_linear: _DenseLinear

    @classmethod
    def create(
        cls, stores: Mapping[OverlayChannel, OverlayVersionStore]
    ) -> _OverlayHarness:
        manager = _AdapterManager()
        dense_linear = _DenseLinear()
        runner = _Runner(
            _LoRAManager(manager),
            _LoRAConfig(max_lora_rank=8),
            _RequestStates({}),
            _LoRAState(),
            _InputBatch(),
            _CausalModel(_Decoder((_Layer(_MLP(dense_linear)),))),
            _ParallelConfig(),
            _VllmConfig(),
        )
        return cls(
            manager=manager,
            runner=runner,
            runtime=_Runtime(
                reqid_to_promptidx={"active": 0, "new": 1, "base": 2},
                reqid_to_sampleidx={"active": 0, "new": 0, "base": 0},
                workload_execution=_Execution(stores),
            ),
            state=OverlayBackendState(),
            dense_linear=dense_linear,
        )

    def bind(self) -> VLLMParameterOverlayController:
        controller = maybe_bind_publisher(
            runtime=self.runtime,
            runner=self.runner,
            state=self.state,
        )
        assert controller is not None
        return controller


def _payload(value: float) -> OverlayPayload:
    prefix = "base_model.model.model.layers.0.self_attn"
    return OverlayPayload(
        OverlayRepresentation.LOW_RANK_ADD,
        {
            f"{prefix}.q_proj.lora_A.default.weight": torch.full((2, 4), value),
            f"{prefix}.q_proj.lora_B.default.weight": torch.full((4, 2), value),
            f"{prefix}.v_proj.lora_A.default.weight": torch.full((2, 4), value),
            f"{prefix}.v_proj.lora_B.default.weight": torch.full((4, 2), value),
            "__tllm_lora_alpha__": torch.tensor(4.0),
        },
    )


def _store(value: float) -> tuple[OverlayChannel, OverlayVersionStore]:
    channel = OverlayChannel("tlm", "qv-lora")
    store = OverlayVersionStore()
    store.stage(
        "tx:1", OverlayPublication(OverlayLineage(channel), 1, _payload(value))
    )
    store.commit("tx:1")
    return channel, store


def test_controller_materializes_and_keeps_request_version_stable() -> None:
    channel, store = _store(1.0)
    harness = _OverlayHarness.create({channel: store})
    controller = harness.bind()

    assert controller.sync_versions() == 1
    controller.pin_request("active")
    first = store.pinned(
        OverlayLeaseKey(
            RequestIdentity("active", 0, 0), OverlayLineage(channel)
        )
    )
    assert first is not None
    store.stage(
        "tx:2",
        OverlayPublication(OverlayLineage(channel), 1, _payload(2.0)),
    )
    store.commit("tx:2")
    assert controller.sync_versions() == 1

    controller.pin_request("active")
    controller.pin_request("new")
    active = store.pinned(
        OverlayLeaseKey(
            RequestIdentity("active", 0, 0), OverlayLineage(channel)
        )
    )
    new = store.pinned(
        OverlayLeaseKey(
            RequestIdentity("new", 1, 0), OverlayLineage(channel)
        )
    )
    assert active == first
    assert new is not None and new.ref != first.ref
    model = harness.manager.models[next(iter(harness.manager.models))]
    assert set(model.loras) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    }
    assert controller.finish_requests(("active",)).requests == (
        RequestIdentity("active", 0, 0),
    )


def test_epoch_rebind_releases_old_leases_removes_adapters_and_allocates_new_ids() -> None:
    channel, first_store = _store(1.0)
    _, second_store = _store(2.0)
    harness = _OverlayHarness.create({channel: first_store})
    first = harness.bind()
    assert first.sync_versions() == 1
    first.pin_request("active")
    first_id = next(iter(harness.manager.models))

    harness.runtime.workload_execution = _Execution({channel: second_store})
    with pytest.raises(RuntimeError, match="requests are active"):
        harness.bind()
    harness.runtime.workload_execution = _Execution({channel: first_store})
    first.finish_requests(("active",))

    harness.runtime.workload_execution = _Execution({channel: second_store})
    second = harness.bind()
    assert second is not first
    assert first_store.pinned(
        OverlayLeaseKey(
            RequestIdentity("active", 0, 0), OverlayLineage(channel)
        )
    ) is None
    assert harness.manager.models == {}

    assert second.sync_versions() == 1
    second.pin_request("active")
    assert next(iter(harness.manager.models)) != first_id


def test_close_removes_remaining_adapters_after_one_remove_failure() -> None:
    channel, store = _store(1.0)
    harness = _OverlayHarness.create({channel: store})
    controller = harness.bind()
    assert controller.sync_versions() == 1
    controller.pin_request("active")
    first_id = next(iter(harness.manager.models))
    store.stage(
        "tx:2",
        OverlayPublication(OverlayLineage(channel), 1, _payload(2.0)),
    )
    store.commit("tx:2")
    assert controller.sync_versions() == 1
    harness.manager.fail_after_remove = first_id
    controller.finish_requests(("active",))

    with pytest.raises(RuntimeError, match="could not remove"):
        controller.close()

    assert harness.manager.models == {}
    assert store.pinned(
        OverlayLeaseKey(
            RequestIdentity("active", 0, 0), OverlayLineage(channel)
        )
    ) is None


def test_dense_versions_apply_by_packed_request_row_and_restore_on_close() -> None:
    target = OverlayTarget(0, "mlp", "down_proj.weight")
    channel = OverlayChannel("inplace_ttt", "layer-0", target)
    store = OverlayVersionStore()
    identities = (
        RequestIdentity("active", 0, 0),
        RequestIdentity("new", 1, 0),
    )
    weights = (
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    refs: list[OverlayVersionRef] = []
    for ordinal, (identity, weight) in enumerate(
        zip(identities, weights, strict=True), start=1
    ):
        lineage = OverlayLineage(channel, identity)
        transaction = f"dense:{ordinal}"
        store.stage(
            transaction,
            OverlayPublication(
                lineage,
                1,
                OverlayPayload(
                    OverlayRepresentation.DENSE_REPLACE,
                    {"weight": weight},
                ),
            ),
        )
        version = store.commit(transaction)
        store.advance_request(OverlayLeaseKey(identity, lineage), version.ref)
        refs.append(version.ref)

    harness = _OverlayHarness.create({channel: store})
    original_forward = harness.dense_linear.forward
    controller = harness.bind()
    assert controller.sync_versions() == 2
    plan = controller.prepare_dense_step(
        step_id=7,
        columns=PackedRowPlanInputs(
            request_ids=("active-a1B2c3D4", "new", "base"),
            scheduled_tokens=(2, 1, 1),
            prompt_tokens=(3, 1, 1),
            computed_tokens=(0, 1, 1),
        ),
    )
    assignments = {assignment.version: assignment for assignment in plan.assignments}
    assert assignments[refs[0]].packed_rows == (0, 1)
    assert assignments[refs[0]].phases == (
        OverlayRowPhase.PREFILL,
        OverlayRowPhase.PREFILL,
    )
    assert assignments[refs[1]].packed_rows == (2,)
    assert assignments[refs[1]].phases == (OverlayRowPhase.DECODE,)

    hidden = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 2.0],
            [4.0, -2.0, 1.0],
            [0.25, 0.5, 0.75],
        ],
        dtype=torch.float64,
    )
    expected = F.linear(hidden, harness.dense_linear.weight)
    expected[:2] = F.linear(hidden[:2], weights[0])
    expected[2:3] = F.linear(hidden[2:3], weights[1])
    output = harness.dense_linear.forward(hidden)[0]
    torch.testing.assert_close(output, expected)

    controller.clear_dense_step()
    torch.testing.assert_close(
        harness.dense_linear.forward(hidden)[0],
        F.linear(hidden, harness.dense_linear.weight),
    )
    speculative = controller.prepare_dense_step(
        step_id=8,
        columns=PackedRowPlanInputs(
            request_ids=("new",),
            scheduled_tokens=(3,),
            prompt_tokens=(1,),
            computed_tokens=(1,),
        ),
    )
    assert speculative.assignments == (
        OverlayRowAssignment(
            target=target,
            version=refs[1],
            packed_rows=(0, 1, 2),
            phases=(
                OverlayRowPhase.DECODE,
                OverlayRowPhase.DECODE,
                OverlayRowPhase.DECODE,
            ),
        ),
    )
    controller.clear_dense_step()
    assert controller.finish_requests(("active",)).requests == (
        identities[0],
    )
    assert "active-a1B2c3D4" not in harness.state.request_identities
    assert "active-a1B2c3D4" not in harness.state.dense_request_ids
    assert controller.sync_versions() == 0
    handle = harness.state.dense_targets[target]
    assert not handle.has_version(refs[0])
    assert handle.has_version(refs[1])

    assert controller.finish_requests(("new",)).requests == (identities[1],)
    assert controller.sync_versions() == 0
    assert target not in harness.state.dense_targets
    assert harness.dense_linear.forward is original_forward

    controller.finish_requests(("base",))
    controller.close()
    assert harness.dense_linear.forward is original_forward


def test_dense_target_validation_rejects_physical_aliases() -> None:
    first = OverlayChannel(
        "inplace_ttt",
        "first",
        OverlayTarget(0, "mlp", "down_proj.weight"),
    )
    alias = OverlayChannel(
        "inplace_ttt",
        "alias",
        OverlayTarget(-1, "mlp", "down_proj.weight"),
    )
    harness = _OverlayHarness.create(
        {first: OverlayVersionStore(), alias: OverlayVersionStore()}
    )

    with pytest.raises(RuntimeError, match="alias the same physical"):
        harness.bind()


def test_typed_capture_records_real_mlp_and_down_proj_operands_once() -> None:
    target = OverlayTarget(0, "mlp", "down_proj.weight")
    channel = OverlayChannel(
        "inplace_ttt", "layer-0", target, capture_operands=True
    )
    harness = _OverlayHarness.create({channel: OverlayVersionStore()})
    mlp = harness.runner.model.model.layers[0].mlp
    original_mlp_forward = mlp.forward
    original_down_forward = harness.dense_linear.forward
    base_weight = harness.dense_linear.weight.detach().clone()
    controller = harness.bind()
    assert controller.sync_versions() == 0
    assert controller.requires_dense_step()
    assert not controller.has_dense_versions()
    plan = controller.prepare_dense_step(
        step_id=9,
        columns=PackedRowPlanInputs(
            request_ids=("active",),
            scheduled_tokens=(2,),
            prompt_tokens=(2,),
            computed_tokens=(0,),
        ),
    )
    assert plan.captures == (
        OverlayCaptureRows(
            target=target,
            request=RequestIdentity("active", 0, 0),
            phase=OverlayRowPhase.PREFILL,
            packed_rows=(0, 1),
            token_offsets=(0, 1),
        ),
    )
    mlp_input = torch.tensor(
        [[1.0, 2.0], [-0.5, 0.25]], dtype=torch.float64
    )
    expected_down_input = torch.cat(
        (mlp_input, mlp_input.sum(dim=1, keepdim=True)), dim=1
    )
    with pytest.raises(RuntimeError, match="every signed target"):
        controller.stage_dense_operand_captured()
    mlp.forward(mlp_input)
    assert controller.stage_dense_operand_captured() == 1
    (captured_batch,) = harness.runtime.workload_execution.captured
    (capture,) = captured_batch.captures
    assert capture.step_id == 9
    assert capture.target == target
    assert capture.rows == plan.captures
    torch.testing.assert_close(capture.mlp_input, mlp_input)
    torch.testing.assert_close(capture.down_proj_input, expected_down_input)
    torch.testing.assert_close(capture.base_weight, base_weight)

    controller.clear_dense_step()
    controller.finish_requests(("active",))
    controller.close()
    assert mlp.forward is original_mlp_forward
    assert harness.dense_linear.forward is original_down_forward


def test_global_lora_and_request_dense_versions_share_one_request_lifecycle() -> None:
    lora_channel, lora_store = _store(1.0)
    target = OverlayTarget(0, "mlp", "down_proj.weight")
    dense_channel = OverlayChannel("inplace_ttt", "layer-0", target)
    identity = RequestIdentity("active", 0, 0)
    dense_lineage = OverlayLineage(dense_channel, identity)
    dense_store = OverlayVersionStore()
    dense_store.stage(
        "dense:1",
        OverlayPublication(
            dense_lineage,
            1,
            OverlayPayload(
                OverlayRepresentation.DENSE_REPLACE,
                {"weight": torch.zeros((2, 3), dtype=torch.float64)},
            ),
        ),
    )
    dense_version = dense_store.commit("dense:1")
    dense_store.advance_request(
        OverlayLeaseKey(identity, dense_lineage), dense_version.ref
    )
    harness = _OverlayHarness.create(
        {lora_channel: lora_store, dense_channel: dense_store}
    )
    controller = harness.bind()
    assert controller.sync_versions() == 2
    active_lora = controller.pin_request("active")
    assert active_lora is not None
    controller.prepare_dense_step(
        step_id=1,
        columns=PackedRowPlanInputs(
            request_ids=("active",),
            scheduled_tokens=(1,),
            prompt_tokens=(1,),
            computed_tokens=(1,),
        ),
    )

    lora_store.stage(
        "tx:2",
        OverlayPublication(
            OverlayLineage(lora_channel), 1, _payload(2.0)
        ),
    )
    lora_store.commit("tx:2")
    assert controller.sync_versions() == 1
    repinned_lora = controller.pin_request("active")
    assert repinned_lora is not None
    assert repinned_lora.lora_int_id == active_lora.lora_int_id
    new_lora = controller.pin_request("new")
    assert new_lora is not None
    assert new_lora.lora_int_id != active_lora.lora_int_id

    controller.clear_dense_step()
    assert controller.finish_requests(("active",)).requests == (identity,)
    assert dense_store.lineages() == ()
    assert lora_store.latest(OverlayLineage(lora_channel)) is not None
    assert controller.sync_versions() == 0
    assert target not in harness.state.dense_targets
    controller.finish_requests(("new",))
    controller.close()


def test_finished_request_is_staged_before_its_first_dense_row_plan() -> None:
    target = OverlayTarget(0, "mlp", "down_proj.weight")
    channel = OverlayChannel("inplace_ttt", "layer-0", target)
    identity = RequestIdentity("active", 0, 0)
    lineage = OverlayLineage(channel, identity)
    store = OverlayVersionStore()
    store.stage(
        "dense:1",
        OverlayPublication(
            lineage,
            1,
            OverlayPayload(
                OverlayRepresentation.DENSE_REPLACE,
                {
                    "weight": torch.zeros(
                        (2, 3), dtype=torch.float64
                    )
                },
            ),
        ),
    )
    version = store.commit("dense:1")
    store.advance_request(OverlayLeaseKey(identity, lineage), version.ref)
    harness = _OverlayHarness.create({channel: store})
    controller = harness.bind()
    assert controller.sync_versions() == 1

    harness.runtime.workload_execution.lifecycle_ports = 0
    with pytest.raises(RuntimeError, match="signed request-finished port"):
        controller.finish_requests(("active",))
    assert store.lineages() == (lineage,)
    harness.runtime.workload_execution.lifecycle_ports = 1
    assert controller.finish_requests(("active",)).requests == (identity,)
    assert harness.runtime.workload_execution.finished == [
        RequestFinishedBatch((identity,))
    ]
    assert store.lineages() == ()
    assert controller.sync_versions() == 0
    assert target not in harness.state.dense_targets


@pytest.mark.parametrize(
    ("parallel", "model_config", "message"),
    (
        (_ParallelConfig(tensor_parallel_size=2), _ModelConfig(), "tensor parallel"),
        (_ParallelConfig(pipeline_parallel_size=2), _ModelConfig(), "pipeline parallel"),
        (_ParallelConfig(), _ModelConfig(quantization="gptq"), "unquantized"),
    ),
)
def test_dense_target_validation_rejects_unsupported_backend_facts(
    parallel: _ParallelConfig,
    model_config: _ModelConfig,
    message: str,
) -> None:
    channel = OverlayChannel(
        "inplace_ttt",
        "layer-0",
        OverlayTarget(0, "mlp", "down_proj.weight"),
    )
    harness = _OverlayHarness.create({channel: OverlayVersionStore()})
    harness.runner.parallel_config = parallel
    harness.runner.vllm_config = _VllmConfig(model_config)

    with pytest.raises(RuntimeError, match=message):
        harness.bind()
