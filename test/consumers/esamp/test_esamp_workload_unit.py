from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import torch

from tllm.consumers.esamp.consumer import ESampConsumer
from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.model import LowRankGatedResidualModel
from tllm.contracts.hidden_locator import HiddenPhase, HiddenLocatorCapability
from tllm.contracts.packed_rows import (
    PackedPhaseRow,
    PackedPhaseRowBatch,
    PackedSamplerRowBatch,
    PackedSamplerRowLink,
)
from tllm.ports.sampler import PackedSamplerGuidanceBatch
from tllm.consumers.esamp.initializers.svd import (
    SVDModelBankInitializerConfig,
)
from tllm.consumers.esamp.legacy_workload import build_esamp_workload
from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.backends.vllm.v027_mrv1 import V027MRv1Driver
from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
from tllm.contracts.backend_capability import BackendCapability
from tllm.consumers.esamp.workload import ESampConfig, ESampWorkload
from tllm.contracts.request_identity import (
    RequestFinishedBatch,
    RequestIdentity,
    RequestKeyField,
)
from tllm.runtime.request_trainable_state import (
    IndexedRequestIdentityBatch,
    IndexedRequestIdentityRow,
    RequestIdentityBatch,
    RequestStateSelectionBatch,
    RequestStateRuntimeService,
    StateEffectOperation,
)
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime import residual_runtime
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_plugins import reconstruct_worker_bindings
from tllm.runtime.workload_bindings import (
    LocalRuntimeServices,
    SamplerRuntimeBinding,
    WorkerRuntimeBinding,
)
from tllm.runtime.mp.worker_extension import (
    TllmWorkerExtension,
    _decision_requires_hidden_taps,
)
from tllm.ports.residual_stream import ResidualLocator
from tllm.schedule import (
    AsyncBatchCaptureMode,
    HostCommandKind,
    MainWorkloadTimeline,
    RegionCaptureMode,
    RegionHostSubmission,
    RegionStreamClass,
    ScheduleStateStorage,
)
from tllm.workload import declare_workload
from tllm.workgraph.physical import CapturePolicy


def _config(**overrides: object) -> ESampConfig:
    values: dict[str, object] = {
        "intervention": True,
        "distiller_beta": 0.25,
        "distiller_hidden_dim": 3,
    }
    values.update(overrides)
    return ESampConfig(**values)  # type: ignore[arg-type]


def _decode_state(step_rows: int = 2) -> ResidualRuntimeState:
    state = ResidualRuntimeState()
    state.decode_count = step_rows
    state.decode_request_ids = [f"request-{row}" for row in range(step_rows)]
    state.decode_prompt_idxs = [10 + row for row in range(step_rows)]
    state.decode_sample_idxs = [row for row in range(step_rows)]
    return state


class _Model:
    def __init__(self) -> None:
        self.lm_head = torch.nn.Linear(4, 7, bias=False)


def _view(
    step_id: int,
    rows: int = 2,
    *,
    request_ids: tuple[str, ...] | None = None,
    prompt_idxs: tuple[int, ...] | None = None,
    sample_idxs: tuple[int, ...] | None = None,
) -> SamplerStepView:
    return SamplerStepView(
        engine_step_id=step_id,
        phase="decode",
        logits=torch.empty(rows, 7),
        sampling_metadata=object(),
        decode_count=rows,
        request_ids=request_ids or tuple(f"request-{row}" for row in range(rows)),
        prompt_idxs=prompt_idxs or tuple(10 + row for row in range(rows)),
        sample_idxs=sample_idxs or tuple(row for row in range(rows)),
        prompt_idx_tensor=None,
        sample_idx_tensor=None,
        source_hidden=torch.empty(rows, 4),
        device=torch.device("cpu"),
        model=_Model(),
        runner=object(),
    )


def _capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        backend_id="vllm-0.27-mrv2",
        runner_family="mrv2",
        sampler_stages=frozenset(),
        native_min_p=False,
        tuple_sampler_return=False,
        split_execute_and_sample=True,
        graph_safety=GraphSafety.EAGER_ONLY,
        eager_extensions=frozenset(
            {
                BackendCapability.PACKED_PHASE_ROWS,
                BackendCapability.PACKED_SAMPLER_ROWS,
                BackendCapability.PACKED_HIDDEN_BLOCK_OUTPUT,
            }
        ),
        hidden_locators=frozenset({HiddenLocatorCapability.BLOCK_OUTPUT_DECODE}),
    )


def _timeline() -> MainWorkloadTimeline:
    return export_main_workload_timeline(
        _capabilities(), graph_mode="eager", static_headroom_bytes=1 << 20
    )


def _specialized_workload(**overrides: object) -> ESampWorkload:
    workload = ESampWorkload(_config(**overrides))
    workload.cold_specialize(
        RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
    )
    return workload


def _request_workload(
    *,
    rows: int = 4,
    intervention: bool = False,
    distiller_beta: float = 0.0,
) -> ESampWorkload:
    workload = ESampWorkload(
        ESampConfig(
            per_request_models=True,
            distiller_hidden_dim=2,
            distiller_lr=1e-2,
            intervention=intervention,
            distiller_beta=distiller_beta,
        )
    )
    workload.cold_specialize(
        RuntimeSpecializationSpec(
            rows, 4, torch.float32, torch.device("cpu")
        )
    )
    return workload


def _request_rows(
    *,
    step_id: int,
    identities: tuple[RequestIdentity, ...],
) -> PackedPhaseRowBatch:
    return PackedPhaseRowBatch(
        plan_epoch=1,
        step_id=step_id,
        total_rows=len(identities),
        rows=tuple(
            PackedPhaseRow(
                packed_row_index=row_index,
                request_ordinal=row_index,
                raw_request_id=identity.request_id,
                token_offset=0,
                phase=HiddenPhase.DECODE,
                identity=identity,
                decode_plan_ordinal=row_index,
            )
            for row_index, identity in enumerate(identities)
        ),
    )


def _sampler_rows(
    rows: PackedPhaseRowBatch,
) -> PackedSamplerRowBatch:
    return PackedSamplerRowBatch(
        row_batch=rows,
        links=tuple(
            PackedSamplerRowLink(
                sampler_row_index=row_index,
                packed_row_index=row_index,
            )
            for row_index in range(rows.total_rows)
        ),
    )


def _run_request_step(
    workload: ESampWorkload,
    service: RequestStateRuntimeService[LowRankGatedResidualModel],
    *,
    step_id: int,
    identities: tuple[RequestIdentity, ...],
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[
    PackedSamplerGuidanceBatch,
    tuple[torch.Tensor, ...],
    RequestStateSelectionBatch,
]:
    service.stage_identities(step_id, RequestIdentityBatch(identities))
    selection = service.selection_for_step(step_id)
    rows = _request_rows(step_id=step_id, identities=identities)
    sampler_rows = _sampler_rows(rows)

    prediction_read = service.before(
        StateEffectOperation.READ, selection, step_id
    )
    guidance, training_source, training_target = workload.request_predict(
        rows, sampler_rows, source, target, selection
    )
    service.after(prediction_read)

    loss_read = service.before(StateEffectOperation.READ, selection, step_id)
    loss_batch = workload.request_loss(
        rows, training_source, training_target, selection
    )
    service.after(loss_read)

    transition = service.before(
        StateEffectOperation.TRANSITION, selection, step_id
    )
    workload.request_train(rows, loss_batch, selection)
    service.after(transition)
    service.finish_step(step_id)
    return guidance, loss_batch.losses, selection


def _publish_guidance(workload: ESampWorkload, step_id: int) -> torch.Tensor:
    """Drive the real signed sink path: task output -> auto-bound store."""
    prediction, _materialized, _target, _sample_indices = workload.predict(
        torch.randn(2, 4),
        torch.randn(2, 4),
        torch.zeros(2, dtype=torch.long),
    )
    with mock.patch.object(residual_runtime, "RUNTIME", _decode_state(2)):
        workload.native_workload_resources().ports.outputs["guidance"].publish(
            prediction, step_id
        )
    return prediction


class ESampRequestWorkloadUnitTest(unittest.TestCase):
    def _legacy_reference(self) -> LowRankGatedResidualModel:
        train_model = LowRankGatedResidualModel(
            hidden_size=4,
            rank=2,
            use_output_layernorm=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        LowRankGatedResidualModel(
            hidden_size=4,
            rank=2,
            use_output_layernorm=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        return train_model

    def _reference_optimizer(
        self, model: LowRankGatedResidualModel
    ) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            model.parameters(),
            lr=1e-2,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            foreach=False,
            fused=None,
        )

    def test_request_parent_shares_one_lease_update_and_version_with_exact_oracle(
        self,
    ) -> None:
        torch.manual_seed(41)
        reference = self._legacy_reference()
        reference_optimizer = self._reference_optimizer(reference)
        torch.manual_seed(41)
        workload = _request_workload()
        service = workload.request_distiller

        identities = (
            RequestIdentity("parent-a", 7, 0),
            RequestIdentity("parent-a", 7, 1),
        )
        source = torch.tensor(
            [[0.2, -0.3, 0.4, -0.5], [0.6, 0.7, -0.8, 0.9]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[-0.1, 0.5, 0.3, -0.7], [0.8, -0.6, 0.2, 0.4]],
            dtype=torch.float32,
        )

        expected_loss = torch.nn.functional.mse_loss(
            reference(source).float(), target.float()
        )
        reference_optimizer.zero_grad(set_to_none=True)
        expected_loss.backward()
        reference_optimizer.step()

        guidance, losses, selection = _run_request_step(
            workload,
            service,
            step_id=3,
            identities=identities,
            source=source,
            target=target,
        )

        self.assertEqual(guidance.affected_row_ids, ())
        self.assertEqual(guidance.total_rows, 2)
        self.assertEqual(tuple(guidance.pred_hidden.shape), (0, 4))
        self.assertFalse(guidance.pred_hidden.requires_grad)
        self.assertIsNone(guidance.pred_hidden.grad_fn)
        self.assertEqual(selection.before_versions, (0,))
        self.assertEqual(len(selection.leases), 1)
        self.assertEqual(
            tuple(row.lease.slot for row in selection.rows),
            (selection.leases[0].slot, selection.leases[0].slot),
        )
        self.assertEqual(len(losses), 1)
        torch.testing.assert_close(losses[0], expected_loss)
        selection_versions = service.versions()
        self.assertEqual(len(selection_versions), 1)
        self.assertEqual(selection_versions[0].key.request_id, "parent-a")
        self.assertEqual(selection_versions[0].version, 1)
        service.stage_identities(4, RequestIdentityBatch((identities[0],)))
        inspection = service.selection_for_step(4)
        trained_module = service.groups(inspection)[0].module
        for actual, expected in zip(
            trained_module.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected)
        read_audit = service.before(
            StateEffectOperation.READ, inspection, 4
        )
        inspection_rows = _request_rows(
            step_id=4, identities=(identities[0],)
        )
        next_guidance, _, _ = workload.request_predict(
            inspection_rows,
            _sampler_rows(inspection_rows),
            source[:1],
            target[:1],
            inspection,
        )
        service.after(read_audit)
        self.assertEqual(inspection.before_versions, (1,))
        self.assertEqual(next_guidance.affected_row_ids, (0,))
        torch.testing.assert_close(
            next_guidance.pred_hidden,
            reference(source[:1]),
        )
        service.finish_step(4)
        self.assertEqual(
            workload.native_workload_resources().services.diagnostics(),
            {"loss_count": 1, "update_count": 1, "guidance_rows": 0},
        )

        service.retire_requests(RequestFinishedBatch(identities))
        self.assertEqual(service.versions(), ())

    def test_request_parents_remain_isolated_and_cleanup_releases_all_slots(self) -> None:
        torch.manual_seed(83)
        reference_a = self._legacy_reference()
        reference_b = self._legacy_reference()
        optimizer_a = self._reference_optimizer(reference_a)
        optimizer_b = self._reference_optimizer(reference_b)
        torch.manual_seed(83)
        workload = _request_workload(rows=3)
        service = workload.request_distiller

        parent_a_rows = (
            RequestIdentity("parent-a", 11, 0),
            RequestIdentity("parent-a", 11, 1),
        )
        parent_b = RequestIdentity("parent-b", 13, 0)
        identities = (parent_a_rows[0], parent_b, parent_a_rows[1])
        source = torch.tensor(
            [
                [0.1, 0.2, -0.3, 0.4],
                [-0.5, 0.6, 0.7, -0.8],
                [0.9, -1.0, 1.1, 1.2],
            ],
            dtype=torch.float32,
        )
        target = -source

        parent_a_indices = torch.tensor((0, 2), dtype=torch.long)
        parent_b_indices = torch.tensor((1,), dtype=torch.long)
        expected_loss_a = torch.nn.functional.mse_loss(
            reference_a(source.index_select(0, parent_a_indices)).float(),
            target.index_select(0, parent_a_indices).float(),
        )
        expected_loss_b = torch.nn.functional.mse_loss(
            reference_b(source.index_select(0, parent_b_indices)).float(),
            target.index_select(0, parent_b_indices).float(),
        )
        optimizer_a.zero_grad(set_to_none=True)
        expected_loss_a.backward()
        optimizer_a.step()
        optimizer_b.zero_grad(set_to_none=True)
        expected_loss_b.backward()
        optimizer_b.step()

        guidance, losses, selection = _run_request_step(
            workload,
            service,
            step_id=5,
            identities=identities,
            source=source,
            target=target,
        )

        self.assertEqual(guidance.affected_row_ids, ())
        self.assertEqual(guidance.total_rows, 3)
        self.assertEqual(tuple(guidance.pred_hidden.shape), (0, 4))
        self.assertEqual(selection.before_versions, (0, 0))
        self.assertEqual(len(selection.leases), 2)
        self.assertEqual(len(losses), 2)
        torch.testing.assert_close(losses[0], expected_loss_a)
        torch.testing.assert_close(losses[1], expected_loss_b)
        versions = {version.key.request_id: version.version for version in service.versions()}
        self.assertEqual(versions, {"parent-a": 1, "parent-b": 1})

        service.stage_identities(
            7, RequestIdentityBatch((parent_a_rows[0], parent_b))
        )
        inspection = service.selection_for_step(7)
        modules = {
            group.lease.key.request_id: group.module
            for group in service.groups(inspection)
        }
        for actual, expected in zip(
            modules["parent-a"].parameters(),
            reference_a.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(
            modules["parent-b"].parameters(),
            reference_b.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        service.finish_step(7)

        _run_request_step(
            workload,
            service,
            step_id=6,
            identities=(parent_a_rows[0],),
            source=source[:1],
            target=target[:1],
        )
        versions = {version.key.request_id: version.version for version in service.versions()}
        self.assertEqual(versions, {"parent-a": 2, "parent-b": 1})

        service.stage_identities(8, RequestIdentityBatch((parent_b,)))
        b_selection = service.selection_for_step(8)
        b_read = service.before(StateEffectOperation.READ, b_selection, 8)
        b_rows = _request_rows(step_id=8, identities=(parent_b,))
        b_guidance, _, _ = workload.request_predict(
            b_rows,
            _sampler_rows(b_rows),
            source[1:2],
            target[1:2],
            b_selection,
        )
        service.after(b_read)
        self.assertEqual(b_guidance.affected_row_ids, (0,))
        self.assertEqual(b_guidance.total_rows, 1)
        torch.testing.assert_close(
            b_guidance.pred_hidden, reference_b(source[1:2])
        )
        for actual, expected in zip(
            service.groups(b_selection)[0].module.parameters(),
            reference_b.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        service.finish_step(8)
        service.retire_requests(RequestFinishedBatch(identities))
        self.assertEqual(service.versions(), ())

    def test_request_guidance_maps_links_while_training_signed_decode_rows(self) -> None:
        workload = _request_workload(
            intervention=True,
            distiller_beta=0.5,
        )
        service = workload.request_distiller
        parent_a = RequestIdentity("parent-a", 41, 0)
        parent_b = RequestIdentity("parent-b", 92, 0)
        first_source = torch.tensor(
            [[0.2, -0.3, 0.4, -0.5]], dtype=torch.float32
        )
        first_guidance, _losses, first_selection = _run_request_step(
            workload,
            service,
            step_id=20,
            identities=(parent_a,),
            source=first_source,
            target=-first_source,
        )
        self.assertEqual(first_selection.before_versions, (0,))
        self.assertEqual(first_guidance.affected_row_ids, ())

        source = torch.tensor(
            [
                [0.1, 0.2, -0.3, 0.4],
                [-0.5, 0.6, 0.7, -0.8],
            ],
            dtype=torch.float32,
        )
        service.stage_indexed_identities(
            21,
            IndexedRequestIdentityBatch(
                (IndexedRequestIdentityRow(0, parent_a),)
            ),
        )
        selection = service.selection_for_step(21)
        self.assertEqual(selection.before_versions, (1,))
        rows = PackedPhaseRowBatch(
            plan_epoch=1,
            step_id=21,
            total_rows=2,
            rows=(
                PackedPhaseRow(
                    packed_row_index=0,
                    request_ordinal=0,
                    raw_request_id="parent-a",
                    token_offset=0,
                    phase=HiddenPhase.DECODE,
                    identity=parent_a,
                    decode_plan_ordinal=0,
                ),
                PackedPhaseRow(
                    packed_row_index=1,
                    request_ordinal=1,
                    raw_request_id="parent-b",
                    token_offset=0,
                    phase=HiddenPhase.PREFILL,
                    identity=parent_b,
                    decode_plan_ordinal=None,
                ),
            ),
        )
        sampler_rows = PackedSamplerRowBatch(
            row_batch=rows,
            links=(
                PackedSamplerRowLink(0, 0),
                PackedSamplerRowLink(1, 0),
                PackedSamplerRowLink(2, 1),
            ),
        )
        read_audit = service.before(
            StateEffectOperation.READ, selection, 21
        )
        guidance, training_source, training_target = (
            workload.request_predict(
                rows,
                sampler_rows,
                source,
                -source,
                selection,
            )
        )
        service.after(read_audit)
        self.assertEqual(guidance.affected_row_ids, (0, 1))
        self.assertEqual(guidance.total_rows, 3)
        expected = service.groups(selection)[0].module(source[:1])
        torch.testing.assert_close(
            guidance.pred_hidden,
            torch.cat((expected, expected), dim=0),
        )
        loss_read = service.before(StateEffectOperation.READ, selection, 21)
        loss_batch = workload.request_loss(
            rows, training_source, training_target, selection
        )
        service.after(loss_read)
        self.assertEqual(len(loss_batch.losses), 1)
        transition = service.before(
            StateEffectOperation.TRANSITION, selection, 21
        )
        workload.request_train(rows, loss_batch, selection)
        service.after(transition)
        self.assertEqual(
            {
                version.key.request_id: version.version
                for version in service.versions()
            },
            {"parent-a": 2},
        )
        service.finish_step(21)

    def test_reused_slot_matches_the_next_legacy_owner_initialization(self) -> None:
        torch.manual_seed(117)
        first_reference = self._legacy_reference()
        second_reference = self._legacy_reference()
        torch.manual_seed(117)
        workload = _request_workload(rows=1)
        service = workload.request_distiller

        first_identity = RequestIdentity("first-parent", 21, 0)
        service.stage_identities(
            9, RequestIdentityBatch((first_identity,))
        )
        first_selection = service.selection_for_step(9)
        first_module = service.groups(first_selection)[0].module
        first_address = next(first_module.parameters()).data_ptr()
        for actual, expected in zip(
            first_module.parameters(),
            first_reference.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        service.finish_step(9)
        service.retire_requests(RequestFinishedBatch((first_identity,)))

        second_identity = RequestIdentity("second-parent", 22, 0)
        service.stage_identities(
            10, RequestIdentityBatch((second_identity,))
        )
        second_selection = service.selection_for_step(10)
        second_module = service.groups(second_selection)[0].module
        self.assertEqual(
            next(second_module.parameters()).data_ptr(), first_address
        )
        for actual, expected in zip(
            second_module.parameters(),
            second_reference.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        self.assertFalse(
            torch.equal(
                next(first_reference.parameters()),
                next(second_reference.parameters()),
            )
        )
        service.finish_step(10)


class ESampRegionUnitTest(unittest.TestCase):
    def test_graph_modes_sign_prediction_and_async_optimizer_capture(self) -> None:
        workload = ESampWorkload(_config())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(
                2, 4, torch.float32, torch.device("cpu")
            )
        )
        for graph_mode, expected_prediction, expected_optimizer in (
            (
                "eager",
                RegionCaptureMode.DISABLED,
                AsyncBatchCaptureMode.DISABLED,
            ),
            (
                "piecewise",
                RegionCaptureMode.REQUIRE,
                AsyncBatchCaptureMode.OPTIMIZER_TRANSITION,
            ),
            (
                "full",
                RegionCaptureMode.REQUIRE,
                AsyncBatchCaptureMode.OPTIMIZER_TRANSITION,
            ),
        ):
            with self.subTest(graph_mode=graph_mode):
                compilation = runtime.compile_workloads(
                    replace(_timeline(), graph_mode=graph_mode)
                )
                regions = {
                    task_name: region
                    for region in compilation.decision.execution_regions
                    for task_name in region.tasks
                }

                self.assertEqual(compilation.graph.schema_version, 5)
                self.assertIs(
                    compilation.graph.states[0].storage,
                    ScheduleStateStorage.OPTIMIZER_STABLE,
                )
                self.assertTrue(regions["predict"].capture_safe)
                self.assertIs(
                    regions["predict"].capture_mode,
                    expected_prediction,
                )
                for task_name in ("loss", "loss__optimize_distiller"):
                    self.assertFalse(regions[task_name].capture_safe)
                    self.assertIs(
                        regions[task_name].capture_mode,
                        RegionCaptureMode.DISABLED,
                    )
                    self.assertIs(
                        regions[task_name].stream,
                        RegionStreamClass.SIDE_FORWARD,
                    )
                    self.assertIs(
                        regions[task_name].host_submission,
                        RegionHostSubmission.ASYNC_SERIAL,
                    )

                predict_advance = next(
                    advance
                    for advance in compilation.decision.runtime_advance_plan
                    if regions["predict"].region_id in advance.region_ids
                )
                self.assertEqual(
                    tuple(command.kind for command in predict_advance.host_commands),
                    (
                        HostCommandKind.WAIT_ASYNC,
                        HostCommandKind.EXECUTE_INLINE,
                    ),
                )
                self.assertTrue(
                    all(
                        command.async_capture_mode
                        is AsyncBatchCaptureMode.DISABLED
                        for command in predict_advance.host_commands
                    )
                )

                tail_advance = next(
                    advance
                    for advance in compilation.decision.runtime_advance_plan
                    if regions["loss"].region_id in advance.region_ids
                )
                self.assertEqual(
                    tuple(command.kind for command in tail_advance.host_commands),
                    (
                        HostCommandKind.WAIT_ASYNC,
                        HostCommandKind.SUBMIT_ASYNC,
                    ),
                )
                submit = tail_advance.host_commands[1]
                self.assertEqual(
                    submit.region_ids,
                    (
                        regions["loss"].region_id,
                        regions["loss__optimize_distiller"].region_id,
                    ),
                )
                self.assertIs(submit.async_capture_mode, expected_optimizer)

    def test_native_dispatch_rejects_every_legacy_only_option(self) -> None:
        cases: tuple[tuple[dict[str, object], str], ...] = (
            ({"consumer_id": "custom"}, "consumer_id"),
            ({"graph_scratch_rows": 8}, "graph_scratch_rows"),
            ({"enable_workgraph_execution": False}, "enable_workgraph_execution"),
            (
                {
                    "per_request_models": True,
                    "per_request_model_bank": True,
                },
                "per_request_model_bank",
            ),
            ({"model_bank_slots": 2}, "model_bank_slots"),
            ({"model_bank_flush_interval": 2}, "model_bank_flush_interval"),
            ({"model_bank_rank": 32}, "model_bank_rank"),
            (
                {"model_bank_initializer": SVDModelBankInitializerConfig()},
                "model_bank_initializer",
            ),
            ({"model_bank_train_cudagraph": True}, "model_bank_train_cudagraph"),
            ({"model_bank_forward_backend": "triton"}, "model_bank_forward_backend"),
            ({"adaptation_pipeline_slots": 2}, "adaptation_pipeline_slots"),
            ({"adaptation_stream_mode": "single"}, "adaptation_stream_mode"),
            ({"adaptation_stream_priority": 1}, "adaptation_stream_priority"),
            ({"trace_per_request_losses": True}, "trace_per_request_losses"),
            ({"trace_interval": 2}, "trace_interval"),
            ({"trace_max_points": 2}, "trace_max_points"),
            ({"compact_capture_lane": True}, "compact_capture_lane"),
            ({"distiller_sampler_backend": "pre_filter_dense"}, "distiller_sampler_backend"),
            (
                {
                    "per_request_models": True,
                    "training_row_selection": "first_per_prompt",
                },
                "training_row_selection",
            ),
            (
                {
                    "per_request_models": True,
                    "model_bank_use_output_layernorm": False,
                },
                "model_bank_use_output_layernorm",
            ),
        )
        for overrides, option in cases:
            with self.subTest(option=option):
                config = ESampConsumerConfig(
                    implementation="native", **overrides
                )
                with self.assertRaisesRegex(ValueError, option):
                    build_esamp_workload(config)

    def test_native_dispatch_preserves_explicit_training_row_selection(self) -> None:
        workload = build_esamp_workload(
            ESampConsumerConfig(
                implementation="native",
                training_row_selection="first_per_prompt",
            )
        )
        self.assertIsInstance(workload, ESampWorkload)
        assert isinstance(workload, ESampWorkload)
        self.assertEqual(
            workload.config.training_row_selection,
            "first_per_prompt",
        )

    def test_native_dispatch_selects_eager_request_state_mode(self) -> None:
        dispatched = build_esamp_workload(
            ESampConsumerConfig(
                implementation="native",
                per_request_models=True,
                enable_distiller_intervention=True,
                distiller_beta=0.5,
            )
        )
        self.assertIs(type(dispatched), ESampWorkload)
        workload = ESampWorkload(
            ESampConfig(
                per_request_models=True,
                intervention=True,
                distiller_beta=0.5,
            )
        )
        self.assertTrue(workload.config.per_request_models)
        self.assertTrue(workload.config.intervention)
        self.assertEqual(workload.config.distiller_beta, 0.5)
        declaration = declare_workload(workload)
        self.assertEqual(
            tuple(task.name for task in declaration.ir.tasks),
            ("request_predict", "request_loss", "request_train"),
        )
        self.assertEqual(
            tuple(item.kind.value for item in declaration.ir.tasks[0].inputs),
            (
                "packed_request_rows",
                "packed_sampler_rows",
                "packed_hidden",
                "packed_hidden",
                "state_selection",
            ),
        )
        self.assertEqual(
            tuple(
                (output.name, output.kind.value, output.target)
                for output in declaration.ir.tasks[0].outputs
            ),
            (
                ("guidance", "sampler", "guidance"),
                (
                    "training_source",
                    "temporary",
                    "temporary:training_source",
                ),
                (
                    "training_target",
                    "temporary",
                    "temporary:training_target",
                ),
            ),
        )
        self.assertIn(
            "guidance",
            workload.native_workload_resources().ports.outputs,
        )
        request_state = declaration.ir.states[1]
        self.assertEqual(request_state.scope.value, "request")
        self.assertEqual(request_state.request_key, (RequestKeyField.REQUEST_ID,))

    def test_request_state_regions_remain_scheduler_selected_eager_work(self) -> None:
        for driver in (V027MRv1Driver(), V027MRv2Driver()):
            with self.subTest(runner=driver.capabilities().runner_family):
                workload = ESampWorkload(
                    ESampConfig(
                        per_request_models=True,
                        distiller_hidden_dim=2,
                    )
                )
                runtime = TllmRuntime(
                    workloads=(workload,),
                    workload_memory_budget_bytes=1 << 20,
                )
                runtime.cold_specialize_workloads(
                    RuntimeSpecializationSpec(
                        4,
                        4,
                        torch.float32,
                        torch.device("cpu"),
                    )
                )
                compilation = runtime.compile_workloads(
                    driver.export_main_workload_timeline(
                        graph_mode="eager",
                        static_headroom_bytes=1 << 20,
                    )
                )
                regions = compilation.decision.execution_regions
                self.assertEqual(
                    tuple(task for region in regions for task in region.tasks),
                    (
                        "request_predict",
                        "request_loss",
                        "request_train",
                    ),
                )
                for region in regions:
                    self.assertIs(region.stream, RegionStreamClass.DEFAULT)
                    self.assertIs(
                        region.host_submission,
                        RegionHostSubmission.INLINE,
                    )
                    self.assertFalse(region.compile_safe)
                    self.assertFalse(region.capture_safe)
                    self.assertIs(
                        region.capture_mode,
                        RegionCaptureMode.DISABLED,
                    )
                physical = compilation.physical
                assert physical is not None
                self.assertTrue(
                    all(
                        node.capture_policy is CapturePolicy.DISABLED
                        for node in physical.nodes
                    )
                )

    def test_direct_request_workload_rejects_unsupported_first_phase_options(self) -> None:
        for overrides, message in (
            (
                {"training_row_selection": "first_per_prompt"},
                "training_row_selection",
            ),
            ({"output_layernorm": False}, "output_layernorm"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    ESampWorkload(
                        ESampConfig(per_request_models=True, **overrides)
                    )

    def test_first_per_prompt_selection_uses_typed_sample_identity(self) -> None:
        workload = _specialized_workload(
            training_row_selection="first_per_prompt"
        )
        source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        target = -source

        prediction, training_source, loss_target, sample_index_tensor = workload.predict(
            source,
            target,
            torch.tensor((0, 1, 0, 2)),
        )
        loss = workload.loss(
            training_source, loss_target, sample_index_tensor
        )

        expected_prediction = workload.distiller(source)
        torch.testing.assert_close(
            prediction, expected_prediction
        )
        assert loss is not None
        torch.testing.assert_close(
            loss,
            torch.nn.functional.mse_loss(
                expected_prediction.index_select(0, torch.tensor((0, 2))).float(),
                target.index_select(0, torch.tensor((0, 2))).float(),
            ),
        )

    def test_native_region_rejects_a_coinstalled_legacy_consumer(self) -> None:
        native = ESampWorkload(_config())
        legacy = ESampConsumer(ESampConsumerConfig(implementation="legacy"))

        with self.assertRaisesRegex(RuntimeError, "native Region authority"):
            TllmRuntime(workloads=(native,), consumers=(legacy,))

    def test_disabled_workload_exports_a_disabled_immutable_worker_spec(self) -> None:
        workload = ESampWorkload(_config(enabled=True))

        workload.set_enabled(False)

        worker = workload.native_workload_resources().worker
        assert isinstance(worker, WorkerRuntimeBinding)
        worker_spec = worker.spec
        self.assertFalse(worker_spec.config["enabled"])
        reconstructed = ESampWorkload.reconstruct(worker_spec.config)
        assert isinstance(reconstructed.services, LocalRuntimeServices)
        self.assertFalse(reconstructed.services.is_enabled())

    def test_new_worker_epoch_refreshes_the_current_enable_snapshot(self) -> None:
        workload = ESampWorkload(_config(enabled=True))
        runtime = TllmRuntime(workloads=(workload,))

        workload.set_enabled(False)
        runtime._refresh_native_worker_specs()

        worker_spec = runtime._workload_worker_specs[0]
        self.assertFalse(worker_spec.config["enabled"])
        self.assertIn(
            ("esamp/enabled", "False"),
            runtime.workload_compile_options.shape_specialization,
        )

    def test_in_process_enable_transition_publishes_a_new_immutable_epoch(self) -> None:
        workload = ESampWorkload(_config(enabled=True))
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )
        initial = runtime.compile_workloads(_timeline())
        initial_model = workload.distiller
        initial_optimizer = workload.optimizer
        initial_state_version = workload.state_version
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                initial, bindings=runtime.workload_execution_bindings
            )
            old_execution = state.workload_execution
            old_store = state.workload_execution_store
            assert old_execution is not None
            assert old_store is not None
            assert old_execution._value_store is not None
            self.assertTrue(old_execution._value_store.is_enabled())
            with self.assertRaisesRegex(RuntimeError, "immutable"):
                old_execution.set_enabled(False)

            self.assertTrue(runtime.transition_native_workload_enabled(False))

            replacement = runtime.workload_compilation
            assert replacement is not None
            self.assertNotEqual(replacement.cache_key, initial.cache_key)
            self.assertNotEqual(
                replacement.decision.signature, initial.decision.signature
            )
            self.assertNotEqual(
                replacement.decision.execution_regions[0].artifact_key,
                initial.decision.execution_regions[0].artifact_key,
            )
            self.assertIs(old_store.current, old_execution)
            from tllm.runtime.workload_execution import StagedExecution

            self.assertIsInstance(old_store.staged, StagedExecution)
            self.assertTrue(old_execution._value_store.is_enabled())
            self.assertIs(workload.distiller, initial_model)
            self.assertIs(workload.optimizer, initial_optimizer)
            self.assertEqual(workload.state_version, initial_state_version)
            worker = workload.native_workload_resources().worker
            assert isinstance(worker, WorkerRuntimeBinding)
            worker_spec = worker.spec
            self.assertFalse(worker_spec.config["enabled"])

            residual_runtime.prepare_physical_workgraph_step(1)
            current = state.workload_execution
            assert current is not None
            assert current._value_store is not None
            self.assertIsNot(current, old_execution)
            self.assertFalse(current._value_store.is_enabled())

    def test_mp_install_cold_binds_hidden_taps_before_first_forward(self) -> None:
        workload = ESampWorkload(_config())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )
        compilation = runtime.compile_workloads(_timeline())
        self.assertTrue(_decision_requires_hidden_taps(compilation.decision))
        extension = TllmWorkerExtension()
        extension.model_runner = mock.Mock(
            device=torch.device("cpu"),
            max_num_reqs=2,
            max_num_tokens=16,
            model_config=mock.Mock(
                dtype=torch.float32,
                get_hidden_size=mock.Mock(return_value=4),
                get_num_layers=None,
            ),
            model=torch.nn.Module(),
        )
        worker = workload.native_workload_resources().worker
        assert isinstance(worker, WorkerRuntimeBinding)
        worker_spec = worker.spec
        state = residual_runtime.RUNTIME
        saved = (
            state.native_schedule_graph,
            state.native_workloads_pending,
            dict(state.residual_raw_paths),
            state.workload_compilation,
            state.workload_execution,
            state.physical_executor,
        )
        try:
            with mock.patch(
                "tllm.runtime.vllm_patch.port_runtime_hooks."
                "setup_runtime_hooks_if_active"
            ) as setup:
                report = extension.tllm_install_workload_decision(
                    compilation.cache_key,
                    compilation.graph.to_json(),
                    compilation.timeline.to_json(),
                    compilation.options.to_json(),
                    compilation.decision.to_json(),
                    compilation.physical.plan_epoch,
                    [worker_spec.to_wire()],
                )
            setup.assert_called_once_with(core=residual_runtime, runner=extension.model_runner)
            self.assertTrue(report["native"])
            self.assertFalse(state.native_workloads_pending)
            self.assertEqual(state.native_schedule_graph, compilation.graph)
            self.assertEqual(len(state.residual_raw_paths), 2)
        finally:
            current = state.workload_execution
            if current is not None and current is not saved[4]:
                current.close()
            (
                state.native_schedule_graph,
                state.native_workloads_pending,
                residual_paths,
                state.workload_compilation,
                state.workload_execution,
                state.physical_executor,
            ) = saved
            state.residual_raw_paths = residual_paths

    def test_worker_spec_reconstructs_exact_signed_graph_and_bindings(self) -> None:
        workload = ESampWorkload(_config())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        spec = RuntimeSpecializationSpec(
            2, 4, torch.float32, torch.device("cpu")
        )
        runtime.cold_specialize_workloads(spec)
        compilation = runtime.compile_workloads(_timeline())
        worker = workload.native_workload_resources().worker
        assert isinstance(worker, WorkerRuntimeBinding)
        worker_spec = worker.spec

        bindings = reconstruct_worker_bindings(
            compilation, (worker_spec,), specialization=spec
        )

        self.assertEqual(
            bindings.declarations[0].ir.semantic_key,
            declare_workload(workload).ir.semantic_key,
        )
        self.assertEqual(set(bindings.ports.outputs), {"guidance"})
        self.assertEqual(set(bindings.state_versions), {"distiller"})
        self.assertIs(type(bindings.sampler), SamplerRuntimeBinding)

    def test_worker_reconstruction_rejects_a_signed_specialization_mismatch(self) -> None:
        workload = ESampWorkload(_config())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        expected = RuntimeSpecializationSpec(
            2, 4, torch.float32, torch.device("cpu")
        )
        runtime.cold_specialize_workloads(expected)
        compilation = runtime.compile_workloads(_timeline())
        worker = workload.native_workload_resources().worker
        assert isinstance(worker, WorkerRuntimeBinding)
        worker_spec = worker.spec

        with self.assertRaisesRegex(RuntimeError, "specialization does not match"):
            reconstruct_worker_bindings(
                compilation,
                (worker_spec,),
                specialization=RuntimeSpecializationSpec(
                    2, 4, torch.float64, torch.device("cpu")
                ),
            )

    def test_signed_physical_regions_drive_one_complete_training_step(self) -> None:
        workload = ESampWorkload(_config())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )
        compilation = runtime.compile_workloads(_timeline())
        signed_tasks = tuple(
            task
            for region in compilation.decision.execution_regions
            for task in region.tasks
        )
        self.assertEqual(
            set(signed_tasks),
            {
                "predict",
                "loss",
                "loss__optimize_distiller",
            },
        )
        self.assertEqual(len(signed_tasks), 3)

        state = _decode_state(2)
        source_locator = ResidualLocator(0, "block_output", "decode")
        target_locator = ResidualLocator(-1, "block_output", "decode")
        state.residual_bindings = {
            "source": ResidualPathBinding(source_locator, "source"),
            "target": ResidualPathBinding(target_locator, "target"),
        }
        state.tap_decode_hidden = {
            "source": torch.randn(2, 4),
            "target": torch.randn(2, 4),
        }
        before = tuple(
            parameter.detach().clone() for parameter in workload.distiller.parameters()
        )

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=runtime.workload_execution_bindings,
            )
            residual_runtime.prepare_physical_workgraph_step(7)
            sampler_binding = workload.native_workload_resources().sampler
            assert isinstance(sampler_binding, SamplerRuntimeBinding)
            sampler_binding.stage_identity(
                7,
                tuple(state.decode_request_ids),
                tuple(state.decode_sample_idxs),
                torch.tensor(state.decode_sample_idxs),
            )
            for region in compilation.decision.execution_regions:
                residual_runtime.advance_workload_anchor(region.placement_anchor)

            execution = state.workload_execution
            assert execution is not None
            self.assertEqual(
                execution.state_versions.observed_version("distiller", None), 1
            )
            stats = residual_runtime.read_and_reset_esamp_stats()

        after = tuple(parameter.detach() for parameter in workload.distiller.parameters())
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, after, strict=True)
            )
        )
        self.assertEqual(workload.state_version, 1)
        self.assertEqual(stats.loss_count, 1)
        services = workload.native_workload_resources().services
        assert isinstance(services, LocalRuntimeServices)
        self.assertEqual(services.diagnostics()["guidance_rows"], 2)

    def test_declaration_has_replica_state_and_no_legacy_adapter(self) -> None:
        workload = ESampWorkload(_config())
        declaration = declare_workload(workload)

        self.assertFalse(hasattr(workload, "legacy_consumer"))
        self.assertEqual(declaration.ir.states[0].scope.value, "replica")
        self.assertEqual(
            tuple(task.name for task in declaration.ir.tasks),
            (
                "predict",
                "loss",
                "loss__optimize_distiller",
            ),
        )
        self.assertEqual(declaration.ir.tasks[0].outputs[0].target, "guidance")
        self.assertEqual(declaration.ir.tasks[0].outputs[1].kind.value, "temporary")

    def test_cold_specialize_forward_update_and_stats(self) -> None:
        workload = ESampWorkload(_config())
        resources = workload.native_workload_resources()
        self.assertEqual(resources.states.snapshots["distiller"](0), {None: 0})
        assert isinstance(resources.services, LocalRuntimeServices)
        diagnostics = resources.services.diagnostics
        self.assertEqual(
            diagnostics(),
            {"guidance_rows": 0, "loss_count": 0, "update_count": 0},
        )
        with self.assertRaises(RuntimeError):
            workload.distiller  # parameters do not exist before cold_specialize

        with torch.inference_mode():
            workload.cold_specialize(
                RuntimeSpecializationSpec(
                    rows=2,
                    hidden_size=4,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
            )
        self.assertIsNotNone(workload.optimizer)
        self.assertFalse(any(parameter.is_inference() for parameter in workload.distiller.parameters()))
        before = tuple(parameter.detach().clone() for parameter in workload.distiller.parameters())
        _publish_guidance(workload, 7)
        self.assertEqual(diagnostics()["guidance_rows"], 2)

        # The signed update path (extern task body is one line of logic).
        loss = torch.nn.functional.mse_loss(
            workload.distiller(torch.randn(2, 4)), torch.randn(2, 4)
        )
        executors = dict(workload.declaration().executors)
        executors["loss__optimize_distiller"](loss)

        after = tuple(parameter.detach() for parameter in workload.distiller.parameters())
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, after, strict=True)))
        self.assertEqual(resources.states.snapshots["distiller"](0), {None: 1})
        self.assertEqual(
            diagnostics(),
            {"guidance_rows": 2, "loss_count": 1, "update_count": 1},
        )

    def test_cold_specialization_rejects_any_changed_static_fact(self) -> None:
        workload = ESampWorkload(_config())
        workload.cold_specialize(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )

        for mismatch in (
            RuntimeSpecializationSpec(3, 4, torch.float32, torch.device("cpu")),
            RuntimeSpecializationSpec(2, 4, torch.float64, torch.device("cpu")),
        ):
            with self.subTest(mismatch=mismatch):
                with self.assertRaisesRegex(RuntimeError, "specialized"):
                    workload.cold_specialize(mismatch)

    def test_sampler_semantic_config_changes_the_signed_cache_key(self) -> None:
        timeline = _timeline()

        def compile_for(beta: float, intervention: bool) -> object:
            runtime = TllmRuntime(
                workloads=(
                    ESampWorkload(
                        _config(
                            distiller_beta=beta,
                            intervention=intervention,
                        )
                    ),
                ),
                workload_memory_budget_bytes=1 << 20,
            )
            runtime.cold_specialize_workloads(
                RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
            )
            return runtime.compile_workloads(timeline)

        baseline = compile_for(0.25, True)
        changed_beta = compile_for(0.75, True)
        disabled_intervention = compile_for(0.25, False)
        self.assertNotEqual(baseline.cache_key, changed_beta.cache_key)
        self.assertNotEqual(baseline.cache_key, disabled_intervention.cache_key)

    def test_sampler_reads_only_same_step_guidance(self) -> None:
        workload = _specialized_workload()
        _publish_guidance(workload, 11)

        sampler = workload.native_workload_resources().sampler
        assert isinstance(sampler, SamplerRuntimeBinding)
        with mock.patch.object(
            workload.distiller,
            "forward",
            side_effect=AssertionError("sampler must not re-run the Region model"),
        ) as forward:
            state = sampler.provider.prepare_step(_view(11))
        forward.assert_not_called()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.affected_row_ids.tolist(), [0, 1])
        self.assertEqual(tuple(state.pred_hidden.shape), (2, 4))
        self.assertIsNone(sampler.provider.prepare_step(_view(12)))
        with self.assertRaisesRegex(RuntimeError, "active step"):
            sampler.provider.prepare_step(
                _view(11, request_ids=("different", "request-1"))
            )


if __name__ == "__main__":
    unittest.main()
