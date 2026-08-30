"""Typed sparse-row sampler-output contracts."""

from __future__ import annotations

from dataclasses import dataclass
from unittest import mock

import pytest
import torch

from tllm.kit.sampler_output import DeclaredSamplerOutput
from tllm.ports.sampler import SamplerHiddenReweightBatch
from tllm.runtime import residual_runtime
from tllm.runtime.sampler_bridge.context import SamplerStepContext
from tllm.runtime.sampler_bridge.exact_backend import build_modified_logits_exact
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_sinks import SinkRecord, SinkStore
from tllm.workload import (
    ConfigField,
    WorkloadConfig,
    declared_config_field,
    sampler,
)


_BETA = ConfigField.floating(0.5)


class _SamplerConfig(WorkloadConfig):
    beta: float = declared_config_field(_BETA)


@dataclass(frozen=True, slots=True)
class _SamplingMetadata:
    pass


class _Model:
    def __init__(self) -> None:
        self.lm_head = torch.nn.Linear(3, 5, bias=False)


class _Runner:
    pass


def _runtime_state() -> ResidualRuntimeState:
    state = ResidualRuntimeState()
    state.decode_request_ids = ["request-0", "request-1", "request-2", "request-3"]
    state.decode_prompt_idxs = [10, 11, 12, 13]
    state.decode_sample_idxs = [0, 0, 1, 1]
    return state


def _view(step_id: int, model: _Model) -> SamplerStepView:
    return SamplerStepView(
        engine_step_id=step_id,
        phase="decode",
        logits=torch.zeros((4, 5)),
        sampling_metadata=_SamplingMetadata(),
        decode_count=4,
        request_ids=("request-0", "request-1", "request-2", "request-3"),
        prompt_idxs=(10, 11, 12, 13),
        sample_idxs=(0, 0, 1, 1),
        prompt_idx_tensor=None,
        sample_idx_tensor=None,
        source_hidden=torch.zeros((4, 3)),
        device=torch.device("cpu"),
        model=model,
        runner=_Runner(),
    )


def test_compact_hidden_output_preserves_full_identity_and_row_mapping() -> None:
    store = SinkStore()
    pred_hidden = torch.tensor(
        ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)),
        dtype=torch.float32,
    )
    payload = SamplerHiddenReweightBatch(pred_hidden, (1, 3), 4)
    with mock.patch.object(residual_runtime, "RUNTIME", _runtime_state()):
        store.publish("guidance", payload, 7)

    match store.lookup("guidance", 7):
        case SinkRecord(
            value=SamplerHiddenReweightBatch() as stored,
            identity=identity,
        ):
            pass
        case unexpected:
            raise AssertionError(f"unexpected sink record {unexpected!r}")
    assert identity.request_ids == (
        "request-0",
        "request-1",
        "request-2",
        "request-3",
    )
    assert identity.prompt_idxs == (10, 11, 12, 13)
    assert identity.sample_idxs == (0, 0, 1, 1)
    assert store.latest_rows("guidance") == 2
    assert stored.affected_row_ids == (1, 3)
    assert stored.pred_hidden.data_ptr() != pred_hidden.data_ptr()

    provider = DeclaredSamplerOutput(
        sampler.hidden_reweight(beta=_BETA),
        _SamplerConfig(),
        store,
        SamplerStepContext(),
    )
    state = provider.prepare_step(_view(7, _Model()))
    assert state is not None
    torch.testing.assert_close(state.pred_hidden, pred_hidden)
    torch.testing.assert_close(
        state.affected_row_ids,
        torch.tensor((1, 3), dtype=torch.long),
    )
    assert state.pred_hidden_row_map is not None
    torch.testing.assert_close(
        state.pred_hidden_row_map,
        torch.tensor((-1, 0, -1, 1), dtype=torch.long),
    )

    modified, generators = build_modified_logits_exact(
        logits=torch.zeros((4, 5)),
        sampling_metadata=_SamplingMetadata(),
        state=state,
        row_ids=torch.tensor((1, 3), dtype=torch.long),
        greedy=True,
    )
    expected_dense = torch.nn.functional.linear(
        pred_hidden,
        state.lm_head_weight,
        state.lm_head_bias,
    )
    torch.testing.assert_close(modified, -0.5 * expected_dense)
    assert generators == {}


def test_empty_hidden_subset_does_not_resolve_a_model_head() -> None:
    store = SinkStore()
    payload = SamplerHiddenReweightBatch(torch.empty((0, 3)), (), 4)
    with mock.patch.object(residual_runtime, "RUNTIME", _runtime_state()):
        store.publish("guidance", payload, 8)
    provider = DeclaredSamplerOutput(
        sampler.hidden_reweight(beta=_BETA),
        _SamplerConfig(),
        store,
        SamplerStepContext(),
    )

    with mock.patch(
        "tllm.kit.sampler_output.resolve_lm_head",
        side_effect=AssertionError("empty subset must not resolve lm_head"),
    ) as resolve:
        assert provider.prepare_step(_view(8, _Model())) is None
    resolve.assert_not_called()
    assert store.latest_rows("guidance") == 0


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            SamplerHiddenReweightBatch(torch.zeros((2, 3)), (1, 1), 4),
            "sorted and unique",
        ),
        (
            SamplerHiddenReweightBatch(torch.zeros((2, 3)), (3, 1), 4),
            "sorted and unique",
        ),
        (
            SamplerHiddenReweightBatch(torch.zeros((1, 3)), (4,), 4),
            "out of bounds",
        ),
        (
            SamplerHiddenReweightBatch(torch.zeros((1, 3)), (), 4),
            "align with affected rows",
        ),
        (
            SamplerHiddenReweightBatch(torch.zeros((0, 3)), (), -1),
            "non-negative",
        ),
    ),
)
def test_compact_hidden_output_rejects_noncanonical_rows(
    payload: SamplerHiddenReweightBatch,
    message: str,
) -> None:
    store = SinkStore()
    with (
        mock.patch.object(residual_runtime, "RUNTIME", _runtime_state()),
        pytest.raises(ValueError, match=message),
    ):
        store.publish("guidance", payload, 9)
