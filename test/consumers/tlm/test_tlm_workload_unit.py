"""Native TLM semantic-DAG, PEFT update, and overlay publication tests."""

from __future__ import annotations

from typing import cast
from unittest import mock

import pytest
import torch
from torch.nn import functional as F

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.tlm.workload import (
    TLMConfig,
    TLMTrainBatch,
    TLMTrainingModel,
    TLMWorkload,
    official_tlm_loss,
)
from tllm.runtime import residual_runtime
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_bindings import LocalRuntimeServices
from tllm.schedule import MainWorkloadTimeline, WorkloadCompilation
from tllm.workload import declare_workload


def _timeline() -> MainWorkloadTimeline:
    return export_main_workload_timeline(
        BackendCapabilities(
            backend_id="vllm-0.27-mrv2",
            runner_family="mrv2",
            sampler_stages=frozenset(),
            native_min_p=False,
            tuple_sampler_return=False,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.EAGER_ONLY,
        ),
        graph_mode="eager",
        static_headroom_bytes=1 << 24,
    )


def _drive(
    runtime: TllmRuntime[TLMWorkload],
    compilation: WorkloadCompilation,
    step_id: int,
) -> tuple[int, int]:
    state = ResidualRuntimeState()
    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(
            compilation, bindings=runtime.workload_execution_bindings
        )
        residual_runtime.prepare_physical_workgraph_step(step_id)
        for region in compilation.decision.execution_regions:
            residual_runtime.advance_workload_anchor(region.placement_anchor)
        execution = state.workload_execution
        assert execution is not None
        return (
            execution.state_versions.observed_version("trainer", None),
            execution.state_versions.observed_version("overlay_versions", None),
        )


def test_official_loss_matches_next_token_weighting() -> None:
    logits = torch.tensor(
        (
            ((3.0, 0.0), (0.0, 3.0), (3.0, 0.0)),
            ((0.0, 3.0), (3.0, 0.0), (0.0, 3.0)),
        )
    )
    labels = torch.tensor(((0, 1, 0), (1, 1, -100)))
    loss, selected = official_tlm_loss(
        logits,
        labels,
        selection_logits=None,
        threshold=0.2,
        lamb=0.5,
        ignore_index=-100,
    )
    row0 = F.cross_entropy(logits[0, :-1], labels[0, 1:])
    row1 = F.cross_entropy(logits[1, :1], labels[1, 1:2])
    rows = torch.stack((row0, row1))
    mask = rows > 0.2
    expected = (
        rows * (0.5 * torch.exp(rows.detach() - 0.2)) * mask
    ).sum() / mask.sum()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(selected, mask)


def test_rust_config_choice_produces_distinct_online_offline_semantic_dags() -> None:
    online = declare_workload(TLMWorkload(TLMConfig(mode="online"))).ir
    offline = declare_workload(TLMWorkload(TLMConfig(mode="offline"))).ir
    assert set(online.schedule) == {
        "train_online",
        "train_online__optimize_trainer",
        "snapshot_online",
        "publish_online",
    }
    assert set(offline.schedule) == {
        "train_offline",
        "train_offline__optimize_trainer",
        "snapshot_offline",
        "publish_offline",
    }
    assert online.semantic_key != offline.semantic_key
def test_tiny_qwen_peft_trains_and_publishes_qv_overlay() -> None:
    pytest.importorskip("peft")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = TLMConfig(
        mode="online",
        rank=2,
        lora_alpha=4,
        threshold=-1.0,
        learning_rate=1e-2,
    )
    spec = RuntimeSpecializationSpec(1, 16, torch.float32, torch.device("cpu"))
    base = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    state = TLMTrainingModel.adapt(base, config, spec)
    before = {
        name: parameter.detach().clone()
        for name, parameter in state.overlay_parameters.items()
    }
    workload = TLMWorkload(config)
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=1 << 24
    )
    with mock.patch.object(TLMTrainingModel, "load", return_value=state):
        runtime.cold_specialize_workloads(spec)
    compilation = runtime.compile_workloads(_timeline())
    batch = TLMTrainBatch(
        input_ids=torch.tensor(((1, 2, 3, 4),)),
        labels=torch.tensor(((1, 2, 3, 4),)),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
    )
    runtime.stage(
        TLMWorkload.batches,
        {
            "input_ids": batch.input_ids,
            "labels": batch.labels,
            "attention_mask": cast(torch.Tensor, batch.attention_mask),
        },
    )

    assert _drive(runtime, compilation, 1) == (1, 1)
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in workload.trainer.overlay_parameters.items()
    )
    publication = runtime.drain(TLMWorkload.publications)[0]
    assert publication.selected_count == 1
    materialized = workload.overlay_versions.store.materialize(publication.version)
    assert "__tllm_lora_alpha__" in materialized
    trainable_names = tuple(
        name for name in materialized if name != "__tllm_lora_alpha__"
    )
    assert trainable_names
    assert all("q_proj" in name or "v_proj" in name for name in trainable_names)
    services = cast(
        LocalRuntimeServices, workload.native_workload_resources().services
    )
    assert services.stats(False).loss_count == 1


def test_offline_batches_keep_one_update_and_version_per_batch() -> None:
    pytest.importorskip("peft")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = TLMConfig(
        mode="offline", rank=2, lora_alpha=4, threshold=-1.0
    )
    spec = RuntimeSpecializationSpec(1, 16, torch.float32, torch.device("cpu"))
    state = TLMTrainingModel.adapt(
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ),
        config,
        spec,
    )
    workload = TLMWorkload(config)
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=1 << 24
    )
    with mock.patch.object(TLMTrainingModel, "load", return_value=state):
        runtime.cold_specialize_workloads(spec)
    compilation = runtime.compile_workloads(_timeline())
    batch = TLMTrainBatch(
        torch.tensor(((1, 2, 3, 4),)),
        torch.tensor(((1, 2, 3, 4),)),
    )
    for step_id in (1, 2):
        runtime.stage(
            TLMWorkload.batches,
            {
                "input_ids": batch.input_ids,
                "labels": batch.labels,
                "attention_mask": torch.empty(0, dtype=torch.long),
            },
        )
        assert _drive(runtime, compilation, step_id) == (step_id, step_id)
    assert tuple(
        publication.version.version
        for publication in runtime.drain(TLMWorkload.publications)
    ) == (1, 2)
