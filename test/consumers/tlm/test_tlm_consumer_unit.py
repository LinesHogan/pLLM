#!/usr/bin/env python3
"""TLM trains internal q/v LoRA and publishes immutable request versions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from tllm.backends.vllm.test_time_lora import VLLMVersionedLoRAPublisher
from tllm.consumers.tlm import (
    HFPEFTTLMTrainer,
    TLMConsumer,
    TLMConsumerConfig,
    TLMTrainBatch,
    TLMTrainResult,
    official_tlm_loss,
)
from tllm.runtime.test_time_lora import LoRAAdapterWeights, LoRAVersionStore


def _batch(rows: int = 2) -> TLMTrainBatch:
    input_ids = torch.arange(rows * 4).reshape(rows, 4) % 7
    return TLMTrainBatch(input_ids=input_ids, labels=input_ids.clone())


def _weights(value: float = 0.0) -> dict[str, LoRAAdapterWeights]:
    return {
        f"model.layers.0.self_attn.{name}": LoRAAdapterWeights(
            torch.full((2, 4), value),
            torch.full((4, 2), value),
            alpha=4,
        )
        for name in ("q_proj", "v_proj")
    }


class _Trainer:
    def __init__(self) -> None:
        self.modes: list[str] = []

    def train_batch(self, batch: TLMTrainBatch, *, mode: str) -> TLMTrainResult:
        self.modes.append(mode)
        return TLMTrainResult(
            loss=float(len(self.modes)),
            selected_count=int(batch.input_ids.shape[0]),
            target_modules=_weights(float(len(self.modes))),
        )


def test_config_uses_official_qv_lora_family_defaults() -> None:
    config = TLMConsumerConfig()
    assert config.rank == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.0
    assert config.learning_rate == 5e-5
    assert config.target_modules == ("q_proj", "v_proj")
    with pytest.raises(ValueError, match="exactly q_proj and v_proj"):
        TLMConsumerConfig(target_modules=("lm_head",))


def test_official_loss_matches_next_token_cross_entropy_weighting() -> None:
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]],
            [[0.0, 3.0], [3.0, 0.0], [0.0, 3.0]],
        ]
    )
    labels = torch.tensor([[0, 1, 0], [1, 1, -100]])
    loss, selected = official_tlm_loss(
        logits,
        labels,
        selection_logits=None,
        threshold=0.2,
        lamb=0.5,
        ignore_index=-100,
    )
    row0 = F.cross_entropy(logits[0, :-1], labels[0, 1:], reduction="mean")
    row1 = F.cross_entropy(
        logits[1, :1], labels[1, 1:2], reduction="mean"
    )
    expected_rows = torch.stack((row0, row1))
    expected_mask = expected_rows > 0.2
    expected = (
        expected_rows
        * (0.5 * torch.exp(expected_rows.detach() - 0.2))
        * expected_mask
    ).sum() / expected_mask.sum()
    torch.testing.assert_close(loss, expected)
    assert torch.equal(selected, expected_mask)


def test_online_predicts_then_trains_and_publishes_a_new_version() -> None:
    trainer = _Trainer()
    consumer = TLMConsumer(TLMConsumerConfig(mode="online"), trainer=trainer)
    calls: list[str] = []

    result = consumer.run_ordered_step(
        predict=lambda: calls.append("predict") or "result",
        batch=_batch(),
    )

    assert result == "result"
    assert calls == ["predict"]
    assert trainer.modes == ["online"]
    assert consumer.version_store.latest_version == 1
    stats = consumer.read_stats()
    assert stats.update_count == stats.loss_count == 1
    assert stats.selected_count == 2
    assert stats.adapter_version == 1


def test_offline_buffers_then_trains_before_predict() -> None:
    trainer = _Trainer()
    consumer = TLMConsumer(TLMConsumerConfig(mode="offline"), trainer=trainer)
    consumer.update_offline(_batch(2))
    consumer.update_offline(_batch(3))
    assert consumer.read_stats().buffered_count == 5

    order: list[str] = []
    consumer.run_ordered_step(
        predict=lambda: order.append("predict"),
        batch=_batch(1),
    )
    assert trainer.modes == ["offline", "offline", "offline"]
    assert order == ["predict"]
    assert consumer.read_stats().buffered_count == 0
    assert consumer.read_stats().adapter_version == 3


def test_request_pins_survive_new_versions_until_explicit_release() -> None:
    store = LoRAVersionStore("tlm")
    store.publish(_weights(1.0))
    assert store.pin_request("active") == 1
    store.publish(_weights(2.0))
    assert store.pin_request("active") == 1
    assert store.pin_request("new") == 2
    assert store.release_request("active")
    assert store.pin_request("active") == 2


class _AdapterManager:
    def __init__(self) -> None:
        self.models: dict[int, object] = {}

    def list_adapters(self) -> dict[int, object]:
        return dict(self.models)

    def add_adapter(self, model: object) -> bool:
        self.models[int(model.id)] = model
        return True


def test_vllm_publisher_materializes_qv_and_applies_mrv1_mapping() -> None:
    store = LoRAVersionStore("tlm")
    store.publish(_weights(1.0))
    adapter_manager = _AdapterManager()
    runner = SimpleNamespace(
        lora_manager=SimpleNamespace(_adapter_manager=adapter_manager),
        lora_config=SimpleNamespace(max_lora_rank=8),
    )
    publisher = VLLMVersionedLoRAPublisher(runner=runner, store=store)
    assert publisher.sync_versions() == 1
    model = next(iter(adapter_manager.models.values()))
    assert set(model.loras) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    }

    input_batch = SimpleNamespace(
        req_id_to_index={"r0": 0, "r1": 1},
        request_lora_mapping=[0, 0],
        lora_id_to_request_ids={},
        lora_id_to_lora_request={},
    )
    assert publisher.apply_mrv1_input_batch(input_batch) == 2
    assert input_batch.request_lora_mapping[0] == input_batch.request_lora_mapping[1]
    store.publish(_weights(2.0))
    publisher.sync_versions()
    publisher.apply_mrv1_input_batch(input_batch)
    assert store.request_version("r0") == 1


def test_peft_trainer_updates_only_qv_lora_on_tiny_qwen() -> None:
    pytest.importorskip("peft")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    consumer = TLMConsumer(
        TLMConsumerConfig(rank=2, lora_alpha=4, threshold=-1.0)
    )
    peft_model = consumer.bind_hf_model(model)
    trainable = [name for name, parameter in peft_model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable)
    assert all("q_proj" in name or "v_proj" in name for name in trainable)

    batch = TLMTrainBatch(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        labels=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
    )
    consumer.update_online(batch)
    snapshot = consumer.version_store.get(1)
    assert snapshot.rank == 2
    assert {name.rsplit(".", 1)[-1] for name in snapshot.target_modules} == {
        "q_proj",
        "v_proj",
    }


def test_peft_trainer_rejects_non_huggingface_modules() -> None:
    pytest.importorskip("peft")

    with pytest.raises(RuntimeError, match="requires a Hugging Face PreTrainedModel"):
        HFPEFTTLMTrainer(torch.nn.Linear(2, 2), TLMConsumerConfig())
