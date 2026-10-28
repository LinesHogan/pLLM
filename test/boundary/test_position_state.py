"""Static allocation layout, split kernel pages, and schema/source tampering evidence."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from pllm import KVOffload, _core
from pllm.backends.vllm.position_state import PositionCacheProvider, PositionMetadata
from pllm.config import ExecutionOptions
from pllm.contracts.codec import encode_boundary
from pllm.workloads.step_loop import StepLoop
from test.workloads.test_recirculation import FACTS, ROWS


@pytest.mark.parametrize("width,allocation", ((896, 16), (3584, 16), (896, 32), (3584, 32)))
def test_predecessors_follow_allocation_pages(width, allocation):
    kernel, permutation = 16, [3, 1]
    cache = torch.full((4 * allocation // kernel, kernel, 1, width // allocation), -999.0)
    layer = SimpleNamespace(kv_cache=cache, layer_name="carry")
    provider = PositionCacheProvider((layer,), (2.0,), allocation, torch.empty(1, width))
    slots = torch.zeros(1, dtype=torch.int64)
    table = torch.tensor(
        [[block * (allocation // kernel) + i for block in permutation for i in range(allocation // kernel)]]
    )
    metadata = PositionMetadata(SimpleNamespace(slot_mapping=slots, block_table_tensor=table))
    context = SimpleNamespace(attn_metadata={"carry": metadata}, slot_mapping={"carry": slots})
    previous = torch.full((1, width), 2.0)
    for offset in range(allocation + 2):
        slots.fill_(permutation[offset // allocation] * allocation + offset % allocation)
        provider.stage(context, torch.tensor([0]), torch.tensor([0]), torch.tensor([offset]), torch.tensor([False]))
        _, entry = provider.entry(())
        torch.testing.assert_close(entry, previous)
        previous = entry + 1
        provider.exit((previous,))
    torch.testing.assert_close(provider.tables[0][3], torch.full((width,), 2.0 + allocation))


@pytest.mark.parametrize("target", ("schema", "effects", "ports"))
def test_position_source_cannot_use_stateless_signing_exemption(target):
    workload = StepLoop()
    facts = replace(FACTS, probe=replace(FACTS.probe, position_states=(("history", 0.0),)))
    attachment = workload.compile_backend(workload.specialize(facts, ExecutionOptions())).attachment
    payload = json.loads(encode_boundary(attachment))
    effect = payload["plan"]["semantic_graph"]["tasks"][0]["replay"]
    if target == "schema":
        payload["probe"]["position_states"] = []
    elif target == "effects":
        effect["role_sensitive"] = False
    else:
        effect["digest"] = ""
    with pytest.raises(ValueError):
        _core.check_backend_attachment(json.dumps(payload))


def test_unsupported_transfer():
    with pytest.raises(ValueError, match="LMCache"):
        KVOffload(connector="LMCacheConnectorV1")


@pytest.mark.parametrize("span", (False, True))
def test_span_saves_every_page_and_fills_unselected_generated_pages(span):
    from unittest.mock import patch

    from pllm.backends.vllm.replay import VllmTokenwiseReplay
    from pllm.contracts.hidden_locator import HiddenPhase
    from pllm.contracts.semantic_graph import TokenProgramEffect
    from pllm.execution.token_program import Save

    cache = torch.full((6, 2, 1, 2), -999.0)
    layer = SimpleNamespace(kv_cache=cache, layer_name="carry")
    replay = SimpleNamespace(
        position_layers=(layer,),
        allocation=2,
        effect=TokenProgramEffect(3, "prefill", (("history", 2.0),)),
        program={"prefill": SimpleNamespace(prompt_pass_is_identity=not span, steps=(Save("history", "tap"),))},
    )
    for start, stop in ((0, 8), (8, 12)):
        rows = tuple(
            replace(
                ROWS.rows[0],
                packed_row_index=i - start,
                token_offset=i,
                phase=HiddenPhase.PREFILL if span or i < 3 else HiddenPhase.DECODE,
            )
            for i in range(start, stop)
        )
        slots = torch.arange(start, stop)
        metadata = PositionMetadata(
            SimpleNamespace(slot_mapping=slots, block_table_tensor=torch.arange(6).reshape(1, -1))
        )
        context = SimpleNamespace(attn_metadata={"carry": metadata}, slot_mapping={"carry": slots})
        hidden = torch.arange(start * 4, stop * 4, dtype=torch.float32).reshape(-1, 4)
        with patch("pllm.backends.vllm.replay.get_forward_context", return_value=context):
            VllmTokenwiseReplay.save_span(
                replay, rows, (rows,), torch.tensor([0]), hidden, {"history": hidden} if span else {}
            )
    torch.testing.assert_close(
        cache.view(6, 4),
        torch.arange(48, dtype=torch.float32).reshape(12, 4)[1::2]
        if span
        else torch.tensor([[4.0, 5.0, 6.0, 7.0], *([[8.0, 9.0, 10.0, 11.0]] * 5)]),
    )
