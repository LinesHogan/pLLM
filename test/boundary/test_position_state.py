"""Static allocation layout, split kernel pages, and schema/source tampering evidence."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from pllm import KVOffload, _core
from pllm.backends.vllm.position_state import PositionCacheProvider, PositionMetadata
from pllm.config import ExecutionOptions
from pllm.execution.codec import encode_boundary
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
        provider.stage(context, replace(ROWS.rows[0], token_offset=offset))
        _, entry = provider.entry(())
        torch.testing.assert_close(entry, previous)
        previous = entry + 1
        provider.exit((previous,))
    torch.testing.assert_close(provider.tables[0][3], torch.full((width,), 2.0 + allocation))


@pytest.mark.parametrize("target", ("schema", "effects", "ports"))
def test_position_source_cannot_use_stateless_signing_exemption(target):
    workload = StepLoop()
    facts = replace(FACTS, probe=replace(FACTS.probe, position_states=(("history", 0.0),)))
    signed = workload.compile_backend(workload.specialize(facts, ExecutionOptions())).attachment
    payload = json.loads(encode_boundary(signed.inputs))
    entry = next(
        function
        for function in payload["plan"]["semantic_graph"]["source_regions"][0]["program"]["functions"]
        if function["name"] == "position.entry@1"
    )
    if target == "schema":
        payload["probe"]["position_states"] = []
    elif target == "effects":
        entry["accesses"] = []
    else:
        entry["outputs"] = []
    with pytest.raises(ValueError):
        _core.compile_backend_attachment(json.dumps(payload))


def test_unsupported_transfer():
    with pytest.raises(ValueError, match="LMCache"):
        KVOffload(connector="LMCacheConnectorV1")
