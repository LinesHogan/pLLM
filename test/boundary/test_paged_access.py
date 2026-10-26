"""Logical KV access: consecutive rows, recorded intervals and device write cells."""

import pytest
import torch

from pllm.execution.paged_access import PagedLayer, bind_paged_access
from test.region_fixtures import paged_rows


def test_rows_record_intervals_and_device_cells():
    caches = tuple(torch.zeros(4, 2, 4, 4) for _ in range(2))
    slots = torch.tensor([4, 5, 8, 9])
    batch = paged_rows()
    plan = bind_paged_access(batch, tuple(PagedLayer(str(i), cache, slots) for i, cache in enumerate(caches)))
    block, offset = plan.cell("1", batch[3])
    assert (block.tolist(), offset.tolist()) == ([2], [1])
    assert set(plan.ranges) == {(layer, request) for layer in ("0", "1") for request in (0, 1)}
    with pytest.raises(ValueError):
        bind_paged_access((batch[0], batch[0]), (PagedLayer("a", caches[0], slots),))
