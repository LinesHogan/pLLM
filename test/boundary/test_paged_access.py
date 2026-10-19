"""Physical KV aliases: shared prefixes, interleaved layers and current writes."""

import pytest
import torch

from pllm.execution.paged_access import PagedLayer, bind_paged_access
from test.region_fixtures import paged_rows


@pytest.mark.parametrize("layout", ("HND", "NHD"))
def test_shared_read_prefix_and_interleaved_layers_keep_distinct_writes(layout):
    root = torch.zeros((4, 2, 2, 4, 4) if layout == "HND" else (4, 2, 4, 2, 4))
    caches = tuple(root[:, :, i] if layout == "HND" else root[:, i].permute(0, 2, 1, 3) for i in range(2))
    table, slots = torch.tensor([[0, 1], [0, 2]]), torch.tensor([4, 5, 8, 9])
    layers = tuple(PagedLayer(str(i), cache, table, slots) for i, cache in enumerate(caches))
    batch = paged_rows()
    plan = bind_paged_access(batch, layers)
    expected = {
        cache[1 if row.request_ordinal == 0 else 2, head, row.token_offset % 4].data_ptr()
        for cache in caches
        for row in batch
        for head in range(2)
    }
    assert {write.address for write in plan.writes} == expected
    assert len(plan.writes) == len(expected) == 16
    assert plan.slot("1", batch[3]) == (2, 1)
    assert torch.count_nonzero(root) == 0


def test_write_aliases_and_page_slot_disagreement_fail_before_mutation():
    cache = torch.arange(4 * 2 * 4 * 4, dtype=torch.float32).reshape(4, 2, 4, 4)
    before = cache.clone()
    table, slots = torch.tensor([[0, 1], [0, 2]]), torch.tensor([4, 5, 8, 9])
    with pytest.raises(ValueError, match="slot differs"):
        bind_paged_access(paged_rows(), (PagedLayer("a", cache, table, torch.tensor([4, 5, 8, 10])),))
    with pytest.raises(ValueError, match="aliases"):
        bind_paged_access(
            paged_rows(), (PagedLayer("a", cache, torch.tensor([[0, 1], [0, 1]]), torch.tensor([4, 5, 4, 5])),)
        )
    # Distinct current slots can still conflict with another request's history.
    with pytest.raises(ValueError, match="aliases"):
        bind_paged_access(
            paged_rows((4, 5, 8, 9)), (PagedLayer("a", cache, torch.tensor([[0, 1, 3], [1, 3, 2]]), slots),)
        )
    # Separate logical layer names do not establish separate storage.
    with pytest.raises(ValueError, match="aliases"):
        bind_paged_access(paged_rows(), (PagedLayer("a", cache, table, slots), PagedLayer("b", cache, table, slots)))
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
