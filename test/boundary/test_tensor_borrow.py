"""Physical read-only storage guarantees used by received pages and model readers."""

import pytest
import torch

from pllm.contracts.tensor_borrow import BorrowedResource
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView, validate_readonly_views


def test_readonly_borrows_allow_shared_reads_and_reject_changed_values():
    weight = torch.tensor([3.0, 7.0])
    view = ReadonlyTensorView(
        BorrowedResource("weights", (2,)),
        tuple(ReadonlyTensorCell.bind((i,), (weight,)) for i in range(2)),
    )
    validate_readonly_views((view,), ())
    for index in range(2):
        value = view.tensor((index,), 0)
        torch.testing.assert_close(value, weight, rtol=0, atol=0)
        assert value.data_ptr() == weight.data_ptr()
    weight.add_(1)
    with pytest.raises(ValueError):
        validate_readonly_views((view,), ())


def test_readonly_views_reject_physical_aliases_with_writable_storage():
    weight = torch.arange(8, dtype=torch.float32)
    view = ReadonlyTensorView(BorrowedResource("weights", ()), (ReadonlyTensorCell.bind((), (weight[2:6],)),))
    storage = weight.untyped_storage()
    protected = ((str(weight.device), storage.data_ptr(), storage.nbytes()),)
    with pytest.raises(ValueError):
        validate_readonly_views((view,), protected)
    assert view.allocation_ranges == protected
