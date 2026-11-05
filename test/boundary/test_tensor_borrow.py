"""Physical read-only storage guarantees used by received pages and model readers."""

import pytest
import torch

from pllm.contracts.semantic_graph import TaskResultInputSource, TaskSymbol
from pllm.contracts.tensor_borrow import BorrowedResource
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView, validate_readonly_views
from pllm.execution.tensor_frame import TensorFrame, result_port
from pllm.execution.tensor_program import TensorBatch
from pllm.testing import step_input


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


@pytest.mark.gpu
def test_task_results_live_until_cuda_readers_finish():
    """A released result keeps its values while another stream's read is still queued."""
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    source = TaskResultInputSource(TaskSymbol("lookup"), "value")
    frame = TensorFrame(step_input(()), ())
    with torch.cuda.stream(producer):
        value = torch.ones(65536, device="cuda")
        ready = torch.cuda.Event()
        ready.record()
    batch = TensorBatch(RequestStateSelection((), ()), (), (), consumer, results={(source.task, source.output): value})
    with torch.cuda.stream(consumer):
        output = torch.empty_like(value)
        consumer.wait_event(ready)
        torch.cuda._sleep(100_000_000)
        borrowed = result_port(source).read(frame, batch)
        output.copy_(borrowed)
    del borrowed, value
    batch.results.clear()
    with torch.cuda.stream(producer):
        poison = torch.empty_like(output)
        poison.fill_(float("nan"))
    producer.synchronize()
    consumer.synchronize()
    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
