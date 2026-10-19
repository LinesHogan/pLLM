"""Masked page copies used by captured tensor steps for fields AdamW does not overwrite."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_pages(
    source: torch.Tensor,
    destination: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    source_stride: tl.constexpr,
    destination_stride: tl.constexpr,
    width: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    column = tl.program_id(1) * 256 + tl.arange(0, 256)
    src = tl.load(sources + row)
    dst = tl.load(destinations + row)
    valid = (dst >= 0) & (column < width)
    value = tl.load(source + src * source_stride + column, valid, 0)
    tl.store(destination + dst * destination_stride + column, value, valid)


def copy_pages(
    source: torch.Tensor, destination: torch.Tensor, sources: torch.Tensor, destinations: torch.Tensor
) -> None:
    """Copy the table's active rows, leaving every other destination byte untouched."""
    if not source.is_cuda:
        active = destinations >= 0
        destination.index_copy_(0, destinations[active], source.index_select(0, sources[active]))
        return
    width = source[0].numel()
    if width:
        _copy_pages[(len(destinations), (width + 255) // 256)](
            source, destination, sources, destinations, source.stride(0), destination.stride(0), width
        )


@triton.jit
def _gather_rows(
    descriptor: torch.Tensor,
    rows: torch.Tensor,
    output: torch.Tensor,
    width: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    column = tl.program_id(1) * 256 + tl.arange(0, 256)
    # Triton lowers the annotated Torch argument to a typed device pointer.
    address = tl.load(descriptor).to(tl.pointer_type(output.dtype.element_ty))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    stride0 = tl.load(descriptor + 1)
    stride1 = tl.load(descriptor + 2)
    index = tl.load(rows + row)
    value = tl.load(address + index * stride0 + column * stride1, column < width, 0)
    tl.store(output + row * width + column, value, column < width)


def gather_rows(descriptor: torch.Tensor, rows: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Read a sampler tensor whose current address travels in the packed step table."""
    if not source.is_cuda:
        return source.index_select(0, rows)
    output = source.new_empty((len(rows), *source.shape[1:]))
    width = output[0].numel()
    _gather_rows[(len(rows), (width + 255) // 256)](descriptor, rows, output, width)
    return output
