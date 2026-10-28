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
    row = tl.program_id(0) // tl.cdiv(width, 256)
    column = (tl.program_id(0) % tl.cdiv(width, 256)) * 256 + tl.arange(0, 256)
    src = tl.load(sources + row)
    dst = tl.load(destinations + row)
    valid = (dst >= 0) & (column < width)
    value = tl.load(source + src * source_stride + column, valid, 0)
    tl.store(destination + dst * destination_stride + column, value, valid)


@torch.library.custom_op("pllm::copy_pages", mutates_args={"destination"})
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
        _copy_pages[(len(destinations) * ((width + 255) // 256),)](
            source, destination, sources, destinations, source.stride(0), destination.stride(0), width
        )


@triton.jit
def _gather_rows(
    descriptor: torch.Tensor,
    rows: torch.Tensor,
    output: torch.Tensor,
    width: tl.constexpr,
) -> None:
    row = tl.program_id(0) // tl.cdiv(width, 256)
    column = (tl.program_id(0) % tl.cdiv(width, 256)) * 256 + tl.arange(0, 256)
    # Triton lowers the annotated Torch argument to a typed device pointer.
    address = tl.load(descriptor).to(tl.pointer_type(output.dtype.element_ty))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    stride0 = tl.load(descriptor + 1)
    stride1 = tl.load(descriptor + 2)
    index = tl.load(rows + row)
    value = tl.load(address + index * stride0 + column * stride1, column < width, 0)
    tl.store(output + row * width + column, value, column < width)


@torch.library.custom_op("pllm::gather_rows", mutates_args=())
def gather_rows(descriptor: torch.Tensor, rows: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Read a sampler tensor whose current address travels in the packed step table."""
    if not source.is_cuda:
        return source.index_select(0, rows)
    output = source.new_empty((len(rows), *source.shape[1:]))
    width = output[0].numel()
    _gather_rows[(len(rows) * ((width + 255) // 256),)](descriptor, rows, output, width)
    return output


def no_allocation(*operands: object) -> None:
    """Fake implementation for mutation-only kernels with caller-owned inputs and scratch."""


copy_pages.register_fake(no_allocation)  # pyright: ignore[reportUnknownMemberType]


@gather_rows.register_fake  # pyright: ignore[reportUnknownMemberType]
def gather_rows_meta(descriptor: torch.Tensor, rows: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """The gathered shape without dereferencing the descriptor's device address."""
    return source.new_empty((len(rows), *source.shape[1:]))


def chunk_capacity(rows: int, chunk: int) -> int:
    """Chunk-aligned room for arriving rows and at most chunk minus one pending rows."""
    return ((rows + chunk - 2) // chunk + 1) * chunk


def prepend_rows(
    tails: tuple[torch.Tensor, ...],
    rows: tuple[torch.Tensor, ...],
    length: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, ...]:
    """Join device-counted tails into a chunk-aligned capacity without reading the count."""
    capacity = chunk_capacity(len(rows[0]), chunk)
    indices = torch.arange(capacity, device=length.device)
    joined: list[torch.Tensor] = []
    for tail, value in zip(tails, rows, strict=True):
        source = torch.cat((tail, value, value.new_zeros((1, *value.shape[1:]))))
        offsets = torch.where(indices < length, indices, indices - length + len(tail))
        offsets = torch.where(indices < length + len(value), offsets, len(source) - 1)
        joined.append(source.index_select(0, offsets))
    return tuple(joined)
