"""Page copies and row gathers used by captured tensor steps and pinned host tables."""

import torch
import triton
import triton.language as tl
from torch._functorch.autograd_function import VmapInfo


@triton.jit
def _copy_pages(
    source: torch.Tensor,
    destination: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    source_stride: tl.constexpr,
    destination_stride: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
) -> None:
    row = tl.program_id(0) // tl.cdiv(width, block)
    column = (tl.program_id(0) % tl.cdiv(width, block)) * block + tl.arange(0, block)
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
        # Larger tiles avoid thousands of idle blocks in masked optimizer fallback copies.
        block = 2048
        _copy_pages[(len(destinations) * ((width + block - 1) // block),)](
            source, destination, sources, destinations, source.stride(0), destination.stride(0), width, block
        )


@triton.jit
def _append_tail(
    pool: torch.Tensor,
    rows: torch.Tensor,
    bounds: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    lengths: torch.Tensor,
    generations: torch.Tensor,
    capacity: int,
    page_stride: tl.constexpr,
    row_stride: tl.constexpr,
    column_stride: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
) -> None:
    lane = tl.program_id(1).to(tl.int64)
    dst = tl.load(destinations + lane)
    if dst >= 0:
        src = tl.load(sources + lane)
        length = tl.load(lengths + lane)
        gen = tl.load(generations + lane)
        prev_gen = tl.load(bounds + dst * 2)
        prev_dirty = tl.load(bounds + dst * 2 + 1)
        dirty = tl.where((prev_gen == gen - 1) | (prev_gen == gen), prev_dirty, capacity)
        first = tl.where(src == dst, length, 0) * width
        end = tl.maximum(length + 1, dirty) * width
        for tile in tl.range(tl.program_id(0), tl.cdiv(end - first, block), tl.num_programs(0)):
            offset = first + tile * block + tl.arange(0, block)
            history = tl.load(pool + src * page_stride + offset, offset < length * width, 0)
            row = tl.load(rows + lane * row_stride + offset % width * column_stride, offset // width == length, 0)
            value = tl.where(offset // width == length, row.to(history.dtype), history)
            tl.store(pool + dst * page_stride + offset, value, offset < end)


@triton.jit
def _commit_tail_bounds(
    bounds: torch.Tensor,
    destinations: torch.Tensor,
    lengths: torch.Tensor,
    generations: torch.Tensor,
) -> None:
    lane = tl.program_id(0).to(tl.int64)
    dst = tl.load(destinations + lane)
    if dst >= 0:
        tl.store(bounds + dst * 2, tl.load(generations + lane))
        tl.store(bounds + dst * 2 + 1, tl.load(lengths + lane) + 1)


def _write_tail_cpu(
    pool: torch.Tensor,
    rows: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    length: torch.Tensor,
    dirty: torch.Tensor,
) -> None:
    pos = torch.arange(pool.shape[1], device=pool.device)
    lanes, offsets = torch.where((pos < length[:, None]) & (src != dst)[:, None])
    pool[dst[lanes], offsets] = pool[src[lanes], offsets]
    pool[dst, length] = rows.to(pool)
    lanes, offsets = torch.where((pos > length[:, None]) & (pos < dirty[:, None]))
    pool[dst[lanes], offsets] = 0


@torch.library.custom_op("pllm::append_tail", mutates_args={"pool", "bounds"})
def append_tail(
    pool: torch.Tensor,
    rows: torch.Tensor,
    bounds: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    lengths: torch.Tensor,
    generations: torch.Tensor,
) -> None:
    """Inherit the live prefix, append one row, clear stale recycled rows and record the new dirty bound."""
    if not pool.is_cuda:
        active = destinations >= 0
        src, dst = sources[active], destinations[active]
        gen, length = generations[active], lengths[active]
        prev_gen, prev_dirty = bounds[dst, 0], bounds[dst, 1]
        dirty = torch.where((prev_gen == gen - 1) | (prev_gen == gen), prev_dirty, pool.shape[1])
        # Invalidate before mutating so an interrupted append forces a full-page clear on reuse.
        bounds[dst, 0] = -2
        _write_tail_cpu(pool, rows[active], src, dst, length, dirty)
        bounds[dst, 0] = gen
        bounds[dst, 1] = length + 1
        return
    capacity, width = pool.shape[1], pool.shape[2]
    block = 1024
    grid = min(256, triton.cdiv(capacity * width, block)), len(rows)
    _append_tail[grid](
        pool,
        rows,
        bounds,
        sources,
        destinations,
        lengths,
        generations,
        capacity,
        pool.stride(0),
        rows.stride(0),
        rows.stride(1),
        width,
        block,
    )
    _commit_tail_bounds[(len(rows),)](bounds, destinations, lengths, generations)


@triton.jit
def _clear_tail(
    pool: torch.Tensor,
    bounds: torch.Tensor,
    destination: torch.Tensor,
    page_stride: tl.constexpr,
    elements: tl.constexpr,
    block: tl.constexpr,
) -> None:
    dst = tl.load(destination)
    if dst >= 0:
        offset = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
        zero = tl.zeros((block,), pool.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        tl.store(pool + dst * page_stride + offset, zero, offset < elements)
        if tl.program_id(0) == 0:
            tl.store(bounds + dst * 2 + 1, dst - dst)


@torch.library.custom_op("pllm::clear_tail", mutates_args={"pool", "bounds"})
def clear_tail(pool: torch.Tensor, bounds: torch.Tensor, destination: torch.Tensor) -> None:
    """Zero a completed tail page and reset its dirty-row bound."""
    if not pool.is_cuda:
        active = destination[destination >= 0]
        pool[active] = 0
        bounds[active, 1] = 0
        return
    elements = pool[0].numel()
    if elements:
        block = 2048
        _clear_tail[(triton.cdiv(elements, block),)](pool, bounds, destination, pool.stride(0), elements, block)


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
    """Read a row tensor whose current address travels in the packed step table."""
    if not source.is_cuda:
        return source.index_select(0, rows)
    output = source.new_empty((len(rows), *source.shape[1:]))
    width = output[0].numel()
    _gather_rows[(len(rows) * ((width + 255) // 256),)](descriptor, rows, output, width)
    return output


@triton.jit
def _host_rows(
    source: torch.Tensor,
    rows: torch.Tensor,
    output: torch.Tensor,
    height: int,
    width: tl.constexpr,
    stride0: int,
    stride1: int,
    count: int,
    block: tl.constexpr,
) -> None:
    # Row counts and strides vary per step and view; as runtime arguments they reuse one compiled kernel.
    tiles = tl.cdiv(width, block)
    for tile in tl.range(tl.program_id(0), count * tiles, tl.num_programs(0)):
        row = tile // tiles
        column = tile % tiles * block + tl.arange(0, block)
        index = tl.load(rows + row).to(tl.int64)
        index = tl.where(index < 0, index + height, index)
        value = tl.load(source + index * stride0 + column * stride1, column < width, 0)
        tl.store(output + row * width + column, value, column < width)


@torch.library.custom_op("pllm::host_rows", mutates_args=())
def host_rows(source: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Read GPU-indexed pinned rows with a bounded grid, leaving SMs for the forward."""
    output = source.new_empty((*rows.shape, source.shape[1]))
    if output.numel():
        block = 2048
        tiles = rows.numel() * triton.cdiv(source.shape[1], block)
        sms = int(torch.cuda.get_device_properties(source.device).multi_processor_count)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        programs = min(tiles, max(1, sms // 4))
        _host_rows[(programs,)](
            source,
            rows.contiguous(),
            output,
            source.shape[0],
            source.shape[1],
            source.stride(0),
            source.stride(1),
            rows.numel(),
            block,
        )
    return output


@host_rows.register_fake  # pyright: ignore[reportUnknownMemberType]
def host_rows_meta(source: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """The ordinary row-gather shape, without accessing pinned storage."""
    return source.new_empty((*rows.shape, source.shape[1]))


@host_rows.register_vmap  # pyright: ignore[reportUnknownMemberType]
def host_rows_vmap(
    info: VmapInfo, in_dims: tuple[int | None, ...], source: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Fold mapped indices into one gather of the shared, two-dimensional table."""
    source_dim, row_dim = in_dims
    assert source_dim is None and row_dim is not None
    return host_rows(source, rows.movedim(row_dim, 0)), 0


def no_allocation(*operands: object) -> None:
    """Fake implementation for mutation-only kernels with caller-owned inputs and scratch."""


append_tail.register_fake(no_allocation)  # pyright: ignore[reportUnknownMemberType]
clear_tail.register_fake(no_allocation)  # pyright: ignore[reportUnknownMemberType]
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
