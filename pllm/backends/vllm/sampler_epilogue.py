"""Ordered dense additions in the sampler-input copy used by the vLLM adapter.

vLLM's temperature, filter and copy launchers have no dense additive operand. Its
sparse logit-bias interface cannot express workload tensors. This one copy kernel
keeps upstream processing and sampling intact while avoiding intermediate logits.
"""

import torch
import triton
import triton.language as tl

from pllm.execution.cuda_stream import upload


@triton.jit
def _copy_additions(
    Logits: torch.Tensor,
    Additions: "tuple[torch.Tensor, ...]",
    RowMap: torch.Tensor | None,
    Output: torch.Tensor,
    LS: tl.constexpr,
    VS: tl.constexpr,
    STRIDES: tl.constexpr,
    FULL: tl.constexpr,
    ROWS: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tokens = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    valid = tokens < VOCAB
    loaded = tl.load(Logits + row * LS + tokens * VS, valid, other=0)
    dtype = loaded.dtype
    value = loaded.to(tl.float32)
    for stage in tl.static_range(len(Additions)):
        selected = row
        if RowMap is not None:
            if not FULL[stage]:  # pyright: ignore[reportIndexIssue]
                selected = tl.load(RowMap + stage * ROWS + row)
        prediction = (
            tl.load(
                Additions[stage] + selected * STRIDES[stage][0] + tokens * STRIDES[stage][1],  # pyright: ignore[reportIndexIssue, reportUnknownArgumentType]
                valid & (selected >= 0),
                other=0,
            )
            .to(dtype)
            .to(tl.float32)
        )
        # Each declaration previously materialized logits in the incoming dtype.
        value = tl.where(selected >= 0, (value + prediction).to(dtype).to(tl.float32), value)
    tl.store(Output + row * VOCAB + tokens, value, valid)


def copy_additions(
    logits: torch.Tensor,
    additions: tuple[torch.Tensor, ...],
    rows: tuple[tuple[int, ...], ...],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Copy in the requested dtype, retaining each addition's incoming-logit rounding."""
    count = logits.shape[0]
    full = tuple(selected == tuple(range(count)) for selected in rows)
    row_map = None
    if not all(full):
        mapping = [[-1] * count for _ in rows]
        for stage, selected in enumerate(rows):
            for index, row in enumerate(selected):
                mapping[stage][row] = index
        row_map = upload(mapping, logits.device)
    return _copy(logits, list(additions), row_map, list(full), dtype)


@torch.library.custom_op("pllm::sampler_additions", mutates_args=())
def _copy(
    logits: torch.Tensor,
    additions: list[torch.Tensor],
    row_map: torch.Tensor | None,
    full: list[bool],
    dtype: torch.dtype,
) -> torch.Tensor:
    # wrap_triton treats tensor tuples as constants and retains their storage across steps.
    # A custom op keeps variable-length additions as runtime tensor operands.
    count, vocab = logits.shape
    output = torch.empty(logits.shape, dtype=dtype, device=logits.device)
    _copy_additions[(count, triton.cdiv(vocab, 1024))](
        logits,
        tuple(additions),
        row_map,
        output,
        logits.stride(0),
        logits.stride(1),
        tuple(tensor.stride() for tensor in additions),
        tuple(full),
        count,
        vocab,
        BLOCK=1024,
        enable_fp_fusion=False,  # pyright: ignore[reportCallIssue]
    )
    return output


def _copy_meta(
    logits: torch.Tensor,
    additions: list[torch.Tensor],
    row_map: torch.Tensor | None,
    full: list[bool],
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(logits.shape, dtype=dtype, device=logits.device)


_copy.register_fake(_copy_meta)  # pyright: ignore[reportUnknownMemberType]
