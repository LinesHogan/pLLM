"""Fixed-storage right-action affine windows used by the region executor."""

import torch
import triton
import triton.language as tl

from pllm.contracts.tensor_policy import ScanForm


@triton.jit
def _coefficients(
    a: torch.Tensor,
    b: torch.Tensor,
    active: torch.Tensor,
    elements: tl.constexpr,
    matrix: tl.constexpr,
    width: tl.constexpr,
    total: tl.constexpr,
    decay: tl.constexpr,
    scale: tl.constexpr,
    block: tl.constexpr,
) -> None:
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < total
    eye = tl.cast(index % matrix // width == index % width, tl.float32)
    live = tl.load(active + index // elements, mask, 0)
    gram = tl.load(a + index, mask, 0)
    cross = tl.load(b + index, mask, 0)
    tl.store(a + index, tl.where(live, decay * (eye - scale * gram), eye), mask)
    tl.store(b + index, tl.where(live, scale * cross, 0), mask)


class RightMatrixBuffers:
    """One invocation or graph slot owns all coefficient, prefix and output storage."""

    def __init__(self, carry: torch.Tensor, window: int, chunk: int) -> None:
        self.a = carry.new_empty((window, *carry.shape))
        self.b, self.next_a, self.next_b, self.initial = (torch.empty_like(self.a) for _ in range(4))
        self.states = carry.new_empty((window + 1, *carry.shape))
        self.final = torch.empty_like(carry)
        self.output = carry.new_empty((window, carry.shape[0], chunk, carry.shape[1]))


def _tree(buffers: RightMatrixBuffers) -> tuple[torch.Tensor, torch.Tensor]:
    a, b, next_a, next_b = buffers.a, buffers.b, buffers.next_a, buffers.next_b
    for level in range(len(a).bit_length() - 1):
        offset = 1 << level
        next_a[:offset].copy_(a[:offset])
        next_b[:offset].copy_(b[:offset])
        # Earlier action is on the left: (A1*A2, B1*A2+B2).
        torch.bmm(a[:-offset].flatten(0, 1), a[offset:].flatten(0, 1), out=next_a[offset:].flatten(0, 1))
        torch.bmm(b[:-offset].flatten(0, 1), a[offset:].flatten(0, 1), out=next_b[offset:].flatten(0, 1))
        next_b[offset:].add_(b[offset:])
        a, b, next_a, next_b = next_a, next_b, a, b
    return a, b


def launch_window(
    inputs: torch.Tensor,
    keys: torch.Tensor,
    carry: torch.Tensor,
    active: torch.Tensor,
    buffers: RightMatrixBuffers,
    scale: float,
    decay: float,
    form: ScanForm,
) -> None:
    """Consume staged [window, heads, rows, width] tensors and device active masks."""
    x, k = inputs.flatten(0, 1), keys.flatten(0, 1)
    torch.bmm(k.transpose(1, 2), k, out=buffers.a.flatten(0, 1))
    torch.bmm(x.transpose(1, 2), k, out=buffers.b.flatten(0, 1))
    _coefficients[((buffers.a.numel() + 255) // 256,)](
        buffers.a,
        buffers.b,
        active,
        carry.numel(),
        carry.shape[-1] ** 2,
        carry.shape[-1],
        buffers.a.numel(),
        decay,
        scale,
        256,
        enable_fp_fusion=False,  # pyright: ignore[reportCallIssue]
    )
    buffers.states[0].copy_(carry)
    if form is ScanForm.HILLIS_STEELE:
        a, b = _tree(buffers)
        buffers.initial.copy_(carry)
        torch.bmm(buffers.initial.flatten(0, 1), a.flatten(0, 1), out=buffers.states[1:].flatten(0, 1))
        buffers.states[1:].add_(b)
    else:
        for index in range(len(buffers.a)):
            torch.bmm(buffers.states[index], buffers.a[index], out=buffers.states[index + 1])
            buffers.states[index + 1].add_(buffers.b[index])
    buffers.final.copy_(torch.where(active.any(), buffers.states[-1], carry))
    torch.bmm(x, buffers.states[:-1].flatten(0, 1).transpose(1, 2), out=buffers.output.flatten(0, 1))
