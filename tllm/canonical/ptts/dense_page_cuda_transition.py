"""Full-root COW writes for canonical CUDA dense-page pools."""

from __future__ import annotations

import torch

from tllm.contracts.tensor_layout import TensorLayoutSpec


def validate_cuda_full_root_transition_targets(
    *,
    target_page_ids: tuple[int, ...],
    target_tensors: tuple[tuple[torch.Tensor, ...], ...],
    layout: TensorLayoutSpec,
    pools: tuple[torch.Tensor, ...],
    device: torch.device,
) -> None:
    """Require complete stream-local root images for every reserved target."""

    if len(target_tensors) != len(target_page_ids):
        raise ValueError("CUDA full-root images must align with transitions")
    for root_tensors in target_tensors:
        if len(root_tensors) != len(layout.fields):
            raise ValueError("CUDA full-root field count differs from layout")
        for tensor, field, pool in zip(
            root_tensors,
            layout.fields,
            pools,
            strict=True,
        ):
            if (
                tuple(tensor.shape) != field.shape
                or tensor.dtype is not pool.dtype
                or tensor.device != device
            ):
                raise ValueError(
                    "CUDA full-root tensor differs from signed layout"
                )


def copy_cuda_full_root_transition_targets(
    *,
    target_page_ids: tuple[int, ...],
    target_tensors: tuple[tuple[torch.Tensor, ...], ...],
    pools: tuple[torch.Tensor, ...],
    stream: torch.cuda.Stream,
) -> None:
    """Copy prevalidated complete root images into private COW pages."""

    with torch.cuda.stream(stream), torch.no_grad():
        for page_id, root_tensors in zip(
            target_page_ids,
            target_tensors,
            strict=True,
        ):
            for pool, tensor in zip(pools, root_tensors, strict=True):
                pool[page_id].copy_(tensor)


__all__ = [
    "copy_cuda_full_root_transition_targets",
    "validate_cuda_full_root_transition_targets",
]
