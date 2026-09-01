"""Full-root COW writes for the canonical CPU dense-page provider."""

from __future__ import annotations

import torch

from tllm.contracts.tensor_layout import TensorLayoutSpec


def copy_full_root_transition_targets(
    *,
    target_page_ids: tuple[int, ...],
    target_tensors: tuple[tuple[torch.Tensor, ...], ...],
    layout: TensorLayoutSpec,
    pools: tuple[torch.Tensor, ...],
    device: torch.device,
) -> None:
    """Validate complete root images, then write only private COW pages."""

    if len(target_tensors) != len(target_page_ids):
        raise ValueError("dense-page full-root images must align with transitions")
    for root_tensors in target_tensors:
        if len(root_tensors) != len(layout.fields):
            raise ValueError("dense-page full-root field count differs from layout")
        for tensor, field, pool in zip(
            root_tensors,
            layout.fields,
            pools,
            strict=True,
        ):
            if (
                tuple(tensor.shape) != field.shape
                or tensor.dtype != pool.dtype
                or tensor.device != device
            ):
                raise ValueError(
                    "dense-page full-root tensor differs from signed CPU layout"
                )
    with torch.no_grad():
        for page_id, root_tensors in zip(
            target_page_ids,
            target_tensors,
            strict=True,
        ):
            for pool, tensor in zip(pools, root_tensors, strict=True):
                pool[page_id].copy_(tensor)


__all__ = ["copy_full_root_transition_targets"]
