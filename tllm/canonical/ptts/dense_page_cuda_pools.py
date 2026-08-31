"""Fixed CUDA tensor pools and eager kernels for dense-page state."""

from __future__ import annotations

import torch

from tllm.canonical.ptts.dense_page_adamw import (
    DensePageAdamWConfig,
    DensePageAdamWFieldBinding,
    apply_cuda_dense_page_adamw,
)
from tllm.canonical.ptts.dense_page_cuda_types import (
    CudaDensePageAdmission,
    CudaDensePageInitialRoot,
)
from tllm.canonical.ptts.dense_page_validation import DensePageSignedSpec
from tllm.contracts.tensor_layout import LayoutFieldId, TensorDType


def _cuda_dtype(dtype: TensorDType) -> torch.dtype:
    match dtype:
        case TensorDType.BOOL:
            return torch.bool
        case TensorDType.UINT8:
            return torch.uint8
        case TensorDType.INT32:
            return torch.int32
        case TensorDType.INT64:
            return torch.int64
        case TensorDType.FLOAT16:
            return torch.float16
        case TensorDType.BFLOAT16:
            return torch.bfloat16
        case TensorDType.FLOAT32:
            return torch.float32
        case TensorDType.FLOAT64:
            return torch.float64


class CudaDensePagePools:
    """Cold fixed-address pools; logical roots remain owned by the ledger."""

    __slots__ = (
        "_layout",
        "_selected",
        "_device",
        "_pools",
        "_addresses",
        "_pool_bytes",
    )

    def __init__(
        self,
        signed: DensePageSignedSpec,
        device: torch.device,
    ) -> None:
        with torch.cuda.device(device):
            pools = tuple(
                torch.zeros(
                    (signed.physical_rows, *field.shape),
                    dtype=_cuda_dtype(field.dtype),
                    device=device,
                )
                for field in signed.layout.fields
            )
        pool_bytes = sum(
            int(pool.numel()) * int(pool.element_size()) for pool in pools
        )
        if pool_bytes != signed.selected_plan.reserved_bytes:
            raise RuntimeError(
                "signed reserved bytes differ from CUDA pool allocation"
            )
        self._layout = signed.layout
        self._selected = signed.selected_plan
        self._device = device
        self._pools = pools
        self._addresses = tuple(pool.data_ptr() for pool in pools)
        self._pool_bytes = pool_bytes

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def pool_bytes(self) -> int:
        return self._pool_bytes

    @property
    def addresses(self) -> tuple[int, ...]:
        return self._addresses

    def require_stable(self) -> None:
        physical_rows = (
            self._selected.max_admitted_lineages
            * self._selected.max_live_versions_per_lineage
        )
        for field, pool, address in zip(
            self._layout.fields,
            self._pools,
            self._addresses,
            strict=True,
        ):
            if (
                pool.data_ptr() != address
                or tuple(pool.shape) != (physical_rows, *field.shape)
                or pool.dtype is not _cuda_dtype(field.dtype)
                or pool.device != self._device
            ):
                raise RuntimeError("CUDA dense-page pool signature changed")

    def new_zero_root(
        self,
        stream: torch.cuda.Stream,
    ) -> CudaDensePageInitialRoot:
        self._require_stream(stream)
        with torch.cuda.stream(stream):
            return CudaDensePageInitialRoot(
                stream,
                tuple(
                    torch.zeros(
                        field.shape,
                        dtype=pool.dtype,
                        device=self._device,
                    )
                    for field, pool in zip(
                        self._layout.fields,
                        self._pools,
                        strict=True,
                    )
                ),
            )

    def require_initial_root(self, root: CudaDensePageInitialRoot) -> None:
        if len(root.tensors) != len(self._layout.fields):
            raise ValueError("CUDA admission field count differs from layout")
        for initial, field, pool in zip(
            root.tensors,
            self._layout.fields,
            self._pools,
            strict=True,
        ):
            if (
                tuple(initial.shape) != field.shape
                or initial.dtype is not pool.dtype
                or initial.device != self._device
            ):
                raise ValueError("CUDA admission tensor differs from signed layout")

    def submit_admission(
        self,
        admissions: tuple[CudaDensePageAdmission, ...],
        target_page_ids: tuple[int, ...],
        stream: torch.cuda.Stream,
    ) -> None:
        self._require_stream(stream)
        with torch.cuda.stream(stream), torch.no_grad():
            for admission, page_id in zip(
                admissions,
                target_page_ids,
                strict=True,
            ):
                for pool, initial in zip(
                    self._pools,
                    admission.initial.tensors,
                    strict=True,
                ):
                    pool[page_id].copy_(initial)

    def require_adamw(
        self,
        fields: DensePageAdamWFieldBinding,
        gradients: tuple[torch.Tensor, ...],
        transition_count: int,
    ) -> None:
        if (
            type(fields) is not DensePageAdamWFieldBinding
            or type(gradients) is not tuple
            or any(type(value) is not torch.Tensor for value in gradients)
        ):
            raise TypeError("CUDA AdamW requires exact fields and tensors")
        field_ids = (
            int(fields.parameter),
            int(fields.gradient),
            int(fields.step),
            int(fields.exp_avg),
            int(fields.exp_avg_sq),
        )
        if any(field_id >= len(self._layout.fields) for field_id in field_ids):
            raise ValueError("CUDA AdamW field ID is outside the signed layout")
        parameter = self._layout.fields[int(fields.parameter)]
        gradient = self._layout.fields[int(fields.gradient)]
        exp_avg = self._layout.fields[int(fields.exp_avg)]
        exp_avg_sq = self._layout.fields[int(fields.exp_avg_sq)]
        step = self._layout.fields[int(fields.step)]
        if (
            parameter.shape != gradient.shape
            or parameter.shape != exp_avg.shape
            or parameter.shape != exp_avg_sq.shape
            or parameter.dtype != gradient.dtype
            or parameter.dtype != exp_avg.dtype
            or parameter.dtype != exp_avg_sq.dtype
            or step.shape != ()
            or step.dtype is not TensorDType.FLOAT32
            or parameter.dtype
            not in (
                TensorDType.FLOAT16,
                TensorDType.BFLOAT16,
                TensorDType.FLOAT32,
                TensorDType.FLOAT64,
            )
        ):
            raise ValueError("CUDA AdamW fields differ from signed layout")
        gradient_pool = self._pools[int(fields.gradient)]
        if len(gradients) != transition_count or any(
            tuple(value.shape) != tuple(gradient_pool.shape[1:])
            or value.dtype is not gradient_pool.dtype
            or value.device != self._device
            for value in gradients
        ):
            raise ValueError("CUDA AdamW gradients differ from signed layout")

    def submit_adamw(
        self,
        *,
        source_page_ids: tuple[int, ...],
        target_page_ids: tuple[int, ...],
        fields: DensePageAdamWFieldBinding,
        config: DensePageAdamWConfig,
        gradients: tuple[torch.Tensor, ...],
        stream: torch.cuda.Stream,
    ) -> None:
        self._require_stream(stream)
        with torch.cuda.stream(stream):
            apply_cuda_dense_page_adamw(
                pools=self._pools,
                source_page_ids=source_page_ids,
                target_page_ids=target_page_ids,
                fields=fields,
                config=config,
                gradients=gradients,
            )

    def snapshot(
        self,
        page_id: int,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, ...]:
        self._require_stream(stream)
        with torch.cuda.stream(stream):
            return tuple(
                pool[page_id].detach().clone() for pool in self._pools
            )

    def field_rows(
        self,
        page_ids: tuple[int, ...],
        field_id: LayoutFieldId,
        stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        self._require_stream(stream)
        if (
            type(page_ids) is not tuple
            or not page_ids
            or type(field_id) is not int
            or int(field_id) < 0
            or int(field_id) >= len(self._pools)
        ):
            raise ValueError("CUDA field-row projection is invalid")
        with torch.cuda.stream(stream):
            indices = torch.tensor(
                page_ids,
                dtype=torch.long,
                device=self._device,
            )
            return self._pools[int(field_id)].index_select(0, indices)

    def _require_stream(self, stream: torch.cuda.Stream) -> None:
        if type(stream) is not torch.cuda.Stream or stream.device != self._device:
            raise RuntimeError("CUDA dense-page pool stream uses another device")


__all__ = ["CudaDensePagePools"]
