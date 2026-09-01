"""Cold physical facts for canonical dense-page providers."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from tllm.contracts.state_representation import (
    StateDeviceKind,
    StateProviderPlan,
    StateRepresentation,
)
from tllm.contracts.tensor_layout import (
    TensorDType,
    TensorLayoutBinding,
    TensorLayoutSpec,
)


@dataclass(frozen=True, slots=True)
class DensePageCpuProviderFacts:
    """Cold physical facts advertised by the strict CPU dense-page provider."""

    provider_id: str
    max_live_versions_per_lineage: int

    def __post_init__(self) -> None:
        if (
            type(self.provider_id) is not str
            or type(self.max_live_versions_per_lineage) is not int
            or not self.provider_id
            or self.max_live_versions_per_lineage < 2
        ):
            raise ValueError(
                "CPU dense-page facts require an ID and at least two live versions"
            )

    def materialize_plan(
        self,
        *,
        resource: str,
        layout: TensorLayoutSpec,
        binding: TensorLayoutBinding,
        max_admitted_lineages: int,
    ) -> StateProviderPlan:
        return _materialize_dense_plan(
            resource=resource,
            provider_id=self.provider_id,
            device_kind=StateDeviceKind.CPU,
            device_ordinal=0,
            layout=layout,
            binding=binding,
            max_admitted_lineages=max_admitted_lineages,
            max_live_versions_per_lineage=self.max_live_versions_per_lineage,
            capabilities=(
                "device/cpu@1",
                "optimizer/adamw@1",
                "state/exact_versions@1",
                "transition/full_root@1",
            ),
        )


@dataclass(frozen=True, slots=True)
class DensePageCudaProviderFacts:
    """Cold physical facts for one explicit CUDA dense-page provider."""

    provider_id: str
    device_ordinal: int
    max_live_versions_per_lineage: int

    def __post_init__(self) -> None:
        if (
            type(self.provider_id) is not str
            or type(self.device_ordinal) is not int
            or type(self.max_live_versions_per_lineage) is not int
            or not self.provider_id
            or self.device_ordinal < 0
            or self.max_live_versions_per_lineage < 2
        ):
            raise ValueError(
                "CUDA dense-page facts require a device, ID, and two versions"
            )

    def materialize_plan(
        self,
        *,
        resource: str,
        layout: TensorLayoutSpec,
        binding: TensorLayoutBinding,
        max_admitted_lineages: int,
    ) -> StateProviderPlan:
        return _materialize_dense_plan(
            resource=resource,
            provider_id=self.provider_id,
            device_kind=StateDeviceKind.CUDA,
            device_ordinal=self.device_ordinal,
            layout=layout,
            binding=binding,
            max_admitted_lineages=max_admitted_lineages,
            max_live_versions_per_lineage=self.max_live_versions_per_lineage,
            capabilities=(
                "device/cuda@1",
                "optimizer/adamw@1",
                "state/exact_versions@1",
                "state/fenced_reclaim@1",
                "transition/full_root@1",
            ),
        )


DensePageProviderFacts = DensePageCpuProviderFacts | DensePageCudaProviderFacts


def _materialize_dense_plan(
    *,
    resource: str,
    provider_id: str,
    device_kind: StateDeviceKind,
    device_ordinal: int,
    layout: TensorLayoutSpec,
    binding: TensorLayoutBinding,
    max_admitted_lineages: int,
    max_live_versions_per_lineage: int,
    capabilities: tuple[str, ...],
) -> StateProviderPlan:
    if (
        type(resource) is not str
        or not resource
        or type(max_admitted_lineages) is not int
        or max_admitted_lineages < 1
    ):
        raise ValueError("dense-page resource capacity is invalid")
    root_bytes = sum(
        prod(field.shape) * _dtype_bytes(field.dtype)
        for field in layout.fields
    )
    return StateProviderPlan(
        resource=resource,
        provider_id=provider_id,
        representation=StateRepresentation.DENSE_PAGE_ARENA,
        device_kind=device_kind,
        device_ordinal=device_ordinal,
        layout_id=binding.layout_id,
        layout_namespace=binding.layout_namespace,
        layout_digest=binding.layout_digest,
        max_admitted_lineages=max_admitted_lineages,
        max_live_versions_per_lineage=max_live_versions_per_lineage,
        root_page_count=1,
        reserved_bytes=(
            root_bytes
            * max_admitted_lineages
            * max_live_versions_per_lineage
        ),
        capture_stable=False,
        capabilities=capabilities,
    )


def _dtype_bytes(dtype: TensorDType) -> int:
    match dtype:
        case TensorDType.BOOL | TensorDType.UINT8:
            return 1
        case TensorDType.FLOAT16 | TensorDType.BFLOAT16:
            return 2
        case TensorDType.INT32 | TensorDType.FLOAT32:
            return 4
        case TensorDType.INT64 | TensorDType.FLOAT64:
            return 8


__all__ = [
    "DensePageCpuProviderFacts",
    "DensePageCudaProviderFacts",
    "DensePageProviderFacts",
]
