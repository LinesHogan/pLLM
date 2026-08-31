"""Cold physical facts for the CPU dense-page provider."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from tllm.contracts.state_representation import (
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
        if (
            type(resource) is not str
            or not resource
            or type(max_admitted_lineages) is not int
            or max_admitted_lineages < 1
        ):
            raise ValueError("CPU dense-page resource capacity is invalid")
        root_bytes = sum(
            prod(field.shape) * _dtype_bytes(field.dtype)
            for field in layout.fields
        )
        return StateProviderPlan(
            resource=resource,
            provider_id=self.provider_id,
            representation=StateRepresentation.DENSE_PAGE_ARENA,
            layout_id=binding.layout_id,
            layout_namespace=binding.layout_namespace,
            layout_digest=binding.layout_digest,
            max_admitted_lineages=max_admitted_lineages,
            max_live_versions_per_lineage=self.max_live_versions_per_lineage,
            root_page_count=1,
            reserved_bytes=(
                root_bytes
                * max_admitted_lineages
                * self.max_live_versions_per_lineage
            ),
            capture_stable=False,
            capabilities=("optimizer/adamw@1", "state/exact_versions@1"),
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


__all__ = ["DensePageCpuProviderFacts"]
