"""Cold physical capacity facts for the shared CPU/CUDA PageBundle arena."""

from dataclasses import dataclass
from math import prod

from tllm.contracts.state_representation import StateDeviceKind, StateProviderPlan, StateRepresentation
from tllm.contracts.tensor_layout import TensorDType, TensorLayoutBinding, TensorLayoutSpec


@dataclass(frozen=True, slots=True)
class PageBundleProviderFacts:
    provider_id: str
    max_live_versions_per_lineage: int
    device_kind: StateDeviceKind = StateDeviceKind.CPU
    device_ordinal: int = 0

    def materialize_plan(self, *, resource: str, layout: TensorLayoutSpec,
                         binding: TensorLayoutBinding, max_admitted_lineages: int) -> StateProviderPlan:
        sizes = {
            TensorDType.BOOL: 1, TensorDType.UINT8: 1, TensorDType.INT32: 4, TensorDType.INT64: 8,
            TensorDType.FLOAT16: 2, TensorDType.BFLOAT16: 2, TensorDType.FLOAT32: 4, TensorDType.FLOAT64: 8,
        }
        root_bytes = sum(prod(field.shape) * sizes[field.dtype] for field in layout.fields)
        return StateProviderPlan(
            resource=resource, provider_id=self.provider_id,
            representation=StateRepresentation.DENSE_PAGE_ARENA,
            device_kind=self.device_kind, device_ordinal=self.device_ordinal,
            layout_id=binding.layout_id, layout_namespace=binding.layout_namespace, layout_digest=binding.layout_digest,
            max_admitted_lineages=max_admitted_lineages,
            max_live_versions_per_lineage=self.max_live_versions_per_lineage,
            root_page_count=len(layout.fields),
            reserved_bytes=root_bytes * max_admitted_lineages * self.max_live_versions_per_lineage,
            capture_stable=False,
            capabilities=tuple(sorted((
                f"device/{self.device_kind.value}@1", "optimizer/adamw@1", "state/exact_versions@1",
                "transition/full_root@1", *(("state/fenced_reclaim@1",) if self.device_kind is StateDeviceKind.CUDA else ()),
            ))),
        )
