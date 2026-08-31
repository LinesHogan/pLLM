#!/usr/bin/env python3
"""GPU smoke for fixed-root forward, COW AdamW, and view completion."""

from __future__ import annotations

import json

import torch

from tllm.contracts.request_identity import (
    RequestFinishedBatch,
    RequestIdentity,
    RequestKeyField,
)
from tllm.contracts.request_state import (
    RequestIdentityBatch,
    StateEffectOperation,
)
from tllm.runtime.fixed_batched_slot_adapter import (
    FixedBatchedSlotLayout,
    FixedBatchedSlotModule,
)
from tllm.runtime.paged_state import TensorFieldId
from tllm.runtime.request_trainable_state import (
    RequestStateRuntimeService,
    RequestTrainableStateStore,
)
from tllm.runtime.stable_packed_state import (
    FrozenPackedTensorField,
    FrozenPackedTensorLayout,
    PackedLayoutId,
)


class _CudaScalarBank(FixedBatchedSlotModule):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.full((1, 1), 2.0, device=device)
        )
        self.bias: torch.Tensor
        self.register_buffer(
            "bias",
            torch.full((1, 1), 0.5, device=device),
        )

    def fixed_batched_slot_layout(self) -> FixedBatchedSlotLayout:
        return FixedBatchedSlotLayout(
            int(self.weight.shape[0]),
            (self.weight,),
            (self.bias,),
        )

    def install_fixed_batched_slot_layout(
        self,
        layout: FixedBatchedSlotLayout,
    ) -> None:
        match layout.slot_parameters, layout.slot_buffers:
            case (weight,), (bias,):
                self.weight = weight
                self.bias = bias
            case _:
                raise ValueError("CUDA scalar bank received the wrong layout")

    def forward(
        self,
        page_ids: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        return source * self.weight.index_select(0, page_ids) + self.bias.index_select(
            0,
            page_ids,
        )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("fixed batched paged-state smoke requires CUDA")
    device = torch.device("cuda:0")
    store = RequestTrainableStateStore[_CudaScalarBank].fixed_batched(
        resource="state",
        key_fields=(RequestKeyField.REQUEST_ID,),
        epoch=1,
        bank=_CudaScalarBank(device),
        lr=0.05,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=0.0,
    )
    service = RequestStateRuntimeService(store)
    views = service.stable_view_service(
        view_epoch=1,
        layout=FrozenPackedTensorLayout(
            PackedLayoutId(1),
            row_capacity=1,
            lease_capacity=1,
            fields=(
                FrozenPackedTensorField(
                    TensorFieldId(0),
                    (1,),
                    torch.float32,
                ),
            ),
        ),
        buffer_count=1,
    )
    identity = RequestIdentity("gpu", 0, 0)
    identities = RequestIdentityBatch((identity,))
    stream = torch.cuda.Stream(device=device)
    source = torch.full((1, 1), 3.0, device=device)

    service.stage_identities(0, identities)
    first_selection = service.selection_for_step(0, stream)
    read_audit = service.before(
        StateEffectOperation.READ,
        first_selection,
        0,
    )
    first_view = views.create_cuda(
        0,
        first_selection,
        (source,),
        stream,
    )
    views.order_cuda(first_view, stream)
    with torch.cuda.stream(stream), torch.inference_mode(False), torch.enable_grad():
        first_page_ids = first_view.version_epoch.page_ids[:1, 0]
        weight_pool = first_view.version_epoch.state_operands[0]
        bias_pool = first_view.version_epoch.state_operands[1]
        prediction = (
            first_view.packed_tensors.tensors[0][:1]
            * weight_pool.index_select(0, first_page_ids)
            + bias_pool.index_select(0, first_page_ids)
        )
        loss = prediction.square().mean()
    service.after(read_audit)
    transition_audit = service.before(
        StateEffectOperation.TRANSITION,
        first_selection,
        0,
    )
    with torch.cuda.stream(stream):
        service.optimize(first_selection, (loss,))
    service.after(transition_audit)
    views.complete_cuda(first_view, stream)
    service.finish_step(0)
    stream.synchronize()
    views.collect_completed()

    old_page = int(first_page_ids[0].item())
    old_weight = float(weight_pool[old_page].item())
    if old_weight != 2.0:
        raise AssertionError("version-zero parameter page was mutated in place")

    service.stage_identities(1, identities)
    second_selection = service.selection_for_step(1, stream)
    second_read_audit = service.before(
        StateEffectOperation.READ,
        second_selection,
        1,
    )
    second_view = views.create_cuda(
        1,
        second_selection,
        (source,),
        stream,
    )
    views.order_cuda(second_view, stream)
    with torch.cuda.stream(stream):
        new_page_tensor = second_view.version_epoch.page_ids[:1, 0]
        new_weight_tensor = second_view.version_epoch.state_operands[0].index_select(
            0,
            new_page_tensor,
        )
    views.complete_cuda(second_view, stream)
    service.after(second_read_audit)
    service.finish_step(1)
    stream.synchronize()

    new_page = int(new_page_tensor[0].item())
    new_weight = float(new_weight_tensor[0, 0].item())
    if new_page == old_page or new_weight == old_weight:
        raise AssertionError("COW optimizer did not publish a distinct updated root")
    (published_version,) = service.versions()
    service.retire_requests(RequestFinishedBatch((identity,)))

    print(
        json.dumps(
            {
                "new_page": new_page,
                "new_weight": new_weight,
                "old_page": old_page,
                "old_weight": old_weight,
                "version": published_version.version,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
