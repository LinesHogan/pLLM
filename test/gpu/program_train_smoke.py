#!/usr/bin/env python3
"""GPU smoke: Program consumer training + stage capture/replay.

Compiles a tiny distiller Program, runs consume_flow on CUDA bundles
(update counter and loss must be nonzero), then captures one stage-sliced
flow execution into a raw CUDA graph and replays it with new data,
asserting the replayed outputs match the eager reference.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


class TinyDistiller(torch.nn.Module):
    def __init__(self, lr: float = 0.1) -> None:
        super().__init__()
        self.model = torch.nn.Linear(8, 8)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    from tllm.workgraph.frontend import extern, flow, hidden, local, result, sink
    from tllm.ports.base import TimingDeadline

    @flow(
        inputs={
            "source": hidden(layer=0, site="block_output", phase="decode"),
            "target": hidden(layer=-1, site="block_output", phase="decode"),
        },
        outputs={"prediction": sink("guidance"), "loss": local()},
        state="distiller",
    )
    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.model(source)
        return prediction, F.mse_loss(prediction.float(), target.detach().float())

    @extern(
        inputs={"loss": result("forward", "loss")},
        after=("forward",),
        state="distiller",
        deadline=TimingDeadline.NEXT_PREPARE,
    )
    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0

    from tllm.contracts.gpu_stage import DeviceTensorLease
    from tllm.contracts.port_bundle import BundleKey, PortBundle
    from tllm.contracts.request_meta_view import RowBatchMeta
    from tllm.program import StateSpec, compile_program
    from tllm.workgraph.frontend import StateScope

    class NullSink:
        @property
        def name(self) -> str:
            return "guidance"

        def publish(self, value: torch.Tensor, meta: object) -> None:
            del meta
            assert value.is_cuda

    device = torch.device("cuda")
    torch.manual_seed(0)
    instance = TinyDistiller().to(device)
    examples = {
        "forward": (torch.randn(4, 8, device=device), torch.randn(4, 8, device=device))
    }
    consumer = compile_program(
        instance,
        states={"distiller": StateSpec(scope=StateScope.REPLICA, optimizer=torch.optim.SGD)},
        examples=examples,
        sinks=(NullSink(),),
    )

    def bundle(source: torch.Tensor, target: torch.Tensor, step_id: int) -> PortBundle:
        rows = int(source.shape[0])
        return PortBundle(
            key=BundleKey(
                engine_step_id=step_id, phase="decode", request_id="req", sample_idx=0
            ),
            entries={
                "device_lease": DeviceTensorLease(
                    entries={"source": source, "target": target}, active_rows=rows
                ),
                "request_meta": RowBatchMeta(
                    request_ids=tuple(f"req-{i}" for i in range(rows)),
                    prompt_idxs=tuple(0 for _ in range(rows)),
                    sample_idxs=tuple(0 for _ in range(rows)),
                    phase="decode",
                    engine_step_id=step_id,
                ),
            },
        )

    for step in (1, 2):
        consumer.consume_flow(
            0,
            bundle(
                torch.randn(4, 8, device=device),
                torch.randn(4, 8, device=device),
                step,
            ),
            None,
        )
    assert consumer.update_count == 2, consumer.update_count
    assert consumer.loss_count == 2, consumer.loss_count
    assert consumer.loss_sum != 0.0, consumer.loss_sum
    slot = consumer.bank("distiller").slot_for(None)
    assert slot.version == 2, slot.version

    # CUDA graph capture/replay of one sliced flow execution.
    plan_executor = None
    for _ordinal, (_node, _plan, executor) in consumer._flows.items():
        plan_executor = executor
    assert plan_executor is not None
    static_source = torch.randn(4, 8, device=device)
    static_target = torch.randn(4, 8, device=device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(2):
            plan_executor.run(static_source, static_target)
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = plan_executor.run(static_source, static_target)
    static_source.copy_(torch.randn(4, 8, device=device))
    static_target.copy_(torch.randn(4, 8, device=device))
    graph.replay()
    ref_prediction, ref_loss = instance.forward(static_source, static_target)
    assert torch.allclose(static_out["prediction"], ref_prediction)
    assert torch.allclose(static_out["loss"], ref_loss)

    print("SMOKE_OK program_train_gpu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
