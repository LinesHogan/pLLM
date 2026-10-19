"""Causal chunk mathematics over request-owned, per-target buffers."""

from dataclasses import dataclass
from typing import Annotated

import torch

import pllm
from pllm import ConditionalLayerReplay, ParameterTarget, TensorUpdate
from pllm import authoring as a
from pllm.contracts.semantic_graph import TaskSymbol
from pllm.execution.chunk_transition import ChunkTransition, chunk_memory, chunk_region, tune_chunk_scan
from pllm.execution.shared_tensor import tensor_dtype
from pllm.execution.tensor_binding import OverlayTransition, TensorBinding, TensorOutput
from pllm.execution.tensor_frame import TensorFrame

Input = Annotated[torch.Tensor, pllm.Operand.INPUT]
Target = Annotated[torch.Tensor, pllm.Operand.TARGET]
Weight = Annotated[torch.Tensor, pllm.Operand.WEIGHT]
Delta = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "delta")]
TailInput = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_input")]
TailTarget = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_target")]
TailLength = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_length")]


@dataclass(frozen=True, slots=True)
class InPlaceTTTConfig:
    targets: tuple[ParameterTarget, ...]
    chunk_size: int = 4096
    learning_rate: float = 1e-3
    update_prefill: bool = True
    update_decode: bool = False
    replay: ConditionalLayerReplay | None = None

    def __post_init__(self) -> None:
        if (
            not self.targets
            or tuple(sorted(set(self.targets))) != self.targets
            or self.chunk_size < 1
            or self.learning_rate <= 0.0
        ):
            raise ValueError("In-Place-TTT mathematical configuration is invalid")


@dataclass(frozen=True)
class InPlaceTTTWorkload(pllm.Workload):
    name = "inplace_ttt.dense_fast_weight"
    config: InPlaceTTTConfig

    def declaration(self, facts: a.BackendCompileFacts | None = None) -> TensorBinding:
        b = super().declaration(facts)
        if facts is not None:
            transition = b.operations[TaskSymbol("transition")].function
            assert isinstance(transition, OverlayTransition)
            b.memory_probes[TaskSymbol("transition")] = lambda facts, options: chunk_memory(
                self.config.learning_rate,
                self.config.chunk_size,
                transition.function,
                type(self).transition is _TRANSITION,
                facts,
            )
        return b

    def operations(
        self, facts: a.BackendCompileFacts, plan: a.PlanBundle, b: TensorBinding
    ) -> a.MathBinding[a.TensorStepInput, TensorFrame, TensorOutput]:
        if type(self).transition is not _TRANSITION:
            return super().operations(facts, plan, b)
        # Export is cold authoring work, before any request owns private pages.
        regions = tuple(
            chunk_region(
                self.config.learning_rate,
                self.config.chunk_size,
                target.input_size,
                target.output_size,
                tensor_dtype(facts.probe.tensor_dtype),
                facts.device,
                transposed,
            )
            for target in facts.targets
            for transposed in (False, True)
        )
        transition = ChunkTransition(self.config.chunk_size, regions, facts.probe.max_packed_rows)
        rows = facts.probe.max_packed_rows + facts.max_admitted_lineages * (self.config.chunk_size - 1)
        workspace = next(
            (fact.workspace_bytes for fact in plan.timeline.task_resources if fact.task == "transition"), None
        )
        if workspace is None:
            workspace = b.memory_probes[TaskSymbol("transition")](facts, pllm.ExecutionOptions())[0].workspace
        sources = {
            (cells[0].input_signatures[0].shape, cells[0].input_signatures[1].stride): cells for _, cells in regions
        }
        transition.scans = {
            layout: tune_chunk_scan(scan, sources[layout], rows, workspace, plan.timeline.region_policy)
            for layout, scan in transition.scans.items()
        }
        if any(scan.parallel for scan in transition.scans.values()):
            transition.streams = (torch.cuda.Stream(device=facts.device),)
        b.lower_overlay("transition", transition)
        return super().operations(facts, plan, b)

    @property
    def dense_fast_weights(self) -> pllm.RequestState:
        tail = max(1, self.config.chunk_size - 1)
        return pllm.RequestState(
            {
                "delta": pllm.Buffer(lambda inputs, outputs: (outputs, inputs)),
                "tail_input": pllm.Buffer(lambda inputs, outputs: (tail, inputs)),
                "tail_target": pllm.Buffer(lambda inputs, outputs: (tail, outputs)),
                "tail_length": pllm.Buffer((), dtype=torch.int64),
            },
            namespace="canonical.dense_fast_weight.state@1",
            keys=(pllm.RequestKeyField.REQUEST_ID, pllm.RequestKeyField.PROMPT_IDX, pllm.RequestKeyField.SAMPLE_IDX),
        )

    @pllm.on(pllm.Phase.MODEL_UPDATE)
    @pllm.overlay(
        state="dense_fast_weights",
        targets=pllm.Config("config.targets"),
        weight="delta",
        prefill=pllm.Config("config.update_prefill"),
        decode=pllm.Config("config.update_decode"),
        replay=pllm.Config("config.replay"),
    )
    def transition(
        self,
        hidden: Input,
        target: Target,
        base: Weight,
        delta: Delta,
        tail_input: TailInput,
        tail_target: TailTarget,
        tail_length: TailLength,
    ) -> pllm.TensorUpdate:
        prefix = int(tail_length.item())
        inputs = torch.cat((tail_input[:prefix], hidden), dim=0)
        targets = torch.cat((tail_target[:prefix], target), dim=0)
        chunk = self.config.chunk_size
        trained = (inputs.shape[0] // chunk) * chunk
        outputs: list[torch.Tensor] = []
        # Each chunk emits against its input version before updating the next chunk's weights.
        for start in range(0, trained, chunk):
            x, y = inputs[start : start + chunk], targets[start : start + chunk]
            outputs.append(x @ (base + delta).T)
            delta.add_((self.config.learning_rate * (y.T @ x)).to(delta))
        remaining = inputs.shape[0] - trained
        tail_input.zero_()
        tail_target.zero_()
        tail_input[:remaining].copy_(inputs[trained:])
        tail_target[:remaining].copy_(targets[trained:])
        tail_length.fill_(remaining)
        if remaining:
            outputs.append(inputs[trained:] @ (base + delta).T)
        output = torch.cat(outputs, dim=0)[-hidden.shape[0] :]
        return TensorUpdate(
            output,
            changed=trained > 0,
            update_count=trained // chunk,
            trained_rows=trained,
            metrics={"tail_rows": remaining},
        )


__all__ = ["InPlaceTTTConfig", "InPlaceTTTWorkload"]

_TRANSITION = InPlaceTTTWorkload.transition
