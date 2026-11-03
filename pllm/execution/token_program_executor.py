"""Execute a traced token program's steps on a batch of independent token positions."""

from collections.abc import Callable, Mapping

import torch

from pllm.execution.model_sites import StateInput
from pllm.execution.token_program import KV, ApplyUpdates, Cell, Save, Segment, TokenProgram

Layer = Callable[[int, torch.Tensor, torch.Tensor | None, KV], tuple[torch.Tensor, torch.Tensor]]
# The residual stream after a layer from its (fresh output, residual): their sum, unless a custom residual owns it.
Stream = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]
Adaptive = Callable[[Segment, torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]


def add(layer: int, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return hidden + residual


def _row_value(
    program: TokenProgram, values: Mapping[str, torch.Tensor], name: str, seed: torch.Tensor
) -> torch.Tensor:
    """A field used directly as a stream supplies the same stream to every row."""
    value = values[name]
    return value.expand_as(seed) if any(name == parameter for parameter, _ in program.parameters) else value


def execute_token_program(
    program: TokenProgram,
    seed: torch.Tensor,
    positions: torch.Tensor,
    generated: torch.Tensor,
    entries: tuple[torch.Tensor, ...],
    layer: Layer,
    apply_updates: Callable[[], None],
    stream: Stream = add,
    adaptive: Adaptive | None = None,
    *,
    parameters: Mapping[StateInput, torch.Tensor] = {},
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Preserve segment residuals and snapshot taps before subsequent in-place layer calls."""
    values = {program.seed: seed, program.position: positions.reshape(-1, 1), program.generated: generated}
    values.update((name, value) for (_, name), value in zip(program.entries, entries, strict=True))
    values.update((name, parameters[source]) for name, source in program.parameters)
    saved = dict(zip((name for name, _ in program.entries), entries, strict=True))
    output = seed
    for step in program.steps:
        if isinstance(step, Segment):
            hidden = _row_value(program, values, step.input, seed)
            if step.active is None:
                outputs = execute_segment(step, hidden, layer, stream)
            else:
                assert adaptive is not None
                active = _row_value(program, values, step.active, seed[:, :1])
                outputs = adaptive(step, hidden, active)
            values.update(zip((name for _, name in step.taps), outputs, strict=True))
        elif isinstance(step, Cell):
            results = step.function(seed, *(values[name] for name in step.inputs))
            values.update(zip(step.outputs, results, strict=True))
        elif isinstance(step, ApplyUpdates):
            apply_updates()
        elif isinstance(step, Save):
            saved[step.name] = values[step.value]
        else:
            output = _row_value(program, values, step.value, seed)
    return output, tuple(saved.values())


def ordinary_pass(seed: torch.Tensor, count: int, layer: Layer, stream: Stream = add) -> torch.Tensor:
    """Run the model's complete stack while retaining its pre-norm residual stream."""
    hidden, residual = seed.clone(), None
    for ordinal in range(count):
        hidden, residual = layer(ordinal, hidden, residual, "commit")
    assert residual is not None
    return stream(count - 1, hidden, residual)


def execute_segment(step: Segment, seed: torch.Tensor, layer: Layer, stream: Stream = add) -> tuple[torch.Tensor, ...]:
    """Run an intact slice, retaining fused residual additions between layers."""
    hidden, residual = seed.clone(), None
    taps = dict(step.taps)
    outputs: dict[int, torch.Tensor] = {}
    for ordinal in range(step.start, step.stop):
        hidden, residual = layer(ordinal, hidden, residual, step.kv)
        if ordinal in taps:
            outputs[ordinal] = stream(ordinal, hidden, residual)
    return tuple(outputs[ordinal] for ordinal, _ in step.taps)
