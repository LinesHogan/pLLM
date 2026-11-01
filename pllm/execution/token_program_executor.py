"""Execute a traced token program's steps on a batch of independent token positions."""

from collections.abc import Callable

import torch

from pllm.execution.token_program import KV, ApplyUpdates, Cell, Save, Segment, TokenProgram

Layer = Callable[[int, torch.Tensor, torch.Tensor | None, KV], tuple[torch.Tensor, torch.Tensor]]


def execute_token_program(
    program: TokenProgram,
    seed: torch.Tensor,
    positions: torch.Tensor,
    generated: torch.Tensor,
    entries: tuple[torch.Tensor, ...],
    layer: Layer,
    apply_updates: Callable[[], None],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Preserve segment residuals and snapshot taps before subsequent in-place layer calls."""
    values = {program.seed: seed, program.position: positions.reshape(-1, 1), program.generated: generated}
    values.update((name, value) for (_, name), value in zip(program.entries, entries, strict=True))
    saved = dict(zip((name for name, _ in program.entries), entries, strict=True))
    output = seed
    for step in program.steps:
        if isinstance(step, Segment):
            hidden, residual = values[step.input].clone(), None
            taps = dict(step.taps)
            for ordinal in range(step.start, step.stop):
                hidden, residual = layer(ordinal, hidden, residual, step.kv)
                if ordinal in taps:
                    values[taps[ordinal]] = hidden + residual
        elif isinstance(step, Cell):
            results = step.function(seed, *(values[name] for name in step.inputs))
            values.update(zip(step.outputs, results, strict=True))
        elif isinstance(step, ApplyUpdates):
            apply_updates()
        elif isinstance(step, Save):
            saved[step.name] = values[step.value]
        else:
            output = values[step.value]
    return output, tuple(saved.values())


def ordinary_pass(seed: torch.Tensor, count: int, layer: Layer) -> torch.Tensor:
    """Run the model's complete stack while retaining its pre-norm residual stream."""
    hidden, residual = seed.clone(), None
    for ordinal in range(count):
        hidden, residual = layer(ordinal, hidden, residual, "commit")
    assert residual is not None
    return hidden + residual
