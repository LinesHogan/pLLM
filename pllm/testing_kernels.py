"""Replay authored math from the CPU schedule on CUDA for testing.check_on_gpu.

The CPU run supplies the inputs, including state and index tensors. Comparing complete
method and cell outputs keeps device rounding in framework bookkeeping out of the oracle.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial

import torch

from pllm.contracts.semantic_graph import TaskSymbol
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.model_sites import DepthInput
from pllm.execution.module_graph import ModuleCall
from pllm.execution.task_memory import SiteArgument, seeded_site_arguments
from pllm.execution.tensor_declaration import TensorDeclaration
from pllm.execution.tensor_frame import TensorMath
from pllm.execution.tensor_task import TensorTask
from pllm.execution.token_program import Cell, TokenProgram


def compare_device(
    name: str,
    function: TensorMath,
    args: tuple[torch.Tensor, ...],
    kwargs: Mapping[str, SiteArgument],
    expected: torch.Tensor | tuple[torch.Tensor, ...],
) -> None:
    """Replay one call on the same seeded inputs and check dtype-default tolerances."""
    gpu_args = tuple(value.detach().to("cuda") for value in args)
    gpu_kwargs = {
        key: value.to(torch.device("cuda")) if isinstance(value, (torch.Tensor, ModuleCall)) else value
        for key, value in kwargs.items()
    }
    mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        actual = function(*gpu_args, **gpu_kwargs)
    finally:
        torch.cuda.set_sync_debug_mode(mode)
    try:
        torch.testing.assert_close(actual, expected, check_device=False)
    except AssertionError as error:
        raise AssertionError(f"[device equivalence] {name}: {error}") from error


@dataclass(frozen=True)
class ReplayCell:
    """A CPU cell with its CUDA replay, used only by the fixed check script."""

    name: str
    function: Callable[..., tuple[torch.Tensor, ...]]
    cuda: Callable[..., tuple[torch.Tensor, ...]]

    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        result = self.function(*args)
        compare_device(self.name, self.cuda, args, {}, result)
        return result


def replay_cells(cpu: dict[str, TokenProgram], cuda: dict[str, TokenProgram]) -> dict[str, TokenProgram]:
    """Pair separately traced cells so factory devices follow the actual method inputs."""
    programs: dict[str, TokenProgram] = {}
    for role, program in cpu.items():
        steps = list(program.steps)
        for index, (step, gpu_step) in enumerate(zip(program.steps, cuda[role].steps, strict=True)):
            if isinstance(step, Cell):
                assert isinstance(gpu_step, Cell)
                steps[index] = replace(step, function=ReplayCell(role, step.function, gpu_step.function))
        programs[role] = replace(program, steps=tuple(steps))
    return programs


def _method_result(task: TensorTask, **inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
    result = task.call(inputs)
    if not task.policy.forward_rows or task.training is not None:
        return result
    mask = inputs[task.policy.forward_rows]
    if isinstance(result, torch.Tensor):
        return task.selected_rows(result, mask)
    return tuple(task.selected_rows(value, mask) for value in result)


def compare_method(
    task: TensorTask,
    inputs: Mapping[str, torch.Tensor],
    expected: torch.Tensor | tuple[torch.Tensor, ...],
) -> None:
    """Replay the same argument adaptation and row selection that produced the CPU output."""
    compare_device(str(task.symbol), partial(_method_result, task), (), inputs, expected)


def check_sites(
    binding: TensorDeclaration,
    facts: BackendCompileFacts,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    device_equivalence: bool,
    results: Mapping[tuple[TaskSymbol, str], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run stateless residual math at each declared place on the check script's token rows."""
    kwargs = seeded_site_arguments(binding, facts, seed, results)
    outputs: dict[str, torch.Tensor] = {}
    for rewrite in binding.sites.rewrites:
        if rewrite.memory is not None:
            continue
        stream = x if rewrite.streams == 1 else x.unsqueeze(1).expand(-1, rewrite.streams, -1)
        for place in rewrite.places:
            for name, function, args, sources in (
                (rewrite.name + ".read", rewrite.read, (stream,), rewrite.read_inputs),
                (
                    rewrite.name,
                    rewrite.write,
                    (stream,) if rewrite.site in ("value", "final") else (stream, y),
                    rewrite.inputs,
                ),
            ):
                if function is None:
                    continue
                arguments = dict(kwargs[name])
                for key, source in sources:
                    if isinstance(source, DepthInput):
                        arguments[key] = rewrite.depth(place)
                result = function(*args, **arguments)
                if device_equivalence:
                    compare_device(name, function, args, arguments, result)
                outputs[f"{name}:{place}"] = result
    return outputs
