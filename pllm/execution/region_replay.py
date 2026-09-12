"""Lower conditional replay's recurrence and stopping rule into a region.

Backend model/KV operations and tracing are invocation-bound external endpoints.
The source contains the loop and the first-iteration/convergence branches.
"""

from collections.abc import Callable

import torch

from pllm.contracts.region_program import (
    Block, Call, Constant, ExternalResource, Function, If, Program, Region,
    Scalar, ScalarKind, ScalarType, TensorType, Value, While,
)
from pllm.contracts.semantic_graph import ConditionalLayerReplay
from pllm.execution.codec import boundary_digest
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import NativePrimitive, PrimitiveResources, external_primitive
from pllm.execution.region_resources import ExternalFunction, RegionPageOwner, RegionResources
from pllm.execution.region_values import Data


def _region(args: tuple[Value, ...], ops: tuple[Constant | Call | Scalar | If | While, ...],
            results: tuple[str, ...]) -> Region:
    return Region(Block(args, ops, results))


def replay_region_workspace_bytes(hidden: int, width: int) -> int:
    """Additional tensor copies over the existing model/convergence workspace.

    One row runs at a time. The entry/loop carry, current result and private
    arguments/results of native calls coexist; eight vectors conservatively
    cover those references without charging once per row or iteration.
    """
    return 8 * width * hidden


class ConditionalReplayRegion:
    def __init__(self, tensor: TensorType, effect: ConditionalLayerReplay) -> None:
        integer, boolean, floating = ScalarType.INT, ScalarType.BOOL, ScalarType.FLOAT
        identity = boundary_digest(tensor, effect)
        self.step = Function("replay.step@1:" + identity, (tensor,), (tensor,), ("model",))
        self.observe = Function("replay.observe@1:" + identity, (tensor, integer), (), ("trace",))
        self.record = Function("replay.record@1:" + identity, (integer, floating, boolean), (), ("trace",))
        relative = Function("replay.relative_l2@1:" + identity, (tensor, tensor), (floating,))
        proceed = Function("replay.proceed@1:" + identity, (floating,), (boolean,))
        finite = Function("replay.finite@1:" + identity, (tensor,), ())
        def distance(inputs: tuple[Data, ...], resources: PrimitiveResources) -> tuple[Data, ...]:
            current, previous = inputs
            if not isinstance(current, torch.Tensor) or not isinstance(previous, torch.Tensor):
                raise ValueError("replay convergence requires tensor operands")
            value = float(((current.float() - previous.float()).square().sum().sqrt()
                / previous.float().square().sum().sqrt().clamp_min(1e-12)).item())
            return (value,)
        def again(inputs: tuple[Data, ...], resources: PrimitiveResources) -> tuple[Data, ...]:
            value, = inputs
            if type(value) is not float:
                raise ValueError("replay relative distance must be a host float")
            # Preserve NaN comparison semantics: the original loop stops only
            # when <= is true; !(threshold < value) would not be equivalent.
            return (not value <= effect.relative_l2_ppm / 1_000_000,)
        def check(inputs: tuple[Data, ...], resources: PrimitiveResources) -> tuple[Data, ...]:
            value, = inputs
            if not isinstance(value, torch.Tensor):
                raise ValueError("replay finite check requires a tensor")
            if not torch.isfinite(value).all().item():
                raise RuntimeError("conditional replay produced nonfinite carry")
            return ()
        args = (Value("carry", tensor), Value("iteration", integer), Value("again", boolean))
        condition = _region(args, (), ("again",))
        first = _region((Value("current", tensor), Value("previous", tensor)), (
            Constant((), (Value("relative", floating),), 0.),
            Constant((), (Value("proceed", boolean),), True),
        ), ("relative", "proceed"))
        later = _region(first.block.arguments, (
            Call(("current", "previous"), (Value("relative", floating),), relative.name),
            Call(("relative",), (Value("proceed", boolean),), proceed.name),
        ), ("relative", "proceed"))
        body = _region(args, (
            Call(("carry",), (Value("current", tensor),), self.step.name),
            Call(("current", "iteration"), (), self.observe.name),
            Constant((), (Value("zero", integer),), 0),
            Scalar(("iteration", "zero"), (Value("first", boolean),), ScalarKind.EQUAL),
            If(("first", "current", "carry"), (Value("relative", floating), Value("proceed", boolean)), first, later),
            Call(("current",), (), finite.name),
            Call(("iteration", "relative", "first"), (), self.record.name),
            Constant((), (Value("one", integer),), 1),
            Scalar(("iteration", "one"), (Value("next_iteration", integer),), ScalarKind.ADD),
        ), ("current", "next_iteration", "proceed"))
        outputs = tuple(Value("final_" + v.name, v.type) for v in args)
        self.program = Program(_region((args[0],), (
            Constant((), (args[1],), 0), Constant((), (args[2],), True),
            While(tuple(v.name for v in args), outputs, effect.max_iterations, condition, body),
        ), tuple(v.name for v in outputs)),
            (self.step, self.observe, self.record, relative, proceed, finite),
            (ExternalResource("model"), ExternalResource("trace")))
        cells = {f.name: external_primitive(f) for f in (self.step, self.observe, self.record)}
        cells.update({relative.name: NativePrimitive(relative, distance), proceed.name: NativePrimitive(proceed, again),
                      finite.name: NativePrimitive(finite, check)})
        schedule = analyze_region_program(self.program, optimize=False).reference_schedule
        self.execution = RegionExecution(self.program, lower_region_schedule(self.program, schedule),
            lower_region_bindings(self.program, schedule), cells)

    def run(self, owner: RegionPageOwner, carry: torch.Tensor, *,
            step: Callable[[torch.Tensor], torch.Tensor], observe: Callable[[torch.Tensor, int], None],
            record: Callable[[int, float | None], None]) -> None:
        def model(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            value, = inputs
            if not isinstance(value, torch.Tensor):
                raise ValueError("replay requires a tensor carry")
            return (step(value).detach(),)
        def observed(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            value, iteration = inputs
            if not isinstance(value, torch.Tensor) or type(iteration) is not int:
                raise ValueError("replay observation has invalid operands")
            observe(value, iteration)
            return ()
        def recorded(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            iteration, relative, first = inputs
            if type(iteration) is not int or type(relative) is not float or type(first) is not bool:
                raise ValueError("replay trace has invalid operands")
            record(iteration, None if first else relative)
            return ()
        resources = RegionResources(self.program, owner, (), (
            ExternalFunction(self.step, model), ExternalFunction(self.observe, observed), ExternalFunction(self.record, recorded)))
        self.execution.run({"carry": carry}, pages=resources)
