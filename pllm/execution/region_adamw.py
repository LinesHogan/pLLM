"""Import the existing AdamW kernel without replacing its numerical path."""

from collections.abc import Sequence
from math import prod

import torch

from pllm.contracts.region_program import Block, Call, Function, Program, Region, TensorType, Value
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.codec import boundary_digest
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import NativePrimitive, PrimitiveResources
from pllm.execution.region_storage import analyze_region_capacity
from pllm.execution.region_values import Data, check_data
from pllm.execution.tensor_program import FieldSlots
from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw


def adamw_cell(parameter: TensorType, config: AdamWConfig) -> NativePrimitive:
    """Five explicit state fields plus a gradient, returning all five new fields.

    Snapshot/publication stay in the containing region. CPU uses the original
    PyTorch scalar path and CUDA the original capturable foreach path.
    """
    if parameter.dtype not in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"):
        raise ValueError("AdamW import requires floating parameters")
    state = (parameter, parameter, TensorType((), "torch.float32", parameter.device), parameter, parameter)
    declaration = Function("native.adamw@1:" + boundary_digest(parameter, config), (*state, parameter), state)
    fields = AdamWFields(*(LayoutFieldId(i) for i in range(5)))

    def update(inputs: tuple[Data, ...], resources: PrimitiveResources) -> tuple[Data, ...]:
        tensors: list[torch.Tensor] = []
        for value in inputs:
            if not isinstance(value, torch.Tensor):
                raise ValueError("AdamW import requires tensor operands")
            tensors.append(value)
        root = tuple(tensors[:5])
        apply_adamw((root,), fields, config, (tensors[5],))
        return root

    return NativePrimitive(declaration, update)


class AdamWRegion:
    """Cold-lowered pure update; writeback borrows the caller's private fields.

    The source program can be composed before scheduling. This adapter currently
    bridges one fixed optimizer batch into the original transition/publication
    lifecycle; it neither allocates pages nor publishes them.
    """

    def __init__(self, parameters: tuple[TensorType, ...], config: AdamWConfig) -> None:
        cell = adamw_batch_cell(parameters, config)
        signature = cell.declaration
        inputs = tuple(Value(f"input{i}", type_) for i, type_ in enumerate(signature.inputs))
        outputs = tuple(Value(f"output{i}", type_) for i, type_ in enumerate(signature.outputs))
        self.program = Program(
            Region(
                Block(
                    inputs,
                    (Call(tuple(v.name for v in inputs), outputs, signature.name),),
                    tuple(v.name for v in outputs),
                )
            ),
            (signature,),
        )
        schedule = analyze_region_program(self.program, optimize=False).reference_schedule
        self._execution = RegionExecution(
            self.program,
            lower_region_schedule(self.program, schedule),
            lower_region_bindings(self.program, schedule),
            {signature.name: cell},
        )
        self._names = tuple(v.name for v in inputs)
        self._parameters = parameters
        self.groups = len(parameters)
        self.capacity = analyze_region_capacity(self.program, schedule, {signature.name: cell})

        # The native wrapper's copies are source-derived. This adapter also
        # allocates one gradient slot per parameter inside adamw_batch_cell.
        # Original kernel/loss workspace and private pages remain separately
        # reserved by the owner; capacity marks native internals as unknown.
        def bytes_(type_: TensorType) -> int:
            widths = {"torch.float16": 2, "torch.bfloat16": 2, "torch.float32": 4, "torch.float64": 8}
            return prod(type_.shape) * widths[type_.dtype]

        self.copy_workspace_bytes = sum(device.peak_tensor_bytes_upper_bound for device in self.capacity.devices) + sum(
            bytes_(parameter) for parameter in parameters
        )

    def __call__(
        self, roots: tuple[tuple[torch.Tensor, ...], ...], gradients: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        if len(roots) != len(self._parameters) or len(gradients) != len(roots) or any(len(root) != 5 for root in roots):
            raise ValueError("AdamW region requires one five-field root and gradient per parameter")
        for value, type_ in zip(
            (value for root in roots for value in root), self.program.functions[0].outputs, strict=True
        ):
            check_data(value, type_)
        arguments = tuple(
            value
            for root, gradient in zip(roots, gradients, strict=True)
            for value in (root[0], root[2], root[3], root[4], gradient)
        )
        result = self._execution.run(dict(zip(self._names, arguments, strict=True))).values
        tensors: list[torch.Tensor] = []
        for value in result:
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("AdamW region returned a non-tensor")
            tensors.append(value)
        return tuple(tuple(tensors[i : i + 5]) for i in range(0, len(tensors), 5))

    def update(
        self, roots: Sequence[tuple[torch.Tensor, ...]], slots: FieldSlots, gradients: tuple[torch.Tensor, ...]
    ) -> None:
        """Copy one instance's mathematical results into its already-issued write."""
        if len(roots) != 1 or len(roots[0]) != 5 * len(self._parameters):
            raise ValueError("instance AdamW writeback requires its exact signed optimizer fields")
        (root,) = roots
        groups = tuple(root[i : i + 5] for i in range(0, len(root), 5))
        updated = self(groups, gradients)
        with torch.no_grad():
            for target, source in zip(root, (value for group in updated for value in group), strict=True):
                target.copy_(source)


def adamw_batch_cell(parameters: tuple[TensorType, ...], config: AdamWConfig) -> NativePrimitive:
    """One original foreach batch, with parameter, step, moments and new gradient.

    The old gradient slot is scratch, so it is not a mathematical input. Each
    parameter contributes five outputs in the original PageBundle field order.
    Parameters may differ in shape and dtype; their original batch order remains.
    """
    if not parameters or len({p.device for p in parameters}) != 1:
        raise ValueError("AdamW batch requires parameters on one device")
    states = tuple(adamw_cell(parameter, config).declaration.outputs for parameter in parameters)
    inputs = tuple(
        type_
        for parameter, state in zip(parameters, states, strict=True)
        for type_ in (state[0], state[2], state[3], state[4], parameter)
    )
    outputs = tuple(type_ for state in states for type_ in state)
    declaration = Function("native.adamw_batch@1:" + boundary_digest(parameters, config), inputs, outputs)
    fields = AdamWFields(*(LayoutFieldId(i) for i in range(5)))

    def update(inputs: tuple[Data, ...], resources: PrimitiveResources) -> tuple[Data, ...]:
        tensors: list[torch.Tensor] = []
        for value in inputs:
            if not isinstance(value, torch.Tensor):
                raise ValueError("AdamW batch requires tensor operands")
            tensors.append(value)
        roots = tuple(
            (tensors[i], torch.empty_like(tensors[i]), tensors[i + 1], tensors[i + 2], tensors[i + 3])
            for i in range(0, len(tensors), 5)
        )
        gradients = tuple(tensors[i + 4] for i in range(0, len(tensors), 5))
        apply_adamw(roots, fields, config, gradients)
        return tuple(tensor for root in roots for tensor in root)

    return NativePrimitive(declaration, update)
