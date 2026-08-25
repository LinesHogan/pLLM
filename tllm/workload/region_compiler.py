"""Process-local callable materialization for signed ExecutionRegions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import torch

from tllm.schedule.contracts import (
    CompileOptions,
    ExecutionRegion,
    RegionAutogradPolicy,
    RegionCompileCommand,
    RegionCompilerAdapter,
    RegionInputBinding,
    RegionImplementationKind,
    RegionOutputBinding,
    RegionTaskInvocation,
    ScheduleDecision,
    ScheduleEffect,
)
from tllm.workload.frontend import WorkloadDeclaration


@dataclass(frozen=True, slots=True)
class RegionCallableArtifact:
    """One signed Region artifact bound to process-local task callables."""

    region_id: str
    artifact_key: str
    input_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    callable: Callable[..., tuple[object, ...]]
    compiled: bool
    compile_command: RegionCompileCommand | None = None
    input_bindings: tuple[RegionInputBinding, ...] = ()
    output_bindings: tuple[RegionOutputBinding, ...] = ()
    effects: tuple[ScheduleEffect, ...] = ()

    def execute(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
        if not isinstance(inputs, Mapping):
            raise TypeError("Region callable inputs must be a mapping")
        supplied = set(inputs)
        expected = set(self.input_keys)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            raise RuntimeError(
                "Region callable input binding mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if self.compile_command is None:
            # Extern Regions have no compiler command and are bound through
            # their signed backend contract. They still execute as a complete
            # singleton Region here.
            return _execute_with_grad(self.callable, self.input_keys, self.output_keys, inputs)
        if self.compile_command.autograd_policy is not RegionAutogradPolicy.ENABLE_GRAD:
            raise RuntimeError("unsupported signed Region autograd policy")
        return _execute_with_grad(self.callable, self.input_keys, self.output_keys, inputs)


def _execute_with_grad(
    callable_artifact: Callable[..., tuple[object, ...]],
    input_keys: tuple[str, ...],
    output_keys: tuple[str, ...],
    inputs: Mapping[str, object],
) -> Mapping[str, object]:
    # The typed command already selected the only currently supported context.
    # This function does not infer a policy from Region tasks or Python state.
    with torch.inference_mode(False):
        with torch.enable_grad():
            result = callable_artifact(*(inputs[key] for key in input_keys))
    if not isinstance(result, tuple) or len(result) != len(output_keys):
        raise RuntimeError("Region callable returned an invalid output tuple")
    return dict(zip(output_keys, result, strict=True))


class RegionArtifactRegistry:
    """Exact process-local artifact binding for one signed Region decision."""

    __slots__ = ("_artifacts",)

    def __init__(
        self,
        decision: ScheduleDecision,
        artifacts: tuple[RegionCallableArtifact, ...],
    ) -> None:
        if not isinstance(decision, ScheduleDecision) or decision.schema_version != 2:
            raise TypeError("decision must be a ScheduleDecision schema v2")
        if not isinstance(artifacts, tuple) or any(
            not isinstance(artifact, RegionCallableArtifact) for artifact in artifacts
        ):
            raise TypeError(
                "artifacts must be tuple[RegionCallableArtifact, ...]"
            )
        by_region: dict[str, RegionCallableArtifact] = {}
        for artifact in artifacts:
            if artifact.region_id in by_region:
                raise RuntimeError(
                    f"duplicate local Region artifact {artifact.region_id!r}"
                )
            by_region[artifact.region_id] = artifact
        expected = {region.region_id: region for region in decision.execution_regions}
        missing = sorted(set(expected) - set(by_region))
        unexpected = sorted(set(by_region) - set(expected))
        if missing or unexpected:
            raise RuntimeError(
                "local Region artifact coverage mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        mismatched = sorted(
            region_id
            for region_id, region in expected.items()
            if by_region[region_id].artifact_key != region.artifact_key
        )
        if mismatched:
            raise RuntimeError(
                "local Region artifact key mismatch: " + ", ".join(mismatched)
            )
        self._artifacts = MappingProxyType(by_region)

    def resolve(self, region_id: str) -> RegionCallableArtifact:
        try:
            return self._artifacts[region_id]
        except KeyError as exc:
            raise RuntimeError(
                f"no local callable artifact for ExecutionRegion {region_id!r}"
            ) from exc

    @property
    def artifacts(self) -> tuple[RegionCallableArtifact, ...]:
        return tuple(self._artifacts.values())


@dataclass(frozen=True, slots=True)
class _TaskBinding:
    task_id: str
    executor: Callable[..., object]


def _task_bindings(
    declarations: tuple[WorkloadDeclaration, ...],
) -> dict[str, _TaskBinding]:
    if not declarations or any(
        not isinstance(declaration, WorkloadDeclaration)
        for declaration in declarations
    ):
        raise TypeError(
            "declarations must be a non-empty tuple[WorkloadDeclaration, ...]"
        )
    multiple = len(declarations) > 1
    bindings: dict[str, _TaskBinding] = {}
    for declaration in declarations:
        workload = declaration.ir

        def qualify(name: str, *, workload_id: str = workload.workload_id) -> str:
            return f"{workload_id}/{name}" if multiple else name

        for task in workload.tasks:
            task_id = qualify(task.name)
            if task_id in bindings:
                raise ValueError(f"duplicate Region task binding {task_id!r}")
            bindings[task_id] = _TaskBinding(
                task_id=task_id,
                executor=declaration.executor(task.name),
            )
    return bindings


def _normalize_task_outputs(
    invocation: RegionTaskInvocation,
    result: object,
) -> tuple[object, ...]:
    count = len(invocation.outputs)
    if count == 0:
        if result is not None:
            raise RuntimeError(
                f"Region task {invocation.task_id!r} declares no outputs but returned a value"
            )
        return ()
    if count == 1:
        return (result,)
    if not isinstance(result, tuple) or len(result) != count:
        raise RuntimeError(
            f"Region task {invocation.task_id!r} must return a {count}-tuple"
        )
    return result


def _torch_compile_adapter(
    combined: Callable[..., tuple[object, ...]],
    command: RegionCompileCommand,
    torch_compile: Callable[..., Callable[..., object]] | None,
) -> Callable[..., tuple[object, ...]]:
    compiler = torch.compile if torch_compile is None else torch_compile
    compiled = compiler(combined, fullgraph=command.fullgraph, dynamic=command.dynamic)
    if not callable(compiled):
        raise RuntimeError("torch_compile adapter did not return a callable Region artifact")
    return cast(Callable[..., tuple[object, ...]], compiled)


def _eager_adapter(
    combined: Callable[..., tuple[object, ...]],
    command: RegionCompileCommand,
    _torch_compile: Callable[..., Callable[..., object]] | None,
) -> Callable[..., tuple[object, ...]]:
    if command.fullgraph or command.dynamic:
        raise RuntimeError("eager adapter received an invalid signed Region command")
    return combined


_COMPILER_ADAPTERS: Mapping[
    RegionCompilerAdapter,
    Callable[
        [
            Callable[..., tuple[object, ...]],
            RegionCompileCommand,
            Callable[..., Callable[..., object]] | None,
        ],
        Callable[..., tuple[object, ...]],
    ],
] = MappingProxyType(
    {
        RegionCompilerAdapter.TORCH_COMPILE: _torch_compile_adapter,
        RegionCompilerAdapter.EAGER: _eager_adapter,
    }
)


def materialize_region_callable(
    declarations: tuple[WorkloadDeclaration, ...],
    region: ExecutionRegion,
    options: CompileOptions,
    *,
    torch_compile: Callable[..., Callable[..., object]] | None = None,
) -> RegionCallableArtifact:
    """Compose one Region and optionally pass the complete callable to torch.compile."""

    if not isinstance(region, ExecutionRegion):
        raise TypeError("region must be ExecutionRegion")
    if not isinstance(options, CompileOptions):
        raise TypeError("options must be CompileOptions")
    bindings = _task_bindings(declarations)
    try:
        region_bindings = tuple(
            (invocation, bindings[invocation.task_id])
            for invocation in region.task_invocations
        )
    except KeyError as exc:
        raise RuntimeError(
            f"ExecutionRegion references an unavailable local task {exc.args[0]!r}"
        ) from exc

    input_keys = [binding.binding_id for binding in region.input_bindings]
    output_keys = [binding.binding_id for binding in region.output_bindings]
    output_sources = [binding.value_binding_id for binding in region.output_bindings]

    def combined(*external: object) -> tuple[object, ...]:
        if len(external) != len(input_keys):
            raise RuntimeError(
                f"ExecutionRegion {region.region_id!r} expected {len(input_keys)} inputs"
            )
        values_by_binding = dict(zip(input_keys, external, strict=True))
        for invocation, binding in region_bindings:
            try:
                arguments = {
                    task_input.port_name: values_by_binding[
                        task_input.value_binding_id
                    ]
                    for task_input in invocation.inputs
                }
            except KeyError as exc:
                raise RuntimeError(
                    "signed ExecutionRegion task invocation references an "
                    f"unavailable value binding {exc.args[0]!r}"
                ) from exc
            result = binding.executor(**arguments)
            task_values = _normalize_task_outputs(invocation, result)
            for output, value in zip(
                invocation.outputs, task_values, strict=True
            ):
                values_by_binding[output.value_binding_id] = value
        return tuple(values_by_binding[source] for source in output_sources)

    command = region.compile_command
    if region.implementation_kind is RegionImplementationKind.COMPILED and command is None:
        raise RuntimeError("compiled ExecutionRegion is missing its signed compile command")
    if region.implementation_kind is RegionImplementationKind.EXTERN and command is not None:
        raise RuntimeError("extern ExecutionRegion cannot carry a compile command")
    executable: Callable[..., tuple[object, ...]] = combined
    if command is not None:
        try:
            adapter = _COMPILER_ADAPTERS[command.adapter]
        except KeyError as exc:
            raise RuntimeError("unsupported signed Region compiler adapter") from exc
        executable = adapter(combined, command, torch_compile)
    return RegionCallableArtifact(
        region_id=region.region_id,
        artifact_key=region.artifact_key,
        input_keys=tuple(input_keys),
        output_keys=tuple(output_keys),
        callable=executable,
        compiled=(
            command is not None
            and command.adapter is RegionCompilerAdapter.TORCH_COMPILE
        ),
        compile_command=command,
        input_bindings=region.input_bindings,
        output_bindings=region.output_bindings,
        effects=region.effects,
    )


def materialize_region_artifacts(
    declarations: tuple[WorkloadDeclaration, ...],
    decision: ScheduleDecision,
    options: CompileOptions,
    *,
    torch_compile: Callable[..., Callable[..., object]] | None = None,
) -> tuple[RegionCallableArtifact, ...]:
    """Materialize every signed Region without changing its partition or order."""

    if not isinstance(decision, ScheduleDecision) or decision.schema_version != 2:
        raise TypeError("decision must be a ScheduleDecision schema v2")
    artifacts = tuple(
        materialize_region_callable(
            declarations,
            region,
            options,
            torch_compile=torch_compile,
        )
        for region in decision.execution_regions
    )
    if tuple(artifact.artifact_key for artifact in artifacts) != tuple(
        region.artifact_key for region in decision.execution_regions
    ):
        raise RuntimeError("local Region artifacts do not match the signed decision")
    return artifacts


__all__ = [
    "RegionArtifactRegistry",
    "RegionCallableArtifact",
    "materialize_region_artifacts",
    "materialize_region_callable",
]
