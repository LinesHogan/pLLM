"""Process-local callable materialization for signed ExecutionRegions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import FunctionType, MappingProxyType
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
    ScheduleEffectKind,
)
from tllm.workload.frontend import WorkloadDeclaration


class RegionParityError(RuntimeError):
    """Fail-closed numerical parity violation for a signed ExecutionRegion.

    Raised when a compiled or captured Region artifact diverges from its
    serial oracle (the same composed callable, run eagerly).  There is no
    silent fallback: the decision is rejected, never silently re-run.
    """


def region_dtype_tolerance(dtype: torch.dtype) -> float:
    """Dtype-aware tolerance for Region parity checks."""
    if dtype in (torch.float64,):
        return 1e-10
    if dtype in (torch.float16, torch.bfloat16):
        return 2e-2
    return 1e-5


def compare_region_outputs(
    region_id: str,
    reference: tuple[object, ...],
    candidate: tuple[object, ...],
    *,
    tolerance: tuple[float, float] | None = None,
    input_floor: float = 0.0,
) -> None:
    """Fail closed unless candidate matches the serial oracle within tolerance.

    ``tolerance`` is the (rtol, atol) pair an approximation workload declares;
    when absent, dtype-aware defaults apply (the exact-policy contract).
    ``input_floor`` is the loosest dtype tolerance among the Region's
    floating-point inputs: an output is a function of those inputs, so its
    compiled-vs-oracle noise floor is set by the lowest-precision input, not
    by its own dtype (an fp32 loss derived from bf16 hidden states carries
    bf16-scale noise).  Real divergence (wrong graph, wrong state) is orders
    of magnitude larger, so the gate remains fail-closed.
    """
    if len(reference) != len(candidate):
        raise RegionParityError(
            f"ExecutionRegion {region_id!r} parity check failed: "
            f"output count {len(candidate)} != oracle {len(reference)}"
        )
    for index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
        if torch.is_tensor(expected) or torch.is_tensor(actual):
            if not (torch.is_tensor(expected) and torch.is_tensor(actual)):
                raise RegionParityError(
                    f"ExecutionRegion {region_id!r} parity check failed: "
                    f"output {index} type mismatch"
                )
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise RegionParityError(
                    f"ExecutionRegion {region_id!r} parity check failed: "
                    f"output {index} shape/dtype mismatch "
                    f"({tuple(actual.shape)}/{actual.dtype} != "
                    f"{tuple(expected.shape)}/{expected.dtype})"
                )
            if tolerance is None:
                rtol = atol = max(region_dtype_tolerance(expected.dtype), input_floor)
            else:
                rtol, atol = tolerance
            if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
                raise RegionParityError(
                    f"ExecutionRegion {region_id!r} parity check failed: "
                    f"output {index} diverges from the serial oracle"
                )
        elif actual != expected:
            raise RegionParityError(
                f"ExecutionRegion {region_id!r} parity check failed: "
                f"output {index} value mismatch"
            )


_MUTATING_EFFECT_KINDS = frozenset(
    {
        ScheduleEffectKind.STATE_WRITE,
        ScheduleEffectKind.STATE_TRANSITION,
        ScheduleEffectKind.STATE_TRANSITION_MAYBE,
        ScheduleEffectKind.VERSION_PUBLISH,
        ScheduleEffectKind.EXTERN_IO,
    }
)


def _effects_are_replayable(effects: tuple[ScheduleEffect, ...]) -> bool:
    """A Region may be executed twice for parity only when it cannot mutate."""
    return all(effect.kind not in _MUTATING_EFFECT_KINDS for effect in effects)


def _region_parity_tolerance(
    declarations: tuple[WorkloadDeclaration, ...],
    region: ExecutionRegion,
) -> tuple[float, float] | None:
    """Resolve the declared approximation criteria covering one Region.

    Returns None for exact-only Regions (dtype-aware defaults apply).  A
    Region touching a non-exact workload requires that workload's declared
    criteria; conflicting declarations across covered workloads fail closed.
    """
    multiple = len(declarations) > 1
    owner: dict[str, WorkloadDeclaration] = {}
    for declaration in declarations:
        for task in declaration.ir.tasks:
            task_id = (
                f"{declaration.ir.workload_id}/{task.name}" if multiple else task.name
            )
            owner[task_id] = declaration
    covered = {
        owner[invocation.task_id]
        for invocation in region.task_invocations
        if invocation.task_id in owner
    }
    policies = {declaration.ir.approximation.value for declaration in covered}
    if len(policies) > 1:
        # The signed planner never merges across approximation policies; a
        # mixed Region here came from a supplied (untrusted) decision, so the
        # parity gate fails closed rather than letting a loose tolerance wash
        # over an exact workload's outputs.
        raise RegionParityError(
            f"ExecutionRegion {region.region_id!r} mixes approximation policies "
            f"{sorted(policies)} across workloads; the signed planner forbids "
            "cross-policy Region merges"
        )
    non_exact = tuple(
        declaration
        for declaration in covered
        if declaration.ir.approximation.value != "exact"
    )
    if not non_exact:
        return None
    criteria = {declaration.approximation_criteria for declaration in non_exact}
    if None in criteria:
        raise RegionParityError(
            f"ExecutionRegion {region.region_id!r} runs under an approximation policy "
            "but the covering workload declares no approximation criteria; "
            "declare approximation_criteria on @workload or keep the exact policy"
        )
    if len(criteria) != 1:
        raise RegionParityError(
            f"ExecutionRegion {region.region_id!r} mixes conflicting approximation "
            "criteria across workloads; split the declarations or unify the criteria"
        )
    declared = next(iter(criteria))
    assert declared is not None
    return (declared.rtol, declared.atol)


def _snapshot_rng(
    devices: set[torch.device],
) -> tuple[torch.Tensor, dict[torch.device, torch.Tensor]]:
    cpu_state = torch.random.get_rng_state()
    cuda_states = {
        device: torch.cuda.get_rng_state(device)
        for device in devices
        if device.type == "cuda"
    }
    return cpu_state, cuda_states


def _restore_rng(snapshot: tuple[torch.Tensor, dict[torch.device, torch.Tensor]]) -> None:
    cpu_state, cuda_states = snapshot
    torch.random.set_rng_state(cpu_state)
    for device, state in cuda_states.items():
        torch.cuda.set_rng_state(state, device)


class _FirstCallParityGate:
    """Run the serial oracle and the compiled artifact once, fail on divergence.

    The first invocation executes the eager composition (serial oracle) and
    the compiled artifact on identical RNG state and compares outputs.  Later
    invocations go straight to the compiled artifact.
    """

    __slots__ = ("_region_id", "_eager", "_compiled", "_verified", "_tolerance")

    def __init__(
        self,
        region_id: str,
        eager: Callable[..., tuple[object, ...]],
        compiled: Callable[..., tuple[object, ...]],
        tolerance: tuple[float, float] | None = None,
    ) -> None:
        self._region_id = region_id
        self._eager = eager
        self._compiled = compiled
        self._verified = False
        self._tolerance = tolerance

    def __call__(self, *args: object) -> tuple[object, ...]:
        if self._verified:
            return self._compiled(*args)
        devices = {
            arg.device for arg in args if torch.is_tensor(arg)
        }
        input_floor = max(
            (
                region_dtype_tolerance(arg.dtype)
                for arg in args
                if torch.is_tensor(arg) and arg.is_floating_point()
            ),
            default=0.0,
        )
        snapshot = _snapshot_rng(devices)
        reference = self._eager(*args)
        _restore_rng(snapshot)
        candidate = self._compiled(*args)
        compare_region_outputs(
            self._region_id,
            reference,
            candidate,
            tolerance=self._tolerance,
            input_floor=input_floor,
        )
        self._verified = True
        return candidate


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
    # Declared (rtol, atol) when an approximation workload covers this Region;
    # None means the exact-policy dtype-aware defaults.
    parity_tolerance: tuple[float, float] | None = None
    # "first_call": a serial-oracle parity gate runs on the first invocation;
    # "first_call_approximation": same gate with declaration-supplied criteria;
    # "deferred_state_effects": compiled Region with mutating effects, which
    # cannot be re-executed for parity (its production gate is the signed
    # state-version audit, which runs on every execution);
    # "eager": no compiled artifact, parity is trivially satisfied.
    parity: str = "eager"

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
    executors: Mapping[str, Callable[..., object]]

    def executor(self, implementation_id: str) -> Callable[..., object]:
        try:
            return self.executors[implementation_id]
        except KeyError as exc:
            raise RuntimeError(
                "ExecutionRegion references an unavailable local implementation "
                f"for task {self.task_id!r}: {implementation_id!r}"
            ) from exc


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
            executors: dict[str, Callable[..., object]] = {}
            for implementation in task.implementations:
                try:
                    executor = declaration.executor(task.name, implementation.key)
                except KeyError as exc:
                    raise RuntimeError(
                        "WorkloadDeclaration is missing a callable for task "
                        f"{task_id!r}, implementation {implementation.key!r}"
                    ) from exc
                if implementation.key in executors:
                    raise RuntimeError(
                        "WorkloadDeclaration has duplicate implementation binding "
                        f"for task {task_id!r}: {implementation.key!r}"
                    )
                executors[implementation.key] = executor
            bindings[task_id] = _TaskBinding(
                task_id=task_id,
                executors=MappingProxyType(executors),
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


def _fresh_region_code(
    function: Callable[..., tuple[object, ...]],
    region_id: str,
) -> Callable[..., tuple[object, ...]]:
    """Give one materialized Region its own Dynamo code-object identity.

    Every ``combined`` closure below shares one module-level code object, and
    Dynamo's recompile cache is per code object: without this, repeated
    materializations in one process (many Regions, or epoch re-materialization)
    accumulate guard entries on a single frame until the cache-size limit
    rejects compilation or silently degrades the Region to eager.  A fresh
    code object per materialization gives each Region its own cache budget.

    Global lifecycle knobs (``torch._dynamo.reset`` / ``cache_size_limit``)
    are deliberately not touched: the process also hosts the externally owned
    main model's compilation state, which must not be clobbered.
    """

    safe_name = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in region_id
    )
    fresh_code = function.__code__.replace(co_name=f"region_{safe_name}")
    return cast(
        Callable[..., tuple[object, ...]],
        FunctionType(
            fresh_code,
            function.__globals__,
            name=fresh_code.co_name,
            closure=function.__closure__,
        ),
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
            (
                invocation,
                bindings[invocation.task_id].executor(implementation_id),
            )
            for invocation, implementation_id in zip(
                region.task_invocations,
                region.implementation_ids,
                strict=True,
            )
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
        for invocation, executor in region_bindings:
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
            result = executor(**arguments)
            task_values = _normalize_task_outputs(invocation, result)
            for output, value in zip(
                invocation.outputs, task_values, strict=True
            ):
                values_by_binding[output.value_binding_id] = value
        return tuple(values_by_binding[source] for source in output_sources)

    combined = _fresh_region_code(combined, region.region_id)
    command = region.compile_command
    if region.implementation_kind is RegionImplementationKind.COMPILED and command is None:
        raise RuntimeError("compiled ExecutionRegion is missing its signed compile command")
    if region.implementation_kind is RegionImplementationKind.EXTERN and command is not None:
        raise RuntimeError("extern ExecutionRegion cannot carry a compile command")
    executable: Callable[..., tuple[object, ...]] = combined
    parity = "eager"
    tolerance = _region_parity_tolerance(declarations, region)
    if command is not None:
        try:
            adapter = _COMPILER_ADAPTERS[command.adapter]
        except KeyError as exc:
            raise RuntimeError("unsupported signed Region compiler adapter") from exc
        executable = adapter(combined, command, torch_compile)
        if command.adapter is RegionCompilerAdapter.TORCH_COMPILE:
            if _effects_are_replayable(region.effects):
                executable = _FirstCallParityGate(
                    region.region_id, combined, executable, tolerance
                )
                parity = (
                    "first_call_approximation" if tolerance is not None else "first_call"
                )
            else:
                # Mutating effects cannot be re-executed for a shadow parity
                # run; their production gate is the signed state-version
                # audit, which runs on every execution.
                parity = "deferred_state_effects"
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
        parity_tolerance=tolerance,
        parity=parity,
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
    "RegionParityError",
    "compare_region_outputs",
    "materialize_region_artifacts",
    "materialize_region_callable",
    "region_dtype_tolerance",
]
