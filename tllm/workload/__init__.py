"""Public auxiliary-workload declaration API."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from tllm.workload.frontend import (
    AdamSpec,
    AdamWSpec,
    AffineAccess,
    AffineExpr,
    After,
    ApproximationCriteria,
    Before,
    Between,
    ConfigPredicate,
    Domain,
    ExecutorRegistry,
    ExternalEffect,
    ExternImplementation,
    HiddenPhase,
    HiddenSite,
    Index,
    NativeWorkloadDefinition,
    RequestKey,
    PublishedModule,
    ServiceState,
    State,
    TrainableState,
    StateEffect,
    WorkloadAuthority,
    WorkloadDeclaration,
    WorkloadDefinition,
    WorkloadEffect,
    current,
    declare_workload,
    definition_for_workload_class,
    extern,
    extern_io,
    hidden,
    host_output,
    maybe_transition,
    maybe_optimize,
    next,
    optimize,
    sampler,
    provided,
    read,
    request_key,
    result,
    staged,
    sink,
    step_id,
    task,
    task_declaration,
    temporary,
    transition,
    when,
    workload,
)
from tllm.workload.config import (
    ConfigField,
    ConfigKind,
    NumericConstraint,
    WorkloadConfig,
    config_field,
)
from tllm.workload.staging import StagedLifecycle, StagedPort
from tllm.workload.host import HostOutput
from tllm.workload.ir import (
    ApproximationPolicy,
    EffectKind,
    OptimizerSpec,
    StateScope,
    StateStorage,
    WorkloadEdgeKind,
    WorkloadIR,
)
from tllm.workload.lowering import lower_schedule_graph
from tllm.workload.region_compiler import (
    RegionArtifactRegistry,
    RegionCallableArtifact,
    materialize_region_artifacts,
    materialize_region_callable,
)

if TYPE_CHECKING:
    from tllm.program.decl import StateSpec


def lower_program(
    instance: object,
    *,
    states: "Mapping[str, StateSpec]",
    workload_id: str | None = None,
) -> WorkloadIR:
    from tllm.workload.compat import lower_program as lower_legacy_program

    return lower_legacy_program(
        instance,
        states=states,
        workload_id=workload_id,
    )


def dependence_effect_fingerprint(
    ir: WorkloadIR,
) -> tuple[tuple[str, str, str, str], ...]:
    from tllm.workload.compat import (
        dependence_effect_fingerprint as legacy_fingerprint,
    )

    return legacy_fingerprint(ir)

__all__ = [
    "AdamSpec",
    "AdamWSpec",
    "AffineAccess",
    "AffineExpr",
    "After",
    "ApproximationCriteria",
    "ApproximationPolicy",
    "Before",
    "Between",
    "Domain",
    "EffectKind",
    "ExecutorRegistry",
    "ExternalEffect",
    "ExternImplementation",
    "ConfigField",
    "ConfigKind",
    "ConfigPredicate",
    "HiddenPhase",
    "HiddenSite",
    "HostOutput",
    "Index",
    "NativeWorkloadDefinition",
    "OptimizerSpec",
    "NumericConstraint",
    "RequestKey",
    "PublishedModule",
    "ServiceState",
    "RegionArtifactRegistry",
    "RegionCallableArtifact",
    "State",
    "TrainableState",
    "StateEffect",
    "StateScope",
    "StateStorage",
    "StagedPort",
    "StagedLifecycle",
    "WorkloadDeclaration",
    "WorkloadDefinition",
    "WorkloadEffect",
    "WorkloadAuthority",
    "WorkloadEdgeKind",
    "WorkloadIR",
    "current",
    "dependence_effect_fingerprint",
    "declare_workload",
    "definition_for_workload_class",
    "extern",
    "extern_io",
    "WorkloadConfig",
    "config_field",
    "hidden",
    "host_output",
    "lower_program",
    "lower_schedule_graph",
    "materialize_region_artifacts",
    "materialize_region_callable",
    "maybe_transition",
    "maybe_optimize",
    "next",
    "optimize",
    "sampler",
    "provided",
    "read",
    "request_key",
    "result",
    "sink",
    "step_id",
    "staged",
    "task",
    "task_declaration",
    "temporary",
    "transition",
    "when",
    "workload",
]
