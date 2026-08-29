"""Public auxiliary-workload declaration API."""

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
    Domain,
    ExecutorRegistry,
    ExternImplementation,
    HiddenPhase,
    HiddenSite,
    Index,
    RequestKey,
    PublishedModule,
    State,
    StateEffect,
    WorkloadAuthority,
    WorkloadDeclaration,
    current,
    declare_workload,
    extern,
    extern_io,
    hidden,
    host_output,
    maybe_transition,
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
    temporary,
    transition,
    workload,
)
from tllm.workload.config import (
    ConfigField,
    NumericConstraint,
    WorkloadConfig,
    config_field,
)
from tllm.workload.staging import StagedPort
from tllm.workload.host import HostOutput
from tllm.workload.ir import (
    ApproximationPolicy,
    EffectKind,
    OptimizerSpec,
    StateScope,
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
    # Type-checking view of the legacy bridge.  At runtime these names load
    # lazily via __getattr__ below so `import tllm.workload` never pulls the
    # legacy Program/workgraph frontend into a native-only process.
    from tllm.workload.compat import dependence_effect_fingerprint, lower_program


def __getattr__(name: str) -> object:
    # The legacy adapter bridge (deleted at retirement) is the only legacy
    # contact point; keep it explicit and lazy.
    if name in ("dependence_effect_fingerprint", "lower_program"):
        from tllm.workload import compat

        return getattr(compat, name)
    raise AttributeError(f"module 'tllm.workload' has no attribute {name!r}")

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
    "ExternImplementation",
    "ConfigField",
    "HiddenPhase",
    "HiddenSite",
    "HostOutput",
    "Index",
    "OptimizerSpec",
    "NumericConstraint",
    "RequestKey",
    "PublishedModule",
    "RegionArtifactRegistry",
    "RegionCallableArtifact",
    "State",
    "StateEffect",
    "StateScope",
    "StagedPort",
    "WorkloadDeclaration",
    "WorkloadAuthority",
    "WorkloadEdgeKind",
    "WorkloadIR",
    "current",
    "dependence_effect_fingerprint",
    "declare_workload",
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
    "temporary",
    "transition",
    "workload",
]
