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
    State,
    StateEffect,
    WorkloadAuthority,
    WorkloadDeclaration,
    current,
    declare_workload,
    extern_io,
    hidden,
    maybe_transition,
    next,
    provided,
    read,
    request_key,
    result,
    sink,
    step_id,
    task,
    temporary,
    transition,
    workload,
)
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
    "HiddenPhase",
    "HiddenSite",
    "Index",
    "OptimizerSpec",
    "RequestKey",
    "RegionArtifactRegistry",
    "RegionCallableArtifact",
    "State",
    "StateEffect",
    "StateScope",
    "WorkloadDeclaration",
    "WorkloadAuthority",
    "WorkloadEdgeKind",
    "WorkloadIR",
    "current",
    "dependence_effect_fingerprint",
    "declare_workload",
    "extern_io",
    "hidden",
    "lower_program",
    "lower_schedule_graph",
    "materialize_region_artifacts",
    "materialize_region_callable",
    "maybe_transition",
    "next",
    "provided",
    "read",
    "request_key",
    "result",
    "sink",
    "step_id",
    "task",
    "temporary",
    "transition",
    "workload",
]
