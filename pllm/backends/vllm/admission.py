"""CPU admission and graph policy shared by engine creation and pllm.explain."""

from collections.abc import Iterator

from pllm.backends.vllm import compat
from pllm.config import ModelConfig
from pllm.execution.compile_facts import BackendRequirements


def model_hooks(requirements: BackendRequirements) -> bool:
    """Whether projections, residuals or attention need hooks in the model's graph."""
    return bool(
        requirements.parameter_targets
        or requirements.custom_residual
        or requirements.custom_attention
        or requirements.activation_layers
    )


def eager_reasons(config: ModelConfig, requirements: BackendRequirements) -> Iterator[tuple[str, str]]:
    """Every model-eager cause, keyed by setting or public effect name in engine priority order."""
    if config.cuda_graphs is False:
        yield "cuda_graphs", "ModelConfig(cuda_graphs=False)"
    if requirements.custom_forward:
        yield "custom_forward", "@pllm.custom_forward schedules model slices and owns their wave graphs"


def admit_workload(config: ModelConfig, requirements: BackendRequirements) -> str:
    """Reject unsupported engine settings and return the model forward eager reason."""
    compat.admit_parallel(config, requirements)
    if config.enable_prefix_caching:
        requirements.check_prefix_cache()
    if any(layout.keep != "all" for layout in requirements.kv_layouts) and (
        requirements.custom_forward or requirements.position_states
    ):
        raise ValueError("static KV layouts cannot accompany PositionState or @pllm.custom_forward (including kv_copy)")
    reason = next((reason for _, reason in eager_reasons(config, requirements)), "")
    if config.cuda_graphs is True and reason:
        raise ValueError(f'{reason}; set ModelConfig(cuda_graphs="auto") or cuda_graphs=False')
    return reason
