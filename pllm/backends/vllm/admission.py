"""CPU admission and graph policy shared by engine creation and pllm.explain."""

from collections.abc import Iterator

from pllm.authoring import BackendRequirements
from pllm.backends.vllm import compat
from pllm.config import ModelConfig


def eager_reasons(config: ModelConfig, requirements: BackendRequirements) -> Iterator[tuple[str, str]]:
    """Every model-eager cause, keyed by setting or public effect name in engine priority order."""
    if config.cuda_graphs is False:
        yield "cuda_graphs", "ModelConfig(cuda_graphs=False)"
    if config.runner == "mrv1":
        yield "runner", 'runner="mrv1" runs workloads without CUDA graphs'
    if requirements.exact_decode and config.cuda_graphs == "auto":
        yield "train_in_forward", "exact train_in_forward decode updates run outside CUDA graphs"
    if requirements.activation_layers:
        yield "ActivationPatch", "activation patches run outside CUDA graphs"
    if requirements.custom_forward:
        yield "custom_forward", "@pllm.custom_forward schedules model slices and owns their wave graphs"
    if requirements.stateful_residual:
        yield "custom_residual", "stateful @pllm.custom_residual advances per-request chunks outside CUDA graphs"
    if requirements.instance_adapter:
        yield "Adapter", "CUDA-graph overlays do not support adapters backed by shared InstanceState"


def admit_workload(config: ModelConfig, requirements: BackendRequirements) -> str:
    """Reject unsupported engine settings and return the model forward eager reason."""
    compat.admit_parallel(config, requirements)
    if requirements.custom_residual and config.enable_prefix_caching:
        raise ValueError(
            "disable prefix caching (enable_prefix_caching=False) when a workload rewrites model sites; cached KV "
            "would not match"
        )
    if requirements.position_states and config.runner == "mrv1":
        raise ValueError('PositionState and @pllm.custom_forward need runner="mrv2"')
    reason = next((reason for _, reason in eager_reasons(config, requirements)), "")
    if config.cuda_graphs is True and reason:
        raise ValueError(f'{reason}; set ModelConfig(cuda_graphs="auto") or cuda_graphs=False')
    return reason
