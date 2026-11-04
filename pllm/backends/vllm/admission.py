"""CPU admission and graph policy shared by engine creation and pllm.explain."""

from collections.abc import Iterator

from pllm.authoring import BackendRequirements
from pllm.backends.vllm import compat
from pllm.config import ModelConfig
from pllm.contracts.hidden_locator import HiddenPhase


def model_capture_bytes(config: ModelConfig, requirements: BackendRequirements) -> int:
    """Fund the model pool and each decode conditional's separate library workspace by default."""
    if HiddenPhase.DECODE in requirements.exact_phases and config.execution.capture_memory_bytes is None:
        from pllm.backends.vllm.model_shape import ModelShape

        shape = ModelShape.of_config(config, requirements.parameter_targets)
        # The request bucket reuses each conditional's dense temporaries and library scratch.
        return (64 << 20) + sum(5 * 4 * target.input_size * target.output_size + (64 << 20) for target in shape.targets)
    return config.execution.model_capture_bytes


def eager_reasons(config: ModelConfig, requirements: BackendRequirements) -> Iterator[tuple[str, str]]:
    """Every model-eager cause, keyed by setting or public effect name in engine priority order."""
    if config.cuda_graphs is False:
        yield "cuda_graphs", "ModelConfig(cuda_graphs=False)"
    if config.runner == "mrv1":
        yield "runner", 'runner="mrv1" runs workloads without CUDA graphs'
    if requirements.activation_layers:
        yield "ActivationPatch", "activation patches run outside CUDA graphs"
    if requirements.custom_forward:
        yield "custom_forward", "@pllm.custom_forward schedules model slices and owns their wave graphs"
    if requirements.instance_adapter:
        yield "Adapter", "CUDA-graph overlays do not support adapters backed by shared InstanceState"


def admit_workload(config: ModelConfig, requirements: BackendRequirements) -> str:
    """Reject unsupported engine settings and return the model forward eager reason."""
    if requirements.input_token_history >= 0 and config.runner != "mrv2":
        raise ValueError('pllm.input_tokens needs runner="mrv2"')
    if requirements.input_token_history >= 0 and requirements.custom_residual and config.tensor_parallel_size != 1:
        raise ValueError(
            "pllm.input_tokens feeding @pllm.custom_residual needs tensor_parallel_size=1 "
            "(custom_residual's TP support is deferred)"
        )
    compat.admit_parallel(config, requirements)
    if (requirements.custom_residual or requirements.custom_attention) and config.enable_prefix_caching:
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
