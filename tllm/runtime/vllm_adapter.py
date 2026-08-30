"""Quarantined dynamic adapter for the external vLLM object graph."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast
import weakref

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.outputs import RequestOutput

_runner_owners: "weakref.WeakKeyDictionary[Any, object]" = weakref.WeakKeyDictionary()


def bind_runner_owner(runner: object, owner: object) -> None:
    _runner_owners[cast(Any, runner)] = owner


def create_llm(args: tuple[object, ...], kwargs: dict[str, object]) -> LLM:
    from tllm.runtime import residual_runtime

    return residual_runtime.make_llm(*args, **kwargs)


def generate_requests(
    llm: LLM,
    prompts: Sequence[TokensPrompt],
    sampling_params: SamplingParams,
) -> list[RequestOutput]:
    return llm.generate(prompts, sampling_params, use_tqdm=False)


def bind_inprocess_runner(llm: object, owner: object) -> None:
    dynamic = cast(Any, llm)
    try:
        driver = dynamic.llm_engine.engine_core.model_executor.driver_worker
        runner = driver.model_runner
    except AttributeError:
        return
    bind_runner_owner(runner, owner)


def install_generate_boundary(
    llm: object,
    owner: object,
    before_generate: Callable[[], None],
    after_generate: Callable[[Sequence[RequestOutput]], None],
) -> None:
    dynamic = cast(Any, llm)
    installed_owner: object | None = getattr(
        dynamic, "_tllm_shape_specialization_poll_owner", None
    )
    if installed_owner is owner:
        return
    if installed_owner is not None:
        raise RuntimeError(
            "LLM already has a shape-specialization poll owned by another runtime"
        )
    generate: Any = getattr(dynamic, "generate", None)
    if generate is None:
        return

    def generate_at_runtime_boundary(*args: Any, **kwargs: Any) -> Any:
        before_generate()
        from tllm.runtime import residual_runtime
        from tllm.runtime.mp.control import ControlPlane
        from tllm.runtime.vllm_patch.capture_runner import (
            make_mapping_add_request,
        )

        engine = dynamic.llm_engine
        original_add_request = engine.add_request
        engine.add_request = make_mapping_add_request(
            runtime=residual_runtime.RUNTIME,
            plane=ControlPlane.try_bind(dynamic),
            orig_add_request=original_add_request,
            request_prompt_indices=None,
            request_sample_indices=None,
        )
        try:
            outputs = generate(*args, **kwargs)
        finally:
            engine.add_request = original_add_request
        after_generate(outputs)
        return outputs

    dynamic._tllm_shape_specialization_poll_owner = owner
    dynamic.generate = generate_at_runtime_boundary


def legacy_runner_graph_enabled(runner: object) -> bool:
    """Decode legacy vLLM graph flags at the foreign-object boundary."""

    dynamic = cast(Any, runner)
    if getattr(dynamic, "use_cuda_graph", None) is True:
        return True
    config: Any = getattr(dynamic, "compilation_config", None)
    if config is None:
        return False
    level: Any = getattr(config, "level", 0)
    return type(level) is int and level > 0


__all__ = [
    "bind_inprocess_runner",
    "bind_runner_owner",
    "create_llm",
    "generate_requests",
    "install_generate_boundary",
    "legacy_runner_graph_enabled",
]
