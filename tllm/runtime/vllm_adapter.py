"""Quarantined dynamic adapter for the external vLLM object graph."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast
import weakref

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.inputs import PromptType
    from vllm.outputs import RequestOutput

_runner_owners: "weakref.WeakKeyDictionary[Any, object]" = weakref.WeakKeyDictionary()


def bind_runner_owner(runner: object, owner: object) -> None:
    _runner_owners[cast(Any, runner)] = owner


def create_llm(args: tuple[object, ...], kwargs: dict[str, object]) -> LLM:
    from tllm.runtime import residual_runtime

    return residual_runtime.make_llm(*args, **kwargs)


def generate_requests(
    llm: LLM,
    prompts: Sequence[PromptType],
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
    abort_generate: Callable[[tuple[str, ...]], None],
    request_prompt_indices: Callable[[], Sequence[int] | None],
    request_sample_indices: Callable[[], Sequence[int] | None],
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
        plane = ControlPlane.try_bind(dynamic)
        prompt_indices = request_prompt_indices()
        sample_indices = request_sample_indices()
        mapping_prepublished = plane is not None and (
            prompt_indices is not None or sample_indices is not None
        )
        original_add_request = engine.add_request
        original_request_counter: Any = None
        original_run_engine: Any = None
        prepublished_request_ids: tuple[str, ...] = ()
        scheduling_resumed = False
        if mapping_prepublished:
            original_request_counter = dynamic.request_counter
            original_run_engine = dynamic._run_engine
        try:
            if mapping_prepublished:
                engine.sleep(level=0, mode="keep")
                if prompt_indices is not None:
                    mapping_count = len(prompt_indices)
                else:
                    assert sample_indices is not None
                    mapping_count = len(sample_indices)
                assert plane is not None
                request_ids = tuple(
                    next(original_request_counter) for _ in range(mapping_count)
                )
                prepublished_request_ids = tuple(
                    str(request_id) for request_id in request_ids
                )
                dynamic.request_counter = iter(request_ids)
                plane.push_request_mapping(
                    [
                        (
                            request_id,
                            index
                            if prompt_indices is None
                            else int(prompt_indices[index]),
                            None
                            if sample_indices is None
                            else int(sample_indices[index]),
                        )
                        for index, request_id in enumerate(
                            prepublished_request_ids
                        )
                    ]
                )

                def run_after_bulk_admission(
                    *run_args: Any, **run_kwargs: Any
                ) -> Any:
                    nonlocal scheduling_resumed
                    if not scheduling_resumed:
                        engine.wake_up(tags=["scheduling"])
                        scheduling_resumed = True
                    return original_run_engine(*run_args, **run_kwargs)

                dynamic._run_engine = run_after_bulk_admission
            engine.add_request = make_mapping_add_request(
                runtime=residual_runtime.RUNTIME,
                plane=None if mapping_prepublished else plane,
                orig_add_request=original_add_request,
                request_prompt_indices=prompt_indices,
                request_sample_indices=sample_indices,
            )
            outputs = generate(*args, **kwargs)
        except Exception:
            if prepublished_request_ids:
                abort_generate(prepublished_request_ids)
            raise
        finally:
            engine.add_request = original_add_request
            if mapping_prepublished:
                dynamic.request_counter = original_request_counter
                dynamic._run_engine = original_run_engine
                if not scheduling_resumed:
                    engine.wake_up(tags=["scheduling"])
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
