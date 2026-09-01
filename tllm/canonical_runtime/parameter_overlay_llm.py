"""Public construction boundary for the canonical parameter-overlay runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from tllm.canonical.workloads.inplace_ttt import InPlaceTTTWorkload
from tllm.canonical_runtime.parameter_overlay_deployment import (
    CanonicalParameterOverlayDeployment,
)
from tllm.canonical_runtime.parameter_overlay_runner import (
    CanonicalParameterOverlayGPUModelRunner,
)
from tllm.canonical_runtime.parameter_overlay_worker import (
    CanonicalParameterOverlayGPUWorker,
)
from tllm.canonical_runtime.runtime import CanonicalVllmConfig, CanonicalVllmDType

from vllm import RequestOutput, SamplingParams
from vllm.entrypoints.llm import LLM
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.executor.uniproc_executor import UniProcExecutor


_MP_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"
_V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
_WORKER_CLASS = (
    "tllm.canonical_runtime.parameter_overlay_worker."
    "CanonicalParameterOverlayGPUWorker"
)


@dataclass(slots=True)
class CanonicalParameterOverlayLlm:
    _llm: LLM
    runner: CanonicalParameterOverlayGPUModelRunner
    deployment: CanonicalParameterOverlayDeployment
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("canonical parameter-overlay LLM is already closed")
        self.runner.detach_parameter_overlay()
        self._llm.llm_engine.engine_core.shutdown()
        self._closed = True

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = False,
    ) -> list[RequestOutput]:
        if self._closed or sampling_params.n != 1:
            raise RuntimeError("canonical parameter-overlay generation requires n=1")
        try:
            outputs = self._llm.generate(
                prompts,
                sampling_params,
                use_tqdm=use_tqdm,
            )
        except Exception:
            self.runner.abort_parameter_overlay_requests()
            raise
        self.runner.retire_completed_parameter_overlay_requests()
        return outputs


def create_canonical_parameter_overlay_llm(
    workload: InPlaceTTTWorkload,
    config: CanonicalVllmConfig,
) -> CanonicalParameterOverlayLlm:
    mp_env = os.environ.get(_MP_ENV)
    v2_env = os.environ.get(_V2_RUNNER_ENV)
    if mp_env not in (None, "0") or v2_env not in (None, "1"):
        raise RuntimeError("canonical parameter overlay conflicts with runner environment")
    match config.dtype:
        case CanonicalVllmDType.BFLOAT16:
            model_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
        case CanonicalVllmDType.FLOAT16:
            model_dtype = "float16"
        case CanonicalVllmDType.FLOAT32:
            model_dtype = "float32"
    os.environ[_MP_ENV] = "0"
    os.environ[_V2_RUNNER_ENV] = "1"
    try:
        llm = LLM(
            model=config.model,
            dtype=model_dtype,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="uni",
            worker_cls=_WORKER_CLASS,
            enforce_eager=True,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
            trust_remote_code=config.trust_remote_code,
            enable_prefix_caching=config.enable_prefix_caching,
            enable_chunked_prefill=False,
        )
    finally:
        if mp_env is None:
            os.environ.pop(_MP_ENV)
        else:
            os.environ[_MP_ENV] = mp_env
        if v2_env is None:
            os.environ.pop(_V2_RUNNER_ENV)
        else:
            os.environ[_V2_RUNNER_ENV] = v2_env
    match llm.llm_engine.engine_core:
        case InprocClient() as client:
            pass
        case _:
            raise RuntimeError("parameter-overlay LLM did not create in-process client")
    match client.engine_core.model_executor:
        case UniProcExecutor() as executor:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("parameter-overlay LLM did not create uni executor")
    match executor.driver_worker.worker:
        case CanonicalParameterOverlayGPUWorker() as worker:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("parameter-overlay LLM did not create signed worker")
    match worker.model_runner:
        case CanonicalParameterOverlayGPUModelRunner() as runner:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("parameter-overlay LLM did not create MRv2 runner")
    try:
        deployment = runner.attach_parameter_overlay(workload)
    except Exception:
        client.shutdown()
        raise
    return CanonicalParameterOverlayLlm(llm, runner, deployment)


__all__ = [
    "CanonicalParameterOverlayLlm",
    "create_canonical_parameter_overlay_llm",
]
