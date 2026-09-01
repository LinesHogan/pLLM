"""Typed construction boundary for the standalone canonical vLLM runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from tllm.canonical.workloads.esamp_request import ESampRequestWorkload
from tllm.canonical_runtime.deployment import CanonicalEsampVllmDeployment
from tllm.canonical_runtime.runner import CanonicalGPUModelRunner
from tllm.canonical_runtime.worker import CanonicalGPUWorker

from vllm import RequestOutput, SamplingParams
from vllm.entrypoints.llm import LLM
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.executor.uniproc_executor import UniProcExecutor


_MP_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"
_V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
_WORKER_CLASS = "tllm.canonical_runtime.worker.CanonicalGPUWorker"


class CanonicalVllmDType(StrEnum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


@dataclass(frozen=True, slots=True)
class CanonicalVllmConfig:
    """Cold backend facts supported by the first MRv2 vertical slice."""

    model: str
    dtype: CanonicalVllmDType
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    seed: int = 0
    trust_remote_code: bool = False
    enable_prefix_caching: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.model) is not str
            or not self.model
            or type(self.dtype) is not CanonicalVllmDType
            or type(self.max_model_len) is not int
            or self.max_model_len < 2
            or type(self.max_num_seqs) is not int
            or self.max_num_seqs < 1
            or type(self.max_num_batched_tokens) is not int
            or self.max_num_batched_tokens < self.max_num_seqs
            or type(self.gpu_memory_utilization) is not float
            or not 0.0 < self.gpu_memory_utilization < 1.0
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.trust_remote_code) is not bool
            or type(self.enable_prefix_caching) is not bool
        ):
            raise ValueError("canonical vLLM configuration is not closed")


@dataclass(slots=True)
class CanonicalEsampLlm:
    """Own the public vLLM object and its exact canonical runner deployment."""

    _llm: LLM
    runner: CanonicalGPUModelRunner
    deployment: CanonicalEsampVllmDeployment
    _closed: bool = False

    def __post_init__(self) -> None:
        if (
            type(self._llm) is not LLM
            or type(self.runner) is not CanonicalGPUModelRunner
            or type(self.deployment) is not CanonicalEsampVllmDeployment
            or type(self._closed) is not bool
            or self._closed
        ):
            raise TypeError("canonical ESamp LLM requires exact owned objects")

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("canonical ESamp LLM was already closed")
        self.runner.detach_esamp()
        self._llm.llm_engine.engine_core.shutdown()
        self._closed = True

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = False,
    ) -> list[RequestOutput]:
        """Run one synchronous n=1 family and retire it at the public finish."""

        if self._closed:
            raise RuntimeError("canonical ESamp LLM is closed")
        if (
            type(prompts) not in (str, list)
            or (type(prompts) is str and not prompts)
            or (
                type(prompts) is list
                and (
                    not prompts
                    or any(type(prompt) is not str or not prompt for prompt in prompts)
                )
            )
            or type(sampling_params) is not SamplingParams
            or sampling_params.n != 1
            or type(use_tqdm) is not bool
        ):
            raise ValueError("canonical ESamp generate requires non-empty text and n=1")
        try:
            outputs = self._llm.generate(
                prompts,
                sampling_params,
                use_tqdm=use_tqdm,
            )
        except Exception:
            self.runner.abort_esamp_requests()
            raise
        self.runner.retire_completed_esamp_requests()
        return outputs


def create_canonical_esamp_llm(
    workload: ESampRequestWorkload,
    config: CanonicalVllmConfig,
) -> CanonicalEsampLlm:
    """Construct, probe, sign, and attach one real in-process MRv2 runner."""

    if (
        type(workload) is not ESampRequestWorkload
        or type(config) is not CanonicalVllmConfig
    ):
        raise TypeError("canonical vLLM construction requires exact inputs")
    mp_env = os.environ.get(_MP_ENV)
    v2_env = os.environ.get(_V2_RUNNER_ENV)
    if mp_env not in (None, "0") or v2_env not in (None, "1"):
        raise RuntimeError(
            "canonical vLLM conflicts with an explicit MP or legacy-runner environment"
        )
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
            raise RuntimeError("canonical vLLM did not construct an in-process client")
    match client.engine_core.model_executor:
        case UniProcExecutor() as executor:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("canonical vLLM did not construct a uni-process executor")
    match executor.driver_worker.worker:
        case CanonicalGPUWorker() as worker:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("canonical vLLM did not construct its signed worker")
    match worker.model_runner:
        case CanonicalGPUModelRunner() as runner:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("canonical vLLM did not construct its MRv2 runner")
    try:
        deployment = runner.attach_esamp(workload)
    except Exception:
        client.shutdown()
        raise
    return CanonicalEsampLlm(llm, runner, deployment)


__all__ = [
    "CanonicalEsampLlm",
    "CanonicalVllmConfig",
    "CanonicalVllmDType",
    "create_canonical_esamp_llm",
]
