"""The single-worker engine configured by pllm's LLM constructor.

The nested worker is accessed only in in-process mode; EngineCore clients in either
mode support shutdown. These declarations describe the configured object graph.
"""

from pllm.backends.vllm.mrv1 import Mrv1Runner
from pllm.backends.vllm.runner import WorkloadModelRunner
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.config import VllmConfig

class _Worker:
    model_runner: WorkloadModelRunner | Mrv1Runner

class _WorkerWrapper:
    worker: _Worker

class _Executor:
    driver_worker: _WorkerWrapper

class _EngineCore:
    model_executor: _Executor

class _CoreClient:
    engine_core: _EngineCore
    def shutdown(self) -> None: ...

class LLMEngine:
    vllm_config: VllmConfig
    engine_core: _CoreClient
    output_processor: OutputProcessor
    def abort_request(self, request_ids: list[str], *, internal: bool) -> None: ...
