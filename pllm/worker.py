"""The upstream EngineCore worker and its closed workload-command boundary."""

from __future__ import annotations

import os
from dataclasses import replace

import torch
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

from pllm.backends.vllm import compat
from pllm.backends.vllm.control import (
    BackendCompletions,
    BackendImport,
    BackendOnlineImport,
    BackendTrain,
    reported_call,
    run_import_state,
    run_train_batch,
)
from pllm.backends.vllm.runner import BackendInstall, BackendWorkerState, WorkloadModelRunner
from pllm.backends.vllm.runtime import BackendCommand, BackendStage, probe_backend
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.host_batch import map_host_batch
from pllm.contracts.semantic_graph import TaskId
from pllm.execution.compile_facts import BackendRequirements
from pllm.execution.state_feed import encode_feed
from pllm.execution.trace import TraceSelection
from pllm.workload_transport import decode_workload


class GPUWorker(compat.Worker):
    def shutdown(self) -> None:
        compat.release_offload_pins(self.vllm_config)
        super().shutdown()

    def _make_model_runner(self) -> WorkloadModelRunner:
        if self.parallel_config.tensor_parallel_size > 1:
            # Prioritize TP collectives over the replicated training lane, whose stream has priority 0.
            stream = torch.cuda.Stream(device=self.device, priority=-1)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            torch.cuda.set_stream(stream)
        return WorkloadModelRunner(self.vllm_config, self.device)

    @property
    def workload_runner(self) -> WorkloadModelRunner:
        runner = self.model_runner
        assert isinstance(runner, WorkloadModelRunner)
        return runner

    def workload_command(self, command: str, payload: str) -> str:
        runner = self.workload_runner
        operation = BackendCommand(command)
        with torch.inference_mode(False):
            match operation:
                case BackendCommand.IMPORT_STATE:
                    return reported_call(
                        lambda: run_import_state(
                            runner.require_session().driver, decode_boundary(payload, BackendImport)
                        )
                    )
                case BackendCommand.TRAIN_BATCH:
                    return reported_call(
                        lambda: run_train_batch(runner.require_session().driver, decode_boundary(payload, BackendTrain))
                    )
                case BackendCommand.FEED:
                    return reported_call(
                        lambda: encode_feed(runner.require_session().driver.feed(decode_boundary(payload, str)))
                    )
                case BackendCommand.STAGE_ONLINE_IMPORT:
                    return reported_call(lambda: self._online_import(decode_boundary(payload, BackendOnlineImport)))
                case BackendCommand.STAGE_COMPLETIONS:
                    return reported_call(lambda: runner.stage_completions(decode_boundary(payload, BackendCompletions)))
                case BackendCommand.FINISH_COMPLETIONS:
                    return reported_call(
                        lambda: runner.finish_completions(decode_boundary(payload, BackendCompletions), self.rank)
                    )
                case BackendCommand.PROBE:
                    return encode_boundary(probe_backend(decode_boundary(payload, BackendRequirements), runner))
                case BackendCommand.INSTALL:
                    install = decode_boundary(payload, BackendInstall)
                    return reported_call(
                        lambda: runner.install_workload(
                            decode_workload(install.definition),
                            install.facts,
                            install.options,
                            install.generated_kv,
                            self.rank,
                        )
                    )
                case BackendCommand.STAGE:
                    stage = decode_boundary(payload, BackendStage)
                    peer = stage.driver_pid
                    stream = torch.cuda.current_stream(runner.device)
                    host_batch = map_host_batch(stage.batch, lambda tensor: tensor.map(peer, stream).to(runner.device))
                    runner.require_session().stage_input(
                        host_batch,
                        stage.identities,
                        None if stage.completion is None or self.rank != 0 else stage.completion.map(peer, stream),
                        samples_per_prompt=stage.samples_per_prompt,
                    )
                case BackendCommand.FINISH:
                    runner.require_session().finish_requests()
                case BackendCommand.ABORT:
                    session = runner.require_session()
                    session.abort_active()
                    session.retire_all()
                case BackendCommand.DIAGNOSTICS:
                    diagnostics = replace(
                        runner.require_session().diagnostics(),
                        kv_cache_blocks=runner.kv_cache_config.num_blocks,
                        kv_cache_max_concurrency=get_max_concurrency_for_kv_cache_config(
                            runner.vllm_config, runner.kv_cache_config
                        ),
                    )
                    return encode_boundary(BackendWorkerState(os.getpid(), diagnostics))
                case BackendCommand.TRACE:
                    return encode_boundary(runner.require_session().trace.snapshot())
                case BackendCommand.PROFILE:
                    runner.require_session().trace.profile_tasks(decode_boundary(payload, tuple[TaskId, ...]))
                case BackendCommand.SELECT_TRACE:
                    runner.require_session().trace.configure(decode_boundary(payload, TraceSelection))
                case BackendCommand.DETACH:
                    runner.detach_workload(closing=bool(payload) and decode_boundary(payload, bool))
            return ""

    def _online_import(self, command: BackendOnlineImport) -> None:
        driver = self.workload_runner.require_session().driver
        if command.update is None:
            for key in command.request_ids:
                driver.discard_online_import(key)
        else:
            update = command.update.update()
            for key in command.request_ids:
                driver.stage_online_import(key, update)
