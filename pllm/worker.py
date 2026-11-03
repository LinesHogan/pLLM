"""The upstream EngineCore worker and its closed workload-command boundary."""

from __future__ import annotations

import os
from functools import partial

import torch

from pllm.authoring import BackendWorkload, DataT, ResultT, map_host_batch
from pllm.backends.vllm import compat
from pllm.backends.vllm.mrv1 import Mrv1Runner
from pllm.backends.vllm.runner import BackendWorkerState, WorkloadModelRunner
from pllm.backends.vllm.runtime import (
    BackendCommand,
    BackendInstall,
    BackendProbe,
    BackendStage,
    probe_backend,
)
from pllm.backends.vllm.training import CausalTraining
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.semantic_graph import TaskId
from pllm.execution.trace import TraceSelection
from pllm.workload_transport import decode_workload


class GPUWorker(compat.Worker):
    def _make_model_runner(self) -> WorkloadModelRunner | Mrv1Runner:
        if self.parallel_config.tensor_parallel_size > 1:
            # Prioritize TP collectives over the replicated training lane, whose stream has priority 0.
            stream = torch.cuda.Stream(device=self.device, priority=-1)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            torch.cuda.set_stream(stream)
        return (
            WorkloadModelRunner(self.vllm_config, self.device)
            if self.use_v2_model_runner
            else Mrv1Runner(self.vllm_config, self.device)
        )

    @property
    def workload_runner(self) -> WorkloadModelRunner | Mrv1Runner:
        runner = self.model_runner
        assert isinstance(runner, (WorkloadModelRunner, Mrv1Runner))
        return runner

    def _install(self, workload: BackendWorkload[DataT, ResultT], install: BackendInstall) -> None:
        """Bind the driver's plan to the workload this process imported, refusing drifted sources."""
        runner = self.workload_runner
        plan = install.attachment.plan
        declaration = workload.funded_declaration(install.facts, causal=partial(CausalTraining, runner.decoder))
        if (declaration.graph, declaration.layouts) != (plan.semantic_graph, plan.tensor_layout_request):
            raise ValueError(
                "the EngineCore worker imported a workload (code or pretrained weights) that differs from the driver's; "
                "run both processes from the same installed sources and do not rewrite pretrained files while the engine starts"
            )
        runner.attach_workload(
            workload.bind_backend(install.facts, install.attachment, declaration, is_output_rank=self.rank == 0),
            install.cache_policy,
        )

    def workload_command(self, command: str, payload: str) -> str:
        runner = self.workload_runner
        operation = BackendCommand(command)
        assert not payload or operation in (
            BackendCommand.PROBE,
            BackendCommand.INSTALL,
            BackendCommand.STAGE,
            BackendCommand.PROFILE,
            BackendCommand.SELECT_TRACE,
        )
        with torch.inference_mode(False):
            match operation:
                case BackendCommand.PROBE:
                    probe = decode_boundary(payload, BackendProbe)
                    return encode_boundary(probe_backend(probe.requirements, runner, publish_lag=probe.publish_lag))
                case BackendCommand.INSTALL:
                    install = decode_boundary(payload, BackendInstall)
                    self._install(decode_workload(install.definition), install)
                    return encode_boundary(BackendWorkerState(os.getpid(), runner.require_session().diagnostics()))
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
                    return encode_boundary(BackendWorkerState(os.getpid(), runner.require_session().diagnostics()))
                case BackendCommand.TRACE:
                    return encode_boundary(runner.require_session().trace.snapshot())
                case BackendCommand.PROFILE:
                    runner.require_session().trace.profile_tasks(decode_boundary(payload, tuple[TaskId, ...]))
                case BackendCommand.SELECT_TRACE:
                    runner.require_session().trace.configure(decode_boundary(payload, TraceSelection))
                case BackendCommand.DETACH:
                    runner.detach_workload()
            return ""
