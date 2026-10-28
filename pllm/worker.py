"""The upstream EngineCore worker and its closed workload-command boundary."""

from __future__ import annotations

import os

import torch
from vllm.v1.worker.gpu_worker import Worker

from pllm.authoring import BackendWorkload, DataT, ResultT, map_auxiliary
from pllm.backends.vllm.mrv1 import Mrv1Runner
from pllm.backends.vllm.runner import WorkloadModelRunner
from pllm.backends.vllm.runtime import (
    BackendCommand,
    BackendInstall,
    BackendProbe,
    BackendStage,
    probe_backend,
)
from pllm.contracts.auxiliary import validate_auxiliary
from pllm.contracts.backend_attachment import AttachmentTopology, SignedBackendAttachment
from pllm.contracts.semantic_graph import TaskId
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.trace import TraceSelection
from pllm.workload_transport import decode_workload as decode_workload, encode_workload as encode_workload


class GPUWorker(Worker):
    def init_device(self) -> None:
        super().init_device()
        topology = (
            AttachmentTopology.ENGINE_CORE
            if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "1"
            else AttachmentTopology.IN_PROCESS
        )
        self.model_runner = (
            WorkloadModelRunner(self.vllm_config, self.device, topology)
            if self.use_v2_model_runner
            else Mrv1Runner(self.vllm_config, self.device, topology)
        )

    @property
    def workload_runner(self) -> WorkloadModelRunner | Mrv1Runner:
        match self.model_runner:
            case WorkloadModelRunner() | Mrv1Runner() as runner:
                return runner
            case _:
                raise RuntimeError("canonical worker has not initialized its runner")

    def _install(self, workload: BackendWorkload[DataT, ResultT], attachment: SignedBackendAttachment) -> None:
        runner = self.workload_runner
        visibility = attachment.inputs.plan.timeline.visibility
        facts = probe_backend(
            workload.backend_requirements(),
            runner,
            attachment.inputs.plan.timeline,
            publish_lag=0 if visibility is None else visibility.lag,
        )
        runner.attach_workload(workload.bind_backend(facts, attachment))

    def workload_command(self, command: str, payload: str) -> str:
        runner = self.workload_runner
        operation = BackendCommand(command)
        if (
            operation
            not in (
                BackendCommand.PROBE,
                BackendCommand.INSTALL,
                BackendCommand.STAGE,
                BackendCommand.PROFILE,
                BackendCommand.OBSERVE,
            )
            and payload
        ):
            raise ValueError("this canonical command has no payload")
        with torch.inference_mode(False):
            match operation:
                case BackendCommand.PROBE:
                    probe = decode_boundary(payload, BackendProbe)
                    return encode_boundary(probe_backend(probe.requirements, runner, publish_lag=probe.publish_lag))
                case BackendCommand.INSTALL:
                    install = decode_boundary(payload, BackendInstall)
                    # The driver compares the rebuilt compilation signature against its own plan.
                    self._install(decode_workload(install.definition), install.attachment)
                    return encode_boundary(runner.worker_state())
                case BackendCommand.STAGE:
                    stage = decode_boundary(payload, BackendStage)
                    validate_auxiliary(
                        stage.auxiliary, runner.require_session().attachment.inputs.plan, lambda tensor: tensor.field
                    )
                    peer = os.getppid()
                    stream = torch.cuda.current_stream(runner.device)
                    auxiliary = map_auxiliary(
                        stage.auxiliary, lambda tensor: tensor.map(peer, stream).to(runner.device)
                    )
                    runner.stage_workload_input(
                        auxiliary,
                        stage.identities,
                        None if stage.completion is None else stage.completion.map(peer, stream),
                    )
                case BackendCommand.FINISH:
                    runner.retire_completed_requests()
                case BackendCommand.ABORT:
                    runner.abort_workload_requests()
                case BackendCommand.DIAGNOSTICS:
                    return encode_boundary(runner.worker_state())
                case BackendCommand.TRACE:
                    return encode_boundary(runner.trace.snapshot())
                case BackendCommand.PROFILE:
                    runner.trace.profile_tasks(decode_boundary(payload, tuple[TaskId, ...]))
                case BackendCommand.OBSERVE:
                    runner.trace.configure(decode_boundary(payload, TraceSelection))
                case BackendCommand.DETACH:
                    runner.detach_workload()
            return ""
