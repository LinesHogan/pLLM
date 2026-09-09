"""Closed built-in authoring boundary for the upstream EngineCore worker."""

from __future__ import annotations

from pllm.contracts.auxiliary import validate_auxiliary

import os
import torch

from pllm.authoring import BackendAuthoring, BackendRequirements, BackendWorkload, map_auxiliary, DataT, ResultT
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.trace import TraceSelection
from pllm.workloads.esamp_request import ESampRequestWorkload
from pllm.workloads.dummy import DummyConfig, DummyWorkload
from pllm.workloads.esamp_request import ESampRequestConfig
from pllm.workloads.inplace_ttt import InPlaceTTTWorkload, InPlaceTTTConfig
from pllm.workloads.override import OverRIDEConfig, OverRIDEWorkload
from pllm.workloads.lora import OutputLoRAConfig, OutputLoRAWorkload
from pllm.workloads.recirculation import RecirculationConfig, RecirculationWorkload
from pllm.workloads.reft import ActivationConfig, ActivationWorkload
from pllm.workloads.tlm import HfQvConfig, HfQvModel, TLMConfig, TLMWorkload
from pllm.backends.vllm.runtime import BackendCommand, BackendInstall, BackendStage, WorkloadModelRunner, probe_backend
from pllm.backends.vllm.mrv1 import Mrv1Runner
from pllm.contracts.backend_attachment import AttachmentTopology, SignedBackendAttachment
from pllm.contracts.semantic_graph import TaskId
from vllm.v1.worker.gpu_worker import Worker


def encode_workload(workload: BackendAuthoring) -> str:
    match workload:
        case DummyWorkload(config=config):
            if workload != DummyWorkload(config):
                raise ValueError("EngineCore cannot transport local tensor replacements; use in-process execution")
            return encode_boundary(config)
        case ESampRequestWorkload(config=config):
            if type(workload) is not ESampRequestWorkload:
                raise ValueError("EngineCore cannot transport custom native subclasses; use in-process execution")
            return encode_boundary(config)
        case InPlaceTTTWorkload(config=config):
            return encode_boundary(config)
        case OverRIDEWorkload(config=config):
            if workload != OverRIDEWorkload(config):
                raise ValueError("EngineCore cannot transport local tensor replacements; use in-process execution")
            return encode_boundary(config)
        case OutputLoRAWorkload(config=config):
            if type(workload) is not OutputLoRAWorkload:
                raise ValueError("EngineCore cannot transport custom native subclasses; use in-process execution")
            return encode_boundary(config)
        case RecirculationWorkload(config=config):
            return encode_boundary(config)
        case ActivationWorkload(config=config):
            return encode_boundary(config)
        case TLMWorkload(config=config, model=HfQvModel() as model):
            if model.device.type != "meta":
                raise ValueError("EngineCore TLM requires HfQvModel.load(config, device=meta); live models are not transported")
            return encode_boundary((config, model.config))
        case _:
            raise ValueError("EngineCore has no closed definition for this workload; Python objects are not transported")


class GPUWorker(Worker):
    def init_device(self) -> None:
        super().init_device()
        topology = (AttachmentTopology.ENGINE_CORE if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "1"
            else AttachmentTopology.IN_PROCESS)
        self.model_runner = (WorkloadModelRunner(self.vllm_config, self.device, topology) if self.use_v2_model_runner
            else Mrv1Runner(self.vllm_config, self.device, topology))

    @property
    def workload_runner(self) -> WorkloadModelRunner | Mrv1Runner:
        match self.model_runner:
            case WorkloadModelRunner() | Mrv1Runner() as runner:
                return runner
            case _:
                raise RuntimeError("canonical worker has not initialized its runner")

    def _install(self, workload: BackendWorkload[DataT, ResultT], attachment: SignedBackendAttachment) -> None:
        runner = self.workload_runner
        facts = probe_backend(workload.backend_requirements(), runner, attachment.inputs.plan.timeline)
        runner.attach_workload(workload.bind_backend(facts, attachment))

    def workload_command(self, command: str, payload: str) -> str:
        runner = self.workload_runner
        operation = BackendCommand(command)
        if operation not in (BackendCommand.PROBE, BackendCommand.INSTALL, BackendCommand.STAGE,
                             BackendCommand.PROFILE, BackendCommand.OBSERVE) and payload:
            raise ValueError("this canonical command has no payload")
        with torch.inference_mode(False):
            match operation:
                case BackendCommand.PROBE:
                    return encode_boundary(probe_backend(decode_boundary(payload, BackendRequirements), runner))
                case BackendCommand.INSTALL:
                    install = decode_boundary(payload, BackendInstall)
                    # The original graph names a closed native binding, never a Python type path.
                    match install.attachment.inputs.plan.semantic_graph.workload_id:
                        case "dummy":
                            self._install(DummyWorkload(decode_boundary(install.definition, DummyConfig)), install.attachment)
                        case "esamp.request":
                            self._install(ESampRequestWorkload(decode_boundary(install.definition, ESampRequestConfig)), install.attachment)
                        case "inplace_ttt.dense_fast_weight":
                            self._install(InPlaceTTTWorkload(decode_boundary(install.definition, InPlaceTTTConfig)), install.attachment)
                        case "override":
                            self._install(OverRIDEWorkload(decode_boundary(install.definition, OverRIDEConfig)), install.attachment)
                        case "lora.output_head":
                            self._install(OutputLoRAWorkload(decode_boundary(install.definition, OutputLoRAConfig)), install.attachment)
                        case "recirculation":
                            self._install(RecirculationWorkload(decode_boundary(install.definition, RecirculationConfig)), install.attachment)
                        case "activation":
                            self._install(ActivationWorkload(decode_boundary(install.definition, ActivationConfig)), install.attachment)
                        case "tlm.qv_lora":
                            config, model = decode_boundary(install.definition, tuple[TLMConfig, HfQvConfig])
                            generator = torch.cuda.default_generators[runner.device.index or 0]
                            cpu_state, gpu_state = torch.random.get_rng_state(), generator.get_state()
                            try:
                                torch.random.set_rng_state(torch.Generator().manual_seed(runner.model_config.seed).get_state())
                                generator.manual_seed(runner.model_config.seed)
                                self._install(TLMWorkload(config, HfQvModel.load(model, runner.device)), install.attachment)
                            finally:
                                torch.random.set_rng_state(cpu_state)
                                generator.set_state(gpu_state)
                        case _:
                            raise ValueError("signed graph has no closed native worker binding")
                    return encode_boundary(runner.worker_state())
                case BackendCommand.STAGE:
                    stage = decode_boundary(payload, BackendStage)
                    validate_auxiliary(stage.auxiliary, runner.require_session().attachment.inputs.plan,
                        lambda tensor: tensor.field)
                    peer = os.getppid()
                    stream = torch.cuda.current_stream(runner.device)
                    auxiliary = map_auxiliary(stage.auxiliary, lambda tensor: tensor.map(peer, stream).to(runner.device))
                    runner.stage_workload_input(auxiliary, stage.identities,
                        None if stage.completion is None else stage.completion.map(peer, stream))
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
