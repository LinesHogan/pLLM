"""The upstream EngineCore worker and its closed workload-command boundary."""

from __future__ import annotations

import os
from functools import partial

import torch
from vllm.v1.worker.gpu_worker import Worker

from pllm.authoring import BackendWorkload, DataT, ResultT, map_auxiliary
from pllm.backends.vllm.mrv1 import Mrv1Runner
from pllm.backends.vllm.prefix_cache import CachePolicy
from pllm.backends.vllm.runner import BackendWorkerState, WorkloadModelRunner
from pllm.backends.vllm.runtime import (
    BackendCommand,
    BackendInstall,
    BackendProbe,
    BackendStage,
    probe_backend,
)
from pllm.backends.vllm.training import CausalTraining
from pllm.contracts.backend_attachment import AttachmentTopology, BackendAttachment
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.semantic_graph import StateScope, TaskId
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
        runner = self.model_runner
        assert isinstance(runner, (WorkloadModelRunner, Mrv1Runner))
        return runner

    def _install(
        self, workload: BackendWorkload[DataT, ResultT], attachment: BackendAttachment, cache_policy: CachePolicy
    ) -> None:
        runner = self.workload_runner
        plan = attachment.plan
        visibility = plan.timeline.visibility
        facts = probe_backend(
            workload.backend_requirements(),
            runner,
            plan.timeline,
            publish_lag=0 if visibility is None else visibility.lag,
        )
        max_lineages = (
            facts.max_admitted_lineages
            if any(state.scope is StateScope.REQUEST for state in plan.semantic_graph.states)
            else 1
        )
        candidates = tuple(
            provider.materialize_plan(
                resource=entry.resource,
                layout=entry.layout,
                binding=binding,
                max_admitted_lineages=max_lineages,
                graph=plan.semantic_graph,
            )
            for entry, binding in zip(
                plan.tensor_layout_request.layouts, plan.tensor_layout_decision.bindings, strict=True
            )
            for provider in facts.providers
            if any(
                selected.resource == entry.resource and selected.provider_id == provider.provider_id
                for selected in plan.states.plans
            )
        )
        if (facts.probe, candidates) != (attachment.probe, plan.states.plans):
            raise ValueError("worker backend facts changed; compile the workload against the current backend")
        runner.attach_workload(
            workload.bind_backend(facts, attachment, causal=partial(CausalTraining, runner.decoder)), cache_policy
        )

    def workload_command(self, command: str, payload: str) -> str:
        runner = self.workload_runner
        operation = BackendCommand(command)
        assert not payload or operation in (
            BackendCommand.PROBE,
            BackendCommand.INSTALL,
            BackendCommand.STAGE,
            BackendCommand.PROFILE,
            BackendCommand.OBSERVE,
        )
        with torch.inference_mode(False):
            match operation:
                case BackendCommand.PROBE:
                    probe = decode_boundary(payload, BackendProbe)
                    return encode_boundary(probe_backend(probe.requirements, runner, publish_lag=probe.publish_lag))
                case BackendCommand.INSTALL:
                    install = decode_boundary(payload, BackendInstall)
                    # The driver compares the rebuilt compilation signature against its own plan.
                    self._install(decode_workload(install.definition), install.attachment, install.cache_policy)
                    return encode_boundary(BackendWorkerState(os.getpid(), runner.require_session().diagnostics()))
                case BackendCommand.STAGE:
                    stage = decode_boundary(payload, BackendStage)
                    peer = os.getppid()
                    stream = torch.cuda.current_stream(runner.device)
                    auxiliary = map_auxiliary(
                        stage.auxiliary, lambda tensor: tensor.map(peer, stream).to(runner.device)
                    )
                    runner.require_session().stage_input(
                        auxiliary,
                        stage.identities,
                        None if stage.completion is None else stage.completion.map(peer, stream),
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
                case BackendCommand.OBSERVE:
                    runner.require_session().trace.configure(decode_boundary(payload, TraceSelection))
                case BackendCommand.DETACH:
                    runner.detach_workload()
            return ""
