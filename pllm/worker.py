"""By-reference authoring boundary for the upstream EngineCore worker."""

from __future__ import annotations

from pllm.contracts.auxiliary import validate_auxiliary

import dataclasses
from hashlib import sha256
import importlib
import inspect
import os
from types import ModuleType
import typing
from typing import TYPE_CHECKING, cast

import torch

from pllm.authoring import BackendAuthoring, BackendRequirements, BackendWorkload, map_auxiliary, DataT, ResultT
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.trace import TraceSelection
from pllm.backends.vllm.runtime import BackendCommand, BackendInstall, BackendStage, WorkloadModelRunner, probe_backend
from pllm.backends.vllm.mrv1 import Mrv1Runner
from pllm.contracts.backend_attachment import AttachmentTopology, SignedBackendAttachment
from pllm.contracts.semantic_graph import TaskId
from pllm.contracts.workload_reference import (
    WorkloadFieldKind, WorkloadFieldTransport, WorkloadReference, WorkerTransportable,
)
from vllm.v1.worker.gpu_worker import Worker

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _resolve(module: ModuleType, qualname: str) -> object:
    value: object = module
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def _module_digest(module: ModuleType) -> str:
    source = inspect.getsourcefile(module)
    if source is None:
        raise ValueError(f"workload module {module.__name__!r} has no source file; "
            "the worker must import the same sources the driver signed")
    with open(source, "rb") as handle:
        return sha256(handle.read()).hexdigest()


def _importable(module_name: str, qualname: str) -> bool:
    return module_name != "__main__" and "<" not in qualname


def _encode_field(workload: BackendAuthoring, field: dataclasses.Field[object], hint: type[object] | None) -> WorkloadFieldTransport:
    value = getattr(workload, field.name)
    if isinstance(value, WorkerTransportable):
        payload = encode_boundary(value.transport_payload())
        reference = type(value)
        return WorkloadFieldTransport(field.name, WorkloadFieldKind.WORKER_TRANSPORT, payload,
            reference.__module__, reference.__qualname__)
    if inspect.isroutine(value) or isinstance(value, type):
        module_name = getattr(value, "__module__", None) or ""
        qualname = getattr(value, "__qualname__", "")
        if not _importable(module_name, qualname):
            raise ValueError(f"EngineCore field {field.name!r} holds a closure or local definition; "
                "use a module-level function the worker can import, or keep multiprocessing=False")
        module = importlib.import_module(module_name)
        if _resolve(module, qualname) is not value:
            raise ValueError(f"EngineCore field {field.name!r} does not resolve to itself in {module_name!r}; "
                "the worker must import the exact object the driver signed")
        return WorkloadFieldTransport(field.name, WorkloadFieldKind.REFERENCE, "", module_name, qualname)
    try:
        payload = encode_boundary(value)
        if hint is None:
            raise ValueError("no declared annotation")
        if decode_boundary(payload, hint) != value:
            raise ValueError("closed round trip changed the value")
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"EngineCore field {field.name!r} of type {type(value).__name__} is not transportable: "
            "fields must be closed values (primitives or typed contracts), importable module-level references, "
            "or WorkerTransportable; live models, lambdas and open handles cannot travel") from exc
    return WorkloadFieldTransport(field.name, WorkloadFieldKind.VALUE, payload, "", "")


def encode_workload(workload: BackendAuthoring) -> str:
    """Transport built-in and user workloads identically: class coordinates plus frozen fields."""
    cls = type(workload)
    module_name, qualname = cls.__module__, cls.__qualname__
    if not _importable(module_name, qualname):
        raise ValueError(f"EngineCore transports workloads by reference: {cls.__name__} is defined in "
            f"{module_name!r}, which a spawned worker cannot import. Move the class into an importable .py "
            "module (notebook and __main__ scopes stay in-process), or keep the default multiprocessing=False")
    module = importlib.import_module(module_name)
    if _resolve(module, qualname) is not cls:
        raise ValueError(f"EngineCore cannot transport {module_name}:{qualname}: the module resolves to a "
            "different class than the driver holds; reload the driver from identical sources")
    parameters = getattr(cls, "__dataclass_params__", None)
    if parameters is None or not parameters.frozen:
        raise ValueError(f"EngineCore transports workloads by reference: {cls.__name__} must be a frozen "
            "dataclass (a fieldless one is fine) whose fields are closed-transportable; add "
            "@dataclass(frozen=True) or keep multiprocessing=False")
    hints = typing.get_type_hints(cls)
    fields = tuple(_encode_field(workload, field, hints.get(field.name))
        for field in dataclasses.fields(cast("type[DataclassInstance]", cls)) if field.init)
    return encode_boundary(WorkloadReference(module_name, qualname, _module_digest(module), fields))


def decode_workload(payload: str, *, device: torch.device, seed: int) -> BackendAuthoring:
    """Rebuild the referenced workload inside the worker; the driver verifies the signature."""
    reference = decode_boundary(payload, WorkloadReference)
    try:
        module = importlib.import_module(reference.module)
    except ImportError as exc:
        raise ValueError(f"EngineCore worker cannot import workload module {reference.module!r}; the class "
            "must live in a module importable by the worker, and the module must stay import-safe "
            "(no GPU work at import time)") from exc
    if _module_digest(module) != reference.module_digest:
        raise ValueError(f"workload module {reference.module!r} differs from the driver's signed copy; "
            "restart the driver from identical sources instead of editing modules between signing and spawn")
    cls = _resolve(module, reference.qualname)
    parameters = getattr(cls, "__dataclass_params__", None)
    if not isinstance(cls, type) or parameters is None or not parameters.frozen:
        raise ValueError(f"referenced workload {reference.module}:{reference.qualname} is not a frozen dataclass class")
    hints = typing.get_type_hints(cls)
    arguments: dict[str, object] = {}
    for field in reference.fields:
        match field.kind:
            case WorkloadFieldKind.VALUE:
                hint = hints.get(field.name)
                if hint is None:
                    raise ValueError(f"referenced workload has no annotation for transported field {field.name!r}")
                arguments[field.name] = decode_boundary(field.payload, hint)
            case WorkloadFieldKind.REFERENCE:
                arguments[field.name] = _resolve(importlib.import_module(field.module), field.qualname)
            case WorkloadFieldKind.WORKER_TRANSPORT:
                target = _resolve(importlib.import_module(field.module), field.qualname)
                if not isinstance(target, WorkerTransportable):
                    raise ValueError(f"transport target {field.module}:{field.qualname} is not WorkerTransportable")
                arguments[field.name] = target.transport_rebuild(
                    decode_boundary(field.payload, target.transport_payload_type()), device=device, seed=seed)
    result = cls(**arguments)
    if not isinstance(result, BackendAuthoring):
        raise ValueError(f"referenced workload {reference.module}:{reference.qualname} is not backend authoring")
    return result


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
                    # The reference names an importable frozen class; the driver compares
                    # the rebuilt compilation signature against its own signed plan.
                    authoring = decode_workload(install.definition, device=runner.device,
                        seed=runner.model_config.seed)
                    if not isinstance(authoring, BackendWorkload):
                        raise ValueError("referenced workload is not a backend workload")
                    self._install(cast(BackendWorkload[object, object], authoring), install.attachment)
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
