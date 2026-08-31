"""Strongly typed author lifecycle for native workloads."""

from __future__ import annotations

import ast
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Generic, Protocol, TypeVar, cast

import torch

from tllm.contracts.consumer_stats import TrainingStats
from tllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.request_trainable_state import (
    RequestStateLossBatch,
    RequestStateRuntimeService,
    RequestTrainableStateStore,
)
from tllm.ports.sampler import (
    PackedSamplerGuidanceBatch,
    SamplerHiddenReweightBatch,
)
from tllm.kit.sampler_output import DeclaredSamplerOutput
from tllm.kit.backend import (
    NoBackendAdapter,
    WorkloadBackendAdapter,
)
from tllm.runtime.sampler_bridge.context import SamplerStepContext
from tllm.runtime.workload_sinks import SinkStore
from tllm.runtime.parameter_overlay import OverlayChannel, OverlayVersionStore
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    WorkloadWorkerSpec,
)
from tllm.runtime.workload_bindings import (
    LocalRuntimeServices,
    NoRuntimeServices,
    NoSamplerBinding,
    RuntimePortRegistry,
    RuntimeInputBinding,
    RuntimeOutputBinding,
    RuntimeStateRegistry,
    StateSnapshot,
    RuntimeServices,
    SamplerRuntimeBinding,
    WorkerRuntimeBinding,
    WorkloadRuntimeBinding,
)
from tllm.workload.config import ConfigAtom, WorkloadConfig
from tllm.workload.frontend import (
    NoConfiguredValue,
    PublishedModule,
    RequestTrainableState,
    ServiceState,
    State,
    StateDescriptor,
    SamplerOutput,
    TrainableState,
    WorkloadDeclaration,
    NativeWorkloadDefinition,
    declare_workload,
    definition_for_workload_class,
)
from tllm.workload.ir import OptimizerSpec
from tllm.workload.ir import StateStorage
from tllm.workload.ir import StateScope
from tllm.workload.ir import EffectKind
from tllm.workload.ir import ValueSourceKind
from tllm.contracts.workload_ports import (
    PortTransactionStatus,
    StagedLifecycle,
)
from tllm.workload.staging import RuntimePortStore, StagedPort
from tllm.contracts.request_identity import RequestFinishedBatch
from tllm.runtime.parameter_overlay import DenseOperandCaptureBatch
from tllm.workload.host import HostOutput, RuntimeHostOutputStore

ConfigT = TypeVar("ConfigT", bound=WorkloadConfig)
StateValueT = TypeVar("StateValueT", bound=torch.nn.Module)
PortValueT = TypeVar("PortValueT")


@dataclass(frozen=True, slots=True)
class _Unspecialized:
    pass


@dataclass(slots=True)
class _Specialized:
    spec: RuntimeSpecializationSpec
    values: dict[str, torch.nn.Module]
    optimizers: dict[str, torch.optim.Optimizer]
    versions: dict[str, int]
    request_state_services: dict[str, RequestStateRuntimeService[torch.nn.Module]]


class _WorkloadAccounting:
    def __init__(self) -> None:
        self._losses: list[torch.Tensor] = []
        self._updates = 0

    def record(self, loss: torch.Tensor) -> None:
        self._losses.append(loss.detach().float().clone())
        self._updates += 1

    def stats(self) -> Mapping[str, int]:
        return MappingProxyType(
            {"loss_count": len(self._losses), "update_count": self._updates}
        )

    def read(self, sync: bool = True) -> TrainingStats:
        if not self._losses:
            return TrainingStats(loss_avg=0.0, loss_count=0)
        total = torch.stack(self._losses).sum()
        if sync and total.device.type == "cuda":
            torch.cuda.current_stream(total.device).synchronize()
        result = TrainingStats(
            loss_avg=float(total.item()) / len(self._losses),
            loss_count=len(self._losses),
        )
        self._losses.clear()
        return result


@dataclass(frozen=True, slots=True)
class _NoSampler:
    pass


@dataclass(frozen=True, slots=True)
class _SamplerBinding:
    policy: SamplerOutput
    provider: DeclaredSamplerOutput


_SamplerState = _NoSampler | _SamplerBinding


class _OptimizerTransitionOwner(Protocol):
    def execute_optimizer_device_step(
        self, state_name: str, loss: torch.Tensor
    ) -> None: ...

    def commit_optimizer_transition(
        self, state_name: str, loss: torch.Tensor
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _WorkloadOptimizerTransition:
    owner: _OptimizerTransitionOwner
    state_name: str

    def device_step(self, loss: torch.Tensor) -> None:
        self.owner.execute_optimizer_device_step(self.state_name, loss)

    def commit(self, loss: torch.Tensor) -> None:
        self.owner.commit_optimizer_transition(self.state_name, loss)


class Workload(NativeWorkload, Generic[ConfigT]):
    """Native workload with typed config, state, and non-optional public accessors."""

    backend_type: ClassVar[type[WorkloadBackendAdapter]] = NoBackendAdapter

    def __init__(self, config: ConfigT) -> None:
        self._initialize(config, type(self).backend_type.create(config))

    def _initialize(
        self, config: ConfigT, backend: WorkloadBackendAdapter
    ) -> None:
        self.config = config
        self.backend = backend
        self._lifecycle: _Unspecialized | _Specialized = _Unspecialized()
        self._accounting = _WorkloadAccounting()
        self._port_store = RuntimePortStore()
        self._staged_ports_by_key = {
            port.key: port for port in type(self)._staged_ports()
        }
        self._host_output_store = RuntimeHostOutputStore()
        self._sinks = SinkStore()
        self._sampler_context = SamplerStepContext()
        self._sampler = self._create_sampler_binding()
        self._resources = self._assemble_resources()

    @classmethod
    def _definition(cls) -> NativeWorkloadDefinition:
        definition = definition_for_workload_class(cls)
        if not isinstance(definition, NativeWorkloadDefinition):
            raise TypeError(f"{cls.__qualname__} has no native workload definition")
        return definition

    @classmethod
    def _states(
        cls,
    ) -> tuple[StateDescriptor, ...]:
        return cls._definition().states

    @classmethod
    def _staged_ports(cls) -> tuple[StagedPort[object], ...]:
        return cls._definition().staged_ports

    @classmethod
    def _host_outputs(cls) -> tuple[HostOutput[object], ...]:
        return cls._definition().host_outputs

    @property
    def specialization(self) -> RuntimeSpecializationSpec:
        match self._lifecycle:
            case _Specialized(spec=spec):
                return spec
            case _:
                raise RuntimeError(f"{type(self).__name__} is not specialized")

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        match self._lifecycle:
            case _Specialized(optimizers=optimizers) if len(optimizers) == 1:
                return next(iter(optimizers.values()))
            case _Specialized():
                raise RuntimeError(f"{type(self).__name__} has no single optimizer")
            case _:
                raise RuntimeError(f"{type(self).__name__} is not specialized")

    @property
    def state_version(self) -> int:
        match self._lifecycle:
            case _Specialized(versions=versions) if len(versions) == 1:
                return next(iter(versions.values()))
            case _Specialized():
                raise RuntimeError(f"{type(self).__name__} has no single State")
            case _Unspecialized():
                active_state_names = {
                    effect.state
                    for task in declare_workload(self).ir.tasks
                    for effect in task.effects
                    if effect.kind is not EffectKind.EXTERN_IO
                }
                active_replica_states = tuple(
                    state
                    for state in type(self)._states()
                    if (
                        state.scope is not StateScope.REQUEST
                        and state.name in active_state_names
                    )
                )
                if len(active_replica_states) == 1:
                    return 0
                raise RuntimeError(f"{type(self).__name__} has no single State")

    def state_version_for(self, state: State[StateValueT]) -> int:
        match self._lifecycle:
            case _Specialized(versions=versions):
                return versions[state.name]
            case _Unspecialized():
                return 0

    def state_value(self, state: State[StateValueT]) -> StateValueT:
        match self._lifecycle:
            case _Specialized(values=values):
                return cast(StateValueT, values[state.name])
            case _:
                raise RuntimeError(f"State {state.name!r} is not specialized")

    def request_state_value(
        self,
        state: RequestTrainableState[StateValueT],
    ) -> RequestStateRuntimeService[StateValueT]:
        if state not in type(self)._states():
            raise ValueError("request state belongs to another workload")
        match self._lifecycle:
            case _Specialized(request_state_services=services):
                return cast(RequestStateRuntimeService[StateValueT], services[state.name])
            case _:
                raise RuntimeError(f"State {state.name!r} is not specialized")

    def cold_specialize(self, spec: RuntimeSpecializationSpec) -> None:
        match self._lifecycle:
            case _Specialized(spec=current) if current == spec:
                return
            case _Specialized():
                raise RuntimeError(
                    f"{type(self).__name__} is specialized for different runtime facts"
                )
            case _Unspecialized():
                pass

        values: dict[str, torch.nn.Module] = {}
        optimizers: dict[str, torch.optim.Optimizer] = {}
        request_state_services: dict[
            str, RequestStateRuntimeService[torch.nn.Module]
        ] = {}
        active_state_names = {
            effect.state
            for task in declare_workload(self).ir.tasks
            for effect in task.effects
            if effect.kind is not EffectKind.EXTERN_IO
        }
        with torch.inference_mode(False), torch.enable_grad():
            for state in type(self)._states():
                if state.name not in active_state_names:
                    continue
                match state:
                    case RequestTrainableState(optimizer=optimizer, key=key):
                        def module_factory() -> torch.nn.Module:
                            return state.initializer(self, spec)

                        def optimizer_factory(
                            module: torch.nn.Module,
                        ) -> torch.optim.Optimizer:
                            return _build_optimizer(
                                optimizer,
                                module,
                                self.config,
                                spec.device,
                            )

                        request_state_services[state.name] = RequestStateRuntimeService(
                            RequestTrainableStateStore(
                                resource=state.name,
                                key_fields=key.fields,
                                capacity=spec.rows,
                                epoch=1,
                                device=spec.device,
                                module_factory=module_factory,
                                optimizer_factory=optimizer_factory,
                            )
                        )
                    case TrainableState(optimizer=optimizer):
                        value = state.initializer(self, spec)
                        values[state.name] = value
                        optimizers[state.name] = _build_optimizer(
                            optimizer,
                            value,
                            self.config,
                            spec.device,
                        )
                    case PublishedModule():
                        value = state.initializer(self, spec)
                        values[state.name] = value
                        pass
                    case State():
                        value = state.initializer(self, spec)
                        values[state.name] = value
                        pass
        self.backend.specialize(spec)
        self._lifecycle = _Specialized(
            spec=spec,
            values=values,
            optimizers=optimizers,
            versions={
                state.name: 0
                for state in type(self)._states()
                if (
                    state.scope is not StateScope.REQUEST
                    and state.name in active_state_names
                )
            },
            request_state_services=request_state_services,
        )
        self._resources = self._assemble_resources()

    def optimize_state(
        self, state: TrainableState[StateValueT], loss: torch.Tensor
    ) -> None:
        self.execute_optimizer_device_step(state.name, loss)
        self.commit_optimizer_transition(state.name, loss)

    def optimize_request_state(
        self,
        state: RequestTrainableState[StateValueT],
        batch: RequestStateLossBatch,
    ) -> None:
        """Apply a closed per-lease loss batch through its declared service."""

        service = self.request_state_value(state)
        if batch.selection.resource != service.resource:
            raise ValueError("request state loss batch belongs to another state")
        service.optimize(batch.selection, batch.losses)
        for loss in batch.losses:
            self._accounting.record(loss)

    def execute_optimizer_device_step(
        self, state_name: str, loss: torch.Tensor
    ) -> None:
        match self._lifecycle:
            case _Specialized(optimizers=optimizers):
                optimizer = optimizers[state_name]
            case _:
                raise RuntimeError(f"State {state_name!r} is not specialized")
        with torch.inference_mode(False), torch.enable_grad():
            optimizer.zero_grad(set_to_none=False)
            torch.autograd.backward(loss)
            optimizer.step()

    def commit_optimizer_transition(
        self, state_name: str, loss: torch.Tensor
    ) -> None:
        match self._lifecycle:
            case _Specialized(versions=versions):
                versions[state_name] += 1
            case _:
                raise RuntimeError(f"State {state_name!r} is not specialized")
        self._accounting.record(loss)

    def publish_module(
        self,
        state: PublishedModule[torch.nn.Module],
        values: Mapping[str, torch.Tensor],
    ) -> None:
        module = self.state_value(state)
        module.load_state_dict(values, strict=True)
        match self._lifecycle:
            case _Specialized(versions=versions):
                versions[state.name] += 1
            case _:
                raise RuntimeError(f"State {state.name!r} is not specialized")

    def advance_state(self, state: ServiceState[StateValueT]) -> int:
        """Advance service state from inside its signed transition task."""

        if state not in type(self)._states():
            raise ValueError("service state belongs to another workload")
        match self._lifecycle:
            case _Specialized(versions=versions):
                versions[state.name] += 1
                return versions[state.name]
            case _:
                raise RuntimeError(f"State {state.name!r} is not specialized")

    def residual_paths(self) -> Mapping[ResidualLocator, str]:
        active_hidden = {
            (input_.layer, input_.site, input_.phase)
            for task in declare_workload(self).ir.tasks
            for input_ in task.inputs
            if input_.kind is ValueSourceKind.HIDDEN
        }
        paths: dict[ResidualLocator, str] = {}
        for source in type(self)._definition().hidden_inputs:
            if (source.layer, source.site.value, source.phase.value) not in active_hidden:
                continue
            match source.path:
                case NoConfiguredValue():
                    continue
                case configured:
                    module_path = configured.resolve(self.config)
            match source.phase:
                case HiddenPhase.PREFILL:
                    phase = "prefill"
                case HiddenPhase.DECODE:
                    phase = "decode"
                case HiddenPhase.NONE:
                    raise RuntimeError("hidden input has no execution phase")
            paths[
                ResidualLocator(
                    int(source.layer), source.site.value, phase
                )
            ] = module_path
        return MappingProxyType(paths)

    def packed_hidden_resources(
        self,
    ) -> tuple[PackedHiddenResourceBinding, ...]:
        active_packed_hidden = {
            (
                input_.layer,
                HiddenLayerKind(input_.layer_kind),
                input_.site,
            )
            for task in declare_workload(self).ir.tasks
            for input_ in task.inputs
            if input_.kind is ValueSourceKind.PACKED_HIDDEN
        }
        bindings: dict[
            tuple[str, HiddenLayerKind, HiddenSite],
            PackedHiddenResourceBinding,
        ] = {}
        for source in type(self)._definition().packed_hidden_inputs:
            locator = (source.layer, source.layer_kind, source.site)
            if locator not in active_packed_hidden:
                continue
            match source.path:
                case NoConfiguredValue():
                    continue
                case configured:
                    module_path = configured.resolve(self.config)
            binding = PackedHiddenResourceBinding(
                layer=source.layer,
                layer_kind=source.layer_kind,
                site=source.site,
                module_path=module_path,
            )
            if locator in bindings and bindings[locator] != binding:
                raise RuntimeError(
                    "packed hidden locator has conflicting module paths"
                )
            bindings[locator] = binding
        return tuple(sorted(bindings.values(), key=lambda item: item.locator_key))

    def _create_sampler_binding(self) -> _SamplerState:
        outputs = type(self)._definition().sampler_outputs
        if not outputs:
            return _NoSampler()
        policy = outputs[0]
        return _SamplerBinding(
            policy=policy,
            provider=DeclaredSamplerOutput(
                policy, self.config, self._sinks, self._sampler_context
            ),
        )

    def native_workload_resources(self) -> WorkloadRuntimeBinding:
        return self._resources

    def parameter_overlay_bindings(
        self,
    ) -> Mapping[OverlayChannel, OverlayVersionStore]:
        return MappingProxyType({})

    def stage_port(self, port: StagedPort[PortValueT], value: PortValueT) -> int:
        if port not in type(self)._staged_ports():
            raise ValueError("staged port belongs to another workload")
        return port.stage(self._port_store, value)

    def stage_request_finished(self, batch: RequestFinishedBatch) -> int:
        staged = 0
        for port in type(self)._staged_ports():
            if port.lifecycle is StagedLifecycle.REQUEST_FINISHED:
                port.stage(self._port_store, batch)
                staged += 1
        return staged

    def request_finished_port_keys(self) -> tuple[str, ...]:
        return tuple(
            port.key
            for port in type(self)._staged_ports()
            if port.lifecycle is StagedLifecycle.REQUEST_FINISHED
        )

    def stage_dense_operand_captured(
        self, batch: DenseOperandCaptureBatch
    ) -> int:
        staged = 0
        for port in type(self)._staged_ports():
            if port.lifecycle is StagedLifecycle.DENSE_OPERAND_CAPTURED:
                port.stage(self._port_store, batch)
                staged += 1
        return staged

    def prepare_port(self, key: str, transaction: str, value: object) -> None:
        self._staged_ports_by_key[key]
        self._port_store.prepare(key, transaction, value)

    def owns_port(self, key: str) -> bool:
        return key in self._staged_ports_by_key

    def owns_output(self, key: str) -> bool:
        return any(port.key == key for port in type(self)._host_outputs())

    def drain_output(self, key: str) -> tuple[object, ...]:
        port = next(port for port in type(self)._host_outputs() if port.key == key)
        return port.drain(self._host_output_store)

    def commit_port(self, key: str, transaction: str) -> int:
        return self._port_store.commit(key, transaction)

    def abort_port(self, key: str, transaction: str) -> None:
        self._port_store.abort(key, transaction)

    def port_transaction_status(
        self, key: str, transaction: str
    ) -> PortTransactionStatus:
        return self._port_store.transaction_status(key, transaction)

    def acknowledge_remote_port_consumed(
        self, key: str, transaction: str
    ) -> None:
        self._port_store.acknowledge_remote_consumed(key, transaction)

    def drain_port(self, port: HostOutput[PortValueT]) -> tuple[PortValueT, ...]:
        if port not in type(self)._host_outputs():
            raise ValueError("host output belongs to another workload")
        return port.drain(self._host_output_store)

    def set_enabled(self, enabled: bool) -> None:
        enabled_name = type(self)._definition().enabled.name
        self.config = self.config.replace(**{enabled_name: enabled})
        self._resources = self._assemble_resources()

    def declaration(self) -> WorkloadDeclaration:
        return declare_workload(self)

    @classmethod
    def reconstruct(cls, raw: Mapping[str, object]) -> WorkloadRuntimeBinding:
        config_type = cls._definition().config_type
        config_wire = {
            name: raw[name]
            for name in config_type.field_names()
        }
        config = cast(
            ConfigT,
            config_type.from_wire(
                cast(Mapping[str, ConfigAtom], config_wire)
            ),
        )
        instance = cls(config)
        instance.backend = cls.backend_type.reconstruct(config, raw)
        instance._resources = instance._assemble_resources()
        return instance.native_workload_resources()

    def _assemble_resources(self) -> WorkloadRuntimeBinding:
        definition = type(self)._definition()
        active_ir = declare_workload(self).ir
        active_inputs = {
            input_.source for task in active_ir.tasks for input_ in task.inputs
        }
        active_outputs = {
            output.target for task in active_ir.tasks for output in task.outputs
        }
        active_states = {
            effect.state
            for task in active_ir.tasks
            for effect in task.effects
            if effect.kind is not EffectKind.EXTERN_IO
        }

        def snapshot_for(state: State[torch.nn.Module]) -> StateSnapshot:
            def snapshot(_step_id: int) -> Mapping[Hashable, int]:
                return MappingProxyType({None: self.state_version_for(state)})

            return snapshot

        state_snapshots: dict[str, StateSnapshot] = {}
        optimizer_transitions: dict[str, _WorkloadOptimizerTransition] = {}
        for state in type(self)._states():
            match state:
                case State() if state.name in active_states:
                    state_snapshots[state.name] = snapshot_for(state)
                    if state.storage is StateStorage.OPTIMIZER_STABLE:
                        optimizer_transitions[state.name] = _WorkloadOptimizerTransition(
                            self, state.name
                        )
                case RequestTrainableState():
                    pass
                case State():
                    pass
        provided_inputs = {
            port.key: RuntimeInputBinding(
                lambda step_id, port=port: port.take(self._port_store, step_id)
            )
            for port in type(self)._staged_ports()
            if port.key in active_inputs
        }
        sampler_providers: dict[ValueSourceKind, RuntimeInputBinding[object]] = {
            ValueSourceKind.SAMPLER_REQUEST_IDS: RuntimeInputBinding(
                self._sampler_context.request_ids
            ),
            ValueSourceKind.SAMPLER_SAMPLE_IDXS: RuntimeInputBinding(
                self._sampler_context.sample_indices
            ),
            ValueSourceKind.SAMPLER_SAMPLE_IDX_TENSOR: RuntimeInputBinding(
                self._sampler_context.sample_index_tensor
            ),
            ValueSourceKind.SAMPLER_LOGITS: RuntimeInputBinding(
                self._sampler_context.logits
            ),
            ValueSourceKind.SAMPLED_TOKENS: RuntimeInputBinding(
                self._sampler_context.tokens
            ),
        }
        provided_inputs.update(
            {
                kind.value: sampler_providers[kind]
                for kind in definition.sampler_inputs
                if kind.value in active_inputs
            }
        )
        backend_binding = self.backend.runtime_binding()
        output_sinks: dict[str, RuntimeOutputBinding[object]]
        match self._sampler:
            case _NoSampler():
                output_sinks = {}
                sampler_binding = NoSamplerBinding()
            case _SamplerBinding(policy=policy, provider=sampler):
                if policy.name not in active_outputs:
                    output_sinks = {}
                    sampler_binding = NoSamplerBinding()
                else:
                    def publish_sampler(value: object, step_id: int) -> None:
                        match value:
                            case (
                                torch.Tensor()
                                | SamplerHiddenReweightBatch()
                                | PackedSamplerGuidanceBatch()
                            ):
                                self._sinks.publish(
                                    policy.name, value, step_id
                                )
                            case _:
                                raise TypeError(
                                    "sampler output must be a Tensor or "
                                    "SamplerHiddenReweightBatch"
                                )

                    output_sinks = {
                        policy.name: RuntimeOutputBinding(publish_sampler)
                    }
                    sampler_binding = SamplerRuntimeBinding(
                        sampler, self._sampler_context.prepare_identity
                    )
        output_sinks.update(
            {
                port.key: RuntimeOutputBinding(
                    port.runtime_publisher(self._host_output_store)
                )
                for port in type(self)._host_outputs()
                if port.key in active_outputs
            }
        )
        active_backend_outputs = {
            name: binding
            for name, binding in backend_binding.ports.outputs.items()
            if name in active_outputs
        }
        backend_output_overlap = set(output_sinks).intersection(active_backend_outputs)
        if backend_output_overlap:
            raise RuntimeError(
                "workload and backend declare duplicate output bindings: "
                + ", ".join(sorted(backend_output_overlap))
            )
        output_sinks.update(active_backend_outputs)
        active_backend_inputs = {
            name: binding
            for name, binding in backend_binding.ports.inputs.items()
            if name in active_inputs
        }
        services = self.runtime_services()
        match self._lifecycle:
            case _Specialized(request_state_services=all_request_services):
                request_state_services = {
                    name: service
                    for name, service in all_request_services.items()
                    if name in active_states
                }
            case _Unspecialized():
                request_state_services = {}
        return WorkloadRuntimeBinding(
            owner=self,
            ports=RuntimePortRegistry(
                inputs={**provided_inputs, **active_backend_inputs},
                outputs=output_sinks,
            ),
            states=RuntimeStateRegistry(snapshots=state_snapshots),
            services=services,
            sampler=sampler_binding,
            specialization_key={
                "workload": definition.workload_id,
                **self.config.specialization_key(),
                **self.backend.specialization_key(),
            },
            residual_paths=self.residual_paths(),
            packed_hidden_resources=self.packed_hidden_resources(),
            worker=WorkerRuntimeBinding(
                WorkloadWorkerSpec(
                    factory=f"{type(self).__module__}:{type(self).__qualname__}.reconstruct",
                    config={
                        **self.config.to_wire(),
                        **self.backend.worker_wire(),
                    },
                )
            ),
            backend=backend_binding,
            parameter_overlays=self.parameter_overlay_bindings(),
            optimizer_transitions=optimizer_transitions,
            request_state_services=request_state_services,
        )

    def runtime_services(self) -> RuntimeServices:
        enabled = type(self)._definition().enabled.resolve(self.config)
        return (
            LocalRuntimeServices(
                cold_specialize=self.cold_specialize,
                diagnostics=self._diagnostics,
                stats=self._accounting.read,
                set_enabled=self.set_enabled,
                is_enabled=lambda: enabled,
                synchronize=lambda: None,
            )
            if type(self)._states() or self.backend.requires_specialization
            else NoRuntimeServices()
        )

    def _diagnostics(self) -> Mapping[str, int]:
        values = dict(self._accounting.stats())
        match self._sampler:
            case _NoSampler():
                pass
            case _SamplerBinding(policy=policy):
                values[f"{policy.name}_rows"] = self._sinks.latest_rows(policy.name)
        return MappingProxyType(values)


def _build_optimizer(
    spec: OptimizerSpec,
    value: torch.nn.Module,
    config: WorkloadConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    parameters = tuple(value.parameters())
    options = dict(spec.options)
    lr = _optimizer_float(options["lr"], config)
    eps = float.fromhex(options["eps"])
    betas = _optimizer_betas(options["betas"])
    match spec.kind:
        case "adam":
            optimizer: torch.optim.Optimizer = (
                torch.optim.Adam(
                    parameters,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    foreach=False,
                    fused=True,
                    capturable=True,
                )
                if device.type == "cuda"
                else torch.optim.Adam(
                    parameters,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    foreach=False,
                    fused=None,
                )
            )
        case "adamw":
            optimizer = (
                torch.optim.AdamW(
                    parameters,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=float.fromhex(options["weight_decay"]),
                    foreach=False,
                    fused=True,
                    capturable=True,
                )
                if device.type == "cuda"
                else torch.optim.AdamW(
                    parameters,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=float.fromhex(options["weight_decay"]),
                    foreach=False,
                    fused=None,
                )
            )
        case unsupported:
            raise ValueError(f"unsupported optimizer kind {unsupported!r}")
    if device.type == "cuda":
        from tllm.runtime.torch_adapter import initialize_adam_capture_storage

        initialize_adam_capture_storage(optimizer, parameters)
    return optimizer


def _optimizer_float(encoded: str, config: WorkloadConfig) -> float:
    if not encoded.startswith("field:"):
        return float.fromhex(encoded)
    value = config.raw_value(encoded.removeprefix("field:"))
    if type(value) is not float:
        raise TypeError("optimizer float field did not resolve to a float")
    return value


def _optimizer_betas(encoded: str) -> tuple[float, float]:
    value: object = ast.literal_eval(encoded)
    if not isinstance(value, tuple):
        raise TypeError("optimizer betas must be a pair of floats")
    items = cast(tuple[object, ...], value)
    if len(items) != 2 or any(type(item) is not float for item in items):
        raise TypeError("optimizer betas must be a pair of floats")
    return cast(tuple[float, float], items)


__all__ = ["Workload"]
