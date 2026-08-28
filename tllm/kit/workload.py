"""Strongly typed author lifecycle for native workloads."""

from __future__ import annotations

import ast
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType
from typing import ClassVar, Generic, Protocol, TypeVar, cast

import torch

from tllm.contracts.consumer_stats import TrainingStats
from tllm.contracts.hidden_locator import HiddenPhase
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.kit.sampler_output import DeclaredSamplerOutput
from tllm.runtime.sampler_bridge.context import SamplerStepContext
from tllm.runtime.workload_sinks import SinkStore
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    NativeWorkloadResources,
    WorkloadWorkerSpec,
)
from tllm.workload.config import ConfigAtom, WorkloadConfig
from tllm.workload.frontend import (
    NoConfiguredValue,
    PublishedState,
    State,
    SamplerOutput,
    TrainableState,
    WorkloadDeclaration,
    NativeWorkloadDefinition,
    declare_workload,
)
from tllm.workload.ir import OptimizerSpec
from tllm.workload.ir import ValueSourceKind
from tllm.workload.staging import RuntimePortStore, StagedPort
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


class _StateVersionOwner(Protocol):
    @property
    def state_version(self) -> int: ...


class _WorkloadAccounting:
    def __init__(self, owner: _StateVersionOwner) -> None:
        self._owner = owner
        self._losses: list[torch.Tensor] = []
        self._updates = 0

    def state_snapshot(self, _step_id: int) -> Mapping[Hashable, int]:
        return MappingProxyType({None: self._owner.state_version})

    def record(self, loss: torch.Tensor) -> None:
        self._losses.append(loss.detach().float())
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


class Workload(NativeWorkload, Generic[ConfigT]):
    """Native workload with typed config, state, and non-optional public accessors."""

    __tllm_workload__: ClassVar[NativeWorkloadDefinition]
    __tllm_states__: ClassVar[
        tuple[TrainableState[torch.nn.Module] | PublishedState[torch.nn.Module], ...]
    ] = ()
    __tllm_staged_ports__: ClassVar[tuple[StagedPort[object], ...]] = ()
    __tllm_host_outputs__: ClassVar[tuple[HostOutput[object], ...]] = ()

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._lifecycle: _Unspecialized | _Specialized = _Unspecialized()
        self._accounting = _WorkloadAccounting(self)
        self._port_store = RuntimePortStore()
        self._staged_ports_by_key = {
            port.key: port for port in type(self).__tllm_staged_ports__
        }
        self._host_output_store = RuntimeHostOutputStore()
        self._sinks = SinkStore()
        self._sampler_context = SamplerStepContext()
        self._sampler = self._create_sampler_binding()
        self._bind_gradient_tasks()
        self._resources = self._assemble_resources()

    @classmethod
    def _definition(cls) -> NativeWorkloadDefinition:
        return cls.__tllm_workload__

    @classmethod
    def _states(
        cls,
    ) -> tuple[TrainableState[torch.nn.Module] | PublishedState[torch.nn.Module], ...]:
        return cls.__tllm_states__

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
            case _:
                return 0

    def _state_value(self, state: State[StateValueT]) -> StateValueT:
        match self._lifecycle:
            case _Specialized(values=values):
                return cast(StateValueT, values[state.name])
            case _:
                raise RuntimeError(f"State {state.name!r} is not specialized")

    def _bind_gradient_tasks(self) -> None:
        for name in type(self)._definition().gradient_tasks:
            bound = getattr(self, name)

            @wraps(bound)
            def run_with_grad(*args, _bound=bound, **kwargs):
                with torch.inference_mode(False), torch.enable_grad():
                    return _bound(
                        *(value.clone() for value in args),
                        **{key: value.clone() for key, value in kwargs.items()},
                    )

            setattr(self, name, run_with_grad)

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
        with torch.inference_mode(False), torch.enable_grad():
            for state in type(self)._states():
                initializer = getattr(self, state.initializer_name)
                value = cast(torch.nn.Module, initializer(spec))
                values[state.name] = value
                match state:
                    case TrainableState(optimizer=optimizer):
                        optimizers[state.name] = _build_optimizer(
                            cast(OptimizerSpec, optimizer), value, self.config
                        )
                    case PublishedState():
                        pass
        self._lifecycle = _Specialized(
            spec=spec,
            values=values,
            optimizers=optimizers,
            versions={state.name: 0 for state in type(self)._states()},
        )

    def _optimize_state(
        self, state: TrainableState[torch.nn.Module], loss: torch.Tensor
    ) -> None:
        match self._lifecycle:
            case _Specialized(optimizers=optimizers, versions=versions):
                optimizer = optimizers[state.name]
            case _:
                raise RuntimeError(f"State {state.name!r} is not specialized")
        with torch.inference_mode(False), torch.enable_grad():
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        versions[state.name] += 1
        self._accounting.record(loss)

    def _publish_state(
        self,
        state: PublishedState[torch.nn.Module],
        values: Mapping[str, torch.Tensor],
    ) -> None:
        module = self._state_value(state)
        module.load_state_dict(values, strict=True)
        match self._lifecycle:
            case _Specialized(versions=versions):
                versions[state.name] += 1
            case _:
                raise RuntimeError(f"State {state.name!r} is not specialized")

    def residual_paths(self) -> Mapping[ResidualLocator, str]:
        paths: dict[ResidualLocator, str] = {}
        for source in type(self)._definition().hidden_inputs:
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

    def native_workload_resources(self) -> NativeWorkloadResources:
        return self._resources

    def stage_port(self, port: StagedPort[PortValueT], value: PortValueT) -> int:
        if port not in type(self).__tllm_staged_ports__:
            raise ValueError("staged port belongs to another workload")
        return port.stage(self._port_store, value)

    def prepare_port(self, key: str, transaction: str, value: object) -> None:
        self._staged_ports_by_key[key]
        self._port_store.prepare(key, transaction, value)

    def owns_port(self, key: str) -> bool:
        return key in self._staged_ports_by_key

    def commit_port(self, key: str, transaction: str) -> int:
        return self._port_store.commit(key, transaction)

    def abort_port(self, key: str, transaction: str) -> None:
        self._port_store.abort(key, transaction)

    def drain_port(self, port: HostOutput[PortValueT]) -> tuple[PortValueT, ...]:
        if port not in type(self).__tllm_host_outputs__:
            raise ValueError("host output belongs to another workload")
        return port.drain(self._host_output_store)

    def set_enabled(self, enabled: bool) -> None:
        enabled_name = type(self)._definition().enabled.name
        self.config = self.config.replace(**{enabled_name: enabled})
        self._resources = self._assemble_resources()

    def declaration(self) -> WorkloadDeclaration:
        return declare_workload(self)

    @classmethod
    def reconstruct(cls, raw: Mapping[str, object]) -> NativeWorkloadResources:
        config = cast(
            ConfigT,
            cls._definition().config_type.from_wire(
                cast(Mapping[str, ConfigAtom], raw)
            ),
        )
        return cls(config).native_workload_resources()

    def _assemble_resources(self) -> NativeWorkloadResources:
        definition = type(self)._definition()
        enabled = definition.enabled.resolve(self.config)
        state_snapshots = {
            state.name: self._accounting.state_snapshot for state in type(self)._states()
        }
        states = type(self)._states()

        def input_provider(port: StagedPort[object]) -> Callable[[int], object]:
            return lambda step_id: port.take(self._port_store, step_id)

        provided_inputs = {
            port.key: input_provider(port) for port in type(self).__tllm_staged_ports__
        }
        sampler_providers: dict[ValueSourceKind, Callable[[int], object]] = {
            ValueSourceKind.SAMPLER_REQUEST_IDS: self._sampler_context.request_ids,
            ValueSourceKind.SAMPLER_LOGITS: self._sampler_context.logits,
            ValueSourceKind.SAMPLED_TOKENS: self._sampler_context.tokens,
        }
        provided_inputs.update(
            {
                kind.value: sampler_providers[kind]
                for kind in definition.sampler_inputs
            }
        )
        output_sinks: dict[str, Callable[[object, int], None]]
        match self._sampler:
            case _NoSampler():
                output_sinks = {}
                sampler_provider = None
            case _SamplerBinding(policy=policy, provider=sampler):
                def publish_sampler(value: object, step_id: int) -> None:
                    self._sinks.publish(policy.name, cast(torch.Tensor, value), step_id)

                output_sinks = {policy.name: publish_sampler}
                sampler_provider = sampler
        output_sinks.update(
            {
                port.key: port.runtime_publisher(self._host_output_store)
                for port in type(self).__tllm_host_outputs__
            }
        )
        return NativeWorkloadResources(
            workload=self,
            provided_inputs=provided_inputs,
            output_sinks=output_sinks,
            state_snapshots=state_snapshots,
            cold_specializer=self.cold_specialize if states else None,
            diagnostics_reader=self._diagnostics,
            stats_reader=self._accounting.read,
            enabled_setter=self.set_enabled,
            sampler_provider=sampler_provider,
            enabled_predicate=lambda: enabled,
            specialization_key={
                "workload": definition.workload_id,
                **self.config.specialization_key(),
            },
            residual_paths=self.residual_paths(),
            worker_spec=WorkloadWorkerSpec(
                factory=f"{type(self).__module__}:{type(self).__qualname__}.reconstruct",
                config=self.config.to_wire(),
            ),
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
) -> torch.optim.Optimizer:
    parameters = list(value.parameters())
    options: dict[str, object] = {}
    for name, encoded in spec.options:
        if encoded.startswith("field:"):
            options[name] = config.to_wire()[encoded.removeprefix("field:")]
        elif name == "betas":
            options[name] = ast.literal_eval(encoded)
        else:
            options[name] = float.fromhex(encoded)
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[spec.kind]
    return optimizer_type(parameters, foreach=False, **options)  # type: ignore[arg-type]


__all__ = ["Workload"]
