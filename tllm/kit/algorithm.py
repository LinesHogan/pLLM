"""``RegionAlgorithm``: the ergonomics base for native Region workloads.

Design goal: an algorithm author writes only the algorithm — config
validation, parameter initialization, task bodies, and any publication or
sampler logic.  Everything mechanical is owned here once:

- specialization key assembly from the config dataclass (derived from all
  fields by default; ``str``/``int``/``bool``/``float.hex`` encoding),
- ``NativeWorkloadResources`` assembly (state snapshots, stats readers,
  enabled epoch snapshots, worker spec, residual manifest),
- ``cold_specialize`` guard preamble plus parameter/optimizer allocation
  driven by the signed ``State(parameters=..., optimizer=...)`` declaration,
- the ``update`` finalization (zero_grad/backward/step, version bump, stats),
- sink publication storage with runtime-attached row identity, and the
  sampler bridge (authors implement ``modify_candidates`` only),
- grad context for tasks whose signed effects read a trainable State,
- ``build``/``reconstruct``/worker-factory wiring over ``dataclass_json``.

Weight-overlay outlook: the optimizer is built from the signed
``OptimizerSpec`` (open ``kind`` + sorted ``options``) through the single
:meth:`_build_optimizer` seam, so a future fast-weight/overlay update rule
plugs in without touching task signatures or the State contract.

The produced declarations are byte-identical to hand-written ones: the kit
only mechanizes process-local lifecycle code, never the signed surface.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Callable, Hashable, Mapping
from functools import partial, wraps
from types import MappingProxyType
from typing import ClassVar, Generic, TypeVar

import torch

from tllm.common.dataclass_json import dataclass_from_dict, dataclass_to_jsonable
from tllm.kit.resources import RegionResourcesBase
from tllm.kit.sampler import SamplerModifier
from tllm.ports.sampler import CandidateModifierState
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    NativeWorkloadResources,
    WorkloadWorkerSpec,
)
from tllm.workload import EffectKind, State
from tllm.workload.frontend import SinkOutput, TaskDecl

ConfigT = TypeVar("ConfigT")


def _encode_specialization_value(value: object, label: str) -> str:
    """Encode one config field with the established key convention."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        # Layer-index sets and similar small int tuples encode order-sensitively.
        return ",".join(str(item) for item in value)
    raise TypeError(
        f"specialization field {label!r} must be bool/int/float/str/tuple[int, ...]"
    )


def _optimizer_options(spec_options: tuple[tuple[str, str], ...]) -> dict[str, object]:
    """Decode signed optimizer options (hex floats / repr tuples)."""
    decoded: dict[str, object] = {}
    for key, value in spec_options:
        if key == "betas":
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, tuple):
                raise TypeError("optimizer betas option must encode a tuple")
            decoded[key] = parsed
        else:
            decoded[key] = float.fromhex(value)
    return decoded


class RegionAlgorithm(NativeWorkload, Generic[ConfigT]):
    """Base class carrying the native Region workload lifecycle.

    Subclass contract (all hooks optional unless noted):

    - ``algorithm`` / ``config_type``: required class attributes.
    - ``specialization_fields``: specialization-key name -> config field name.
    - ``state_name``: trainable ``State`` attribute; auto-detected when the
      class declares exactly one State.
    - ``validate_config``: reject unsupported configuration early.
    - ``init_parameters``: allocate the declared parameters/modules for one
      cold specialization (required when a trainable state exists).
    - ``provided_inputs`` / ``output_sinks`` / ``residual_paths``: resource
      bindings (default empty).
    - ``enabled_field``: config field toggled by ``set_enabled`` epochs.
    """

    algorithm: ClassVar[str] = ""
    config_type: ClassVar[type | None] = None
    state_name: ClassVar[str | None] = None
    # None (the default) derives the specialization key from every config
    # dataclass field, minus specialization_exclude.  An explicit mapping
    # (key name -> config field name) remains available for legacy surfaces.
    specialization_fields: ClassVar[Mapping[str, str] | None] = None
    specialization_exclude: ClassVar[tuple[str, ...]] = ()
    resources_type: ClassVar[type[RegionResourcesBase] | None] = RegionResourcesBase
    sampler_provider_type: ClassVar[type | None] = None
    enabled_field: ClassVar[str | None] = None
    worker_factory: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.algorithm:
            raise TypeError(f"{cls.__name__} must declare a non-empty algorithm tag")
        if cls.config_type is None:
            raise TypeError(f"{cls.__name__} must declare config_type")

    def __init__(self, config: ConfigT) -> None:
        config_type = type(self).config_type
        if config_type is None or not isinstance(config, config_type):
            raise TypeError(f"config must be {config_type}")
        self.validate_config(config)
        self.config = config
        self._parameters: dict[str, object] | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._specialization: RuntimeSpecializationSpec | None = None
        self._state_version = 0
        self._resources = self.make_resources()
        self._sampler_provider = self._make_sampler_provider()
        self._wrap_trainable_read_tasks()
        self._native_resources = self._assemble_resources()

    def _make_sampler_provider(self) -> object:
        provider_type = type(self).sampler_provider_type
        modifier_overridden = (
            type(self).modify_candidates is not RegionAlgorithm.modify_candidates
        )
        if provider_type is not None:
            if modifier_overridden:
                raise RuntimeError(
                    f"{type(self).__name__} must declare either "
                    "sampler_provider_type or modify_candidates, not both"
                )
            return provider_type(self)
        if not modifier_overridden:
            return None
        sinks = self._declared_sink_names()
        if len(sinks) != 1:
            raise RuntimeError(
                f"{type(self).__name__} implements modify_candidates and must "
                f"declare exactly one sink output (found {len(sinks)})"
            )
        return SamplerModifier(self, sinks[0])

    # -- author hooks ----------------------------------------------------

    def validate_config(self, config: ConfigT) -> None:
        """Reject unsupported configuration before any resource exists."""

    def make_resources(self) -> object:
        resources_type = type(self).resources_type
        return None if resources_type is None else resources_type(self)

    def provided_inputs(self) -> Mapping[str, Callable[[int], object]]:
        return MappingProxyType({})

    def output_sinks(self) -> Mapping[str, Callable[[object, int], None]]:
        return MappingProxyType({})

    def residual_paths(self) -> Mapping[object, str]:
        return MappingProxyType({})

    def extra_specialization(self) -> Mapping[str, str]:
        """Derived specialization-key entries beyond plain config fields."""
        return MappingProxyType({})

    def extra_resource_fields(self) -> Mapping[str, object]:
        """Additional NativeWorkloadResources keyword arguments."""
        return MappingProxyType({})

    def diagnostics_reader(self) -> Callable[[], Mapping[str, int | float]] | None:
        resources = self._resources
        if isinstance(resources, RegionResourcesBase):
            return resources.stats
        return None

    def synchronize(self) -> None:
        """Drain asynchronous work owned by this workload (default: none)."""

    def init_parameters(
        self, spec: RuntimeSpecializationSpec
    ) -> Mapping[str, torch.nn.Parameter | torch.nn.Module]:
        raise NotImplementedError(
            f"{type(self).__name__} declares a trainable State and must implement "
            "init_parameters()"
        )

    def optimizer_overrides(self) -> Mapping[str, object]:
        """Physical optimizer kwargs overriding the nominal signed options."""
        return MappingProxyType({})

    def extra_stats(self) -> Mapping[str, int]:
        """Algorithm-specific counters merged into the resources stats."""
        return MappingProxyType({})

    def modifier_active(self) -> bool:
        """Extra sampler-modifier predicate beyond the signed enabled flag."""
        return True

    def modify_candidates(
        self, value: torch.Tensor, view: SamplerStepView
    ) -> CandidateModifierState:
        """Map this step's sink publication to the sampler modifier payload.

        Overriding this hook installs the framework-owned SamplerModifier:
        the enabled check, publication lookup, and row/phase/device alignment
        are all owned by the kit.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not modify sampler candidates"
        )

    # -- resource exposure -----------------------------------------------

    @property
    def resources(self) -> object:
        return self._resources

    @property
    def sampler_provider(self) -> object:
        return self._sampler_provider

    @property
    def params(self) -> Mapping[str, object]:
        parameters = self._parameters
        if parameters is None:
            raise RuntimeError(
                f"{type(self).__name__} must be cold_specialize()d before execution"
            )
        return MappingProxyType(parameters)

    def param(self, name: str) -> torch.nn.Parameter | torch.nn.Module:
        """One declared State parameter; never None, never Optional-typed."""
        try:
            value = self.params[name]
        except KeyError as exc:
            raise RuntimeError(
                f"{type(self).__name__} has no signed parameter {name!r}"
            ) from exc
        if not isinstance(value, (torch.nn.Parameter, torch.nn.Module)):
            raise RuntimeError(
                f"{type(self).__name__} parameter {name!r} is not a Parameter/Module"
            )
        return value

    @property
    def model(self) -> torch.nn.Module:
        """The single Module-valued State parameter (raises otherwise)."""
        modules = [
            value
            for value in self.params.values()
            if isinstance(value, torch.nn.Module)
        ]
        if len(modules) != 1:
            raise RuntimeError(
                f"{type(self).__name__} must declare exactly one Module parameter "
                f"to use .model (found {len(modules)})"
            )
        return modules[0]

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        return self._optimizer

    @property
    def specialization(self) -> RuntimeSpecializationSpec | None:
        return self._specialization

    @property
    def state_version(self) -> int:
        return self._state_version

    def native_workload_resources(self) -> NativeWorkloadResources:
        return self._native_resources

    # -- specialization identity -------------------------------------------

    def _specialization_key(self) -> Mapping[str, str]:
        key: dict[str, str] = {"algorithm": type(self).algorithm}
        declared = type(self).specialization_fields
        if declared is None:
            excluded = frozenset(type(self).specialization_exclude)
            try:
                config_fields = dataclasses.fields(self.config)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError(
                    f"{type(self).__name__} config must be a dataclass for the "
                    "derived specialization key (or set specialization_fields)"
                ) from exc
            for field in config_fields:
                if field.name in excluded:
                    continue
                key[field.name] = _encode_specialization_value(
                    getattr(self.config, field.name), field.name
                )
        else:
            for name, field_name in declared.items():
                key[name] = _encode_specialization_value(
                    getattr(self.config, field_name), field_name
                )
        for name, value in self.extra_specialization().items():
            if not isinstance(value, str) or not value:
                raise TypeError("extra specialization values must be non-empty str")
            key[name] = value
        return key

    def _config_wire(self) -> Mapping[str, object]:
        raw = dataclass_to_jsonable(self.config)  # type: ignore[arg-type]
        if not isinstance(raw, dict):
            raise TypeError(f"{type(self).__name__} config did not encode as a JSON object")
        return raw

    def _factory(self) -> str:
        override = type(self).worker_factory
        if override is not None:
            return override
        return f"{type(self).__module__}:{type(self).__qualname__}.reconstruct"

    @classmethod
    def reconstruct(cls, raw: Mapping[str, object]) -> NativeWorkloadResources:
        """Spawn-safe worker factory bound by the default worker spec."""
        config = dataclass_from_dict(cls.config_type, dict(raw))  # type: ignore[arg-type]
        return cls(config).native_workload_resources()

    # -- resource assembly ---------------------------------------------------

    def _declared_trainable_state(self) -> State | None:
        declared: dict[str, State] = {}
        for klass in reversed(type(self).__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, State):
                    declared[name] = value
        explicit = type(self).state_name
        if explicit is not None:
            try:
                return declared[explicit]
            except KeyError as exc:
                raise RuntimeError(
                    f"{type(self).__name__} has no declared State named {explicit!r}"
                ) from exc
        if len(declared) == 1:
            return next(iter(declared.values()))
        if declared:
            raise RuntimeError(
                f"{type(self).__name__} declares multiple States; set state_name"
            )
        return None

    def _trainable_state_resource(self, state: State) -> str:
        return state.name

    def _declared_sink_names(self) -> tuple[str, ...]:
        """Sink output names from the signed task declarations (class order)."""
        names: list[str] = []
        for klass in reversed(type(self).__mro__):
            for member in vars(klass).values():
                try:
                    declaration = member.__tllm_task__
                except AttributeError:
                    continue
                if not isinstance(declaration, TaskDecl):
                    continue
                for name, output in declaration.outputs:
                    if isinstance(output, SinkOutput) and name not in names:
                        names.append(name)
        return tuple(names)

    def _wrap_trainable_read_tasks(self) -> None:
        """Run tasks that read a trainable State under grad, automatically.

        The signed effects already say which tasks touch trainable
        parameters, so the ``inference_mode(False) / enable_grad`` preamble
        is owned here instead of inside every task body.
        """
        state = self._declared_trainable_state()
        if state is None or state.optimizer is None:
            return
        for klass in type(self).__mro__:
            for name, member in vars(klass).items():
                try:
                    declaration = member.__tllm_task__
                except AttributeError:
                    continue
                if not isinstance(declaration, TaskDecl):
                    continue
                if not any(
                    effect.kind is EffectKind.READ and effect.state is state
                    for effect in declaration.effects
                ):
                    continue
                if name in self.__dict__:
                    continue  # already wrapped (MRO re-visit)
                bound = getattr(self, name)

                @wraps(bound)
                def grad_wrapped(
                    *args: object, _bound: Callable[..., object] = bound, **kwargs: object
                ) -> object:
                    with torch.inference_mode(False), torch.enable_grad():
                        return _bound(*args, **kwargs)

                setattr(self, name, grad_wrapped)

    def _assemble_resources(self) -> NativeWorkloadResources:
        state = self._declared_trainable_state()
        enabled_field = type(self).enabled_field
        extras = dict(self.extra_resource_fields())
        output_sinks = dict(self.output_sinks())
        declared_sinks = self._declared_sink_names()
        if declared_sinks:
            # Declared sinks publish into the kit SinkStore by default; the
            # runtime attaches the row identity at publish time.
            resources = self._resources
            if not isinstance(resources, RegionResourcesBase):
                raise RuntimeError(
                    f"{type(self).__name__} declares sink outputs and must use "
                    "a RegionResourcesBase resources object"
                )
            for name in declared_sinks:
                if name in output_sinks:
                    continue  # explicit author binding wins (transitional)
                output_sinks[name] = partial(resources.sinks.publish, name)
        kwargs: dict[str, object] = {
            "workload": self,
            "provided_inputs": self.provided_inputs(),
            "output_sinks": output_sinks,
            "sampler_provider": self._sampler_provider,
            "synchronizer": self.synchronize,
            "specialization_key": self._specialization_key(),
            "residual_paths": self.residual_paths(),
            "worker_spec": WorkloadWorkerSpec(
                factory=self._factory(),
                config=self._config_wire(),
            ),
        }
        diagnostics = self.diagnostics_reader()
        if diagnostics is not None:
            kwargs["diagnostics_reader"] = diagnostics
        if state is not None:
            resources = self._resources
            if not isinstance(resources, RegionResourcesBase):
                raise RuntimeError(
                    f"{type(self).__name__} declares a trainable State and must use "
                    "a RegionResourcesBase resources object"
                )
            kwargs["state_snapshots"] = {
                self._trainable_state_resource(state): resources.runtime_state_snapshot
            }
            kwargs["cold_specializer"] = self.cold_specialize
            kwargs["stats_reader"] = resources.read_and_reset_stats
        if enabled_field is not None:
            # The predicate belongs to this immutable resource snapshot; a
            # later enable transition creates a new signed Region epoch.
            enabled_snapshot = bool(getattr(self.config, enabled_field))
            kwargs["enabled_setter"] = self.set_enabled
            kwargs["enabled_predicate"] = (
                lambda enabled=enabled_snapshot: enabled
            )
        kwargs.update(extras)
        return NativeWorkloadResources(**kwargs)  # type: ignore[arg-type]

    def set_enabled(self, enabled: bool) -> None:
        enabled_field = type(self).enabled_field
        if enabled_field is None:
            raise RuntimeError(f"{type(self).__name__} has no enabled_field")
        setattr(self.config, enabled_field, bool(enabled))
        self._native_resources = self._assemble_resources()

    # -- cold specialization --------------------------------------------------

    def cold_specialize(self, spec: RuntimeSpecializationSpec) -> None:
        """Allocate declared parameters and optimizer outside inference mode."""
        if not isinstance(spec, RuntimeSpecializationSpec):
            raise TypeError("spec must be RuntimeSpecializationSpec")
        state = self._declared_trainable_state()
        if state is None:
            raise RuntimeError(f"{type(self).__name__} declares no trainable State")
        previous = self._specialization
        if previous is not None:
            if previous != spec:
                raise RuntimeError(
                    f"{type(self).__name__} is already specialized for another "
                    "static shape, dtype, or device"
                )
            return
        with torch.inference_mode(False):
            with torch.enable_grad():
                allocated: dict[str, object] = dict(self.init_parameters(spec))
                self._check_parameter_names(allocated, state)
                optimizer = self._build_optimizer(state, allocated)
        self._parameters = allocated
        self._optimizer = optimizer
        self._specialization = spec

    def _check_parameter_names(
        self, allocated: Mapping[str, object], state: State
    ) -> None:
        expected = set(state.parameters)
        if set(allocated) != expected:
            raise RuntimeError(
                f"{type(self).__name__} init_parameters must allocate exactly the "
                f"signed State parameters {sorted(expected)}"
            )

    def _build_optimizer(
        self, state: State, allocated: Mapping[str, object]
    ) -> torch.optim.Optimizer | None:
        spec = state.optimizer
        if spec is None:
            return None
        parameters: list[torch.nn.Parameter] = []
        for value in allocated.values():
            if isinstance(value, torch.nn.Parameter):
                parameters.append(value)
            elif isinstance(value, torch.nn.Module):
                parameters.extend(value.parameters())
            else:
                raise TypeError(
                    "init_parameters values must be Parameter or nn.Module"
                )
        kwargs = _optimizer_options(spec.options)
        kwargs.update(self.optimizer_overrides())
        # foreach=False keeps the physical step deterministic and matches the
        # established single-group Region training semantics.
        if spec.kind == "adam":
            return torch.optim.Adam(parameters, foreach=False, **kwargs)  # type: ignore[arg-type]
        if spec.kind == "adamw":
            return torch.optim.AdamW(parameters, foreach=False, **kwargs)  # type: ignore[arg-type]
        raise RuntimeError(
            f"unsupported signed optimizer kind {spec.kind!r}; register the new "
            "kind in RegionAlgorithm._build_optimizer (weight-overlay seam)"
        )

    # -- update finalization ---------------------------------------------------

    def commit_state_update(self) -> None:
        """Land a host-pushed parameter refresh as one signed state transition.

        Unlike :meth:`step_update` there is no loss or optimizer: an external
        training loop published new parameter values (the weight-overlay
        seam), and the signed transition Region calls this after copying them
        into the State parameters so the audit observes the version bump.
        """
        if self._parameters is None:
            raise RuntimeError(
                f"{type(self).__name__} must be cold_specialize()d before execution"
            )
        self._state_version += 1

    def step_update(self, loss: torch.Tensor) -> None:
        """One signed optimizer step: backward, step, version bump, stats."""
        if self._parameters is None or self._optimizer is None:
            raise RuntimeError(
                f"{type(self).__name__} must be cold_specialize()d before execution"
            )
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            raise ValueError(
                f"{type(self).__name__} update requires a differentiable loss Tensor"
            )
        with torch.inference_mode(False):
            with torch.enable_grad():
                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._optimizer.step()
        self._state_version += 1
        resources = self._resources
        if isinstance(resources, RegionResourcesBase):
            resources.record_update(loss)


__all__ = ["RegionAlgorithm"]
