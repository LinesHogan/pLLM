"""Process-local binding slots for signed backend-owned extern Regions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


BackendRegionCallback = Callable[[object, Mapping[str, object], int], object]
BackendRegionStager = Callable[["BackendRegionStageContext"], None]


@dataclass(frozen=True, slots=True)
class BackendRegionStageContext:
    """Opaque backend facts for staging one signed extern Region."""

    implementation_id: str
    backend_event: str
    region_id: str
    slot: "BackendRegionBinderSlot"
    runtime: object
    runner: object
    step_id: int
    num_scheduled_tokens: object | None
    input_batch: object | None

    def __post_init__(self) -> None:
        if not self.implementation_id:
            raise TypeError("backend Region stage implementation_id must be non-empty str")
        if not self.backend_event:
            raise TypeError("backend Region stage backend_event must be non-empty str")
        if not self.region_id:
            raise TypeError("backend Region stage region_id must be non-empty str")
        if self.slot.implementation_id != self.implementation_id:
            raise ValueError("backend Region stage slot does not match implementation_id")
        if self.step_id < 0:
            raise ValueError("backend Region binder step_id must be non-negative")


class BackendRegionStagerRegistry:
    """Backend-owned staging implementations keyed by signed identity.

    Slots remain workload-local resource handles.  This registry owns only the
    backend adapter needed to turn opaque backend facts into a staged plan.
    """

    __slots__ = ("_stagers",)

    def __init__(self, stagers: Mapping[str, BackendRegionStager]) -> None:
        if any(not implementation_id for implementation_id in stagers):
            raise TypeError(
                "backend Region stagers must map non-empty implementation IDs to callables"
            )
        self._stagers = MappingProxyType(dict(stagers))

    def stage(self, context: BackendRegionStageContext) -> None:
        try:
            stager = self._stagers[context.implementation_id]
        except KeyError as exc:
            raise RuntimeError(
                "signed backend Region has no worker-local backend stager for "
                f"{context.implementation_id!r}"
            ) from exc
        stager(context)


@dataclass(slots=True)
class _StagedBackendRegion:
    step_id: int
    plan: object
    callback: BackendRegionCallback
    provided_values: dict[str, object] = field(
        default_factory=lambda: dict[str, object]()
    )


@dataclass(frozen=True, slots=True)
class _OpenBackendRegion:
    pass


@dataclass(frozen=True, slots=True)
class _CompletedBackendRegion:
    step_id: int


_BackendRegionState = (
    _OpenBackendRegion | _StagedBackendRegion | _CompletedBackendRegion
)


class BackendRegionBinderSlot:
    """One-shot local handoff for one signed extern implementation identity."""

    __slots__ = (
        "_implementation_id",
        "_provided_inputs",
        "_backend_stager",
        "_state",
    )

    def __init__(
        self,
        implementation_id: str,
        *,
        provided_inputs: Mapping[str, Callable[[int], object]] | None = None,
        backend_stager: BackendRegionStager | None = None,
    ) -> None:
        if not implementation_id:
            raise TypeError("backend Region binder implementation_id must be non-empty str")
        raw_inputs: Mapping[str, Callable[[int], object]] = (
            {} if provided_inputs is None else provided_inputs
        )
        if any(not name for name in raw_inputs):
            raise TypeError(
                "backend Region binder provided_inputs must map non-empty names to callables"
            )
        self._implementation_id = implementation_id
        self._provided_inputs = MappingProxyType(dict(raw_inputs))
        self._backend_stager = backend_stager
        self._state: _BackendRegionState = _OpenBackendRegion()

    @property
    def implementation_id(self) -> str:
        return self._implementation_id

    def stage_from_backend(self, context: BackendRegionStageContext) -> None:
        """Invoke the workload-owned backend adapter for this signed slot."""

        if context.slot is not self or context.implementation_id != self._implementation_id:
            raise ValueError("backend Region stage context does not match binder slot")
        if self._backend_stager is None:
            raise RuntimeError(
                "signed backend Region has no worker-local backend stager for "
                f"{self._implementation_id!r}"
            )
        self._backend_stager(context)

    def stage(
        self,
        plan: object,
        callback: BackendRegionCallback,
        step_id: int,
    ) -> None:
        """Stage exactly one opaque backend plan for one execution step."""

        if step_id < 0:
            raise ValueError("backend Region binder step_id must be non-negative")
        step = step_id
        match self._state:
            case _StagedBackendRegion(step_id=current_step):
                if current_step == step:
                    raise RuntimeError(
                        "duplicate backend Region stage for "
                        f"{self._implementation_id!r} at step {step}"
                    )
                raise RuntimeError(
                    "stale backend Region stage for "
                    f"{self._implementation_id!r}: step {current_step} remains staged"
                )
            case _CompletedBackendRegion(step_id=completed) if step <= completed:
                if step == completed:
                    raise RuntimeError(
                        "backend Region stage already completed for "
                        f"{self._implementation_id!r} at step {step}"
                    )
                raise RuntimeError(
                    "stale backend Region stage for "
                    f"{self._implementation_id!r}: completed step is {completed}"
                )
            case _OpenBackendRegion() | _CompletedBackendRegion():
                self._state = _StagedBackendRegion(step, plan, callback)

    def _require_staged(self, step_id: int) -> _StagedBackendRegion:
        if step_id < 0:
            raise ValueError("backend Region binder step_id must be non-negative")
        step = step_id
        match self._state:
            case _StagedBackendRegion(step_id=staged_step) as staged:
                if staged_step != step:
                    raise RuntimeError(
                        "backend Region is staged for step "
                        f"{staged_step}, not requested step {step}"
                    )
                return staged
            case _CompletedBackendRegion(step_id=completed) if completed == step:
                raise RuntimeError(
                    "backend Region stage already completed for "
                    f"{self._implementation_id!r} at step {step}"
                )
            case _OpenBackendRegion() | _CompletedBackendRegion():
                raise RuntimeError(
                    "missing staged backend Region for "
                    f"{self._implementation_id!r} at step {step}"
                )

    def provided_input(self, name: str, step_id: int) -> object:
        """Return one lazily materialized provider value for the staged step."""

        if not name:
            raise TypeError("backend Region binder provided input name must be non-empty str")
        staged = self._require_staged(step_id)
        try:
            provider = self._provided_inputs[name]
        except KeyError as exc:
            raise RuntimeError(
                "backend Region binder has no provided input "
                f"{name!r} for {self._implementation_id!r}"
            ) from exc
        if name not in staged.provided_values:
            staged.provided_values[name] = provider(staged.step_id)
        return staged.provided_values[name]

    def plan_for_step(self, step_id: int) -> object:
        """Return the staged opaque backend plan without consuming the slot."""

        return self._require_staged(step_id).plan

    def provided_inputs(self, step_id: int) -> Mapping[str, object]:
        """Return the complete immutable provided-input view for the staged step."""

        self._require_staged(step_id)
        return MappingProxyType(
            {
                name: self.provided_input(name, step_id)
                for name in self._provided_inputs
            }
        )

    def clear(self, step_id: int) -> None:
        """Consume a staged plan without invoking its callback."""

        staged = self._require_staged(step_id)
        self._state = _CompletedBackendRegion(staged.step_id)

    def abort(self, step_id: int) -> bool:
        """Retire an uninvoked stage after its enclosing backend step failed."""

        if step_id < 0:
            raise ValueError("backend Region binder step_id must be non-negative")
        step = step_id
        match self._state:
            case _StagedBackendRegion(step_id=staged_step):
                if staged_step != step:
                    raise RuntimeError(
                        "cannot abort backend Region staged for step "
                        f"{staged_step} from failed step {step}"
                    )
                self._state = _CompletedBackendRegion(step)
                return True
            case _OpenBackendRegion() | _CompletedBackendRegion():
                return False

    def invoke(self, step_id: int) -> object:
        """Invoke and consume the one staged plan for ``step_id`` exactly once."""

        staged = self._require_staged(step_id)
        inputs = self.provided_inputs(staged.step_id)
        # A backend callback can fail, but retrying it would duplicate a
        # possibly side-effectful Region. Consume before crossing that boundary.
        self._state = _CompletedBackendRegion(staged.step_id)
        return staged.callback(staged.plan, inputs, staged.step_id)


__all__ = [
    "BackendRegionBinderSlot",
    "BackendRegionStageContext",
    "BackendRegionStager",
    "BackendRegionStagerRegistry",
]
