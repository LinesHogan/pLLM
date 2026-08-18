#!/usr/bin/env python3
"""Instance-scoped tLLM runtime.

``TllmRuntime`` is the recommended entry point:

    runtime = TllmRuntime(consumers=[EsampConsumer(...)])
    llm = runtime.make_llm(model_name=..., dtype=..., ...)

Module-level ``tllm.make_llm`` / ``tllm.register_consumer`` remain as a thin
façade over a process-wide default runtime for backward compatibility.

Current limitation (tracked by the runtime-instancing migration): backend
hooks are installed on engine classes once per process, and the active
runtime's state is what those hooks see. Creating two engines with
*different* runtimes in one process is not supported yet; the runner ->
runtime routing table (:func:`bind_runner_runtime`) is the seam where
per-runner dispatch will land.
"""

from __future__ import annotations

from typing import Any, Sequence
import weakref

from tllm.runtime.config import TllmRuntimeConfig


# runner -> owning TllmRuntime. Weak on both sides: runners and runtimes are
# collected freely; stale entries vanish.
_RUNNER_RUNTIMES: "weakref.WeakKeyDictionary[Any, TllmRuntime]" = weakref.WeakKeyDictionary()

# The runtime the process-wide class hooks currently dispatch to.
_ACTIVE_RUNTIME: "TllmRuntime | None" = None


def bind_runner_runtime(runner: Any, runtime: "TllmRuntime") -> None:
    _RUNNER_RUNTIMES[runner] = runtime


def resolve_runner_runtime(runner: Any) -> "TllmRuntime | None":
    try:
        return _RUNNER_RUNTIMES.get(runner)
    except TypeError:
        return None


def active_runtime() -> "TllmRuntime":
    if _ACTIVE_RUNTIME is None:
        return default_runtime()
    return _ACTIVE_RUNTIME


class TllmRuntime:
    def __init__(
        self,
        *,
        backend: str = "auto",
        producers: Sequence[Any] = (),
        consumers: Sequence[Any] = (),
        config: TllmRuntimeConfig | None = None,
    ) -> None:
        if backend != "auto":
            raise ValueError(
                f"backend={backend!r}: only 'auto' is supported; backend selection "
                "is structural (tllm.backends)"
            )
        self.config = config or TllmRuntimeConfig()
        self.producers = tuple(producers)
        self._consumers: list[Any] = list(consumers)

    @property
    def consumers(self) -> tuple[Any, ...]:
        return tuple(self._consumers)

    def register_consumer(self, consumer: Any) -> None:
        plane = None
        spec = None
        if active_runtime() is self:
            from tllm.runtime.mp.control import active_control_plane

            plane = active_control_plane()
            if plane is not None:
                from tllm.consumers.registry import spec_for_consumer

                spec = spec_for_consumer(consumer)
        self._consumers.append(consumer)
        if active_runtime() is self:
            from tllm.runtime import residual_runtime
            from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec

            residual_runtime.register_dispatch_consumer(consumer)
            refresh_exported_bootstrap_spec()
            if plane is not None and spec is not None:
                plane.register_consumer(spec.name, spec.config)

    def clear_consumers(self) -> None:
        self._consumers.clear()
        if active_runtime() is self:
            from tllm.runtime import residual_runtime
            from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec
            from tllm.runtime.mp.control import active_control_plane

            residual_runtime.clear_dispatch_consumers()
            refresh_exported_bootstrap_spec()
            plane = active_control_plane()
            if plane is not None:
                plane.clear_consumers()

    def activate(self) -> None:
        """Make this runtime the one global hooks dispatch to.

        A runtime with no consumers of its own leaves any consumers already
        registered through the legacy façade untouched (backward compat).
        """
        global _ACTIVE_RUNTIME
        _ACTIVE_RUNTIME = self
        self._apply_consumers()

    def _apply_consumers(self) -> None:
        if not self._consumers:
            return
        from tllm.runtime import residual_runtime

        residual_runtime.clear_dispatch_consumers()
        for consumer in self._consumers:
            residual_runtime.register_dispatch_consumer(consumer)

    def make_llm(self, *args: Any, **kwargs: Any) -> Any:
        from tllm.runtime import residual_runtime
        from tllm.runtime.mp.control import set_active_control_plane

        set_active_control_plane(None)
        self.activate()
        llm = residual_runtime.make_llm(*args, **kwargs)
        self._bind_engine_runners(llm)
        return llm

    def _bind_engine_runners(self, llm: Any) -> None:
        """Record runner -> runtime routing for every live worker runner."""
        from tllm.runtime.mp.control import ControlPlane, set_active_control_plane

        plane = ControlPlane.try_bind(llm)
        if plane is not None:
            try:
                plane.call("tllm_bind_runner_runtime")
                set_active_control_plane(plane)
                return
            except Exception:
                set_active_control_plane(None)
                pass
        # Fallback for engines without a collective_rpc path (test doubles):
        # walk the in-process executor directly.
        def _bind(worker: Any) -> None:
            runner = getattr(worker, "model_runner", None)
            if runner is not None:
                bind_runner_runtime(runner, self)

        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        executor = getattr(core, "model_executor", None)
        driver = getattr(executor, "driver_worker", None)
        if driver is not None:
            _bind(driver)


_DEFAULT_RUNTIME: "TllmRuntime | None" = None


def default_runtime() -> TllmRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = TllmRuntime()
    return _DEFAULT_RUNTIME
