#!/usr/bin/env python3
"""Runner patch lifecycle for the residual-stream runtime.

Functions here install and drive the vLLM runner patches. They operate on
the "core" module (``tllm.runtime.residual_runtime``) passed explicitly, so
the module-level patch state (``GPUModelRunner``, ``_ORIG_*``,
``_PATCH_INSTALLED``) stays where the hook modules expect it while the logic
lives here.
"""

from __future__ import annotations

from typing import Any


def ensure_runner_binding(core: Any) -> None:
    if (
        core.GPUModelRunner is not None
        and core._ORIG_LOAD_MODEL is not None
        and core._ORIG_PREPARE_INPUTS is not None
        and core._ORIG_EXECUTE_MODEL is not None
    ):
        return

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _GPUModelRunner  # noqa: E402

    core.GPUModelRunner = _GPUModelRunner
    core._ORIG_LOAD_MODEL = _GPUModelRunner.load_model
    core._ORIG_PREPARE_INPUTS = _GPUModelRunner._prepare_inputs
    core._ORIG_EXECUTE_MODEL = _GPUModelRunner.execute_model


def install_runner_patch(core: Any) -> None:
    if core._PATCH_INSTALLED:
        return
    from tllm.backends.vllm import install_backend

    install_backend(core=core)


def make_llm(core: Any, *args: object, **kwargs: object) -> object:
    install_runner_patch(core)
    # Snapshot the runtime for the EngineCore child (multiprocess mode); the
    # tllm general plugin rehydrates from it there. Harmless in-process.
    from tllm.runtime.mp.bootstrap import export_bootstrap_spec_env
    from tllm.runtime.mp.control import TLLM_WORKER_EXTENSION_QUALNAME

    export_bootstrap_spec_env()
    # Mix the tLLM control-plane extension into every worker, in both
    # multiprocess modes, so ControlPlane str-method RPCs resolve identically
    # (in-process under mp=0, EngineCore child under mp=1).
    kwargs.setdefault("worker_extension_cls", TLLM_WORKER_EXTENSION_QUALNAME)
    from tllm.util import tools as _tool_helpers

    llm = _tool_helpers.make_plain_llm(*args, **kwargs)  # pyright: ignore[reportArgumentType] — untyped passthrough by design
    # The legacy ``residual_runtime.make_llm`` entry point bypasses
    # ``TllmRuntime._bind_engine_runners``. Bind its parent-side stats/control
    # channel here so mp=1 callers do not silently read the empty parent state.
    from tllm.runtime.mp.control import ControlPlane, set_active_control_plane

    set_active_control_plane(ControlPlane.try_bind(llm))
    return llm
