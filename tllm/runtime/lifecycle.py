#!/usr/bin/env python3
"""Runner patch lifecycle for the residual-stream runtime.

Functions here install and drive the vLLM runner patches. They operate on
the "core" module (``tllm.runtime.residual_runtime``) passed explicitly, so
the module-level patch state (``GPUModelRunner``, ``_ORIG_*``,
``_PATCH_INSTALLED``) stays where the hook modules expect it while the logic
lives here.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

#: Backend drivers are plugins loaded by id at patch-install time. The
#: runtime never statically imports a concrete backend — backends are
#: adapters *above* the runtime, discovered by name like consumer plugins.
_BACKEND_DRIVER_MODULES = {"vllm": "tllm.backends.vllm"}



def install_runner_patch(core: Any, *, backend: str = "vllm") -> None:
    if core._PATCH_INSTALLED:
        return
    driver = importlib.import_module(_BACKEND_DRIVER_MODULES[str(backend)])
    driver.install_backend(core=core)


def make_plain_llm(
    model_name: str,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    enable_prefix_caching: bool,
    enforce_eager: bool = False,
    seed: int | None = None,
    worker_extension_cls: str | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    enable_lora: bool = False,
    max_lora_rank: int | None = None,
    max_loras: int | None = None,
    max_cpu_loras: int | None = None,
) -> Any:
    """Construct a plain vLLM LLM with tLLM's standard single-GPU defaults.

    Canonical home for engine construction (previously ``tllm.util.tools``,
    which re-exports this). The vLLM import stays function-local so importing
    the runtime never pays for vLLM.
    """
    from vllm import LLM

    kwargs: dict[str, Any] = {}
    if seed is not None:
        kwargs["seed"] = int(seed)
    if worker_extension_cls:
        kwargs["worker_extension_cls"] = str(worker_extension_cls)
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = int(max_num_seqs)
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if enable_lora:
        kwargs["enable_lora"] = True
    if max_lora_rank is not None:
        kwargs["max_lora_rank"] = int(max_lora_rank)
    if max_loras is not None:
        kwargs["max_loras"] = int(max_loras)
    if max_cpu_loras is not None:
        kwargs["max_cpu_loras"] = int(max_cpu_loras)
    return LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=int(max_model_len),
        enforce_eager=bool(enforce_eager),
        dtype=cast(Any, dtype),  # vLLM's ModelDType union; validated by vLLM itself
        enable_prefix_caching=bool(enable_prefix_caching),
        **kwargs,
    )


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
    llm = make_plain_llm(*args, **kwargs)  # pyright: ignore[reportArgumentType] — untyped passthrough by design
    # The legacy ``residual_runtime.make_llm`` entry point bypasses
    # ``TllmRuntime._bind_engine_runners``. Bind its parent-side stats/control
    # channel here so mp=1 callers do not silently read the empty parent state.
    from tllm.runtime.mp.control import ControlPlane, set_active_control_plane

    set_active_control_plane(ControlPlane.try_bind(llm))
    return llm
