#!/usr/bin/env python3
"""Compatibility probes for PyTorch compilation state at the hook boundary."""

from __future__ import annotations

from types import FunctionType, MethodType

import torch


def is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        fn = getattr(compiler, "is_compiling", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        fn = getattr(dynamo, "is_compiling", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass
    return False


def install_capture_compile_identity(
    *,
    model: torch.nn.Module,
    layer_modules: tuple[torch.nn.Module, ...],
    fingerprint: str,
) -> None:
    """Salt vLLM's AOT identity for instance-level capture wrappers."""
    for module in model.modules():
        if not callable(getattr(module, "aot_compile", None)):
            continue
        descendants = tuple(module.modules())
        if not all(any(candidate is layer for candidate in descendants) for layer in layer_modules):
            continue
        installed_identity = getattr(module, "_tllm_capture_compile_identity", None)
        if installed_identity is not None:
            if installed_identity == fingerprint:
                return
            raise RuntimeError(
                "vLLM compile owner already has a different tLLM capture plan"
            )
        if getattr(module, "aot_compiled_fn", None) is not None or bool(
            getattr(module, "compiled", False)
        ):
            raise RuntimeError(
                "capture-resident producer must be installed before vLLM compiles the model"
            )
        bound_forward = module.forward
        function = getattr(bound_forward, "__func__", None)
        if function is None:
            raise RuntimeError("vLLM compiled model forward is not a bound Python method")
        cloned = FunctionType(
            function.__code__,
            function.__globals__,
            name=function.__name__,
            argdefs=function.__defaults__,
            closure=function.__closure__,
        )
        cloned.__kwdefaults__ = function.__kwdefaults__
        cloned.__annotations__ = dict(getattr(function, "__annotations__", {}))
        cloned.__module__ = function.__module__
        cloned.__qualname__ = f"{function.__qualname__}__tllm_capture_{fingerprint}"
        module.forward = MethodType(cloned, module)  # type: ignore[method-assign]
        setattr(module, "_tllm_capture_compile_identity", fingerprint)
        return
    raise RuntimeError(
        "capture-resident producer could not locate the vLLM torch-compile owner"
    )
