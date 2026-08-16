#!/usr/bin/env python3
"""vLLM backend installation entry point.

Probes the installed vLLM structurally, selects matching drivers, and
installs their class hooks. Unknown structures fail fast.
"""

from __future__ import annotations

from typing import Any

from tllm.backends.protocol import UnsupportedBackendError
from tllm.backends.vllm.detect import VllmBackendInfo, probe_vllm
from tllm.backends.vllm.legacy_mrv1 import LegacyMRv1Driver
from tllm.backends.vllm.v027_mrv1 import V027MRv1Driver
from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver


def select_drivers(info: VllmBackendInfo) -> list[Any]:
    drivers: list[Any] = []
    v027_mrv1 = V027MRv1Driver()
    v027_mrv2 = V027MRv2Driver()
    legacy = LegacyMRv1Driver()

    if v027_mrv1.supports(info) or v027_mrv2.supports(info):
        # A 0.25+ install may still run either runner family per engine
        # instance (use_v2_model_runner is a per-config decision), so all
        # supported families get hooks.
        if v027_mrv1.supports(info):
            drivers.append(v027_mrv1)
        if v027_mrv2.supports(info):
            drivers.append(v027_mrv2)
        return drivers
    if legacy.supports(info):
        return [legacy]
    raise UnsupportedBackendError(
        "unrecognised vLLM structure: "
        f"version={info.version} "
        f"mrv1_runner={'yes' if info.mrv1_runner_cls is not None else 'no'} "
        f"mrv1_prepare_inputs_arity={info.mrv1_prepare_inputs_arity} "
        f"mrv1_sample_tokens={info.mrv1_has_sample_tokens} "
        f"mrv2_runner={'yes' if info.mrv2_runner_cls is not None else 'no'} "
        f"sampler_tuple_return={info.sampler_returns_tuple}. "
        "Add an explicit backend driver for this structure."
    )


def install_backend(*, core: Any) -> list[str]:
    """Install hooks for the probed backend exactly once. Returns driver ids."""
    installed: list[str] = list(getattr(core, "_TLLM_BACKEND_DRIVERS", ()) or ())
    if installed:
        return installed
    info = probe_vllm()
    for driver in select_drivers(info):
        driver.install(core=core, info=info)
        installed.append(driver.backend_id)
    core._TLLM_BACKEND_DRIVERS = tuple(installed)
    core._PATCH_INSTALLED = True
    return installed


__all__ = [
    "install_backend",
    "probe_vllm",
    "select_drivers",
    "VllmBackendInfo",
]
