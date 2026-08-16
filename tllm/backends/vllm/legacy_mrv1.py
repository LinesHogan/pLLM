#!/usr/bin/env python3
"""Legacy backend driver for vLLM < 0.25 MRv1 (transition support only).

Reuses the historical patch chain in ``tllm.runtime.vllm_patch``
(InputBatch min-p patch, single-argument ``_prepare_inputs`` wrapper,
tuple-free ``Sampler.sample``). Not actively tested; new work should target
the 0.25+ drivers.
"""

from __future__ import annotations

from typing import Any

from tllm.backends.protocol import (
    BackendCapabilities,
    GraphSafety,
    RunnerBinding,
    SamplerStage,
)


class LegacyMRv1Driver:
    backend_id = "vllm-legacy-mrv1"
    _HOOK_FLAG = "_tllm_legacy_mrv1_hooks"

    def supports(self, info: Any) -> bool:
        major, minor, _ = info.version_tuple
        return bool(
            major == 0
            and minor < 25
            and info.mrv1_runner_cls is not None
            and not info.mrv1_has_sample_tokens
            and info.mrv1_prepare_inputs_arity <= 1
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            runner_family="mrv1",
            sampler_stages=frozenset(
                {
                    SamplerStage.RAW_LOGITS,
                    SamplerStage.AFTER_TEMPERATURE,
                    SamplerStage.AFTER_MIN_P,
                    SamplerStage.AFTER_TOP_K_TOP_P,
                    SamplerStage.BEFORE_DRAW,
                    SamplerStage.AFTER_DRAW,
                }
            ),
            native_min_p=False,
            tuple_sampler_return=False,
            split_execute_and_sample=False,
            graph_safety=GraphSafety.CAPTURE_SAFE,
        )

    def install(self, *, core: Any, info: Any) -> None:
        cls = info.mrv1_runner_cls
        if getattr(cls, self._HOOK_FLAG, False):
            return
        core.GPUModelRunner = cls
        core._ORIG_LOAD_MODEL = cls.load_model
        core._ORIG_PREPARE_INPUTS = cls._prepare_inputs
        core._ORIG_EXECUTE_MODEL = cls.execute_model
        from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks

        _hooks.install_runner_patch(core=core)
        setattr(cls, self._HOOK_FLAG, True)

    def unbind(self, binding: RunnerBinding) -> None:
        binding.active = False
