#!/usr/bin/env python3
"""Unit tests for min-p handling across vLLM versions.

vLLM 0.25+ handles min-p natively via a built-in argmax-invariant logits
processor, so tLLM must NOT patch ``InputBatch`` there. On older versions
tLLM installs a patch that exposes ``min_p`` on ``SamplingMetadata``.
"""

from __future__ import annotations

import unittest

import torch
from vllm import SamplingParams


def _vllm_minor() -> int:
    import vllm

    parts = str(vllm.__version__).split("+", 1)[0].split(".")
    return int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0


class VllmInputBatchMinPPatchUnitTest(unittest.TestCase):
    def test_legacy_patch_exposes_min_p_on_sampling_metadata(self) -> None:
        if _vllm_minor() >= 25:
            self.skipTest("vLLM 0.25+ handles min-p natively; the tLLM patch is not installed")
        from tllm.runtime.vllm_patch.port_runtime_hooks import install_input_batch_min_p_patch

        install_input_batch_min_p_patch()
        from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

        batch = InputBatch(
            max_num_reqs=2,
            max_model_len=16,
            max_num_batched_tokens=16,
            device=torch.device("cpu"),
            pin_memory=False,
            vocab_size=10,
            block_sizes=[16],
        )
        request = CachedRequestState(
            req_id="req0",
            prompt_token_ids=[1, 2],
            mm_kwargs=[],
            mm_positions=[],
            sampling_params=SamplingParams(
                temperature=0.8,
                top_p=1.0,
                top_k=-1,
                min_p=0.2,
                max_tokens=1,
            ),
            pooling_params=None,
            generator=None,
            block_ids=([],),
            num_computed_tokens=2,
            output_token_ids=[],
        )

        batch.add_request(request)
        metadata = batch._make_sampling_metadata()

        self.assertIsInstance(getattr(metadata, "min_p", None), torch.Tensor)
        self.assertAlmostEqual(float(metadata.min_p[0]), 0.2, places=5)

    def test_v027_native_min_p_and_no_tllm_patch(self) -> None:
        if _vllm_minor() < 25:
            self.skipTest("native min-p processor only exists on vLLM 0.25+")
        from tllm.backends.vllm import install_backend
        from tllm.runtime import residual_runtime as rr

        install_backend(core=rr)

        from vllm.v1.worker.gpu_input_batch import InputBatch

        self.assertFalse(
            getattr(InputBatch, "_tllm_min_p_patched", False),
            "tLLM must not patch InputBatch on vLLM 0.25+ (min-p is native)",
        )
        from vllm.v1.sample.logits_processor import builtin

        self.assertTrue(
            any("min_p" in name.lower() or "minp" in name.lower() for name in dir(builtin)),
            "expected a native min-p logits processor in vLLM 0.25+",
        )
        params = SamplingParams(temperature=0.8, min_p=0.2, max_tokens=1)
        self.assertAlmostEqual(float(params.min_p), 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
