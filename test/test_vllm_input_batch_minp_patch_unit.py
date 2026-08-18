#!/usr/bin/env python3
"""Unit tests for min-p handling across vLLM versions.

vLLM 0.25+ handles min-p natively via a built-in argmax-invariant logits
processor, so tLLM must NOT patch ``InputBatch`` there.
"""

from __future__ import annotations

import unittest

from vllm import SamplingParams


def _vllm_minor() -> int:
    import vllm

    parts = str(vllm.__version__).split("+", 1)[0].split(".")
    return int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0


class VllmInputBatchMinPPatchUnitTest(unittest.TestCase):
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
