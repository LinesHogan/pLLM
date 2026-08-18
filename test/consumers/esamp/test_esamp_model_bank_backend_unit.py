#!/usr/bin/env python3
"""Unit tests for ESamp model-bank forward backends."""

from __future__ import annotations

import unittest

from tllm.consumers.esamp.model_bank_backend import normalize_model_bank_forward_backend


class ESampModelBankBackendUnitTest(unittest.TestCase):
    def test_backend_name_normalization(self) -> None:
        self.assertEqual(normalize_model_bank_forward_backend(None), "torch")
        self.assertEqual(normalize_model_bank_forward_backend("reference"), "torch")
        self.assertEqual(normalize_model_bank_forward_backend("triton"), "triton_grouped")


if __name__ == "__main__":
    unittest.main()
