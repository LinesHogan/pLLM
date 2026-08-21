#!/usr/bin/env python3
"""Request identity normalization at the vLLM boundary."""

from __future__ import annotations

import unittest

from tllm.contracts.request_identity import resolve_prompt_sample, resolve_request_identity


class RequestIdentityUnitTest(unittest.TestCase):
    def test_direct_and_explicit_sample_mapping(self) -> None:
        identity = resolve_request_identity(
            "request_with_underscores",
            prompt_indices={"request_with_underscores": 4},
            sample_indices={"request_with_underscores": 2},
        )

        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertEqual(
            (identity.request_id, identity.prompt_idx, identity.sample_idx),
            ("request_with_underscores", 4, 2),
        )

    def test_parallel_child_preserves_parent_with_underscores(self) -> None:
        self.assertEqual(
            resolve_prompt_sample(
                "3_parent_with_underscores",
                prompt_indices={"parent_with_underscores": 7},
            ),
            (7, 3),
        )

    def test_vllm_internal_suffix_resolves_direct_and_parallel_ids(self) -> None:
        mapping = {"external-request": 5}
        self.assertEqual(
            resolve_prompt_sample("external-request-a1B2c3D4", prompt_indices=mapping),
            (5, 0),
        )
        self.assertEqual(
            resolve_prompt_sample("6_external-request-a1B2c3D4", prompt_indices=mapping),
            (5, 6),
        )

    def test_unknown_identity_is_explicit(self) -> None:
        self.assertEqual(resolve_prompt_sample("unknown", prompt_indices={}), (-1, -1))


if __name__ == "__main__":
    unittest.main()
