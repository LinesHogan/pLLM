from __future__ import annotations

import unittest

from tllm.backends.protocol import BackendCapabilities, GraphSafety, SamplerStage
from tllm.workgraph.anchors import (
    BackendCompatibilityError,
    anchor_desc,
    backend_snapshot_from_capabilities,
)
from tllm.workgraph.ir import BackendSnapshot


class Phase1aAnchorTest(unittest.TestCase):
    def test_supported_backend_matrix_has_stable_anchor(self) -> None:
        for family in ("mrv1", "mrv2"):
            for mode in ("eager", "piecewise"):
                capabilities = BackendCapabilities(
                    backend_id=f"vllm-{family}",
                    runner_family=family,
                    sampler_stages=frozenset({SamplerStage.BEFORE_DRAW}),
                    native_min_p=True,
                    tuple_sampler_return=False,
                    split_execute_and_sample=True,
                    graph_safety=GraphSafety.CAPTURE_SAFE,
                )
                backend = backend_snapshot_from_capabilities(capabilities, graph_mode=mode)
                anchor = anchor_desc(backend, "execute_model.post")
                self.assertEqual(anchor.backend, backend.backend_id)
                self.assertEqual(anchor.runner, family)
                self.assertEqual(anchor.graph_mode, mode)
                self.assertTrue(anchor.externally_owned)
                self.assertEqual(anchor.mobility, "pinned")

    def test_unknown_family_or_graph_mode_fails_closed(self) -> None:
        capabilities = BackendCapabilities(
            backend_id="vllm-unknown",
            runner_family="future",
            sampler_stages=frozenset(),
            native_min_p=False,
            tuple_sampler_return=False,
            split_execute_and_sample=False,
        )
        with self.assertRaises(BackendCompatibilityError):
            backend_snapshot_from_capabilities(capabilities, graph_mode="eager")

        capabilities = BackendCapabilities(
            backend_id="vllm-mrv1",
            runner_family="mrv1",
            sampler_stages=frozenset(),
            native_min_p=False,
            tuple_sampler_return=False,
            split_execute_and_sample=False,
        )
        with self.assertRaises(BackendCompatibilityError):
            backend_snapshot_from_capabilities(capabilities, graph_mode="full")

    def test_unknown_hook_is_not_an_anchor(self) -> None:
        capabilities = BackendCapabilities(
            backend_id="vllm-mrv2",
            runner_family="mrv2",
            sampler_stages=frozenset(),
            native_min_p=False,
            tuple_sampler_return=False,
            split_execute_and_sample=True,
        )
        backend = backend_snapshot_from_capabilities(capabilities, graph_mode="eager")
        with self.assertRaises(BackendCompatibilityError):
            anchor_desc(backend, "layer.post")

    def test_direct_backend_snapshot_is_fail_closed(self) -> None:
        for kwargs in (
            {"runner_family": "future"},
            {"graph_mode": "future"},
            {"backend_version": "0.26.9"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    BackendSnapshot(
                        backend_id="vllm-test",
                        runner_family=kwargs.get("runner_family", "mrv1"),
                        graph_mode=kwargs.get("graph_mode", "eager"),
                        backend_version=kwargs.get("backend_version", "0.27.1"),
                    )

    def test_anchor_descriptor_rejects_unknown_hook(self) -> None:
        with self.assertRaises(ValueError):
            # The Python descriptor is also a fail-closed boundary; this is
            # not deferred to the Rust validator.
            from tllm.workgraph.ir import AnchorDesc

            AnchorDesc("vllm", "mrv1", "eager", "layer.post")


if __name__ == "__main__":
    unittest.main()
