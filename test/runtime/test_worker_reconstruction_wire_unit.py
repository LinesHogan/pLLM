from __future__ import annotations

import hashlib
import json
import unittest

from tllm import _core


class WorkerReconstructionWireContractTest(unittest.TestCase):
    """Rust owns graph reconstruction and signed runtime-fact equality."""

    _GRAPH = {
        "workload_key": "worker-reconstruction-wire",
        "tasks": [
            {
                "task_id": "run",
                "timing": {"after": "model.execute", "before": "step.end"},
                "temporary_bytes": 0,
                "implementation_ids": ["python:run@1"],
                "required_capabilities": [],
                "capture_safe": False,
            }
        ],
        "edges": [],
        "values": [],
        "persistent_state_bytes": 0,
        "schema_version": 1,
    }
    _TIMELINE = {
        "backend_id": "worker-reconstruction-wire",
        "backend_version": "0.27.1",
        "runner_family": "mrv2",
        "graph_mode": "eager",
        "anchors": [
            {
                "anchor_id": "model.execute",
                "ordinal": 0,
                "observable": True,
                "graph_boundary": True,
                "visible_values": [],
            },
            {
                "anchor_id": "step.end",
                "ordinal": 1,
                "observable": True,
                "graph_boundary": True,
                "visible_values": [],
            },
        ],
        "regions": [],
        "static_headroom_bytes": 64,
        "capabilities": [],
        "schema_version": 1,
    }

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def _wire(
        self,
        runtime_entries: list[tuple[str, str]] | None = None,
        static_bindings: str = "[]",
    ) -> tuple[str, str, str, str, str]:
        graph = self._json(self._GRAPH)
        timeline = self._json(self._TIMELINE)
        options = self._json(
            {
                "memory_budget_bytes": 64,
                "allow_serial_fallback": True,
                "shape_specialization": runtime_entries or [],
                "static_binding_manifest": static_bindings,
                "schema_version": 1,
            }
        )
        decision = _core.compile_schedule(graph, timeline, options)
        cache_key = hashlib.sha256(
            f"workload-compilation-v3\0{graph}\0{timeline}\0{options}".encode("ascii")
        ).hexdigest()
        return cache_key, graph, timeline, options, decision

    def _validate(
        self,
        *,
        runtime_entries: list[tuple[str, str]] | None = None,
        reconstructed_graph: str | None = None,
        facts: str = "null",
        signed_static_bindings: str = "[]",
        reconstructed_static_bindings: str | None = None,
    ) -> None:
        cache_key, graph, timeline, options, decision = self._wire(
            runtime_entries,
            signed_static_bindings,
        )
        _core.validate_worker_reconstruction_wire(
            cache_key,
            graph,
            timeline,
            options,
            decision,
            1,
            graph if reconstructed_graph is None else reconstructed_graph,
            facts,
            "[]",
            (
                signed_static_bindings
                if reconstructed_static_bindings is None
                else reconstructed_static_bindings
            ),
        )

    def _validate_local(
        self,
        *,
        reconstructed_graph: str | None = None,
        signed_static_bindings: str = "[]",
        reconstructed_static_bindings: str | None = None,
    ) -> None:
        cache_key, graph, timeline, options, decision = self._wire(
            static_bindings=signed_static_bindings
        )
        _core.validate_local_execution_binding_wire(
            cache_key,
            graph,
            timeline,
            options,
            decision,
            1,
            graph if reconstructed_graph is None else reconstructed_graph,
            "[]",
            (
                signed_static_bindings
                if reconstructed_static_bindings is None
                else reconstructed_static_bindings
            ),
        )

    @staticmethod
    def _signed_entries() -> list[tuple[str, str]]:
        return [
            ("runtime/device", "cpu"),
            ("runtime/dtype", "torch.float32"),
            ("runtime/hidden_size", "4"),
            ("runtime/rows", "2"),
        ]

    def test_null_facts_are_valid_only_without_signed_runtime_facts(self) -> None:
        self._validate()
        facts = self._json(
            {
                "device": "cpu",
                "dtype": "torch.float32",
                "hidden_size": 4,
                "rows": 2,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "are required"):
            self._validate(runtime_entries=self._signed_entries())
        with self.assertRaisesRegex(RuntimeError, "unsigned"):
            self._validate(facts=facts)

    def test_signed_runtime_facts_require_an_exact_all_field_match(self) -> None:
        facts = {
            "device": "cpu",
            "dtype": "torch.float32",
            "hidden_size": 4,
            "rows": 2,
        }
        self._validate(runtime_entries=self._signed_entries(), facts=self._json(facts))
        for key, wrong in (
            ("device", "cuda:0"),
            ("dtype", "torch.float16"),
            ("hidden_size", 8),
            ("rows", 3),
        ):
            changed = dict(facts)
            changed[key] = wrong
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError, "specialization does not match"):
                    self._validate(
                        runtime_entries=self._signed_entries(),
                        facts=self._json(changed),
                    )

    def test_runtime_fact_wire_is_closed_and_signed_options_cannot_be_partial(self) -> None:
        facts = self._json(
            {
                "device": "cpu",
                "dtype": "torch.float32",
                "hidden_size": 4,
                "rows": 2,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "contain all"):
            self._validate(runtime_entries=[("runtime/device", "cpu")], facts=facts)
        with self.assertRaisesRegex(ValueError, "unknown field"):
            self._validate(
                facts=(
                    '{"device":"cpu","dtype":"torch.float32",'
                    '"hidden_size":4,"rows":2,"extra":"x"}'
                )
            )
        with self.assertRaisesRegex(ValueError, "duplicate field"):
            self._validate(
                facts=(
                    '{"device":"cpu","device":"cpu",'
                    '"dtype":"torch.float32","hidden_size":4,"rows":2}'
                )
            )
        with self.assertRaisesRegex(ValueError, "missing field"):
            self._validate(
                facts='{"device":"cpu","dtype":"torch.float32","rows":2}'
            )
        with self.assertRaisesRegex(ValueError, "positive rows/hidden_size"):
            self._validate(
                runtime_entries=self._signed_entries(),
                facts='{"device":"cpu","dtype":"torch.float32","hidden_size":0,"rows":2}',
            )
        with self.assertRaisesRegex(ValueError, "invalid nullable RuntimeSpecializationFacts"):
            self._validate(
                runtime_entries=self._signed_entries(),
                facts='{"device":"cpu","dtype":"torch.float32","hidden_size":"4","rows":2}',
            )

    def test_worker_graph_drift_and_untrusted_graph_wire_fail_closed(self) -> None:
        drifted = dict(self._GRAPH)
        drifted["persistent_state_bytes"] = 1
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self._validate(reconstructed_graph=self._json(drifted))
        with self.assertRaisesRegex(ValueError, "unknown field"):
            self._validate(reconstructed_graph='{"workload_key":"x","unexpected":true}')
        with self.assertRaisesRegex(ValueError, "duplicate field"):
            self._validate(
                reconstructed_graph=self._json(self._GRAPH).replace(
                    '"workload_key":"worker-reconstruction-wire"',
                    '"workload_key":"worker-reconstruction-wire",'
                    '"workload_key":"other"',
                )
            )

    def test_local_binding_gate_rejects_graph_drift_before_artifacts_bind(self) -> None:
        self._validate_local()
        drifted = dict(self._GRAPH)
        drifted["persistent_state_bytes"] = 1
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self._validate_local(reconstructed_graph=self._json(drifted))

    def test_runtime_specialization_normalizer_owns_merge_and_exact_match(self) -> None:
        _cache_key, _graph, _timeline, options, _decision = self._wire()
        facts = self._json(
            {
                "device": "cpu",
                "dtype": "torch.float32",
                "hidden_size": 4,
                "rows": 2,
            }
        )
        normalized = _core.normalize_runtime_specialization_options(options, facts)
        normalized_value = json.loads(normalized)
        self.assertEqual(
            normalized_value["shape_specialization"],
            [
                ["runtime/device", "cpu"],
                ["runtime/dtype", "torch.float32"],
                ["runtime/hidden_size", "4"],
                ["runtime/rows", "2"],
            ],
        )
        self.assertEqual(
            _core.normalize_runtime_specialization_options(normalized, facts), normalized
        )
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _core.normalize_runtime_specialization_options(
                normalized,
                facts.replace('"rows":2', '"rows":3'),
            )
        for invalid in (
            "null",
            '{"device":"cpu","dtype":"torch.float32","hidden_size":0,"rows":2}',
            '{"device":"cpu","dtype":"torch.float32","hidden_size":"4","rows":2}',
            '{"device":"cpu","dtype":"torch.float32","hidden_size":4,"rows":2,"extra":true}',
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises((RuntimeError, ValueError)):
                    _core.normalize_runtime_specialization_options(normalized, invalid)

    def test_static_backend_binding_manifest_is_rust_signed_and_exact(self) -> None:
        signed = [
            {
                "workload_id": "recirculation.vllm-prefill",
                "factory": "tllm.consumers.recirculation.workload:RecirculationWorkload.reconstruct",
                "config": {
                    "source_layer": 3,
                    "destination_layer": 1,
                    "alpha": 0.2,
                    "beta": None,
                    "ramp_tokens": 4,
                },
                "specialization_key": [
                    ["algorithm", "recirculation-vllm-typed-region-v2"],
                    ["alpha", "0x1.999999999999ap-3"],
                    ["beta", "default"],
                    ["destination_layer", "1"],
                    ["ramp_tokens", "4"],
                    ["source_layer", "3"],
                ],
                "backend_region_implementation_ids": [
                    "tllm.recirculation:bind_prefill@1"
                ],
            }
        ]
        signed_wire = self._json(signed)
        self._validate(signed_static_bindings=signed_wire)

        for field, value in (
            ("source_layer", 2),
            ("destination_layer", 0),
            ("alpha", 0.3),
            ("beta", 0.1),
            ("ramp_tokens", 8),
        ):
            drifted = json.loads(signed_wire)
            drifted[0]["config"][field] = value
            with self.subTest(config=field):
                with self.assertRaisesRegex(RuntimeError, "static backend binding"):
                    self._validate(
                        signed_static_bindings=signed_wire,
                        reconstructed_static_bindings=self._json(drifted),
                    )

        drifted_key = json.loads(signed_wire)
        drifted_key[0]["specialization_key"][1][1] = "0x1.0000000000000p-2"
        with self.assertRaisesRegex(RuntimeError, "static backend binding"):
            self._validate(
                signed_static_bindings=signed_wire,
                reconstructed_static_bindings=self._json(drifted_key),
            )

    def test_worker_factory_specs_are_checked_before_python_import(self) -> None:
        signed = [
            {
                "workload_id": "worker.test",
                "factory": "worker.module:build",
                "config": {"artifact": "signed"},
                "specialization_key": [],
                "backend_region_implementation_ids": [],
            }
        ]
        cache_key, graph, timeline, options, decision = self._wire(
            static_bindings=self._json(signed)
        )

        def validate(specs: object) -> None:
            _core.validate_worker_reconstruction_specs(
                cache_key,
                graph,
                timeline,
                options,
                decision,
                1,
                self._json(specs),
            )

        validate([{"factory": "worker.module:build", "config": {"artifact": "signed"}}])
        for drifted in (
            [],
            [{"factory": "worker.module:other", "config": {"artifact": "signed"}}],
            [{"factory": "worker.module:build", "config": {"artifact": "changed"}}],
            [
                {"factory": "worker.module:build", "config": {"artifact": "signed"}},
                {"factory": "worker.module:build", "config": {"artifact": "signed"}},
            ],
        ):
            with self.subTest(specs=drifted), self.assertRaisesRegex(
                RuntimeError, "parent-signed static binding"
            ):
                validate(drifted)
        with self.assertRaisesRegex(ValueError, "unknown field"):
            validate(
                [
                    {
                        "factory": "worker.module:build",
                        "config": {"artifact": "signed"},
                        "extra": True,
                    }
                ]
            )

    def test_local_static_recirculation_binding_must_match_live_resources(self) -> None:
        signed = [
            {
                "workload_id": "recirculation.vllm-prefill",
                "factory": "tllm.consumers.recirculation.workload:RecirculationWorkload.reconstruct",
                "config": {
                    "source_layer": 3,
                    "destination_layer": 1,
                    "alpha": 0.2,
                    "beta": None,
                    "ramp_tokens": 4,
                },
                "specialization_key": [
                    ["algorithm", "recirculation-vllm-typed-region-v2"],
                    ["alpha", "0x1.999999999999ap-3"],
                    ["beta", "default"],
                    ["destination_layer", "1"],
                    ["ramp_tokens", "4"],
                    ["source_layer", "3"],
                ],
                "backend_region_implementation_ids": [
                    "tllm.recirculation:bind_prefill@1"
                ],
            }
        ]
        signed_wire = self._json(signed)
        self._validate_local(signed_static_bindings=signed_wire)

        for field, value in (("source_layer", 2), ("alpha", 0.3)):
            drifted = json.loads(signed_wire)
            drifted[0]["config"][field] = value
            with self.subTest(config=field), self.assertRaisesRegex(
                RuntimeError, "static backend binding"
            ):
                self._validate_local(
                    signed_static_bindings=signed_wire,
                    reconstructed_static_bindings=self._json(drifted),
                )

        drifted_specialization = json.loads(signed_wire)
        drifted_specialization[0]["specialization_key"][1][1] = (
            "0x1.0000000000000p-2"
        )
        with self.assertRaisesRegex(RuntimeError, "static backend binding"):
            self._validate_local(
                signed_static_bindings=signed_wire,
                reconstructed_static_bindings=self._json(drifted_specialization),
            )

        drifted_binder = json.loads(signed_wire)
        drifted_binder[0]["backend_region_implementation_ids"] = [
            "tllm.recirculation:other@1"
        ]
        with self.assertRaisesRegex(RuntimeError, "static backend binding"):
            self._validate_local(
                signed_static_bindings=signed_wire,
                reconstructed_static_bindings=self._json(drifted_binder),
            )

    def test_model_layer_count_is_a_signed_runtime_fact(self) -> None:
        entries = [
            ("runtime/device", "cpu"),
            ("runtime/dtype", "torch.float32"),
            ("runtime/hidden_size", "4"),
            ("runtime/model_layer_count", "4"),
            ("runtime/rows", "2"),
        ]
        facts = self._json(
            {
                "device": "cpu",
                "dtype": "torch.float32",
                "hidden_size": 4,
                "rows": 2,
                "model_layer_count": 4,
            }
        )
        self._validate(runtime_entries=entries, facts=facts)
        with self.assertRaisesRegex(RuntimeError, "specialization does not match"):
            self._validate(
                runtime_entries=entries,
                facts=facts.replace('"model_layer_count":4', '"model_layer_count":5'),
            )

    def test_max_prefill_tokens_is_a_signed_runtime_fact(self) -> None:
        entries = [
            ("runtime/device", "cpu"),
            ("runtime/dtype", "torch.float32"),
            ("runtime/hidden_size", "4"),
            ("runtime/max_prefill_tokens", "16"),
            ("runtime/rows", "2"),
        ]
        facts = self._json(
            {
                "device": "cpu",
                "dtype": "torch.float32",
                "hidden_size": 4,
                "max_prefill_tokens": 16,
                "rows": 2,
            }
        )
        self._validate(runtime_entries=entries, facts=facts)
        with self.assertRaisesRegex(RuntimeError, "specialization does not match"):
            self._validate(
                runtime_entries=entries,
                facts=facts.replace('"max_prefill_tokens":16', '"max_prefill_tokens":32'),
            )
        with self.assertRaisesRegex(RuntimeError, "specialization does not match"):
            self._validate(
                runtime_entries=entries,
                facts=facts.replace(',"max_prefill_tokens":16', ''),
            )


if __name__ == "__main__":
    unittest.main()
