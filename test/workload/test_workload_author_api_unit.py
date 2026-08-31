"""Contracts for the compact workload author surface."""

from __future__ import annotations

import unittest
from typing import Literal

import torch
from tllm.contracts.packed_rows import PackedPhaseRowBatch
from tllm.kit.workload import Workload
from tllm.runtime.workload_bindings import LocalRuntimeServices

from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.workload import (
    AdamWSpec,
    Before,
    Between,
    ConfigField,
    ConfigKind,
    State,
    PublishedModule,
    StagedPort,
    WorkloadConfig,
    config_field,
    declared_config_field,
    declare_workload,
    hidden,
    optimize,
    packed_hidden,
    packed_request_rows,
    staged,
    task,
    workload,
    when,
)

_LEARNING_RATE = ConfigField.floating(3e-4)


class _Config(WorkloadConfig):
    enabled: bool = config_field(default=True, kind=ConfigKind.BOOL)
    learning_rate: float = declared_config_field(_LEARNING_RATE)
    layer_path: str = config_field(
        default="model.layers.0", kind=ConfigKind.STR
    )


_MODE: ConfigField[Literal["online", "offline"]] = ConfigField.choice(
    "online", ("online", "offline")
)


class _ModeConfig(WorkloadConfig):
    enabled: bool = config_field(default=True, kind=ConfigKind.BOOL)
    mode: Literal["online", "offline"] = declared_config_field(_MODE)


@workload(id="author.mode", config=_ModeConfig, enabled=_ModeConfig.enabled)
class _ModeWorkload(Workload[_ModeConfig]):
    @task(when=when(_MODE, "online"))
    def online(self) -> None:
        pass

    @task(when=when(_MODE, "offline"))
    def offline(self) -> None:
        pass


@workload(id="author.probe", config=_Config, enabled=_Config.enabled)
class _AuthorWorkload(Workload[_Config]):
    model: State[torch.nn.Linear] = State.replica(
        optimizer=AdamWSpec.configured(_LEARNING_RATE)
    )

    @model.initialize
    def init_model(self, spec: RuntimeSpecializationSpec) -> torch.nn.Module:
        return torch.nn.Linear(
            spec.hidden_size,
            spec.hidden_size,
            bias=False,
            device=spec.device,
            dtype=spec.dtype,
        )

    @task(
        inputs={
            "value": hidden(
                0,
                "block_output",
                "decode",
                path=_Config.layer_path,
            )
        },
        outputs={"loss": optimize(model, at=Before("next.prepare"))},
        timing=Between("model.execute", "sampler.pre_draw"),
    )
    def train(self, value: torch.Tensor) -> torch.Tensor:
        output = self.model(value)
        return output.square().mean()


@workload(id="author.staged", config=_Config, enabled=_Config.enabled)
class _StagedWorkload(Workload[_Config]):
    model: State[torch.nn.Linear] = State.replica(
        optimizer=AdamWSpec.configured(_LEARNING_RATE)
    )
    batch: StagedPort[torch.Tensor] = StagedPort()

    @model.initialize
    def init_model(self, spec: RuntimeSpecializationSpec) -> torch.nn.Module:
        return torch.nn.Linear(spec.hidden_size, spec.hidden_size, bias=False)

    @task(inputs={"batch": staged(batch)})
    def consume(self, batch: torch.Tensor) -> None:
        del batch


@workload(id="author.published", config=_Config, enabled=_Config.enabled)
class _PublishedWorkload(Workload[_Config]):
    intervention: PublishedModule[torch.nn.Linear] = State.published_module()

    @intervention.initialize
    def init_intervention(self, spec: RuntimeSpecializationSpec) -> torch.nn.Module:
        return torch.nn.Linear(spec.hidden_size, spec.hidden_size, bias=False)


_PACKED_MODE: ConfigField[Literal["online", "offline"]] = ConfigField.choice(
    "online", ("online", "offline")
)


class _PackedConfig(WorkloadConfig):
    enabled: bool = config_field(default=True, kind=ConfigKind.BOOL)
    mode: Literal["online", "offline"] = declared_config_field(_PACKED_MODE)
    source_path: str = config_field(
        default="model.layers.0", kind=ConfigKind.STR
    )
    target_path: str = config_field(
        default="model.layers.7", kind=ConfigKind.STR
    )


_ONLINE_PACKED_ROWS = packed_request_rows()
_OFFLINE_PACKED_ROWS = packed_request_rows()


@workload(
    id="author.packed-resources",
    config=_PackedConfig,
    enabled=_PackedConfig.enabled,
)
class _PackedAuthorWorkload(Workload[_PackedConfig]):
    @task(
        when=when(_PACKED_MODE, "online"),
        inputs={
            "rows": _ONLINE_PACKED_ROWS,
            "source": packed_hidden(
                0,
                rows=_ONLINE_PACKED_ROWS,
                path=_PackedConfig.source_path,
            ),
            "target": packed_hidden(
                -1,
                rows=_ONLINE_PACKED_ROWS,
                path=_PackedConfig.target_path,
            ),
        },
    )
    def online(
        self,
        rows: PackedPhaseRowBatch,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        _ = rows, source, target

    @task(
        when=when(_PACKED_MODE, "offline"),
        inputs={
            "rows": _OFFLINE_PACKED_ROWS,
            "hidden": packed_hidden(
                "teacher",
                rows=_OFFLINE_PACKED_ROWS,
                path="model.teacher",
            ),
        },
    )
    def offline(
        self,
        rows: PackedPhaseRowBatch,
        hidden: torch.Tensor,
    ) -> None:
        _ = rows, hidden


class WorkloadAuthorApiUnitTest(unittest.TestCase):
    def test_rust_normalized_choice_selects_one_signed_semantic_variant(self) -> None:
        self.assertEqual(
            declare_workload(_ModeWorkload(_ModeConfig(mode="online"))).ir.schedule,
            ("online",),
        )
        self.assertEqual(
            declare_workload(_ModeWorkload(_ModeConfig(mode="offline"))).ir.schedule,
            ("offline",),
        )
        with self.assertRaisesRegex(ValueError, "must be one of"):
            _ModeConfig(mode="invalid")

    def test_decorator_derives_author_metadata_and_residual_manifest(self) -> None:
        workload_instance = _AuthorWorkload(_Config(layer_path="model.layers.4"))
        resources = workload_instance.native_workload_resources()
        self.assertEqual(
            {locator.layer: path for locator, path in resources.residual_paths.items()},
            {0: "model.layers.4"},
        )
        assert isinstance(resources.services, LocalRuntimeServices)
        self.assertTrue(resources.services.is_enabled())
        self.assertNotIn("algorithm", vars(_AuthorWorkload))
        self.assertNotIn("config_type", vars(_AuthorWorkload))

    def test_packed_resource_manifest_uses_only_active_typed_locators(self) -> None:
        online = _PackedAuthorWorkload(
            _PackedConfig(
                mode="online",
                source_path="model.layers.2",
                target_path="model.layers.9",
            )
        ).native_workload_resources()
        self.assertEqual(
            tuple(
                (
                    binding.layer,
                    binding.layer_kind.value,
                    binding.site.value,
                    binding.module_path,
                )
                for binding in online.packed_hidden_resources
            ),
            (
                ("-1", "index", "block_output", "model.layers.9"),
                ("0", "index", "block_output", "model.layers.2"),
            ),
        )

        offline = _PackedAuthorWorkload(
            _PackedConfig(mode="offline")
        ).native_workload_resources()
        self.assertEqual(
            tuple(
                (
                    binding.layer,
                    binding.layer_kind.value,
                    binding.site.value,
                    binding.module_path,
                )
                for binding in offline.packed_hidden_resources
            ),
            (("teacher", "path", "block_output", "model.teacher"),),
        )

    def test_state_initializer_binds_module_and_configured_optimizer(self) -> None:
        workload_instance = _AuthorWorkload(_Config(learning_rate=3e-4))
        resources = workload_instance.native_workload_resources()
        assert isinstance(resources.services, LocalRuntimeServices)
        resources.services.cold_specialize(
            RuntimeSpecializationSpec(
                rows=2,
                hidden_size=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
        )
        self.assertIsInstance(workload_instance.model, torch.nn.Linear)
        optimizer = workload_instance.optimizer
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        assert optimizer is not None
        self.assertEqual(optimizer.param_groups[0]["lr"], 3e-4)

    def test_optimize_generates_explicit_transition_task(self) -> None:
        declaration = _AuthorWorkload(_Config()).declaration()
        tasks = {task.name: task for task in declaration.ir.tasks}
        self.assertEqual(set(tasks), {"train", "train__optimize_model"})
        self.assertEqual(
            tuple(effect.kind.value for effect in tasks["train"].effects),
            ("read",),
        )
        self.assertEqual(
            tuple(effect.kind.value for effect in tasks["train__optimize_model"].effects),
            ("transition",),
        )
        self.assertEqual(
            tuple(declaration.ir.schedule),
            ("train", "train__optimize_model"),
        )
        self.assertNotIn("train__optimize_model", vars(_AuthorWorkload))
        self.assertIn("train__optimize_model", dict(declaration.executors))

    def test_generated_publish_task_is_definition_owned_not_class_injected(self) -> None:
        declaration = _PublishedWorkload(_Config()).declaration()

        self.assertEqual(
            tuple(task.name for task in declaration.ir.tasks),
            ("publish__intervention",),
        )
        self.assertNotIn("publish__intervention", vars(_PublishedWorkload))
        self.assertIn("publish__intervention", dict(declaration.executors))

    def test_staged_port_is_typed_ir_and_missing_data_skips(self) -> None:
        workload_instance = _StagedWorkload(_Config())
        declaration = workload_instance.declaration()
        self.assertEqual(declaration.ir.tasks[0].inputs[0].kind.value, "staged")
        provider = next(
            iter(workload_instance.native_workload_resources().ports.inputs.values())
        )
        from tllm.contracts.workload_ports import SKIPPED

        self.assertIs(provider.read(1), SKIPPED)
        payload = torch.ones(2, 3)
        sequence = workload_instance.stage_port(_StagedWorkload.batch, payload)
        self.assertEqual(sequence, 1)
        self.assertIs(provider.read(2), payload)
        self.assertIs(provider.read(3), SKIPPED)


if __name__ == "__main__":
    unittest.main()
    ConfigField,
