"""Contracts for the compact workload author surface."""

from __future__ import annotations

import unittest

import torch
from tllm.kit.workload import Workload

from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.workload import (
    AdamWSpec,
    Before,
    Between,
    State,
    StagedPort,
    WorkloadConfig,
    config_field,
    hidden,
    optimize,
    staged,
    task,
    workload,
)


class _Config(WorkloadConfig):
    enabled: bool = config_field(default=True)
    learning_rate: float = config_field(default=3e-4)
    layer_path: str = config_field(default="model.layers.0")


@workload(id="author.probe", config=_Config, enabled=_Config.enabled)
class _AuthorWorkload(Workload[_Config]):
    model: State[torch.nn.Linear] = State.replica(
        optimizer=AdamWSpec(lr=_Config.learning_rate)
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
        optimizer=AdamWSpec(lr=_Config.learning_rate)
    )
    batch: StagedPort[torch.Tensor] = StagedPort()

    @model.initialize
    def init_model(self, spec: RuntimeSpecializationSpec) -> torch.nn.Module:
        return torch.nn.Linear(spec.hidden_size, spec.hidden_size, bias=False)

    @task(inputs={"batch": staged(batch)})
    def consume(self, batch: torch.Tensor) -> None:
        del batch


class WorkloadAuthorApiUnitTest(unittest.TestCase):
    def test_decorator_derives_author_metadata_and_residual_manifest(self) -> None:
        workload_instance = _AuthorWorkload(_Config(layer_path="model.layers.4"))
        resources = workload_instance.native_workload_resources()
        self.assertEqual(
            {locator.layer: path for locator, path in resources.residual_paths.items()},
            {0: "model.layers.4"},
        )
        self.assertTrue(resources.enabled_predicate())  # type: ignore[misc]
        self.assertNotIn("algorithm", vars(_AuthorWorkload))
        self.assertNotIn("config_type", vars(_AuthorWorkload))

    def test_state_initializer_binds_module_and_configured_optimizer(self) -> None:
        workload_instance = _AuthorWorkload(_Config(learning_rate=3e-4))
        resources = workload_instance.native_workload_resources()
        assert resources.cold_specializer is not None
        resources.cold_specializer(
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

    def test_staged_port_is_typed_ir_and_missing_data_skips(self) -> None:
        workload_instance = _StagedWorkload(_Config())
        declaration = workload_instance.declaration()
        self.assertEqual(declaration.ir.tasks[0].inputs[0].kind.value, "staged")
        provider = next(
            iter(workload_instance.native_workload_resources().provided_inputs.values())
        )
        from tllm.workload.runtime_values import SKIPPED

        self.assertIs(provider(1), SKIPPED)
        payload = torch.ones(2, 3)
        sequence = workload_instance.stage_port(_StagedWorkload.batch, payload)
        self.assertEqual(sequence, 1)
        self.assertIs(provider(2), payload)
        self.assertIs(provider(3), SKIPPED)


if __name__ == "__main__":
    unittest.main()
