"""Author lifecycle contracts for typed request-trainable state."""

from __future__ import annotations

import unittest
from typing import cast

import torch

from tllm.kit.workload import Workload
from tllm.runtime.request_trainable_state import (
    RequestStateRuntimeService,
    RequestStateSelectionBatch,
)
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.workload_bindings import LocalRuntimeServices
from tllm.workload import (
    AdamWSpec,
    ConfigField,
    RequestKeyField,
    RequestTrainableState,
    State,
    WorkloadConfig,
    read,
    request_key,
    task,
    workload,
)


_REQUEST_ENABLED = ConfigField.boolean(True)


class _RequestConfig(WorkloadConfig):
    enabled: bool = cast(bool, _REQUEST_ENABLED)


@workload(
    id="author.request-trainable",
    config=_RequestConfig,
    enabled=_REQUEST_ENABLED,
)
class _RequestTrainableWorkload(Workload[_RequestConfig]):
    distiller: RequestTrainableState[torch.nn.Linear] = State[
        torch.nn.Linear
    ].request_trainable(
        key=request_key(RequestKeyField.REQUEST_ID),
        optimizer=AdamWSpec.fixed(1e-3),
    )
    selection = distiller.selection()

    @distiller.initialize
    def init_distiller(
        self, spec: RuntimeSpecializationSpec
    ) -> torch.nn.Linear:
        return torch.nn.Linear(
            spec.hidden_size,
            spec.hidden_size,
            bias=False,
            device=spec.device,
            dtype=spec.dtype,
        )

    @task(inputs={"selection": selection}, effects=(read(selection),))
    def observe(self, selection: RequestStateSelectionBatch) -> None:
        del selection


class RequestTrainableStateAuthorUnitTest(unittest.TestCase):
    def test_cold_specialization_owns_one_rows_sized_eager_service(self) -> None:
        workload_instance = _RequestTrainableWorkload(_RequestConfig())
        before = workload_instance.native_workload_resources()
        self.assertEqual(before.request_state_services, {})
        self.assertEqual(before.states.snapshots, {})
        self.assertEqual(before.optimizer_transitions, {})

        resources = workload_instance.native_workload_resources()
        assert isinstance(resources.services, LocalRuntimeServices)
        resources.services.cold_specialize(
            RuntimeSpecializationSpec(
                rows=3,
                hidden_size=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
        )

        service = workload_instance.distiller
        self.assertIsInstance(service, RequestStateRuntimeService)
        self.assertIs(
            service,
            workload_instance.request_state_value(
                _RequestTrainableWorkload.distiller
            ),
        )
        self.assertIs(
            service,
            workload_instance.native_workload_resources().request_state_services[
                "distiller"
            ],
        )
        self.assertEqual(service.resource, "distiller")
        self.assertEqual(service.capacity, 3)
        self.assertEqual(service.epoch, 1)
        self.assertEqual(
            workload_instance.native_workload_resources().states.snapshots,
            {},
        )
        self.assertEqual(
            workload_instance.native_workload_resources().optimizer_transitions,
            {},
        )

    def test_request_trainable_state_preserves_closed_key_and_dynamic_storage(self) -> None:
        declaration = _RequestTrainableWorkload(_RequestConfig()).declaration()
        state = declaration.ir.states[0]

        self.assertEqual(state.request_key, (RequestKeyField.REQUEST_ID,))
        self.assertEqual(state.storage.value, "dynamic")


if __name__ == "__main__":
    unittest.main()
