"""Worker reconstruction oracle for Rust-issued state layout IDs."""

from __future__ import annotations

import torch

from tllm.backends.timeline import export_main_workload_timeline
from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
from tllm.consumers.esamp.workload import ESampConfig, ESampWorkload
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.workload_bindings import WorkerRuntimeBinding
from tllm.runtime.workload_plugins import reconstruct_worker_bindings


def test_worker_reconstructs_fixed_state_with_signed_graph_layout_ids() -> None:
    workload = ESampWorkload(
        ESampConfig(
            state_mode="model_bank",
            request_capacity=2,
            distiller_hidden_dim=2,
        )
    )
    runtime = TllmRuntime(
        workloads=(workload,),
        workload_memory_budget_bytes=1 << 20,
    )
    specialization = RuntimeSpecializationSpec(
        2,
        4,
        torch.float32,
        torch.device("cpu"),
    )
    runtime.cold_specialize_workloads(specialization)
    compilation = runtime.compile_workloads(
        export_main_workload_timeline(
            V027MRv2Driver().capabilities(),
            graph_mode="eager",
            static_headroom_bytes=1 << 20,
        )
    )
    worker = workload.native_workload_resources().worker
    assert isinstance(worker, WorkerRuntimeBinding)

    reconstructed = reconstruct_worker_bindings(
        compilation,
        (worker.spec,),
        specialization=specialization,
    )

    assert tuple(reconstructed.request_state_services) == (
        "request_distiller_bank",
    )
