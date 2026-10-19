"""Evidence 15: exact native binding to shared plans, layouts and backend facts."""

import json
from dataclasses import replace
from typing import Annotated
from unittest.mock import patch

import pytest
import torch

from pllm import Parameter, _core
from pllm.authoring import BackendCompileFacts, ParameterTargetShape
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentHiddenResource,
    AttachmentTopology,
    BackendPlacementProbe,
    SignedBackendAttachment,
)
from pllm.contracts.hidden_locator import HiddenLayerKind
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.parameter_overlay import (
    ParameterModuleKind,
    ParameterRole,
    ParameterTarget,
)
from pllm.contracts.semantic_graph import (
    HostPayloadSchemaId,
    SemanticOutputKind,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.backend_attachment import (
    compile_backend_attachment,
    lower_backend_attachment,
    validate_backend_attachment_physical,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.tensor_binding import TensorBinding
from pllm.workloads.esamp_request import ESampRequestConfig, ESampRequestWorkload
from pllm.workloads.inplace_ttt import (
    InPlaceTTTConfig,
    InPlaceTTTWorkload,
)


def overlay_facts():
    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=2))
    probe = BackendPlacementProbe(
        "test.backend@1",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        AttachmentTopology.IN_PROCESS,
        AttachmentGraphMode.EAGER,
        StateDeviceKind.CPU,
        0,
        1,
        1,
        TensorDType.FLOAT32,
        3,
        8,
        (),
        (),
    )
    facts = BackendCompileFacts(
        probe,
        RuntimeFacts(torch.device("cpu")).timeline(("state/dense_page_cpu@1", "transition/full_root@1")),
        (PageBundleProviderFacts("test.attachment@1", 4),),
        2,
        4096,
        7,
        (ParameterTargetShape(target, 3, 2),),
    )
    return workload, probe, facts


def sampler_deployment(probe, facts, beta=0.2):
    esamp = ESampRequestWorkload(ESampRequestConfig(rank=2, source_layer=0, target_layer=1, intervention_beta=beta))
    sampler_probe = replace(
        probe,
        hidden_resources=tuple(
            AttachmentHiddenResource(
                PackedHiddenResourceBinding(
                    str(source.layer), HiddenLayerKind.INDEX, source.site, f"model.layers[{source.layer}]"
                ),
                source.phase,
            )
            for source in esamp.backend_requirements().hidden_inputs
        ),
    )
    return esamp.compile_backend(replace(facts, probe=sampler_probe, targets=()))


def test_driver_plans_once_and_worker_binds_only_matching_math_and_facts() -> None:
    workload, probe, facts = overlay_facts()
    with patch.object(PageBundleArena, "__init__", side_effect=AssertionError("planning allocated state")):
        planned = workload.plan_backend(facts)
    with patch.object(_core, "compile_state_schedule", side_effect=AssertionError("worker selected a plan")):
        workload.bind_backend(facts, planned)
    with pytest.raises(ValueError, match="mathematics"):
        InPlaceTTTWorkload(replace(workload.config, learning_rate=0.5)).bind_backend(facts, planned)
    annotations = dict(
        InPlaceTTTWorkload.transition.__annotations__,
        delta=Annotated[torch.Tensor, Parameter("dense_fast_weights", "tail_input")],
    )
    with patch.object(InPlaceTTTWorkload.transition, "__annotations__", annotations):
        with pytest.raises(ValueError, match="mathematics"):
            workload.bind_backend(facts, planned)
    with pytest.raises(ValueError, match="worker backend facts"):
        workload.bind_backend(replace(facts, probe=replace(probe, topology=AttachmentTopology.ENGINE_CORE)), planned)

    class UnconsumedSamplerWorkload(InPlaceTTTWorkload):
        def _build(self, b: TensorBinding) -> None:
            super()._build(b)
            graph = b.graph
            task, other = graph.tasks
            (output,) = task.outputs
            b.graph = replace(
                graph,
                tasks=(
                    replace(
                        task,
                        outputs=(replace(output, kind=SemanticOutputKind.SAMPLER, schema=HostPayloadSchemaId("")),),
                    ),
                    other,
                ),
            )

    with pytest.raises(ValueError, match="composition"):
        UnconsumedSamplerWorkload(workload.config).compile_backend(facts)


def test_signed_attachment_crosses_the_wire_and_rejects_malformed_payloads() -> None:
    workload, probe, facts = overlay_facts()
    for signed in (
        workload.compile_backend(facts).attachment,
        sampler_deployment(probe, facts).attachment,
        sampler_deployment(probe, facts, beta=0.0).attachment,
    ):
        reconstructed = decode_boundary(encode_boundary(signed), SignedBackendAttachment)
        assert reconstructed == signed
        validate_backend_attachment_physical(reconstructed.inputs, reconstructed.decision, reconstructed.physical)
        assert compile_backend_attachment(signed.inputs) == signed.decision
        assert lower_backend_attachment(signed.inputs, signed.decision) == signed.physical
        wire = encode_boundary(signed.inputs)
        for path, key in (
            ((), "plan"),
            (("request",), "state_resource"),
            (("probe",), "model_fingerprint"),
            (("plan",), "tensor_layout_request"),
        ):
            for missing in (True, False):
                payload = json.loads(wire)
                node = payload
                for part in path:
                    node = node[part]
                if missing:
                    del node[key]
                else:
                    node["unknown"] = 0
                with pytest.raises(ValueError):
                    _core.compile_backend_attachment(json.dumps(payload))
        with pytest.raises(ValueError):
            _core.compile_backend_attachment(
                wire.replace('"schema_version":8', '"schema_version":8,"schema_version":8', 1)
            )


def test_vllm_handle_releases_the_engine_on_factory_and_shutdown_failures() -> None:
    pytest.importorskip("vllm")
    import io
    import os
    from types import SimpleNamespace
    from weakref import ref

    from vllm import SamplingParams

    import pllm
    from pllm.backends.vllm import runtime
    from pllm.execution.shared_tensor import CpuFileStorage

    workload, _, facts = overlay_facts()
    overlay = workload.compile_backend(facts).attachment
    # Factory failure must release an already-created engine, before a handle owns it.
    for multiprocessing, message in ((False, "in-process client"), (True, "single-worker quorum")):
        with io.BytesIO(b"owned engine resource") as resource:
            engine = SimpleNamespace(
                llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=resource.close)),
                collective_rpc=lambda *args, **kwargs: (),
            )
            environment = {
                "VLLM_ENABLE_V1_MULTIPROCESSING": "1" if multiprocessing else "0",
                "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            }
            with patch.dict(os.environ, environment), patch.object(runtime, "LLM", return_value=engine):
                with pytest.raises(RuntimeError, match=message):
                    pllm.make_llm(
                        pllm.ModelConfig("unused-local-model", multiprocessing=multiprocessing),
                        workload=ESampRequestWorkload(),
                    )
                assert resource.closed

    # A failed shutdown retains the real shared-input FD until close can retry.
    staged = []
    shutdown_failure = [True]

    def rpc(*args, **kwargs):
        command, payload = kwargs["args"]
        if command == runtime.BackendCommand.STAGE.value:
            staged.append(decode_boundary(payload, runtime.BackendStage))
        return ("",)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic engine failure")

    def shutdown():
        if shutdown_failure[0]:
            raise RuntimeError("synthetic shutdown failure")

    engine = SimpleNamespace(
        owned=torch.empty(128),
        collective_rpc=rpc,
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=shutdown),
            abort_request=fail,
            output_processor=SimpleNamespace(request_states={"0": SimpleNamespace(external_req_id="0")}),
        ),
        sleep=lambda **kwargs: None,
        enqueue=lambda *args, **kwargs: ["0"],
        wake_up=lambda **kwargs: None,
        wait_for_completion=fail,
    )
    owned = ref(engine.owned)
    handle = runtime.VllmLlm(engine, None, None, overlay, facts)
    del engine
    with pytest.raises(RuntimeError, match="synthetic engine failure"):
        handle.generate(["test"], SamplingParams(max_tokens=1), feedback_group="ownership")
    shared = staged[0].completion.storage
    assert isinstance(shared, CpuFileStorage)
    assert os.fstat(shared.descriptor).st_size == facts.probe.hidden_size * 4
    assert owned() is not None
    with pytest.raises(RuntimeError, match="closed"):
        handle.generate(["test"], SamplingParams(max_tokens=1))
    shutdown_failure[0] = False
    handle.close()
    assert owned() is None
    with pytest.raises(OSError):
        os.fstat(shared.descriptor)
    handle.close()
