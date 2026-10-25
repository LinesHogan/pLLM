"""Evidence 15: exact native binding to shared plans, layouts and backend facts."""

import io
import json
import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from weakref import ref

import pytest
import torch

import pllm
from pllm import _core
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
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.backend_attachment import (
    compile_backend_attachment,
    lower_backend_attachment,
    validate_backend_attachment_physical,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.compile_facts import BackendCompileFacts, ParameterTargetShape
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
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


def sampler_deployment(probe, facts):
    esamp = ESampRequestWorkload(ESampRequestConfig(rank=2, source_layer=0, target_layer=1, intervention_beta=0.2))
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


def test_bind_rejects_a_plan_built_for_other_math_or_other_backend_facts() -> None:
    workload, probe, facts = overlay_facts()
    planned = workload.plan_backend(facts)
    with pytest.raises(ValueError):
        InPlaceTTTWorkload(replace(workload.config, learning_rate=0.5)).bind_backend(facts, planned)
    with pytest.raises(ValueError):
        workload.bind_backend(replace(facts, probe=replace(probe, topology=AttachmentTopology.ENGINE_CORE)), planned)


def test_signed_attachment_crosses_the_wire_and_rejects_malformed_payloads() -> None:
    workload, probe, facts = overlay_facts()
    for signed in (workload.compile_backend(facts).attachment, sampler_deployment(probe, facts).attachment):
        reconstructed = decode_boundary(encode_boundary(signed), SignedBackendAttachment)
        assert reconstructed == signed
        validate_backend_attachment_physical(reconstructed.inputs, reconstructed.decision, reconstructed.physical)
        assert compile_backend_attachment(signed.inputs) == signed.decision
        assert lower_backend_attachment(signed.inputs, signed.decision) == signed.physical
        payload = json.loads(encode_boundary(signed.inputs))
        payload["probe"]["unknown"] = 0
        with pytest.raises(ValueError):
            _core.compile_backend_attachment(json.dumps(payload))


@pytest.mark.parametrize("multiprocessing", (False, True))
def test_factory_failure_releases_an_already_created_engine(multiprocessing: bool) -> None:
    pytest.importorskip("vllm")
    from pllm.backends.vllm import runtime

    # A stub engine stands in for vLLM so the factory failure path runs without a GPU.
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
        # Inject the engine at the vLLM constructor boundary.
        with patch.dict(os.environ, environment), patch.object(runtime, "LLM", return_value=engine):
            with pytest.raises(RuntimeError):
                pllm.make_llm(
                    pllm.ModelConfig("unused-local-model", multiprocessing=multiprocessing),
                    workload=ESampRequestWorkload(),
                )
            assert resource.closed


def test_failed_shutdown_retains_the_shared_input_fd_until_close_retries() -> None:
    pytest.importorskip("vllm")
    from vllm import SamplingParams

    from pllm.backends.vllm import runtime
    from pllm.execution.shared_tensor import CpuFileStorage

    workload, _, facts = overlay_facts()
    overlay = workload.compile_backend(facts).attachment
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

    # A stub engine stands in for vLLM so failure paths run without a GPU.
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
    with pytest.raises(RuntimeError):
        handle.generate(["test"], SamplingParams(max_tokens=1), feedback_group="ownership")
    shared = staged[0].completion.storage
    assert isinstance(shared, CpuFileStorage)
    assert owned() is not None
    with pytest.raises(RuntimeError):
        handle.generate(["test"], SamplingParams(max_tokens=1))
    shutdown_failure[0] = False
    handle.close()
    assert owned() is None
    with pytest.raises(OSError):
        os.fstat(shared.descriptor)
    handle.close()
