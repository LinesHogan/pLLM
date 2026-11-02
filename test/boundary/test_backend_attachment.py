"""Evidence 15: exact native binding to shared plans, layouts and backend facts."""

import io
import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from weakref import ref

import pytest
import torch

import pllm
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentHiddenResource,
    BackendAttachment,
    ModelProbe,
)
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.hidden_locator import HiddenLayerKind
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.parameter_overlay import (
    Projection,
    ProjectionShape,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from pllm.workloads.esamp import ESampConfig, ESampWorkload
from test.dense_update import (
    DenseUpdate,
    DenseUpdateConfig,
)


def overlay_facts():
    target = Projection("down_proj", 0)
    workload = DenseUpdate(DenseUpdateConfig((target,), chunk_size=2))
    probe = ModelProbe(
        "c" * 64,
        AttachmentGraphMode.EAGER,
        StateDeviceKind.CPU,
        TensorDType.FLOAT32,
        3,
        8,
        (),
        (),
        targets=(ProjectionShape(target, 3, 2),),
    )
    facts = BackendCompileFacts(
        probe,
        model_timeline(torch.device("cpu"), ("state/dense_page_cpu@1", "transition/full_root@1")),
        (PageBundleProviderFacts("test.attachment@1", 4),),
        2,
        4096,
        7,
    )
    return workload, probe, facts


def sampler_deployment(probe, facts):
    esamp = ESampWorkload(ESampConfig(rank=2, source_layer=0, target_layer=1, intervention_beta=0.2))
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
    return esamp.compile_backend(replace(facts, probe=replace(sampler_probe, targets=())))


def test_attachment_inputs_cross_the_wire() -> None:
    workload, probe, facts = overlay_facts()
    for attachment in (workload.compile_backend(facts).attachment, sampler_deployment(probe, facts).attachment):
        reconstructed = decode_boundary(encode_boundary(attachment), BackendAttachment)
        assert reconstructed == attachment


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
            with pytest.raises((AttributeError, ValueError)):
                pllm.make_llm(
                    pllm.ModelConfig("unused-local-model", multiprocessing=multiprocessing),
                    workload=ESampWorkload(),
                )
            assert resource.closed


def test_failed_shutdown_retains_the_shared_input_fd_until_close_retries() -> None:
    pytest.importorskip("vllm")
    from vllm import SamplingParams

    from pllm.backends.vllm import runtime
    from pllm.backends.vllm.prefix_cache import CachePolicy
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
            vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1)),
            abort_request=fail,
            output_processor=SimpleNamespace(request_states={"0": SimpleNamespace(external_req_id="0")}),
        ),
        sleep=lambda **kwargs: None,
        enqueue=lambda *args, **kwargs: ["0"],
        wake_up=lambda **kwargs: None,
        wait_for_completion=fail,
    )
    owned = ref(engine.owned)
    handle = runtime.VllmLlm(engine, None, overlay, facts, CachePolicy.for_attachment(overlay, "isolate"), None, "")
    del engine
    with pytest.raises(RuntimeError):
        handle.generate_completions(["test"], SamplingParams(max_tokens=1), group_id="ownership")
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
