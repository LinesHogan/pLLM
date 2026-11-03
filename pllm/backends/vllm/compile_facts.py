"""Pure backend planning facts shared by the vLLM probe and CPU diagnostics."""

from dataclasses import replace
from hashlib import sha256

import torch

from pllm.authoring import BackendRequirements
from pllm.backends.vllm.architecture import hidden_binding, kv_rejection
from pllm.backends.vllm.hidden_taps import HiddenTapLayout
from pllm.backends.vllm.model_shape import ModelShape
from pllm.backends.vllm.overlay_graph import graph_mode
from pllm.config import ModelConfig
from pllm.contracts.backend_attachment import (
    TOKEN_CAPTURE_CAPABILITY,
    AttachmentGraphMode,
    AttachmentHiddenResource,
    CausalModelDomain,
    ModelProbe,
)
from pllm.contracts.hidden_locator import HiddenPhase, logit_layers
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import SchedulerTimeline
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline

_REQUEST_CAPABILITIES = (
    "producer/packed_hidden@1",
    "sampler/post_filter_exact@1",
    "sampler/pre_filter_add@1",
)


_OVERLAY_CAPABILITIES = (
    "operand_capture/mlp_down_projection_pair@1",
    "overlay/dense_add@1",
    "overlay/low_rank_add@1",
    "overlay/head_add_fp32@1",
    "operand_capture/projection_input@1",
)


def _digest(fields: tuple[str, ...]) -> str:
    return sha256("\x00".join(fields).encode("utf-8")).hexdigest()


def compile_facts(
    shape: ModelShape,
    requirements: BackendRequirements,
    config: ModelConfig,
    *,
    model_identity: str,
    hidden_layouts: tuple[HiddenTapLayout, ...],
    overlay_graph_bytes: int,
    memory_bytes: int,
    checkpoint: str = "",
    blas_workspace_bytes: int = 0,
) -> BackendCompileFacts:
    """Derive reservations and capabilities without allocating tensors or querying a device."""
    publish_lag = config.execution.publish_lag
    feature = "a PositionState" if requirements.position_states else "a @pllm.custom_forward"
    if requirements.position_states or requirements.custom_forward:
        if rejection := shape.read_only_reason or kv_rejection(shape.architecture):
            raise ValueError(f"{feature} {rejection}")
    if requirements.causal_training and shape.read_only_reason:
        raise ValueError(f"causal training {shape.read_only_reason}")
    if any(layer < 0 or layer >= shape.layer_count for layer in requirements.activation_layers):
        raise ValueError("activation layer is outside the backend model")
    if any(not -shape.layer_count <= layer < shape.layer_count for layer in requirements.layer_logits):
        raise ValueError(
            f"pllm.layer_logits{requirements.layer_logits} names a layer outside the model's {shape.layer_count} layers"
        )
    layer_logits = logit_layers(requirements.layer_logits, shape.layer_count)
    hidden_resources = tuple(
        sorted(
            (
                AttachmentHiddenResource(hidden_binding(source.layer, source.site, shape.layer_count), source.phase)
                for source in requirements.hidden_inputs
            ),
            key=lambda resource: (*resource.resource_key, resource.phase is HiddenPhase.DECODE),
        )
    )
    custom_forward_capabilities = (
        ("model/custom_forward@1", TOKEN_CAPTURE_CAPABILITY)
        if config.runner == "mrv2" and graph_mode(overlay_graph_bytes) is AttachmentGraphMode.EAGER
        else ()
    )
    probe = ModelProbe(
        model_fingerprint=_digest(
            (
                f"model={config.model}",
                f"revision={config.revision or ''}",
                f"class={model_identity}",
                f"hidden_size={shape.hidden_size}",
                f"layers={shape.layer_count}",
                f"vocab_size={shape.vocab_size}",
                *(f"hidden={resource.binding.tap}" for resource in hidden_resources),
                f"layer_logits={layer_logits}",
                *(f"activation_layer={layer}" for layer in requirements.activation_layers),
                *(f"target={target.target}:{target.input_size}:{target.output_size}" for target in shape.targets),
            )
        ),
        hidden_size=shape.hidden_size,
        graph_mode=graph_mode(overlay_graph_bytes),
        device_kind=StateDeviceKind.CUDA,
        tensor_dtype=TensorDType(config.dtype),
        max_packed_rows=config.max_num_batched_tokens,
        causal_domain=CausalModelDomain(config.max_num_seqs, shape.layer_count, config.max_model_len),
        hidden_resources=hidden_resources,
        layer_logits=layer_logits,
        blas_workspace_bytes=blas_workspace_bytes,
        capabilities=tuple(
            sorted(
                set(
                    (
                        *_REQUEST_CAPABILITIES,
                        *_OVERLAY_CAPABILITIES,
                        *custom_forward_capabilities,
                        f"backend/vllm_{config.runner}@1",
                    )
                )
            )
        ),
    )
    observed = model_timeline(
        torch.device("cuda"),
        (
            "state/dense_page_cuda@1",
            "state/fenced_reclaim@1",
            "transition/full_root@1",
            "optimizer/adamw@1",
            *(("backend/tensor_parallel@1",) if config.tensor_parallel_size > 1 else ()),
            *custom_forward_capabilities,
        ),
    )
    required = input_reservations(
        probe,
        hidden_layouts,
        observed,
        config.max_num_seqs,
        shape.padded_vocab_size,
        requirements.sampler_stage,
        requirements.sampler_feedback,
    )
    required = tuple(value + overlay_graph_bytes for value in required)
    # Exact in-forward transitions retain current/private roots, without a
    # post-sampler history. Released GPU reads are fenced by SlotPools on reuse;
    # they do not need another physically reserved root per request.
    capacity = (2 if requirements.exact_decode else 3 + publish_lag) * config.max_num_seqs + 2
    facts = BackendCompileFacts(
        replace(probe, targets=shape.targets, position_states=requirements.position_states),
        replace(observed, reserved_bytes=required),
        (
            PageBundleProviderFacts(
                "page_bundle_cuda@1",
                capacity,
                probe.device_kind,
            ),
        ),
        config.max_num_seqs,
        memory_bytes,
        config.seed,
        shape.vocab_size,
        shape.layer_count,
        checkpoint,
    )
    return facts


def input_reservations(
    probe: ModelProbe,
    hidden_layouts: tuple[HiddenTapLayout, ...],
    timeline: SchedulerTimeline,
    rows: int,
    vocab: int,
    stage: SamplerStage | None,
    feedback: bool,
) -> tuple[int, ...]:
    """Reserve persistent capture storage and sampler scratch across model-step anchors."""
    # Layouts union aliases and phases per module. Slots and capture indices stay
    # resident across steps; task_memory traces consumer tables, gathers and casts.
    hidden = sum(layout.resident_bytes for layout in hidden_layouts)
    # FP32 envelope for original logits, conversion, projection and rowwise arithmetic.
    # Excludes upstream sampling/GEMM library scratch, model weights and KV storage.
    workspace = (
        rows * (36 * vocab + 4 * probe.hidden_size + 8)
        if stage is SamplerStage.PACKED_POST_FILTER_EXACT
        else rows * (12 * vocab + 8)
        if stage is SamplerStage.PACKED_PRE_FILTER_ADD
        else 0
    )
    # Layer-logit projection retains gathered streams, norm operands and both model/FP32 logits.
    width = probe.tensor_dtype.itemsize
    workspace += rows * len(probe.layer_logits) * (2 * width * probe.hidden_size + (width + 4) * vocab)
    retained = rows * (4 * vocab + 8) if feedback else 0
    # StepCapture retains its feedback pointers until the next table is prepared.
    # Budget both old and new private copies during handoff, including next forward.
    private_feedback = 2 * retained
    sampler = next(anchor.ordinal for anchor in timeline.anchors if anchor.symbol == "sampler.pre_draw")
    return tuple(
        hidden
        + private_feedback
        + (workspace if anchor.symbol in ("sampler.pre_draw", "sampler.complete") else 0)
        + (retained if anchor.ordinal >= sampler else 0)
        for anchor in timeline.anchors
    )
