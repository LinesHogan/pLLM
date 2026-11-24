"""Model dimensions and LM-head eligibility shared by CPU diagnostics and loaded backend probes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

import torch
from cattrs import structure
from vllm.config import LoadConfig, ModelConfig as VllmModelConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    pad_vocab_size,
)
from vllm.model_executor.model_loader.weight_utils import get_quant_config
from vllm.model_executor.models import ModelRegistry

from pllm.backends.vllm.architecture import (
    DENSE_ARCHITECTURES,
    HYBRID_NAMES,
    KV_MOE_ARCHITECTURES,
    hidden_binding,
    moe_projection_error,
    post_attention_norm_name,
)
from pllm.config import ModelConfig
from pllm.contracts.hidden_locator import (
    HiddenLayerKind,
    HiddenSite,
    LayerRouteSelector,
    resolve_route_layers,
    route_locator_tag,
)
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.parameter_overlay import Projection, ProjectionName, ProjectionShape, expand_targets

if TYPE_CHECKING:
    from pllm.backends.vllm.architecture import Decoder


_DENSE_NAMES = tuple(cls.__name__ for cls in DENSE_ARCHITECTURES)
_KV_MOE_NAMES = tuple(cls.__name__ for cls in KV_MOE_ARCHITECTURES)


@dataclass(frozen=True)
class _ConfigShape:
    architectures: tuple[str, ...]
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    num_attention_heads: int
    intermediate_size: int = 0
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    logit_scale: float = 1.0
    tie_word_embeddings: bool = False
    final_logit_softcapping: float | None = None
    mlp_only_layers: tuple[int, ...] = ()
    decoder_sparse_step: int = 1
    num_experts: int = 0
    num_local_experts: int = 0
    layer_types: tuple[str, ...] | None = None
    attn_output_gate: bool = True
    v_head_dim: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None

    @property
    def query_width(self) -> int:
        """Features per query head before MLA absorption or dense attention."""
        if self.kv_lora_rank:
            return (self.qk_nope_head_dim or 0) + (self.qk_rope_head_dim or 0)
        return self.head_dim or self.hidden_size // self.num_attention_heads

    def routed_mlp(self, layer: int) -> str | None:
        """Identify routed layers; Qwen MoE configs can also include dense layers."""
        name = self.architectures[0]
        if name == "Qwen3_5MoeForCausalLM":
            return "Qwen3NextSparseMoeBlock"
        if name == "MixtralForCausalLM":
            return "MixtralDecoderLayer.block_sparse_moe"
        if (
            name in _KV_MOE_NAMES
            and (self.num_experts or self.num_local_experts)
            and layer not in self.mlp_only_layers
            and (layer + 1) % self.decoder_sparse_step == 0
        ):
            return name.removesuffix("ForCausalLM") + "SparseMoeBlock"
        return None

    def forward_flops(self) -> int | None:
        """Dense projection matmuls per token, excluding attention over the growing KV history."""
        if self.architectures[0] not in _DENSE_NAMES:
            return None
        head = self.head_dim or self.hidden_size // self.num_attention_heads
        query = head * self.num_attention_heads
        kv = head * (self.num_key_value_heads or self.num_attention_heads)
        parameters = self.num_hidden_layers * self.hidden_size * (2 * query + 2 * kv + 3 * self.intermediate_size)
        return 2 * (parameters + self.hidden_size * pad_vocab_size(self.vocab_size))

    def projections(self, targets: tuple[Projection, ...], size: int) -> tuple[ProjectionShape, ...]:
        if targets and self.architectures[0] not in (*_DENSE_NAMES, *_KV_MOE_NAMES, *HYBRID_NAMES):
            raise ValueError(
                "projection shapes for this architecture are decided at make_llm; use a dense or KV MoE config for CPU checks"
            )
        head = self.head_dim or self.hidden_size // self.num_attention_heads
        kv_heads = self.num_key_value_heads or self.num_attention_heads
        sizes = {
            "q_proj": (self.hidden_size, head * self.num_attention_heads),
            "k_proj": (self.hidden_size, head * kv_heads),
            "v_proj": (self.hidden_size, head * kv_heads),
            "o_proj": (head * self.num_attention_heads, self.hidden_size),
            "gate_proj": (self.hidden_size, self.intermediate_size),
            "up_proj": (self.hidden_size, self.intermediate_size),
            "down_proj": (self.intermediate_size, self.hidden_size),
        }
        expanded = expand_targets(targets, self.num_hidden_layers)
        if any(target.layer is None or target.layer >= self.num_hidden_layers for target in expanded):
            raise ValueError("choose projection layers below the model's num_hidden_layers")
        projections: list[ProjectionShape] = []
        heads = {
            "q_proj": self.num_attention_heads,
            "k_proj": kv_heads,
            "v_proj": kv_heads,
            "o_proj": self.num_attention_heads,
        }
        for target in expanded:
            assert target.layer is not None
            if target.name not in heads and (routed := self.routed_mlp(target.layer)):
                dense = tuple(i for i in range(self.num_hidden_layers) if not self.routed_mlp(i))
                err = moe_projection_error(target.layer, routed)
                if dense:
                    err = ValueError(f"{err}; dense MLP layers: {dense}")
                raise err
            if target.name in heads and self.architectures[0] in HYBRID_NAMES:
                assert self.layer_types is not None
                if self.layer_types[target.layer] != "full_attention":
                    full = tuple(i for i, kind in enumerate(self.layer_types) if kind == "full_attention")
                    raise ValueError(
                        f"layer {target.layer} is a Gated DeltaNet layer; choose a full-attention layer {full} for q/k/v overlays"
                    )
                if target.name == "q_proj" and self.attn_output_gate:
                    raise ValueError("gated attention interleaves query and output-gate rows; overlay k_proj or v_proj")
            inputs, outputs = sizes[target.name]
            partitions = _head_partitions(target.name, heads[target.name], size) if target.name in heads else size
            # o_proj and down_proj are row-parallel and split their input features; the others split their outputs.
            row = target.name in ("o_proj", "down_proj")
            if (inputs if row else outputs) % partitions:
                raise ValueError(f"{target.name} features must divide tensor_parallel_size evenly")
            projections.append(
                ProjectionShape(
                    target,
                    inputs // partitions if row else inputs,
                    outputs if row else outputs // partitions,
                    input_partitions=partitions if row else 1,
                    output_partitions=1 if row else partitions,
                )
            )
        return tuple(projections)


def _head_partitions(name: str, heads: int, size: int) -> int:
    """How many ranks split an attention projection: whole heads per rank, or fewer KV heads replicated."""
    if heads % size and size % heads:  # neither divides the other
        raise ValueError(f"{name} heads must divide tensor_parallel_size or partition evenly across it")
    if name == "q_proj" and heads < size:
        raise ValueError("q_proj needs at least tensor_parallel_size query heads")
    return min(heads, size)


def _head_rejection(model: VllmModelConfig, shape: _ConfigShape, architecture: str) -> str | None:
    from pllm.backends.vllm.architecture import guidance_head_rejection

    # Ask vLLM which method its head would use, without allocating or loading base weights.
    tied_embedding = architecture == "Gemma2ForCausalLM" or (
        architecture in ("Glm4ForCausalLM", "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM") and shape.tie_word_embeddings
    )
    with torch.device("meta"):
        head = (VocabParallelEmbedding if tied_embedding else ParallelLMHead)(
            shape.vocab_size, shape.hidden_size, params_dtype=model.dtype, disable_tp=True
        )
    # GptOss constructs its head without quant_config, even when its experts are quantized.
    if model.quantization is not None and architecture != "GptOssForCausalLM":
        quantization = get_quant_config(model, LoadConfig())
        quantization.maybe_update_config(model.model, hf_config=model.hf_config, revision=model.revision)
        method = quantization.get_quant_method(head, prefix="model.embed_tokens" if tied_embedding else "lm_head")
        if method is not None:
            head.quant_method = method
    scale = shape.logit_scale if architecture in ("LlamaForCausalLM", "MistralForCausalLM", "Phi3ForCausalLM") else 1.0
    return guidance_head_rejection(head, scale, model.head_dtype, shape.final_logit_softcapping)


@dataclass(frozen=True, slots=True)
class ModelShape:
    """Geometry shared by config-only planning and loaded-model probing, without weights."""

    architecture: str
    hidden_size: int
    layer_count: int
    vocab_size: int
    padded_vocab_size: int
    targets: tuple[ProjectionShape, ...]
    read_only_reason: str | None = None
    base_forward_flops_per_token_estimate: int | None = None
    attention_heads: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    guidance_rejection: str | None = None
    v_head_dim: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0

    def hidden_binding(
        self, layer: int, site: HiddenSite, route: LayerRouteSelector | None = None
    ) -> PackedHiddenResourceBinding:
        """Map a declared site or runtime LayerRoute to its residual-add or routed pool binding."""
        if route is not None:
            resolved = replace(route, over=resolve_route_layers(route.over, self.layer_count))
            tag = route_locator_tag(resolved)
            return PackedHiddenResourceBinding(tag, HiddenLayerKind.WILDCARD, site, f"routed:{tag}:{site.value}", 0)
        return hidden_binding(layer, site, self.layer_count, post_attention_norm_name(self.architecture))

    @classmethod
    def of_config(
        cls,
        config: ModelConfig,
        targets: tuple[Projection, ...] = (),
        routed_targets: tuple[tuple[ProjectionName, tuple[int, ...]], ...] = (),
    ) -> Self:
        """Read only config.json; dense projection widths follow the decoder's q/v/down layouts."""
        from pllm.backends.vllm.architecture import admit_architecture

        model = VllmModelConfig(
            model=config.model,
            revision=config.revision,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            trust_remote_code=config.trust_remote_code,
        )
        # vLLM inspects lazy model classes in a subprocess; importing hybrids here would initialize CUDA.
        info, _ = ModelRegistry.inspect_model_cls(model.architectures, model)
        read_only = admit_architecture(info.architecture)
        if (targets or routed_targets) and read_only:
            raise ValueError(f"a parameter overlay {read_only}")
        shape = structure({**model.hf_text_config.to_dict(), "architectures": [info.architecture]}, _ConfigShape)
        resolved_routed = tuple(
            Projection(name, layer)
            for name, over in routed_targets
            for layer in resolve_route_layers(over, shape.num_hidden_layers)
        )
        all_targets = tuple(dict.fromkeys((*targets, *resolved_routed)))
        return cls(
            info.architecture,
            shape.hidden_size,
            shape.num_hidden_layers,
            shape.vocab_size,
            pad_vocab_size(shape.vocab_size),
            shape.projections(all_targets, config.tensor_parallel_size),
            read_only,
            shape.forward_flops(),
            shape.num_attention_heads // config.tensor_parallel_size,
            max(1, (shape.num_key_value_heads or shape.num_attention_heads) // config.tensor_parallel_size),
            shape.query_width,
            shape.intermediate_size,
            _head_rejection(model, shape, info.architecture),
            v_head_dim=shape.v_head_dim or 0,
            kv_lora_rank=shape.kv_lora_rank or 0,
            qk_nope_head_dim=shape.qk_nope_head_dim or 0,
            qk_rope_head_dim=shape.qk_rope_head_dim or 0,
        )

    @classmethod
    def of_model(
        cls,
        decoder: Decoder,
        targets: tuple[Projection, ...],
        dtype: torch.dtype,
        routed_targets: tuple[tuple[ProjectionName, tuple[int, ...]], ...] = (),
    ) -> Self:
        """Inspect the actual loaded projection slices, including architecture-specific admission."""
        from pllm.backends.vllm.architecture import guidance_rejection, lm_head_module
        from pllm.backends.vllm.parameter_overlay import inspect_parameter_targets

        model = decoder.model
        head = lm_head_module(model)
        tp_size = get_tensor_model_parallel_world_size()
        layers = len(model.model.layers)
        resolved_routed = tuple(
            Projection(name, layer) for name, over in routed_targets for layer in resolve_route_layers(over, layers)
        )
        all_targets = tuple(dict.fromkeys((*expand_targets(targets, layers), *resolved_routed)))
        shapes = inspect_parameter_targets(decoder, all_targets, dtype)
        config = structure({**model.config.to_dict(), "architectures": [type(model).__name__]}, _ConfigShape)
        return cls(
            type(model).__name__,
            model.config.hidden_size,
            layers,
            model.config.vocab_size,
            head.num_embeddings_padded,
            shapes,
            decoder.read_only,
            config.forward_flops(),
            config.num_attention_heads // tp_size,
            max(1, (config.num_key_value_heads or config.num_attention_heads) // tp_size),
            config.query_width,
            config.intermediate_size,
            guidance_rejection(head, model.logits_processor),
            v_head_dim=config.v_head_dim or 0,
            kv_lora_rank=config.kv_lora_rank or 0,
            qk_nope_head_dim=config.qk_nope_head_dim or 0,
            qk_rope_head_dim=config.qk_rope_head_dim or 0,
        )
