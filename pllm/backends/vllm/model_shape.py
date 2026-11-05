"""Model dimensions read by CPU diagnostics and by the loaded vLLM backend probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from cattrs import structure
from vllm.config import ModelConfig as VllmModelConfig
from vllm.model_executor.layers.vocab_parallel_embedding import pad_vocab_size
from vllm.model_executor.models import ModelRegistry

from pllm.config import ModelConfig
from pllm.contracts.parameter_overlay import Projection, ProjectionShape, expand_targets

if TYPE_CHECKING:
    import torch

    from pllm.backends.vllm.architecture import Decoder


_DENSE_ARCHITECTURES = ("Qwen2ForCausalLM", "Qwen3ForCausalLM", "LlamaForCausalLM")


@dataclass(frozen=True)
class _ConfigShape:
    architectures: tuple[str, ...]
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int | None = None
    head_dim: int | None = None

    def forward_flops(self) -> int | None:
        """Dense projection matmuls per token, excluding attention over the growing KV history."""
        if self.architectures[0] not in _DENSE_ARCHITECTURES:
            return None
        head = self.head_dim or self.hidden_size // self.num_attention_heads
        query = head * self.num_attention_heads
        kv = head * (self.num_key_value_heads or self.num_attention_heads)
        parameters = self.num_hidden_layers * self.hidden_size * (2 * query + 2 * kv + 3 * self.intermediate_size)
        return 2 * (parameters + self.hidden_size * pad_vocab_size(self.vocab_size))

    def projections(self, targets: tuple[Projection, ...]) -> tuple[ProjectionShape, ...]:
        if targets and self.architectures[0] not in _DENSE_ARCHITECTURES:
            raise ValueError(
                "projection shapes for this architecture are decided at make_llm; use a dense Qwen2, Qwen3 or Llama config for CPU checks"
            )
        head = self.head_dim or self.hidden_size // self.num_attention_heads
        kv_heads = self.num_key_value_heads or self.num_attention_heads
        sizes = {
            "q_proj": (self.hidden_size, head * self.num_attention_heads),
            "v_proj": (self.hidden_size, head * kv_heads),
            "down_proj": (self.intermediate_size, self.hidden_size),
        }
        expanded = expand_targets(targets, self.num_hidden_layers)
        if any(target.layer is None or target.layer >= self.num_hidden_layers for target in expanded):
            raise ValueError("choose projection layers below the model's num_hidden_layers")
        return tuple(ProjectionShape(target, *sizes[target.name]) for target in expanded)


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

    @classmethod
    def of_config(cls, config: ModelConfig, targets: tuple[Projection, ...] = ()) -> Self:
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
        if targets and read_only:
            raise ValueError(f"a parameter overlay {read_only}")
        shape = structure(model.hf_text_config.to_dict(), _ConfigShape)
        return cls(
            info.architecture,
            shape.hidden_size,
            shape.num_hidden_layers,
            shape.vocab_size,
            pad_vocab_size(shape.vocab_size),
            shape.projections(targets),
            read_only,
            shape.forward_flops(),
            shape.num_attention_heads,
            shape.num_key_value_heads or shape.num_attention_heads,
            shape.head_dim or shape.hidden_size // shape.num_attention_heads,
            shape.intermediate_size,
        )

    @classmethod
    def of_model(cls, decoder: Decoder, targets: tuple[Projection, ...], dtype: torch.dtype) -> Self:
        """Inspect the actual loaded projection slices, including architecture-specific admission."""
        from pllm.backends.vllm.parameter_overlay import inspect_parameter_targets

        model = decoder.model
        layers = len(model.model.layers)
        shapes = inspect_parameter_targets(decoder, expand_targets(targets, layers), dtype)
        config = structure(model.config.to_dict(), _ConfigShape)
        return cls(
            type(model).__name__,
            model.config.hidden_size,
            layers,
            model.config.vocab_size,
            model.lm_head.num_embeddings_padded,
            shapes,
            decoder.read_only,
            config.forward_flops(),
            config.num_attention_heads,
            config.num_key_value_heads or config.num_attention_heads,
            config.head_dim or config.hidden_size // config.num_attention_heads,
            config.intermediate_size,
        )
