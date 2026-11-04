"""Install static KV layouts before vLLM builds its native specs, groups and managers."""

from functools import partial

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.v1.kv_cache_interface import KVCacheSpec, RSWASpec, SlidingWindowSpec

from pllm.backends.vllm.architecture import Decoder
from pllm.backends.vllm.model_shape import ModelShape
from pllm.config import ModelConfig
from pllm.contracts.kv_layout import KVCapacity, KVLayout, expand_layouts, storage_layout
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.attention import PAGE_SIZE

LAYOUT_ENV = "PLLM_KV_LAYOUTS"


class SinkSpec(RSWASpec):
    """Keep R-SWA allocation separate from full attention; the native RSWAManager owns blocks."""

    def is_uniform_with_collection(self, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        """Only layers with the same prompt-sink window may share allocation decisions."""
        return all(
            isinstance(spec, RSWASpec) and spec.rswa_window == self.rswa_window for spec in kv_cache_specs.values()
        )


def sink_first(specs: dict[str, KVCacheSpec]) -> dict[str, KVCacheSpec]:
    """vLLM asks the first spec about compatibility; full specs would absorb R-SWA subclasses."""
    return dict(sorted(specs.items(), key=lambda item: not isinstance(item[1], RSWASpec)))


def sink_spec(attention: Attention, window: int, vllm_config: VllmConfig) -> RSWASpec:
    """Use vLLM's prompt-plus-window manager, retaining ordinary unquantized page geometry."""
    return SinkSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=attention.num_kv_heads,
        head_size=attention.head_size,
        dtype=attention.kv_cache_torch_dtype,
        rswa_window=window,
    )


def install_layouts(decoder: Decoder, layouts: tuple[KVLayout, ...]) -> None:
    """Declare native cache ownership before profiling or allocation, including shared chains."""
    if not layouts:
        return
    model = decoder.kv_writable("static KV layouts")
    layouts = expand_layouts(layouts, len(model.model.layers))
    owners = {layout.layer for layout in layouts} | {storage_layout(layouts, layout.layer).layer for layout in layouts}
    for layer in owners:
        attention = model.model.layers[layer].self_attn.attn
        if attention.sliding_window is not None or attention.kv_sharing_target_layer_name is not None:
            raise ValueError("static KV layouts need layers with full attention and their own cache")
        if attention.kv_cache_dtype != "auto":
            raise ValueError("static KV layouts need unquantized KV; set kv_cache_dtype=auto")
    for layout in layouts:
        attention = model.model.layers[layout.layer].self_attn.attn
        storage = storage_layout(layouts, layout.layer)
        if layout.keep == "none":
            target = model.model.layers[storage.layer].self_attn.attn
            attention.kv_sharing_target_layer_name = target.layer_name
        elif layout.keep == "window":
            if layout.sink:
                attention.get_kv_cache_spec = partial(sink_spec, attention, layout.window)
            else:
                attention.sliding_window = layout.window


def layout_capacity(shape: ModelShape, config: ModelConfig, layouts: tuple[KVLayout, ...]) -> tuple[KVCapacity, ...]:
    """Use vLLM's sliding-window admission formula, with FA2's allocation page geometry."""
    owners = {layout.layer: layout for layout in expand_layouts(layouts, shape.layer_count)}
    result: list[KVCapacity] = []
    for layer in range(shape.layer_count):
        layout = owners.get(layer, KVLayout(layer))
        blocks = (config.max_model_len + PAGE_SIZE - 1) // PAGE_SIZE
        if layout.keep == "none":
            blocks = 0
        elif layout.keep == "window" and not layout.sink:
            spec = SlidingWindowSpec(
                block_size=PAGE_SIZE,
                num_kv_heads=shape.kv_heads,
                head_size=shape.head_dim,
                dtype=TensorDType(config.dtype).torch,
                sliding_window=layout.window,
            )
            blocks = spec.max_admission_blocks_per_request(
                max_in_flight_tokens=(2 if config.runner == "mrv2" else 1) * config.max_num_batched_tokens,
                max_model_len=config.max_model_len,
            )
        result.append(
            KVCapacity(
                layout,
                PAGE_SIZE,
                blocks,
                2 * PAGE_SIZE * shape.kv_heads * shape.head_dim * TensorDType(config.dtype).torch.itemsize,
            )
        )
    return tuple(result)
