"""Install static KV layouts before vLLM builds its native specs, groups and managers."""

from collections.abc import Callable, Sequence
from functools import partial

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator, KVCacheCoordinatorNoPrefixCache
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, RSWASpec, SlidingWindowSpec

from pllm.backends.vllm.architecture import Decoder
from pllm.backends.vllm.compat import VLLM_027
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


def allocation_blocks(
    coordinator: KVCacheCoordinator,
    native: Callable[..., int],
    request_id: str,
    num_tokens: int,
    new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
    num_encoder_tokens: int,
    total_computed_tokens: int,
    num_local_computed_tokens: int,
    num_tokens_main_model: int,
    apply_admission_cap: bool = False,
    prefill_end: int = 0,
) -> int:
    """Predict running requests once: these native groups have identical logical page counts.

    Recycling leaves null placeholders, so every group's table has the same
    length even though its live pages differ. Admission and imported prefixes
    still use vLLM's complete predictor; native managers own all mutations.
    """
    first = coordinator.single_type_managers[0]
    blocks = first.req_to_blocks.get(request_id)
    if blocks is not None and not apply_admission_cap and total_computed_tokens <= len(blocks) * first.block_size:
        return max((num_tokens + first.block_size - 1) // first.block_size - len(blocks), 0) * len(
            coordinator.single_type_managers
        )
    kwargs = {} if VLLM_027 else {"prefill_end": prefill_end}
    return native(
        request_id,
        num_tokens,
        new_computed_blocks,
        num_encoder_tokens,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=apply_admission_cap,
        **kwargs,
    )


def streamline_allocation(coordinator: KVCacheCoordinator) -> None:
    """Share running-request predictions only for equal-page native static-layout groups."""
    managers = coordinator.single_type_managers
    if (
        isinstance(coordinator, KVCacheCoordinatorNoPrefixCache)
        and len(managers) > 1
        and all(isinstance(item, (FullAttentionManager, SlidingWindowManager)) for item in managers)
        and len({item.block_size for item in managers}) == 1
    ):
        coordinator.get_num_blocks_to_allocate = partial(
            allocation_blocks, coordinator, coordinator.get_num_blocks_to_allocate
        )
        coordinator.allocate_new_blocks = partial(allocate_pages, coordinator, coordinator.allocate_new_blocks)
        coordinator.remove_skipped_blocks = partial(
            reclaim_pages,
            tuple(item for item in managers if isinstance(item.kv_cache_spec, (SlidingWindowSpec, RSWASpec))),
        )


def _covered(coordinator: KVCacheCoordinator, request_id: str, num_tokens: int) -> bool:
    first = coordinator.single_type_managers[0]
    return num_tokens <= len(first.req_to_blocks.get(request_id, ())) * first.block_size


def allocate_pages(
    coordinator: KVCacheCoordinator,
    native: Callable[..., tuple[list[KVCacheBlock], ...]],
    request_id: str,
    num_tokens: int,
    num_tokens_main_model: int,
    num_encoder_tokens: int = 0,
) -> tuple[list[KVCacheBlock], ...]:
    """Leave covered steps alone; native managers append every newly needed page."""
    if _covered(coordinator, request_id, num_tokens):
        return tuple([] for _ in coordinator.single_type_managers)
    return native(request_id, num_tokens, num_tokens_main_model, num_encoder_tokens)


def reclaim_pages(
    managers: tuple[SingleTypeKVCacheManager, ...],
    request_id: str,
    processed_computed_tokens: int,
    num_prompt_tokens: int | None = None,
) -> None:
    """Only window/R-SWA managers can reclaim pages; full managers would do nothing."""
    for manager in managers:
        manager.remove_skipped_blocks(request_id, processed_computed_tokens, num_prompt_tokens)
