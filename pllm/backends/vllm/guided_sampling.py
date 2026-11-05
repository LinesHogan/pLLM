"""Bounded indirect LM-head projection and draw used by the vLLM post-filter sampler.

vLLM exposes filtering and Gumbel-max, but no draw that projects hidden predictions through
selected embedding rows. This kernel keeps its filter output and calls its Gumbel helper,
including the request seed, token position, precision and greedy behavior.
"""

import torch
import triton
import triton.language as tl
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax

from pllm.backends.vllm.compat import VLLM_027


@triton.jit
def _keys(logits: tl.tensor, tokens: tl.tensor, finite: tl.tensor) -> tl.tensor:
    bits = logits.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    return tl.where(finite, (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - tokens.to(tl.uint32)), 0)


@triton.jit
def _decode(keys: tl.tensor) -> tuple[tl.tensor, tl.tensor]:
    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where(ordered & 0x80000000 != 0, ordered ^ 0x80000000, ~ordered)
    return bits.to(tl.float32, bitcast=True), (0xFFFFFFFF - keys.to(tl.uint32)).to(tl.int32)


@triton.jit
def _draw(
    logits: tl.tensor,
    tokens: tl.tensor,
    mask: tl.tensor,
    row: tl.tensor,
    Mapping: torch.Tensor,
    Temperature: torch.Tensor,
    Seeds: torch.Tensor,
    Positions: torch.Tensor,
    VOCAB: tl.constexpr,
    FP64: tl.constexpr,
    V027: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor]:
    logits = tl.where(mask, logits, float("-inf"))
    if V027:
        value, index = gumbel_block_argmax(
            logits,
            tokens,
            mask,
            row,
            Mapping,
            Temperature,
            Seeds,
            Positions,
            None,
            0,
            None,
            VOCAB,
            APPLY_TEMPERATURE=False,
            USE_FP64=FP64,
        )
    else:
        value, index = gumbel_block_argmax(
            logits,
            tokens,
            mask,
            row,
            Mapping,
            Temperature,
            Seeds,
            Positions,
            None,
            0,
            0,
            None,
            None,
            0,
            VOCAB,
            IS_DRAFTING=False,
            APPLY_TEMPERATURE=False,
            USE_FP64=FP64,
        )
    token = tl.sum(tl.where(tl.arange(0, logits.shape[0]) == index, tokens, 0), 0)
    return value, token


@triton.jit
def _scan_survivors(
    Logits: torch.Tensor,
    Scratch: torch.Tensor,
    row: tl.tensor,
    guided: tl.tensor,
    Mapping: torch.Tensor,
    Temperature: torch.Tensor,
    Seeds: torch.Tensor,
    Positions: torch.Tensor,
    TopP: torch.Tensor | None,
    LS: tl.constexpr,
    VOCAB: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    FP64: tl.constexpr,
    V027: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    offsets = tl.arange(0, BLOCK)
    slots = tl.arange(0, K)
    count = tl.full((), 0, tl.int32)
    best = tl.full((), float("-inf"), tl.float64 if FP64 else tl.float32)
    winner = tl.full((), 0, tl.int32)
    full = guided
    peak_key = tl.full((), 0, tl.uint64)
    maximum = tl.full((), float("-inf"), tl.float32)
    mass = tl.full((), 0, tl.float32)
    for start in range(tl.cdiv(VOCAB, BLOCK)):
        tokens = start * BLOCK + offsets
        logits = tl.load(Logits + row * LS + tokens, tokens < VOCAB, other=float("-inf")).to(tl.float32)
        finite = (logits != float("-inf")) & (logits != float("inf")) & (logits == logits)
        added = tl.sum(finite.to(tl.int32), 0)
        keys = _keys(logits, tokens, finite)
        if TopP is not None:
            full = full & (added == tl.minimum(tl.full((), BLOCK, tl.int32), VOCAB - start * BLOCK))
            if full:
                tile_maximum = tl.max(logits, 0)
                updated_maximum = tl.maximum(maximum, tile_maximum)
                mass = mass * tl.exp(maximum - updated_maximum) + tl.sum(tl.exp(logits - updated_maximum), 0)
                maximum = updated_maximum
                peak_key = tl.maximum(peak_key, tl.max(keys, 0))
        if guided & (count + added <= K):
            ranks = tl.cumsum(finite.to(tl.int32), 0) - 1
            tl.store(Scratch + row * K + count + ranks, keys, finite)
        else:
            if guided & (count > 0) & (count <= K):
                tl.debug_barrier()
                previous = tl.load(Scratch + row * K + slots, slots < count, other=0)
                values, ids = _decode(previous)
                best, winner = _draw(
                    values, ids, previous != 0, row, Mapping, Temperature, Seeds, Positions, VOCAB, FP64, V027
                )
            value, token = _draw(
                logits, tokens, tokens < VOCAB, row, Mapping, Temperature, Seeds, Positions, VOCAB, FP64, V027
            )
            replace = (value > best) | ((value == best) & (token < winner))
            best, winner = tl.maximum(best, value), tl.where(replace, token, winner)
        count += added
    if TopP is not None:
        p = tl.load(TopP + row)
        peak_key = tl.where(full & (p < 1) & (1.0 / mass >= p), peak_key, 0)
    return count, best, winner, peak_key


@triton.jit
def _sample(
    Logits: torch.Tensor,
    Hidden: torch.Tensor,
    Weight: torch.Tensor,
    RowMap: torch.Tensor,
    RowWeights: torch.Tensor | None,
    Scratch: torch.Tensor,
    Output: torch.Tensor,
    Mapping: torch.Tensor,
    Temperature: torch.Tensor,
    Seeds: torch.Tensor,
    Positions: torch.Tensor,
    TopP: torch.Tensor | None,
    LS: tl.constexpr,
    HS: tl.constexpr,
    WS: tl.constexpr,
    VOCAB: tl.constexpr,
    WIDTH: tl.constexpr,
    BETA: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    DOT_ROWS: tl.constexpr,
    DOT_WIDTH: tl.constexpr,
    FP64: tl.constexpr,
    V027: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    prediction_row = tl.load(RowMap + row)
    beta = tl.full((), BETA, tl.float32)
    if RowWeights is not None:
        beta *= tl.load(RowWeights + prediction_row, prediction_row >= 0, other=0).to(tl.float32)
    guided = (prediction_row >= 0) & (beta != 0)
    count, best, winner, peak_key = _scan_survivors(
        Logits,
        Scratch,
        row,
        guided,
        Mapping,
        Temperature,
        Seeds,
        Positions,
        TopP,
        LS,
        VOCAB,
        K,
        BLOCK,
        FP64,
        V027,
    )
    if guided:
        if count > K:
            count = tl.full((), 0, tl.int32)
        if peak_key != 0:
            # The upstream filter skipped its mask; explicitly use the true one-token nucleus.
            _, peak_token = _decode(peak_key)
            for start in range(tl.cdiv(VOCAB, BLOCK)):
                tokens = start * BLOCK + tl.arange(0, BLOCK)
                tl.store(Logits + row * LS + tokens, tl.full((BLOCK,), float("-inf"), tl.float32), tokens < VOCAB)
            tl.store(Scratch + row * K, peak_key)
            count = tl.full((), 1, tl.int32)
            best = tl.full((), float("-inf"), tl.float64 if FP64 else tl.float32)
            winner = peak_token
        tl.debug_barrier()
        # Only finite selected survivors fetch embedding rows. No overflow ever projects the vocabulary.
        for group in tl.range(tl.cdiv(count, DOT_ROWS)):
            slot = group * DOT_ROWS + tl.arange(0, DOT_ROWS)
            selected_keys = tl.load(Scratch + row * K + slot, slot < count, other=0)
            selected_logits, selected_tokens = _decode(selected_keys)
            valid = selected_keys != 0
            hidden_axis = tl.arange(0, DOT_WIDTH)
            hidden = tl.load(Hidden + prediction_row * HS + hidden_axis, hidden_axis < WIDTH, other=0)
            weight = tl.load(
                Weight + selected_tokens[:, None].to(tl.int64) * WS + hidden_axis[None, :],
                valid[:, None] & (hidden_axis[None, :] < WIDTH),
                other=0,
            )
            prediction = tl.sum(weight.to(tl.float32) * hidden[None, :].to(tl.float32), 1)
            # Native unquantized LM heads round their accumulation to the weight dtype.
            prediction = prediction.to(Weight.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
            selected_logits = tl.where(valid, (1 + beta) * selected_logits - beta * prediction, float("-inf"))
            tl.store(Logits + row * LS + selected_tokens, selected_logits, valid)
            value, token = _draw(
                selected_logits, selected_tokens, valid, row, Mapping, Temperature, Seeds, Positions, VOCAB, FP64, V027
            )
            replace = (value > best) | ((value == best) & (token < winner))
            best, winner = tl.maximum(best, value), tl.where(replace, token, winner)
    tl.store(Output + row, winner)


def guided_sample(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    row_map: torch.Tensor,
    row_weights: torch.Tensor | None,
    beta: float,
    capacity: int,
    mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    top_p: torch.Tensor | None = None,
    *,
    fp64: bool,
) -> torch.Tensor:
    """Guide rows with at most K survivors; wider rows keep the upstream unguided distribution.

    Full-vocabulary rows whose top-1 probability >= top_p < 1 use the true one-token nucleus.
    This is an opt-in approximation to vLLM's skipped-mask distribution. Top-1 ties use token ID.
    Scratch is bounded by rows * K; the single kernel reads each filtered logit once.
    """
    rows, vocab = logits.shape
    hidden = hidden.to(weight.dtype).contiguous()
    mapping, positions = mapping.contiguous(), positions.contiguous()
    scratch = torch.empty((rows, capacity), dtype=torch.uint64, device=logits.device)
    output = torch.empty(rows, dtype=torch.int64, device=logits.device)
    _sample[(rows,)](
        logits,
        hidden,
        weight,
        row_map,
        row_weights,
        scratch,
        output,
        mapping,
        temperature,
        seeds,
        positions,
        top_p,
        logits.stride(0),
        hidden.stride(0),
        weight.stride(0),
        vocab,
        hidden.shape[1],
        beta,  # pyright: ignore[reportArgumentType]
        capacity,
        BLOCK=4096,
        DOT_ROWS=16,
        DOT_WIDTH=triton.next_power_of_2(hidden.shape[1]),
        FP64=fp64,
        V027=VLLM_027,
        num_warps=8,  # pyright: ignore[reportCallIssue]
        enable_fp_fusion=False,  # pyright: ignore[reportCallIssue]
    )
    return output
