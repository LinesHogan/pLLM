"""StepLoop on real MRv2: cache-backed recurrence and source-owned token capture."""

import sys
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, replace
from functools import partial
from itertools import product
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F
from vllm import RequestOutput, SamplingParams, _custom_ops as ops
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.vllm_flash_attn import flash_attn_varlen_func

import pllm
from pllm.backends.vllm.position_state import PositionCacheProvider
from pllm.backends.vllm.prefix_cache import CachePolicy
from pllm.backends.vllm.replay import RouteWave, VllmTokenwiseReplay
from pllm.backends.vllm.runtime import BackendCommand, BackendInstall, BackendStage
from pllm.config import KVOffload
from pllm.contracts.codec import encode_boundary
from pllm.contracts.request_identity import RequestIdentity
from pllm.workload_transport import encode_workload
from pllm.workloads.step_loop import StepLoop
from test.gpu.vllm_recirculation_smoke import replay_handle

# Broad-margin greedy fixtures keep BF16 batch-width rounding separate from near-tie token flips.
PROMPTS = ["One two", "Water freezes at"]


@contextmanager
def record_steps(handle):
    """Record native span/wave results and independently verify every recurrent state handoff."""
    records, active, samples, tokens, caches, carries = {}, [], {}, {}, {}, {}
    primitive = handle.runner.require_session()._replay
    stage, replay, span = RouteWave.stage, torch.cuda.CUDAGraph.replay, VllmTokenwiseReplay.save_span

    def external(row):
        return handle._llm.llm_engine.output_processor.request_states[row.identity.request_id].external_req_id

    def staged(wave, rows):
        stage(wave, rows)
        active[:] = rows
        for lane, row in enumerate(rows):
            key = external(row), row.token_offset - 1
            if row.phase.value == "decode" and key in records:
                torch.testing.assert_close(wave.position.entry(())[1][lane : lane + 1], records[key][0], rtol=0, atol=0)

    def replayed(graph):
        replay(graph)
        wave = next((wave for wave in primitive.captures.values() if wave.graph is graph), None)
        if wave is not None:
            pre = wave.position.tables[0][wave.position.indices[0, 0]]
            for lane, row in enumerate(active):
                records[external(row), row.token_offset] = (
                    pre[lane : lane + 1].clone(),
                    wave.output[lane : lane + 1].clone(),
                    len(active),
                )

    def saved(self, rows, groups, requests, hidden, saved_values):
        for lane, row in enumerate(rows):
            records[external(row), row.token_offset] = (
                hidden[lane : lane + 1].clone(),
                self.output[row.packed_row_index : row.packed_row_index + 1].clone(),
                len(rows),
            )
        return span(self, rows, groups, requests, hidden, saved_values)

    forward = primitive.model.forward
    oracle = Oracle(handle.runner.model, handle.runner.require_session()._replay.domain.positions)

    def forwarded(input_ids, *args, **kwargs):
        ids = input_ids.tolist()
        result = forward(input_ids, *args, **kwargs)
        selected = {link.packed_row_index for link in handle.runner.require_session()._step.rows.links}
        for row in primitive.rows:
            identity, position = external(row), row.token_offset
            history = tokens.setdefault(identity, [])
            if len(history) < position:
                continue  # A cache-hit donor is checked after its lost prefix is recomputed from zero.
            history[position:] = [ids[row.packed_row_index]]
            actual_pre, actual_normal, width = records[identity, position]
            previous = carries.get((identity, position - 1))
            expected, cache = oracle.run(
                history,
                position + 1 if row.phase.value == "prefill" else 0,
                cache=caches.get(identity),
                carry=previous,
                start=position,
                widths=[width],
            )
            caches[identity] = cache
            pre, normal, _, sampled = expected[0]
            carries[identity, position] = pre
            for actual, reference in ((actual_pre, pre), (actual_normal, normal)):
                error = (actual.float() - reference.float()).norm() / reference.float().norm()
                assert error < 0.06, (position, float(error))
            if row.packed_row_index in selected:
                samples[identity, position] = sampled
        return result

    with (
        patch.object(RouteWave, "stage", staged),
        patch.object(torch.cuda.CUDAGraph, "replay", replayed),
        patch.object(VllmTokenwiseReplay, "save_span", saved),
        patch.object(primitive.model, "forward", forwarded),
    ):
        yield records, samples


def forced_preemption():
    """Constrain the real allocator and prove tail eviction and prefix-state resumption."""
    preempt, lookup = Scheduler._preempt_request, KVCacheManager.get_computed_blocks
    for capacity, policy in product((5, 4), ("reuse", "isolate")):
        events, hits = {}, []

        def evict(scheduler, request, *args, **kwargs):
            events[request.request_id] = tuple(
                scheduler.kv_cache_manager.get_block_ids(request.request_id)[0][: request.num_computed_tokens // 16]
            )
            return preempt(scheduler, request, *args, **kwargs)

        def resume(manager, request):
            result = lookup(manager, request)
            if request.request_id in events:
                hits.append((events[request.request_id], tuple(result[0].get_block_ids()[0]), result[1]))
            return result

        handle = replay_handle(StepLoop(), length=64, prefix=True, generated_kv=policy, blocks=capacity)
        try:
            with (
                record_steps(handle) as (records, samples),
                patch.object(Scheduler, "_preempt_request", evict),
                patch.object(KVCacheManager, "get_computed_blocks", resume),
            ):
                outputs = handle.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=35, ignore_eos=True))
            for output in outputs:
                for index, token in enumerate(output.outputs[0].token_ids):
                    assert token == samples[output.request_id, len(output.prompt_token_ids) - 1 + index]
            assert events and hits, (capacity, events, hits)
            assert any(
                (hit > 0 and old[0] in kept and old[-1] not in kept) if capacity == 5 else hit == 0
                for old, kept, hit in hits
            ), hits
            print(
                f"Forced preemption PASS policy={policy} blocks={capacity} evictions={events} resumed={hits}",
                flush=True,
            )
        finally:
            handle.close()


def generate_ids(handle, prompts, params=None):
    """Submit exact donor token IDs through the normal workload admission boundary."""
    llm = handle._llm
    llm.sleep(level=0, mode="keep")
    ids = llm.enqueue(
        [{"prompt_token_ids": tokens} for tokens in prompts],
        [
            handle._cache_policy.sampling(param)
            for param in (params or [SamplingParams(temperature=0, max_tokens=3, ignore_eos=True) for _ in prompts])
        ],
        use_tqdm=False,
    )
    identities = tuple(RequestIdentity(name, handle._next_prompt_index + i, 0) for i, name in enumerate(ids))
    handle.command(BackendCommand.STAGE, encode_boundary(BackendStage(identities, None, None)))
    handle._next_prompt_index += len(ids)
    llm.wake_up(tags=["scheduling"])
    outputs = llm.wait_for_completion(RequestOutput, use_tqdm=False)
    handle.command(BackendCommand.FINISH)
    return outputs


def offload_roundtrips(expected):
    """Restore generated-prefix KV/carry from both stock CPU connectors after GPU eviction."""
    stage, lookup = PositionCacheProvider.stage, KVCacheManager.get_computed_blocks
    for connector in ("OffloadingConnector", "SimpleCPUOffloadConnector"):
        local, boundaries = {}, []
        handle = replay_handle(
            StepLoop(),
            length=64,
            prefix=True,
            generated_kv="reuse",
            blocks=6,
            offload=KVOffload(connector, 8 * 1024**2),
        )

        def locate(manager, request):
            result = lookup(manager, request)
            local[request.request_id] = result[1]
            return result

        def staged(provider, context, rows, requests, positions, generated):
            stage(provider, context, rows, requests, positions, generated)
            primitive = handle.runner.require_session()._replay
            for lane, index in enumerate(rows.tolist()):
                row = primitive.rows[index]
                if row.token_offset == 32:
                    boundaries.append(
                        (
                            row.identity.request_id,
                            int(provider.indices[0, 1, lane]),
                            provider.entry(())[1][lane : lane + 1].clone(),
                        )
                    )

        try:
            with (
                patch.object(PositionCacheProvider, "stage", staged),
                patch.object(KVCacheManager, "get_computed_blocks", locate),
            ):
                (donor,) = handle.generate([PROMPTS[0]], SamplingParams(temperature=0, max_tokens=35, ignore_eos=True))
                for word in ("apple ", "river "):
                    handle.generate([word * 48], SamplingParams(temperature=0, max_tokens=2, ignore_eos=True))
                (received,) = generate_ids(handle, [donor.prompt_token_ids + list(donor.outputs[0].token_ids)])
            first, last = boundaries[0], boundaries[-1]
            assert local[last[0]] == 0 and first[1] != last[1], (connector, local, first[:2], last[:2])
            torch.testing.assert_close(first[2].view(torch.uint8), last[2].view(torch.uint8), rtol=0, atol=0)
            assert list(received.outputs[0].token_ids) == expected
            print(f"{connector} full-page asynchronous restore PASS allocation={first[1]}->{last[1]}", flush=True)
        finally:
            handle.close()


def donor_loss():
    """A reused request recomputes lost donor pages using its own prompt roles."""
    handle = replay_handle(StepLoop(), length=64, prefix=True, generated_kv="reuse", blocks=5)
    lookup, hits = KVCacheManager.get_computed_blocks, []
    frontiers, saved = [], []
    native_forward = VllmTokenwiseReplay.forward

    def forwarded(primitive, *args, **kwargs):
        rows = [row.token_offset for row in primitive.rows if row.identity.prompt_idx == 2]
        if rows:
            frontiers.append((rows[0], rows[-1]))
        result = native_forward(primitive, *args, **kwargs)
        if rows:
            provider = primitive.position
            last = next(row for row in primitive.rows if row.identity.prompt_idx == 2 and row.token_offset == rows[-1])
            slot = get_forward_context().slot_mapping[provider.layers[0].layer_name][
                last.packed_row_index : last.packed_row_index + 1
            ]
            saved.append(provider.tables[0][slot // provider.allocation].clone())
        return result

    try:
        (donor,) = handle.generate([PROMPTS[0]], SamplingParams(temperature=0, max_tokens=35, ignore_eos=True))
        receiver = donor.prompt_token_ids + list(donor.outputs[0].token_ids)

        def locate(manager, request):
            result = lookup(manager, request)
            if request.prompt_token_ids == receiver:
                hits.append((request.num_preemptions, result[1]))
            return result

        with (
            patch.object(KVCacheManager, "get_computed_blocks", locate),
            patch.object(
                handle.runner._workload_session._replay.model,
                "forward",
                partial(forwarded, handle.runner._workload_session._replay),
            ),
            record_steps(handle) as (recorded, sampled),
        ):
            outputs = generate_ids(
                handle,
                [[123] * 16, receiver],
                [SamplingParams(temperature=0, max_tokens=count, ignore_eos=True) for count in (35, 8)],
            )
        received = next(output for output in outputs if output.prompt_token_ids == receiver)
        assert hits[0] == (0, 32) and any(preemptions and hit == 0 for preemptions, hit in hits), hits
        reference, _ = Oracle(handle.runner.model, handle.runner.require_session()._replay.domain.positions).run(
            receiver + list(received.outputs[0].token_ids[:-1]),
            len(receiver),
            widths=[
                recorded[received.request_id, p][2]
                for p in range(len(receiver) + len(received.outputs[0].token_ids) - 1)
            ],
        )
        assert frontiers[0][0] == 32 and any(start == 0 for start, _ in frontiers[1:])
        # Contiguous versus paged FA2 rounding accumulates across this long prompt.
        error = (saved[-1].float() - reference[-1][0].float()).norm() / reference[-1][0].float().norm()
        assert error < 0.05, float(error)
        assert (
            received.outputs[0].token_ids[-1]
            == reference[-1][3]
            == sampled[received.request_id, len(receiver) + len(received.outputs[0].token_ids) - 2]
        )
        handle._llm.reset_prefix_cache()
        with record_steps(handle) as (_, cold_samples):
            (cold,) = generate_ids(handle, [receiver], [SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)])
        assert all(
            token == cold_samples[cold.request_id, len(receiver) - 1 + i]
            for i, token in enumerate(cold.outputs[0].token_ids)
        )
        print("Lost adopted donor prefix matches the cache-loss-history oracle PASS", flush=True)
    finally:
        handle.close()


@dataclass(frozen=True)
class ScaledStepLoop(StepLoop):
    """Same cache schema with distinct frozen route mathematics."""

    @pllm.layer_route(tokens="all")
    def loop(self, layers):
        hidden = torch.where(layers.generated, layers.load("history", offset=-1), layers.seed)
        result = layers(hidden=hidden)[-1] * 0.999
        layers.save("history", result)
        return result


def isolated_multiturn():
    """Warm isolate equals cold prompt computation, also across the EngineCore boundary."""
    from test.gpu.vllm_step_loop_smoke import ScaledStepLoop as ImportedStepLoop

    for multiprocessing in (True, False):
        handle = replay_handle(StepLoop(), length=128, prefix=True, multiprocessing=multiprocessing)
        try:
            (donor,) = handle.generate(
                ["One two three four five six seven eight nine ten " * 2],
                SamplingParams(temperature=0, max_tokens=35, ignore_eos=True),
            )
            receiver = donor.prompt_token_ids + list(donor.outputs[0].token_ids)
            (warm,) = generate_ids(handle, [receiver])
            assert warm.num_cached_tokens == len(donor.prompt_token_ids) // 16 * 16 > 0
            handle._llm.reset_prefix_cache()
            (cold,) = generate_ids(handle, [receiver])
            assert warm.outputs[0].token_ids == cold.outputs[0].token_ids and cold.num_cached_tokens == 0
            namespace = handle._cache_policy.namespace
            workload = ImportedStepLoop()
            handle.attachment = workload.plan_backend(handle._facts)
            handle._cache_policy = CachePolicy.for_attachment(handle.attachment, "isolate")
            handle.command(BackendCommand.DETACH)
            if multiprocessing:
                handle.command(
                    BackendCommand.INSTALL,
                    encode_boundary(BackendInstall(handle.attachment, encode_workload(workload), handle._cache_policy)),
                )
            else:
                deployment = workload.bind_backend(handle._facts, handle.attachment)
                handle.runner.attach_workload(deployment, handle._cache_policy)
            assert handle._cache_policy.namespace != namespace
            (changed,) = generate_ids(handle, [receiver])
            assert changed.num_cached_tokens == 0
            print(
                f"Idle same-schema workload replacement has no cross-workload hit PASS EngineCore={multiprocessing}",
                flush=True,
            )
            print(
                f"Isolate multi-turn cold equality PASS EngineCore={multiprocessing} hit={warm.num_cached_tokens}",
                flush=True,
            )
        finally:
            handle.close()


def rms(value, norm):
    """Share the arithmetic primitive so the independent loop preserves BF16 rounding."""
    value = value.to(norm.weight.dtype)
    output = torch.empty_like(value)
    ops.rms_norm(output, value, norm.weight, norm.variance_epsilon)
    return output


class Oracle:
    """Independent eager torch decoder: frozen weights, private KV, no region execution."""

    def __init__(self, model, capacity):
        self.model, self.capacity = model, capacity

    @torch.no_grad()
    def run(self, tokens, prompt_length, *, cache=None, carry=None, start=0, widths=None):
        cache = [[] for _ in self.model.model.layers] if cache is None else [list(items[:start]) for items in cache]
        results = []
        for position, token in enumerate(tokens[start:], start):
            width = 1 if widths is None else widths[position - start]

            def linear(value, weight, bias=None):
                return F.linear(value.expand(width, -1).contiguous(), weight, bias)[:1]

            hidden = (
                F.embedding(torch.tensor([token], device="cuda"), self.model.model.embed_tokens.weight)
                if position < prompt_length
                else carry
            )
            residual = None
            for ordinal, layer in enumerate(self.model.model.layers):
                combined = hidden.float() if residual is None else hidden.float() + residual.float()
                residual, hidden = combined.to(hidden.dtype), rms(combined.to(hidden.dtype), layer.input_layernorm)
                attention = layer.self_attn
                qkv = linear(hidden, attention.qkv_proj.weight, attention.qkv_proj.bias)
                q, k, v = (
                    part.reshape(-1, attention.head_dim)
                    for part in qkv.split([attention.q_size, attention.kv_size, attention.kv_size], -1)
                )
                cos, sin = attention.rotary_emb.cos_sin_cache[position].chunk(2)

                def rotate(value):
                    first, second = value.float().chunk(2, -1)
                    return torch.cat(
                        (first * cos.float() - second * sin.float(), second * cos.float() + first * sin.float()), -1
                    ).to(hidden.dtype)

                q, k = rotate(q), rotate(k)
                cache[ordinal].append(torch.cat((k, v), -1))
                keys, values = torch.stack(cache[ordinal]).chunk(2, -1)
                attended = flash_attn_varlen_func(
                    q=q.unsqueeze(0).expand(width, -1, -1).contiguous(),
                    k=keys.repeat(width, 1, 1),
                    v=values.repeat(width, 1, 1),
                    cu_seqlens_q=torch.arange(width + 1, dtype=torch.int32, device="cuda"),
                    cu_seqlens_k=torch.arange(width + 1, dtype=torch.int32, device="cuda") * (position + 1),
                    max_seqlen_q=1,
                    max_seqlen_k=self.capacity,
                    softmax_scale=attention.scaling,
                    causal=True,
                    fa_version=2,
                ).reshape(width, -1)[:1]
                projected = linear(attended, attention.o_proj.weight, attention.o_proj.bias)
                combined = residual.float() + projected.float()
                residual, hidden = (
                    combined.to(hidden.dtype),
                    rms(combined.to(hidden.dtype), layer.post_attention_layernorm),
                )
                gate, up = linear(hidden, layer.mlp.gate_up_proj.weight, layer.mlp.gate_up_proj.bias).chunk(2, -1)
                activated = torch.empty_like(gate)
                torch.ops._C.silu_and_mul(activated, torch.cat((gate, up), -1))
                hidden = linear(activated, layer.mlp.down_proj.weight, layer.mlp.down_proj.bias)
            carry = hidden + residual
            normal = rms(carry, self.model.model.norm)
            logits = linear(normal, self.model.lm_head.weight)
            results.append((carry.clone(), normal, logits, int(logits.argmax(-1))))
        return results, cache


@torch.no_grad()
def compare(output, records, oracle, *, cache=None, carry=None, start=0):
    tokens = output.prompt_token_ids + list(output.outputs[0].token_ids[:-1])
    # The oracle owns its entire carry/KV recurrence. Match physical GEMM/FA2 widths
    # by repeating its private row, without calling the route executor or reading its state.
    expected, cache = oracle.run(
        tokens,
        len(output.prompt_token_ids),
        cache=cache,
        carry=carry,
        start=start,
        widths=[records[output.request_id, p][2] for p in range(start, len(tokens))],
    )
    for position, (pre, normal, logits, sampled) in enumerate(expected, start):
        actual_pre, actual_normal, _ = records[output.request_id, position]
        actual_logits = F.linear(actual_normal, oracle.model.lm_head.weight)
        for actual, reference in ((actual_pre, pre), (actual_normal, normal), (actual_logits, logits)):
            # Batched bf16 GEMMs and varlen FA2 accumulate different rounding across layers.
            relative = (actual.float() - reference.float()).norm() / reference.float().norm()
            assert relative < 0.06, (position, float(relative))
        if position >= len(output.prompt_token_ids) - 1:
            assert sampled == output.outputs[0].token_ids[position - len(output.prompt_token_ids) + 1], (
                position,
                sampled,
            )
    return expected, cache


def main():
    isolated_multiturn()
    handle = replay_handle(StepLoop(), length=128, prefix=True, generated_kv="reuse")
    primitive = handle.runner._workload_session._replay
    stage, replay = RouteWave.stage, torch.cuda.CUDAGraph.replay
    save_span = VllmTokenwiseReplay.save_span
    records, active = {}, []
    entered, roles = {}, set()
    schedule, capture_token = Scheduler.schedule, RouteWave.capture
    admitted = False
    forward = VllmTokenwiseReplay.forward

    def permuted(primitive, *args, **kwargs):
        context = get_forward_context()
        metadata = {name: copy(item) for name, item in context.attn_metadata.items()}
        blocks, allocation = primitive.position.tables[0].shape[0], primitive.position.allocation

        def page(ids):
            return torch.where(ids == 0, 0, 1 + ids % (blocks - 1))

        slots = {
            name: page(value // allocation) * allocation + value % allocation
            for name, value in context.slot_mapping.items()
        }
        for name, item in metadata.items():
            item.block_table, item.slot_mapping = page(item.block_table), slots[name]
        with (
            override_forward_context(replace(context, attn_metadata=metadata, slot_mapping=slots)),
            torch.profiler.record_function("step_loop.forward"),
        ):
            return forward(primitive, *args, **kwargs)

    def mixed(scheduler, *args, **kwargs):
        nonlocal admitted
        scheduler.max_num_running_reqs = 2 if admitted else 1
        result = schedule(scheduler, *args, **kwargs)
        admitted = True
        scheduler.max_num_running_reqs = 2
        return result

    def external(row):
        return handle._llm.llm_engine.output_processor.request_states[row.identity.request_id].external_req_id

    def cold(wave):
        provider = wave.position
        saved = tuple(table[index[0]].clone() for table, index in zip(provider.tables, provider.indices))
        result = capture_token(wave)
        for table, index, before in zip(provider.tables, provider.indices, saved):
            torch.testing.assert_close(table[index[0]].view(torch.uint8), before.view(torch.uint8), atol=0, rtol=0)
        return result

    def staged(wave, rows):
        stage(wave, rows)
        active[:] = rows
        roles.add(frozenset(item.phase.value for item in primitive.rows))
        for lane, row in enumerate(rows):
            previous = wave.position.entry(())[1][lane : lane + 1]
            if row.phase.value == "decode" and (external(row), row.token_offset - 1) in records:
                torch.testing.assert_close(previous, records[external(row), row.token_offset - 1][0], rtol=0, atol=0)
            if external(row) not in entered:
                entered[external(row)] = previous.clone()

    def span_saved(self, rows, groups, requests, hidden, saved_values):
        provider = self.position
        for group in groups:
            row = group[0]
            if external(row) not in entered:
                packed = torch.tensor([row.packed_row_index], device=hidden.device)
                provider.stage(
                    get_forward_context(),
                    packed,
                    torch.tensor([row.request_ordinal], device=hidden.device),
                    torch.tensor([row.token_offset], device=hidden.device),
                    packed.new_zeros(1),
                )
                entered[external(row)] = provider.entry(())[1].clone()
        for index, row in enumerate(rows):
            records[external(row), row.token_offset] = (
                hidden[index : index + 1].clone(),
                self.output[row.packed_row_index : row.packed_row_index + 1].clone(),
                len(rows),
            )
        return save_span(self, rows, groups, requests, hidden, saved_values)

    def replayed(graph):
        replay(graph)
        capture = next((capture for capture in primitive.captures.values() if capture.graph is graph), None)
        if capture is not None:
            provider = capture.position
            pre = provider.tables[0][provider.indices[0, 0]]
            for lane, row in enumerate(active):
                records[external(row), row.token_offset] = (
                    pre[lane : lane + 1].clone(),
                    capture.output[lane : lane + 1].clone(),
                    len(active),
                )

    try:
        with (
            patch.object(primitive.model, "forward", partial(permuted, primitive)),
            patch.object(Scheduler, "schedule", mixed),
            patch.object(RouteWave, "capture", cold),
            patch.object(RouteWave, "stage", staged),
            patch.object(VllmTokenwiseReplay, "save_span", span_saved),
            patch.object(torch.cuda.CUDAGraph, "replay", replayed),
        ):
            outputs = handle.generate(
                PROMPTS,
                SamplingParams(temperature=0, max_tokens=35, ignore_eos=True),
                use_tqdm=False,
            )
        assert frozenset(("prefill", "decode")) in roles
        assert len(handle.runner.kv_cache_config.kv_cache_groups) == 1
        assert primitive.position.tables[0].shape == (handle.runner.kv_cache_config.num_blocks, 896)
        oracle = Oracle(handle.runner.model, handle.runner.require_session()._replay.domain.positions)
        references = [compare(output, records, oracle) for output in outputs]
        print("Independent torch pre-norm carry, normal hidden, logits and deterministic tokens PASS")
        assert {width for width, _ in primitive.captures} <= {1, 2}
        donor = outputs[0]
        receiver = donor.prompt_token_ids + list(donor.outputs[0].token_ids)
        with (
            patch.object(primitive.model, "forward", partial(permuted, primitive)),
            patch.object(Scheduler, "schedule", mixed),
            patch.object(RouteWave, "capture", cold),
            patch.object(RouteWave, "stage", staged),
            patch.object(VllmTokenwiseReplay, "save_span", span_saved),
            patch.object(torch.cuda.CUDAGraph, "replay", replayed),
        ):
            (received,) = generate_ids(handle, [receiver])
        hit = min(position for request, position in records if request == received.request_id)
        assert hit >= 16 and hit % 16 == 0
        assert received.prompt_token_ids[:hit] == (donor.prompt_token_ids + list(donor.outputs[0].token_ids))[:hit]
        torch.testing.assert_close(entered[received.request_id], records[donor.request_id, hit - 1][0], atol=0, rtol=0)
        compare(
            received,
            records,
            oracle,
            cache=references[0][1],
            carry=references[0][0][hit - 1][0],
            start=hit,
        )
        cold_reference, _ = oracle.run(receiver, len(receiver))
        assert not torch.equal(cold_reference[-1][0], records[received.request_id, len(receiver) - 1][0])
        print(f"Real multi-turn reuse and divergence from isolate cold math PASS hit={hit}")

        with (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profile,
            patch.object(primitive.model, "forward", partial(permuted, primitive)),
        ):
            handle.generate(["Warm profiling"], SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
        assert any(event.name == "step_loop.forward" for event in profile.events())
        for event in profile.events():
            parent = event.cpu_parent
            while parent is not None and parent.name != "step_loop.forward":
                parent = parent.cpu_parent
            if parent is not None:
                assert not any(
                    word in event.name.lower()
                    for word in ("dtoh", "device to host", "synchronize", "local_scalar_dense")
                ), event.name
        print("Warmed whole-forward zero-sync PASS")
    finally:
        handle.close()
    forced_preemption()
    offload_roundtrips(list(received.outputs[0].token_ids))
    donor_loss()


if __name__ == "__main__":
    main()
