"""RTX 4090 adaptive-depth acceptance: native KV oracle, sync-debug replay and paired decode timings."""

import argparse
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.forward_context import override_forward_context

import pllm
from pllm.backends.vllm.adaptive_depth import AdaptiveSlice
from pllm.backends.vllm.architecture import hidden_binding
from pllm.backends.vllm.compat import enable_batch_invariance
from pllm.backends.vllm.custom_forward import ProgramWave
from pllm.backends.vllm.custom_forward_attention import BatchAttentionBuffers
from pllm.backends.vllm.hidden_taps import HiddenTaps
from pllm.contracts.hidden_locator import HiddenSite
from pllm.execution.step_capture import capture_graph
from pllm.execution.token_program_executor import execute_segment
from pllm.execution.torch_private import cuda_graph_pool_bytes
from pllm.testing_gpu import run_gpu_checks
from pllm.workloads.dola import DoLaWorkload


@dataclass(frozen=True)
class Exit(pllm.Workload):
    rule: str = "none"
    exit_layer: int | None = None

    @property
    def gate(self) -> pllm.InstanceState:
        return pllm.InstanceState({"threshold": (1,)}, dtype=torch.float32)

    @property
    def router(self) -> pllm.InstanceState:
        return pllm.InstanceState(init=pllm.FromModule(self.zero_router), dtype=torch.float32)

    def zero_router(self, dimensions: pllm.Dimensions) -> torch.nn.Module:
        """Build a learned head whose initial decision equals a constant gate."""
        router = torch.nn.Linear(dimensions.hidden_size, 1)
        for parameter in router.parameters():
            torch.nn.init.zeros_(parameter)
        return router

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers, threshold=pllm.parameter("gate", "threshold"), router=pllm.module("router")):
        if self.rule == "loop":
            taps = layers()
            hidden = taps[-3]
            for _ in range(2):
                hidden = layers(
                    layers.layer_count - 2,
                    hidden=torch.lerp(taps[-3], hidden, 0.01),
                    active=layers.position % 2 == 0,
                )[-1]
            return taps[-1]
        middle = layers.layer_count // 2 if self.exit_layer is None else self.exit_layer
        hidden = layers(stop=middle)[middle - 1]
        active = router(hidden.float()).sigmoid() > threshold
        if self.rule == "none":
            active = ~active
        elif self.rule == "mixed":
            active = (layers.position % 2 == 0) & active
        return (
            layers(middle, hidden=hidden)[-1]
            if self.rule == "vanilla"
            else layers(middle, hidden=hidden, active=active)[-1]
        )


@dataclass(frozen=True)
class ValueExit(Exit):
    """Exercise value rewrites on both computed and propagated K/V."""

    @pllm.custom_residual(site="value")
    def rewrite(self, v):
        return v * 0.9


@dataclass(frozen=True)
class ReadoutExit(Exit, DoLaWorkload):
    """DoLa reads skipped layers while C2 changes computed MLP residuals."""

    @pllm.custom_residual(site="mlp", layers=(23,))
    def residual(self, x, y, earlier=pllm.residual(22)):
        return x + y * 0.9 + earlier * 0.01


def replay_contracts(config):
    """Check real eviction and frozen-stream DoLa inputs beside C2 rewrites on Qwen2.5-0.5B.

    test_adaptive_depth starts this mode with VLLM_BATCH_INVARIANT=1, so the checks run here, where the taps are
    observed."""
    _kernels = enable_batch_invariance()
    snapshots, current = [], []
    prepare, read = HiddenTaps.prepare, HiddenTaps.read
    sources = tuple(hidden_binding(layer, HiddenSite.RESIDUAL, 24).tap for layer in (21, 22, 23))

    def prepared(taps, rows):
        current[:] = rows.rows
        return prepare(taps, rows)

    def observed(taps, tap, packed):
        value = read(taps, tap, packed)
        if tap in sources[1:]:
            selected = [
                index
                for index, row in enumerate(packed)
                if current[row].role.value == "decode" and (rule == "none" or current[row].token_offset % 2 != 0)
            ]
            if selected:
                snapshots.append((value[selected].clone(), read(taps, sources[0], packed)[selected].clone()))
        return value

    for rule, workload in (
        ("loop", Exit("loop")),
        ("none", ReadoutExit(rule="none", exit_layer=22, layers=(0, 21, 22), plausibility=0.99)),
        ("mixed", ReadoutExit(rule="mixed", exit_layer=22, layers=(0, 21, 22), plausibility=0.99)),
    ):
        with patch.object(HiddenTaps, "prepare", prepared), patch.object(HiddenTaps, "read", observed):
            result = run_gpu_checks(workload, config, None)
        assert any(value.startswith("preempted:") for value in result.passed), result
        assert rule == "loop" or snapshots
        for actual, expected in snapshots:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        snapshots.clear()
        print(type(workload).__name__, result)


def cells(wave):
    return tuple(
        (attention.kv_cache, wave.attention.metadata[attention.layer_name].slot_mapping)
        for attention in wave.custom_forward.attentions
    )


def read_cells(wave):
    return tuple(cache[slots // cache.shape[2], :, slots % cache.shape[2]].clone() for cache, slots in cells(wave))


def write_cells(wave, values):
    for (cache, slots), value in zip(cells(wave), values, strict=True):
        cache[slots // cache.shape[2], :, slots % cache.shape[2]] = value


def full_depth(wave):
    hidden = wave.seed.clone()
    middle = len(wave.custom_forward.model.layers) // 2
    for start, stop in ((0, middle), (middle, len(wave.custom_forward.model.layers))):
        residual = None
        for index in range(start, stop):
            hidden, residual = wave.custom_forward.model.layers[index](wave.positions, hidden, residual)
        hidden = hidden + residual
    return wave.custom_forward.model.norm(hidden)


def reference(wave, active):
    """Separate, full-width masked oracle; direct QKV/RoPE verifies each skipped layer's cache."""
    body = wave.custom_forward.model
    middle = len(body.layers) // 2
    hidden, residual = wave.seed.clone(), None
    for layer in body.layers[:middle]:
        hidden, residual = layer(wave.positions, hidden, residual)
    frozen = hidden + residual
    hidden, residual = frozen.clone(), None
    for ordinal, layer in enumerate(body.layers[middle:], middle):
        # Every skipped layer receives the same frozen stream and its own weights.
        attn = layer.self_attn
        qkv, _ = attn.qkv_proj(layer.input_layernorm(frozen))
        q, k, v = qkv.split((attn.q_size, attn.kv_size, attn.kv_size), -1)
        q, k = attn.rotary_emb(wave.positions, q, k)
        hidden, residual = layer(wave.positions, hidden, residual)
        cache, slots = cells(wave)[ordinal]
        kv = torch.cat(
            (
                k.reshape(len(frozen), attn.num_kv_heads, attn.head_dim),
                v.reshape(len(frozen), attn.num_kv_heads, attn.head_dim),
            ),
            -1,
        )
        live = cache[slots // cache.shape[2], :, slots % cache.shape[2]]
        cache[slots // cache.shape[2], :, slots % cache.shape[2]] = torch.where(active[:, :, None], live, kv)
    return body.norm(torch.where(active, hidden + residual, frozen))


def split_full(wave):
    """Native full-depth outputs at half width, measuring batch numerics on the same history."""
    outputs = []
    for rows in torch.arange(len(wave.seed), device=wave.seed.device).chunk(2):
        buffers = BatchAttentionBuffers(
            wave.attention.context, wave.custom_forward.attentions, wave.custom_forward.domain.positions, len(rows)
        )
        buffers.refresh(
            buffers.bind(wave.attention.context, wave.custom_forward.attentions, wave.positions + 1), rows, rows
        )
        part = SimpleNamespace(seed=wave.seed[rows], positions=wave.positions[rows], custom_forward=wave.custom_forward)
        with override_forward_context(buffers.context):
            outputs.append(full_depth(part))
    return torch.cat(outputs)


class Audit:
    def __init__(self, rule, logits, measure_at):
        self.rule, self.logits, self.measure_at = rule, logits, measure_at
        self.max_error = self.max_bound = 0.0
        self.native_selection = AdaptiveSlice.__call__
        self.waves = {}
        self.baselines = {}
        self.timings = []
        self.checked = 0
        self.steady = []
        self.native_capture = ProgramWave.capture
        self.native_replay = torch.cuda.CUDAGraph.replay
        self.counts = set()
        self.capture_bytes = 0

    def mask(self, wave, active):
        # The acceptance fixture forces persistent request parity; workload routing
        # still accepts only per-token tensor decisions in production.
        if self.rule == "mixed":
            return wave.indices[1].reshape(-1, 1) % 2 == 1
        if self.rule == "buckets":
            count = wave.positions[0] % (len(wave.seed) + 1)
            return (torch.arange(len(wave.seed), device=active.device) < count).reshape(-1, 1)
        return active

    def select(self, selection, step, hidden, active):
        active = self.mask(selection.wave, active)
        return self.native_selection(selection, step, hidden, active)

    def eager_selection(self, selection, step, hidden, active):
        """Diagnostic host selection checks captured buckets against the same native eager kernels."""
        count = int(active.sum())
        self.counts.add(count)
        wave = selection.wave
        if count == len(hidden):
            with wave.custom_forward.sites_rows(wave.indices[0]) as stream:
                return execute_segment(step, hidden, wave.layer, stream)
        selection.propagate(step, hidden, active)
        output = tuple(hidden.clone() for _ in step.taps)
        if count:
            size = next(size for size in selection.widths if count <= size)
            rows = torch.argsort(~active.flatten(), stable=True)[:size]
            selection.compact(step, hidden, active, rows, output)
        return output

    def capture(self, wave):
        self.native_capture(wave)
        self.capture_bytes = max(
            self.capture_bytes,
            cuda_graph_pool_bytes(wave.graph.pool(), wave.seed.device.index or 0)
            + sum(
                captured.adaptive.resident_bytes
                for captured in wave.custom_forward.captures.values()
                if captured.adaptive is not None
            ),
        )
        saved = read_cells(wave)
        with override_forward_context(wave.attention.context):
            graph, output, _ = capture_graph(
                wave.seed.device, lambda: full_depth(wave), restore=lambda: write_cells(wave, saved)
            )
        write_cells(wave, saved)
        self.waves[wave.graph] = wave
        self.baselines[wave.graph] = graph, output

    def replay(self, graph):
        if graph not in self.waves:
            return self.native_replay(graph)
        wave = self.waves[graph]
        baseline, full = self.baselines[graph]
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            self.native_replay(graph)
        finally:
            torch.cuda.set_sync_debug_mode(mode)
        events[1].record()
        live = read_cells(wave)
        self.check_eager(wave, live)
        events[2].record()
        self.native_replay(baseline)
        events[3].record()
        self.check_reference(wave, live, full)
        if self.checked == self.measure_at:
            # Keep both measurements at steady GPU clocks; oracle assertions otherwise idle
            # the GPU before the adaptive replay but leave it warm for the paired baseline.
            for selected in (graph, baseline, graph, baseline):
                for _ in range(10):
                    self.native_replay(selected)
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(100):
                    self.native_replay(selected)
                end.record()
                self.steady.append((start, end))
        write_cells(wave, live)
        self.checked += 1
        self.timings.append(events)

    def check_eager(self, wave, live):
        """Check every computed output and owned KV slot before running the independent oracle."""
        observation, results = wave.observation, wave.results
        with (
            override_forward_context(wave.attention.context),
            patch.object(self, "native_selection", self.eager_selection),
        ):
            eager = wave.body()
        wave.observation, wave.results = observation, results
        torch.testing.assert_close(wave.output, eager, atol=0, rtol=0)
        for value, uncaptured in zip(live, read_cells(wave), strict=True):
            torch.testing.assert_close(value, uncaptured, atol=0, rtol=0)

    def check_reference(self, wave, live, full):
        """Match native propagation bitwise and bound compacted rows by batch numerics."""
        active = self.mask(wave, torch.full((len(wave.seed), 1), self.rule == "all", device=wave.seed.device))
        if self.rule == "mixed":
            split = split_full(wave)
            bound = float((self.logits(full) - self.logits(split)).abs().max())
            error = float(
                (self.logits(wave.output)[active.flatten()] - self.logits(full)[active.flatten()]).abs().max()
            )
            self.max_error, self.max_bound = max(self.max_error, error), max(self.max_bound, bound)
        with override_forward_context(wave.attention.context):
            expected = reference(wave, active)
        if self.rule not in ("mixed", "buckets"):
            torch.testing.assert_close(wave.output, expected, atol=0, rtol=0)
        for value, oracle in zip(live, read_cells(wave), strict=True):
            if self.rule in ("mixed", "buckets"):
                value, oracle = value[~active.flatten()], oracle[~active.flatten()]
            torch.testing.assert_close(value, oracle, atol=0, rtol=0)
        if self.rule == "all":
            torch.testing.assert_close(wave.output, full, atol=0, rtol=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--value-rewrite", action="store_true")
    parser.add_argument("--rule", choices=("none", "all", "mixed", "buckets", "contracts"), default="none")
    args = parser.parse_args()
    config = pllm.ModelConfig(
        args.model,
        max_model_len=max(512, args.tokens + 64),
        max_num_seqs=args.batch,
        max_num_batched_tokens=512,
        gpu_memory_utilization=args.gpu_memory_utilization,
        execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=512 << 20),
    )
    if args.rule == "contracts":
        replay_contracts(config)
        return
    prompts = ["The capital of France is" + " very" * i for i in range(args.batch)]
    params = SamplingParams(max_tokens=args.tokens, temperature=0, ignore_eos=True)
    vanilla = None
    if args.rule == "mixed":
        import gc

        baseline = pllm.make_llm(workload=Exit("vanilla"), config=config)
        vanilla = baseline.generate(prompts, params)
        baseline.close()
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
    handle = pllm.make_llm(workload=ValueExit(args.rule) if args.value_rewrite else Exit(args.rule), config=config)
    audit = Audit(
        args.rule,
        lambda hidden: torch.nn.functional.linear(hidden, handle.runner.model.lm_head.weight),
        min(128, max(1, (args.tokens - 1) // 2)),
    )
    try:
        with (
            patch.object(
                AdaptiveSlice,
                "__call__",
                lambda selection, step, hidden, active: audit.select(selection, step, hidden, active),
            ),
            patch.object(ProgramWave, "capture", lambda wave: audit.capture(wave)),
            patch.object(torch.cuda.CUDAGraph, "replay", lambda graph: audit.replay(graph)),
        ):
            outputs = handle.generate(prompts, params)
        if vanilla is not None:
            assert audit.max_error <= audit.max_bound + 1e-6, (audit.max_error, audit.max_bound)
            for index in range(1, args.batch, 2):
                assert outputs[index].outputs[0].token_ids == vanilla[index].outputs[0].token_ids, (
                    f"request {index}: use VLLM_BATCH_INVARIANT=1 to require identical greedy tokens across GEMM widths"
                )
            print(
                f"non_exited_greedy_equal={args.batch // 2}x{args.tokens} max_logit_error={audit.max_error} batch_numerics_bound={audit.max_bound}"
            )
        torch.cuda.synchronize()
        elapsed = [(a.elapsed_time(b), c.elapsed_time(d)) for a, b, c, d in audit.timings[1:]]
        adaptive, baseline = (sum(values) / len(elapsed) for values in zip(*elapsed))
        steady = [start.elapsed_time(end) / 100 for start, end in audit.steady]
        print(f"steady_replay_ms={steady} ratio={(steady[0] + steady[2]) / (steady[1] + steady[3]):.4f}")
        print(
            f"graph_eager_bitwise_counts={sorted(audit.counts)} capture_bytes={audit.capture_bytes} budget_bytes={config.execution.cuda_graph_pool_budget_bytes}"
        )
        print(
            f"adaptive_depth model={args.model} batch={args.batch} rule={args.rule} checked={audit.checked} adaptive_ms={adaptive:.4f} baseline_ms={baseline:.4f} ratio={adaptive / baseline:.4f} blocking_syncs=0"
        )
    finally:
        handle.close()


if __name__ == "__main__":
    main()
