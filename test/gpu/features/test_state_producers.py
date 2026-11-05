"""Internal q/v projection oracle on Qwen2 for each independent state producer."""

import time
from dataclasses import dataclass, replace
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams

import pllm
from pllm.backends.vllm.sampler import GuidedSampler
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.semantic_graph import StateResource
from pllm.execution.tensor_executor import TensorExecutor
from pllm.workloads.state_producers import DecodeFactorsWorkload, PromptFactorsWorkload, PromptMixtureWorkload
from test.gpu.vllm_inplace_ttt_smoke import ImportedLowRank, open_llm


def _assert_imported_factor_rows(handle, fields, indices):
    """Read the graph's actual page indices, independently of admission and publication order."""
    session = handle.runner.require_session()
    (reader,) = session._overlay.readers
    rows = session._step.rows.row_batch
    count = next(size for size in reader.programs if size >= rows.total_rows)
    slots, positions, owners = reader.ports(reader.table, count)
    for row in rows.request_rows:
        block = int(owners[row.packed_row_index]) // positions.shape[1]
        prompt = indices.index(row.identity.prompt_index)
        for field, (pool, expected) in enumerate(zip(reader.pools, fields, strict=True)):
            torch.testing.assert_close(pool[int(slots[block, field])], expected[prompt].to(pool), rtol=0, atol=0)


@pytest.mark.parametrize("workload", [PromptFactorsWorkload(), ImportedLowRank()])
@pytest.mark.parametrize("chunked", [False, True])
def test_prompt_lora_batch_matches_individual_requests(workload, chunked):
    """Prompt-produced and imported factors follow prompt order across prefill chunks and samples."""
    prompts = [
        "What is the capital of France? Answer in one sentence.",
        "Name three primary colors.",
        "What is two plus two? Explain your answer in one sentence.",
    ]
    indices = (29, 3, 17)
    with open_llm(
        workload,
        seqs=6,
        tokens=8 if chunked else 128,
        model_length=128,
        chunked=chunked,
        cuda_graphs=True,
        capture_memory_bytes=512 << 20,
        memory_budget_bytes=3 << 30,
    ) as handle:
        fields = None
        if isinstance(workload, ImportedLowRank):
            (layout,) = handle.attachment.plan.tensor_layout_request.layouts
            # A small external hypernetwork maps prompt features to distinct LoRA factors.
            features = torch.tensor([[len(p) / 64, sum(p.encode()) % 31 / 31] for p in prompts])
            generator = torch.Generator().manual_seed(19)
            fields = tuple(
                torch.einsum("pc,c...->p...", features, torch.randn(2, *field.shape, generator=generator) * 0.02).to(
                    handle.device
                )
                for field in layout.layout.fields
            )
        sample = GuidedSampler.__call__

        def inspect(sampler, logits, inputs):
            if fields is not None:
                _assert_imported_factor_rows(handle, fields, indices)
            return sample(sampler, logits, inputs)

        with patch.object(GuidedSampler, "__call__", inspect):
            for n in (1, 2):
                # Exercise one adapted decode step; low temperature avoids rounding-sensitive random draws.
                sampling = SamplingParams(max_tokens=2, temperature=0 if n == 1 else 0.1, n=n, seed=71, ignore_eos=True)
                batch = None if fields is None else pllm.io.StateImport(fields)
                outputs = handle.generate(prompts, sampling, batch=batch, prompt_indices=indices)
                for i, prompt in enumerate(prompts):
                    batch = None if fields is None else pllm.io.StateImport(tuple(field[i : i + 1] for field in fields))
                    (single,) = handle.generate([prompt], sampling, batch=batch, prompt_indices=(indices[i],))
                    assert outputs[i].prompt_token_ids == single.prompt_token_ids
                    assert [sample.token_ids for sample in outputs[i].outputs] == [
                        sample.token_ids for sample in single.outputs
                    ]
        diagnostics = handle.diagnostics()
        assert diagnostics.active_states == len(
            handle.runner.require_session()._deployment.execution.directory.instance_states
        )
        assert diagnostics.pending_state_releases == 0


@pytest.mark.parametrize("workload", [PromptFactorsWorkload(), PromptMixtureWorkload(), DecodeFactorsWorkload()])
def test_internal_projection_consumes_pinned_request_factors(workload, monkeypatch):
    with open_llm(workload, seqs=3, chunked=True, memory_budget_bytes=1 << 30) as handle:
        monkeypatch.setattr(handle._require_open(), "request_counter", iter(range(99, 0, -1)))
        session = handle.runner.require_session()
        execution = session._deployment.execution
        overlay = session._overlay
        captures = []
        capture_begin = torch.cuda.CUDAGraph.capture_begin

        def begin(graph, *args, **kwargs):
            captures.append(graph)
            return capture_begin(graph, *args, **kwargs)

        monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_begin", begin)
        checked = []
        produced = {}
        write_outputs = TensorExecutor.write_outputs

        def remember(executor, data, batch, source_rows):
            output = write_outputs(executor, data, batch, source_rows)
            if executor.task.tuple_output:
                for index, row in enumerate(data.rows):
                    produced[row.identity] = tuple(value[index].clone() for value in executor.results)
            return output

        monkeypatch.setattr(TensorExecutor, "write_outputs", remember)
        plan = handle.attachment.plan
        execute = execution.execute_anchor

        def delayed(invocation, anchor):
            tasks = tuple(task for task in plan.schedule.tasks if task.anchor_id == anchor)
            if tasks:
                with torch.cuda.stream(invocation.pin.stream):
                    torch.cuda._sleep(1_000_000)
            execute(invocation, anchor)

        monkeypatch.setattr(execution, "execute_anchor", delayed)
        phases = set()
        torch.cuda.reset_peak_memory_stats()

        def intercept(reader, index):
            emit = reader.emit

            def compare(hidden, desired, pools, table, output, captured, desired_capture):
                base = output.clone()
                emit(hidden, desired, pools, table, output, captured, desired_capture)
                context = session._step.context
                rows = session._step.rows.row_batch.rows
                phases.add(frozenset(row.phase for row in rows))
                for row in rows:
                    ref = context.current_owned_ref(context.selection.ref_for(row.identity, StateResource("factors")))
                    root = execution.snapshot(ref, torch.cuda.current_stream())
                    width, inputs = reader.width, hidden.shape[-1]
                    rank = workload.factor_rank
                    if isinstance(workload, DecodeFactorsWorkload):
                        shapes = tuple(
                            (rank, target.input_size) if side == 0 else (target.output_size, rank)
                            for target in handle.attachment.probe.targets
                            for side in range(2)
                        )
                        fields = tuple(
                            value.reshape(shape)
                            for value, shape in zip(
                                torch.split(root[0], tuple(a * b for a, b in shapes)), shapes, strict=True
                            )
                        )
                    else:
                        fields = root
                        if row.phase is HiddenPhase.DECODE:
                            assert ref.version_id > 0
                            for field, expected_field in zip(fields, produced[row.identity], strict=True):
                                torch.testing.assert_close(field, expected_field)
                    a, b = fields[index * 2 : index * 2 + 2]
                    x = hidden[row.packed_row_index].double()
                    expected = (
                        base[row.packed_row_index, reader.offset : reader.offset + width].double()
                        + (x @ a.double().T) @ b.double().T
                    )
                    torch.testing.assert_close(
                        output[row.packed_row_index, reader.offset : reader.offset + width].double(),
                        expected.to(output.dtype).double(),
                        rtol=0.008,
                        atol=0.008,
                    )
                    checked.append((row.identity.request_id, ref.version_id, index, inputs, width))

            return compare

        for index, reader in enumerate(overlay.readers):
            monkeypatch.setattr(reader, "emit", intercept(reader, index))
        start = time.perf_counter()
        outputs = handle.generate(
            ["Name a color.", "word " * 61, "Name a color."],
            SamplingParams(max_tokens=4, temperature=0, ignore_eos=True),
        )
        elapsed = time.perf_counter() - start
        capture_count = len(captures)

        def forbid_body(*args):
            raise AssertionError("tensor Python body ran after graph warmup")

        with monkeypatch.context() as replay:
            replay.setattr(TensorExecutor, "_body", forbid_body)
            handle.generate(
                ["Name a color.", "A short question about stars."],
                SamplingParams(max_tokens=3, temperature=0, ignore_eos=True),
            )
        assert len(captures) == capture_count
        assert frozenset((HiddenPhase.PREFILL, HiddenPhase.DECODE)) in phases
        assert len(outputs) == 3 and {entry[2] for entry in checked} == {0, 1}
        assert any(entry[1] == 0 for entry in checked) and any(entry[1] > 0 for entry in checked)
        assert any(entry[3] != entry[4] for entry in checked)
        assert handle.diagnostics().active_states == len(execution.directory.instance_states)
        assert handle.diagnostics().pending_state_releases == 0
        print(
            type(workload).__name__,
            "seconds",
            elapsed,
            "peak_allocated",
            torch.cuda.max_memory_allocated(),
            "peak_reserved",
            torch.cuda.max_memory_reserved(),
        )

    assert execution.diagnostics().active_states == 0


@pytest.mark.parametrize("workload", [PromptFactorsWorkload(), DecodeFactorsWorkload()])
def test_composed_readers_replay_model_graphs(workload, monkeypatch):
    from pllm.backends.vllm.parameter_overlay import capacities
    from pllm.execution.step_capture import StepCapture

    results = []
    for graphs in (False, True):
        with open_llm(
            workload, cuda_graphs=graphs, capture_memory_bytes=256 << 20, memory_budget_bytes=1 << 30
        ) as handle:
            if not graphs:
                manager = handle.runner.cudagraph_manager
                dispatch = manager.dispatch
                # Capture pads prefill to a capacity class. Compare the same arithmetic shape.
                manager.dispatch = lambda *args, **kwargs: replace(
                    dispatch(*args, **kwargs),
                    num_tokens=next(n for n in capacities(64) if n >= args[1]),
                )
            runs = []
            run = StepCapture.run

            def record(capture, *args, **kwargs):
                runs.append(capture)
                return run(capture, *args, **kwargs)

            with monkeypatch.context() as checks:
                checks.setattr(StepCapture, "run", record)
                outputs = handle.generate(
                    ["Name a color.", "word " * 31],
                    SamplingParams(max_tokens=5, temperature=0, ignore_eos=True, logprobs=1),
                )
                assert runs
                checks.setattr(TensorExecutor, "_body", lambda *args: pytest.fail("producer body after warmup"))
                outputs += handle.generate(
                    ["word " * 31, "Name a color."],
                    SamplingParams(max_tokens=5, temperature=0, ignore_eos=True, logprobs=1),
                )
            results.append([(o.outputs[0].token_ids, o.outputs[0].cumulative_logprob) for o in outputs])
            if isinstance(workload, PromptFactorsWorkload):
                session = handle.runner.require_session()
                before_execute = session.before_execute

                def preempt(scheduler_output):
                    scheduler_output.preempted_req_ids = set(session._identities)
                    return before_execute(scheduler_output)

                with monkeypatch.context() as replay:
                    replay.setattr(session, "before_execute", preempt)
                    with pytest.raises(RuntimeError, match="would recompute"):
                        handle.generate(["Name a color."], SamplingParams(max_tokens=3, ignore_eos=True))
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "runner,budget,seqs",
    [(runner, 128, seqs) for runner in ("mrv1", "mrv2") for seqs in (2, 128)],
)
def test_chunked_runners_discard_partial_prompt_feedback(runner, budget, seqs):
    from pllm.workloads.esamp import ESampConfig, ESampWorkload

    tokens = []
    prompts = ["word"] * (seqs + 2) if seqs == 128 else ["word " * (budget + 4), "word " * 74]
    for chunked in (False, True):
        with open_llm(
            ESampWorkload(ESampConfig(rank=2)),
            runner=runner,
            seqs=seqs,
            chunked=chunked,
            tokens=budget if chunked else 2176,
            model_length=2176,
            memory_budget_bytes=4 << 30,
            capture_memory_bytes=256 << 20,
        ) as handle:
            outputs = handle.generate(
                prompts,
                SamplingParams(max_tokens=3, temperature=0, ignore_eos=True),
            )
            tokens.append([o.outputs[0].token_ids for o in outputs])
            diagnostics = handle.diagnostics()
            assert diagnostics.loss_count == diagnostics.update_count == 2 * len(prompts)
            # Releases finish asynchronously; their completion is covered by the page-arena tests.
            assert diagnostics.active_states == 0
    assert tokens[0] == tokens[1]


@dataclass(frozen=True)
class PretrainedProbe(pllm.Workload):
    """Export the tutorial's probe scores with hidden rows for an independent arithmetic oracle."""

    state: pllm.InstanceState

    @pllm.export(name="probe", max_rows=2, max_cols=897)
    def score(self, h=pllm.final_hidden(), w=pllm.parameter("state", "w"), b=pllm.parameter("state", "b")):
        scores = torch.sigmoid((h.float() * w).sum(-1, keepdim=True) + b)
        return torch.cat((scores, h.float()), dim=-1)


def test_hf_trained_probe_deploys_from_safetensors(tmp_path):
    import gc

    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pllm import testing

    model = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)
    hf = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).cuda()
    texts = [
        "I love this movie, it is wonderful.",
        "What a fantastic and joyful day.",
        "I hate this movie, it is awful.",
        "What a terrible and miserable day.",
    ]
    with torch.no_grad():
        rows = [
            hf(**tokenizer(text, return_tensors="pt").to("cuda"), output_hidden_states=True)
            .hidden_states[-1][0]
            .float()
            for text in texts
        ]
    features = torch.cat(rows)
    labels = torch.cat([torch.full((len(row),), float(i < 2), device="cuda") for i, row in enumerate(rows)])
    del hf, rows
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    probe = torch.nn.Linear(features.shape[1], 1).cuda()
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(100):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(features).squeeze(-1), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    path = str(tmp_path / "sentiment_probe.safetensors")
    weights = {"w": probe.weight.detach()[0].cpu().contiguous(), "b": probe.bias.detach().cpu()}
    save_file(weights, path)
    del probe, optimizer, features, labels, loss
    gc.collect()
    torch.cuda.empty_cache()
    workload = PretrainedProbe(pllm.InstanceState({"w": (896,), "b": (1,)}, pretrained=path))
    config = pllm.ModelConfig(
        model, max_model_len=64, max_num_seqs=2, max_num_batched_tokens=64, gpu_memory_utilization=0.25
    )
    assert testing.check(workload, config).passed
    assert path in pllm.explain(workload, config).states[0].source
    with pllm.make_llm(config, workload=workload) as handle:
        handle.generate(
            ["The sky is", "What a wonderful day"], SamplingParams(max_tokens=4, temperature=0, ignore_eos=True)
        )
        events = [event for event in handle.trace_snapshot().events if event.tensor_name == "probe"]
        assert len(events) == 3
        for event in events:
            values = event.tensor()
            expected = torch.sigmoid((values[:, 1:] * weights["w"].bfloat16()).sum(-1) + weights["b"].bfloat16())
            torch.testing.assert_close(values[:, 0], expected)
        assert handle.diagnostics().eager_reason == ""
