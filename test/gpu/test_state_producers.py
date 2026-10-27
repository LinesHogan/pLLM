"""Internal q/v projection oracle on Qwen2 for each independent state producer."""

import time
from dataclasses import replace

import pytest
import torch
from vllm import SamplingParams

from pllm.contracts.backend_attachment import AttachmentKind
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.semantic_graph import StateResource
from pllm.execution.backend_attachment import compile_backend_attachment
from pllm.execution.tensor_executor import TensorExecutor
from pllm.workloads.state_producers import DecodeFactors, PromptFactors, PromptMixture
from test.gpu.vllm_inplace_ttt_smoke import open_llm


@pytest.mark.parametrize("workload", [PromptFactors(), PromptMixture(), DecodeFactors()])
def test_internal_projection_consumes_pinned_request_factors(workload, monkeypatch):
    with open_llm(workload, memory_budget_bytes=1 << 30) as handle:
        session = handle.runner.require_session()
        execution = handle.deployment.execution
        overlay = session._overlay
        captures = []
        capture_begin = torch.cuda.CUDAGraph.capture_begin

        def begin(graph, *args, **kwargs):
            captures.append(graph)
            return capture_begin(graph, *args, **kwargs)

        monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_begin", begin)
        checked = []
        launches = []
        plan = handle.attachment.inputs.plan
        execute = execution.execute_anchor

        def delayed(invocation, anchor):
            tasks = tuple(task for task in plan.schedule_physical.tasks if task.anchor_id == anchor)
            if tasks:
                assert all(task.lane == "side" and task.read_origin == "latest" for task in tasks)
                assert invocation.context.pin.stream == session._post_sample.stream
                with torch.cuda.stream(invocation.context.pin.stream):
                    torch.cuda._sleep(1_000_000)
                launches.append(plan.timeline.anchors[anchor].symbol)
            execute(invocation, anchor)

        monkeypatch.setattr(execution, "execute_anchor", delayed)
        copy_inputs = session._post_sample.copy_inputs

        def poison_taps(context):
            hidden = context.data().step.hidden
            copy_inputs(context)
            for snapshot in hidden.snapshots:
                snapshot.storage.fill_(float("nan"))

        monkeypatch.setattr(session._post_sample, "copy_inputs", poison_taps)
        phases = set()
        torch.cuda.reset_peak_memory_stats()

        def intercept(reader, index):
            emit = reader.emit

            def compare(hidden, pools, table, output, captured):
                base = output.clone()
                emit(hidden, pools, table, output, captured)
                context = session._step.invocation.context
                rows = session._step.rows.row_batch.rows
                phases.add(frozenset(row.phase for row in rows))
                for row in rows:
                    ref = context.current_owned_ref(context.selection.ref_for(row.identity, StateResource("factors")))
                    root = execution.snapshot(ref, torch.cuda.current_stream())
                    width, inputs = reader.width, hidden.shape[-1]
                    rank = workload.factor_rank
                    if isinstance(workload, DecodeFactors):
                        shapes = tuple(
                            (rank, target.input_size) if side == 0 else (target.output_size, rank)
                            for target in handle.attachment.inputs.probe.targets
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
            ["A short question about stars.", "word " * 61, "Name a color."],
            SamplingParams(max_tokens=4, temperature=0, ignore_eos=True),
        )
        elapsed = time.perf_counter() - start
        assert len(captures) == (1 if isinstance(workload, DecodeFactors) else 2)

        def forbid_body(*args):
            raise AssertionError("tensor Python body ran after graph warmup")

        with monkeypatch.context() as replay:
            replay.setattr(TensorExecutor, "_body", forbid_body)
            handle.generate(
                ["Name a color.", "A short question about stars."],
                SamplingParams(max_tokens=3, temperature=0, ignore_eos=True),
            )
        assert len(captures) == (1 if isinstance(workload, DecodeFactors) else 2)
        assert frozenset((HiddenPhase.PREFILL, HiddenPhase.DECODE)) in phases
        assert launches and set(launches) == {"state.publish"}
        assert tuple(a.symbol for a in plan.timeline.anchors) == (
            "request.prepare",
            "model.execute",
            "state.publish",
            "sampler.pre_draw",
            "sampler.complete",
            "token.commit",
            "request.finish",
            "next.prepare",
        )
        assert len(outputs) == 3 and {entry[2] for entry in checked} == {0, 1}
        assert any(entry[1] == 0 for entry in checked) and any(entry[1] > 0 for entry in checked)
        assert any(entry[3] != entry[4] for entry in checked)
        assert handle.diagnostics().active_lineages == len(execution.directory.instance_states)
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

    assert execution.diagnostics().active_lineages == 0


@pytest.mark.parametrize("workload", [PromptFactors(), DecodeFactors()])
def test_composed_readers_replay_model_graphs(workload, monkeypatch):
    from pllm.execution.step_capture import StepCapture

    results = []
    for graphs in (False, True):
        with open_llm(
            workload, cuda_graphs=graphs, capture_memory_bytes=256 << 20, memory_budget_bytes=1 << 30
        ) as handle:
            if graphs:
                inputs = handle.attachment.inputs
                with pytest.raises(ValueError, match="Graph overlays"):
                    compile_backend_attachment(
                        replace(
                            inputs, request=replace(inputs.request, attachment_kind=AttachmentKind.ACTIVATION_PATCH)
                        )
                    )
                with pytest.raises(ValueError, match="Graph overlays"):
                    compile_backend_attachment(
                        replace(
                            inputs,
                            probe=replace(
                                inputs.probe,
                                capabilities=tuple(c for c in inputs.probe.capabilities if c != "backend/vllm_mrv2@1"),
                            ),
                        )
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
            assert handle.diagnostics().fallback_count == 0
            if isinstance(workload, PromptFactors):
                session = handle.runner.require_session()
                prepare, saved = session.prepare_input, []

                def recompute(rows):
                    saved.append(rows)
                    return prepare(replace(saved[0], row_batch=replace(saved[0].row_batch, step_id=rows.step_id)))

                with monkeypatch.context() as replay:
                    replay.setattr(session, "prepare_input", recompute)
                    with pytest.raises(RuntimeError, match="would recompute"):
                        handle.generate(["Name a color."], SamplingParams(max_tokens=3, ignore_eos=True))
    assert [tokens for tokens, _ in results[0]] == [tokens for tokens, _ in results[1]]
    # vLLM's fused graph kernels and eager BF16 kernels round differently.
    assert [score for _, score in results[0]] == pytest.approx([score for _, score in results[1]], abs=0.1)


@pytest.mark.parametrize(
    "runner,budget,seqs",
    [(runner, budget, 2) for runner in ("mrv1", "mrv2") for budget in (128, 512, 2048)]
    + [(runner, 128, 128) for runner in ("mrv1", "mrv2")],
)
def test_chunked_runners_discard_partial_prompt_feedback(runner, budget, seqs):
    from pllm.config import RunnerKind
    from pllm.workloads.esamp_request import ESampRequestConfig, ESampRequestWorkload

    tokens = []
    prompts = ["word"] * (seqs + 2) if seqs == 128 else ["word " * (budget + 4), "word " * 74]
    for chunked in (False, True):
        with open_llm(
            ESampRequestWorkload(ESampRequestConfig(rank=2)),
            runner=RunnerKind(runner),
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
            assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
    assert tokens[0] == tokens[1]
