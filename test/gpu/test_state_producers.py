"""Internal q/v projection oracle on Qwen2 for each independent state producer."""

import time

import pytest
import torch
from vllm import SamplingParams

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.semantic_graph import StateResource
from pllm.workloads.state_producers import DecodeFactors, PromptFactors, PromptMixture
from test.gpu.vllm_inplace_ttt_smoke import open_llm


@pytest.mark.parametrize("workload", [PromptFactors(), PromptMixture(), DecodeFactors()])
def test_internal_projection_consumes_pinned_request_factors(workload, monkeypatch):
    with open_llm(workload, memory_budget_bytes=1 << 30) as handle:
        session = handle.runner.require_session()
        execution = handle.deployment.execution
        overlay = session._overlay
        checked = []
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
        assert frozenset((HiddenPhase.PREFILL, HiddenPhase.DECODE)) in phases
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
