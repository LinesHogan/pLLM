"""OverRIDE on a real vLLM model, checked against the captured hidden state and base logits.

Guidance added before filtering must equal the independent head oracle at the exact
read version, feedback must carry the untouched base logits and the drawn tokens,
and a captured completion batch feeds FTTT and TTRL on the GPU.
"""

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import pllm
from pllm.backends.vllm.runner import WorkloadModelRunner
from pllm.backends.vllm.sampler import Vllm027Sampler
from pllm.config import ModelConfig, VllmDType
from pllm.execution.state_execution import StateExecution
from pllm.host import HostRuntimeFacts
from pllm.workloads.fttt import FTTTWorkload
from pllm.workloads.override import OverRIDEConfig, OverRIDEWorkload
from pllm.workloads.ttrl import TTRLConfig, TTRLWorkload


def main() -> None:
    config = OverRIDEConfig(rank=2, num_iterations=3)
    handle = pllm.make_llm(
        workload=OverRIDEWorkload(config),
        config=ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct",
            VllmDType.BFLOAT16,
            64,
            2,
            64,
            0.25,
            seed=7,
            execution=pllm.ExecutionOptions(memory_budget_bytes=1 << 30),
        ),
    )
    execution = handle.deployment.execution
    hidden = None
    base_logits = None
    observed = []
    drawn = {}
    hook = None
    reader_ref = None
    native_guidance = StateExecution.sampler_guidance
    native_add, native_complete = Vllm027Sampler._add_logits, StateExecution.complete_sampler
    try:

        def capture(module, args, output):
            nonlocal hidden
            value = output[0] if isinstance(output, tuple) else output
            hidden = value.detach().float().clone()

        hook = handle.runner.model.model.norm.register_forward_hook(capture)

        def inspect_guidance(owner, invocation, stream=None):
            nonlocal reader_ref
            (reader_ref,) = invocation.context.pin.refs
            return native_guidance(owner, invocation, stream)

        def inspect_add(logits, guidance):
            nonlocal base_logits
            base_logits = logits.detach().clone()
            assert hidden is not None
            # Lagged guidance uses the invocation's exact read pin, while training
            # may already have published a newer version in the directory.
            assert reader_ref is not None
            parameters = execution.snapshot(reader_ref, torch.cuda.current_stream(execution.device))[0]
            width, vocab = hidden.shape[1], logits.shape[1]
            down = parameters[: 4 * width].reshape(2, 2, width)
            up = parameters[4 * width :].reshape(2, vocab, 2)
            rows = guidance.guidance.sampler_rows
            expected = torch.zeros(logits.shape, device=logits.device, dtype=torch.float32)
            for link in rows.links:
                identity = rows.row_batch.rows[link.packed_row_index].identity
                if identity.sample_idx:
                    expected[link.sampler_row_index] = config.lambd * (
                        hidden[link.packed_row_index] @ down[identity.sample_idx - 1].T @ up[identity.sample_idx - 1].T
                    )
            torch.testing.assert_close(guidance.pred_dense_logits, expected, rtol=2e-4, atol=2e-5)
            result = native_add(logits, guidance)
            torch.testing.assert_close(result, logits + expected.to(logits))
            torch.testing.assert_close(logits, base_logits, rtol=0, atol=0)
            observed.append(int(torch.count_nonzero(expected)))
            return result

        def inspect_feedback(owner, invocation, stream=None, *, feedback=None):
            native_complete(owner, invocation, stream, feedback=feedback)
            assert feedback is not None and base_logits is not None
            torch.testing.assert_close(feedback.base_logits, base_logits, rtol=0, atol=0)
            for identity, token in zip(feedback.sampler_rows.identities, feedback.sampled_tokens.tolist(), strict=True):
                drawn.setdefault((identity.prompt_idx, identity.sample_idx), []).append(token)

        with (
            patch.object(Vllm027Sampler, "_add_logits", staticmethod(inspect_add)),
            patch.object(StateExecution, "sampler_guidance", inspect_guidance),
            patch.object(StateExecution, "complete_sampler", inspect_feedback),
        ):
            for iteration in range(3):
                before = execution.diagnostics().update_count
                drawn.clear()
                outputs = handle.generate(
                    ["The sky is", "Water freezes at"],
                    SamplingParams(max_tokens=3, temperature=0.8, top_k=10, top_p=0.9, ignore_eos=True),
                    sample_index=iteration,
                )
                assert len(outputs) == 2
                for index, output in enumerate(outputs):
                    assert drawn[(iteration * 2 + index, iteration)] == list(output.outputs[0].token_ids[1:])
                assert execution.diagnostics().update_count - before == (2 if iteration < 2 else 0)
        # Side-lane training releases its pins when its GPU work completes; check after it has.
        execution.arena.collect_completed(wait=True)
        diagnostics = execution.diagnostics()
        assert diagnostics.loss_count == diagnostics.update_count == 4
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        assert len(observed) == 6 and min(observed[-2:]) > 0
        expected_hidden = {}
        native_sample = WorkloadModelRunner.sample

        def inspect_completion(runner, hidden_states, input_batch, grammar):
            assert hidden is not None
            offset = 0
            for request_id, count in zip(input_batch.req_ids, input_batch.num_scheduled_tokens, strict=True):
                offset += int(count)
                expected_hidden[request_id] = hidden[offset - 1].clone()
            return native_sample(runner, hidden_states, input_batch, grammar)

        with patch.object(WorkloadModelRunner, "sample", inspect_completion):
            batch = handle.generate(
                ["Choose 1 or 2, boxed: "] * 3,
                SamplingParams(
                    max_tokens=16, temperature=0.8, structured_outputs=StructuredOutputsParams(regex=r"\\boxed\{[12]\}")
                ),
                sample_index=2,
                feedback_group="vote",
                prompt_indices=(100, 100, 100),
            )
        stream = torch.cuda.current_stream(execution.device)
        batch.readiness.order_on(stream)
        assert len(batch.completions.entries) == 3
        for completion, row in zip(batch.completions.entries, batch.hidden, strict=True):
            assert completion.identity.prompt_idx == 100
            torch.testing.assert_close(row, expected_hidden[completion.identity.request_id], rtol=0, atol=0)
        facts = HostRuntimeFacts(execution.device, hidden_size=batch.hidden.shape[1])
        feedback = pllm.make_llm(workload=FTTTWorkload(), config=facts)
        edited = feedback.run(batch)
        assert edited is not None and edited.update_count == 1
        reward = pllm.make_llm(
            workload=TTRLWorkload(TTRLConfig(rank=2, votes_per_prompt=3, samples_per_prompt=2)),
            config=replace(facts, max_input_rows=3, memory_budget_bytes=1 << 20),
        )
        reward.admit((), stream)
        trained = reward.step(batch, stream)
        assert trained.loss is not None and reward.diagnostics().update_count == 1
        reward.retire(())
        print(f"OverRIDE: updates={diagnostics.update_count} sampler_checks={len(observed)}; FTTT/TTRL updates=1/1")
    finally:
        if hook is not None:
            hook.remove()
        handle.close()
    assert execution.diagnostics().active_lineages == 0


if __name__ == "__main__":
    main()
