"""ReFT on a real Qwen2 layer: imported parameters patch exactly the first/last prompt rows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from vllm import SamplingParams

import pllm
from pllm.config import ModelConfig
from pllm.workloads.reft import ReFTConfig, ReFTWorkload

BUDGET = pllm.ExecutionOptions(memory_budget_bytes=1 << 30)


def main() -> None:
    handle = pllm.make_llm(
        workload=ReFTWorkload(ReFTConfig((23,), 2, first_tokens=1, last_tokens=1)),
        config=ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", "bfloat16", 64, 2, 64, 0.25, seed=7, execution=BUDGET),
    )
    execution = handle.runner.require_session()._deployment.execution
    baseline = []
    observed = []
    hook = None
    try:
        layer = handle.runner.model.model.layers[23]

        def capture(module, args, output):
            if args[0].numel() > 2:
                baseline.append(tuple(value.detach().clone() for value in output))

        hook = layer.register_forward_hook(capture)
        prompts = ["The sky is", "Water freezes at"]
        sampling = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
        handle.generate(prompts, sampling)
        assert execution.diagnostics().update_count == 0 and len(baseline) == 1
        hook.remove()
        hidden, residual = baseline[0]
        width, device = hidden.shape[1], hidden.device
        generator = torch.Generator(device=device).manual_seed(23)
        parameters = tuple(
            torch.randn(shape, device=device, dtype=hidden.dtype, generator=generator) * 0.02
            for shape in ((2, width), (2,), (width, 2))
        )

        def check(module, args, output):
            if args[0].numel() <= 2:
                return
            positions = args[0]
            starts = (positions == 0).nonzero().flatten().tolist()
            ends = [*starts[1:], positions.numel()]
            selected = torch.tensor(
                sorted({index for begin, end in zip(starts, ends) for index in (begin, end - 1)}), device=device
            )
            weight, bias, rotation = parameters
            expected = hidden.clone()
            rows = hidden[selected]
            delta = ((rows @ weight.T + bias) - rows @ rotation) @ rotation.T
            expected[selected] += delta
            torch.testing.assert_close(output[0], expected, rtol=0.02, atol=0.02)
            torch.testing.assert_close(output[1], residual, rtol=0, atol=0)
            assert torch.count_nonzero(output[0] - hidden) > 0
            observed.append(selected.numel())

        hook = layer.register_forward_hook(check)
        handle.generate(prompts, sampling, batch=pllm.state.StateImport(parameters))
        assert observed == [4]
        diagnostics = execution.diagnostics()
        assert diagnostics.update_count == 1
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == 0
        print(f"ReFT: publications={diagnostics.update_count} patched_prefill_rows={observed[0]}")
    finally:
        if hook is not None:
            hook.remove()
        handle.close()
    assert execution.diagnostics().active_states == 0


if __name__ == "__main__":
    main()
