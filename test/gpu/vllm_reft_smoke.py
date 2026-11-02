"""ReFT on a real Qwen2 layer: imported parameters patch exactly the first/last prompt rows of the last layer's MLP
output, which joins the residual stream in the final norm."""

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
    hooks = []
    try:
        # Layer 23 is the last: its patched MLP output joins the stream inside the final norm.
        norm = handle.runner.model.model.norm

        def capture(module, args):
            if args[0].shape[0] > 2:
                baseline.append(tuple(value.detach().clone() for value in args))

        hooks.append(norm.register_forward_pre_hook(capture))
        prompts = ["The sky is", "Water freezes at"]
        sampling = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
        handle.generate(prompts, sampling)
        assert execution.diagnostics().update_count == 0 and len(baseline) == 1
        hooks.pop().remove()
        hidden, residual = baseline[0]
        width, device = hidden.shape[1], hidden.device
        generator = torch.Generator(device=device).manual_seed(23)
        parameters = tuple(
            torch.randn(shape, device=device, dtype=hidden.dtype, generator=generator) * 0.02
            for shape in ((2, width), (2,), (width, 2))
        )
        positions: list[torch.Tensor] = []
        layer = handle.runner.model.model.layers[23]
        hooks.append(layer.register_forward_pre_hook(lambda module, args: positions.append(args[0])))

        def check(module, args, output):
            if args[0].shape[0] <= 2:
                return
            starts = (positions[-1] == 0).nonzero().flatten().tolist()
            ends = [*starts[1:], positions[-1].numel()]
            selected = torch.tensor(
                sorted({index for begin, end in zip(starts, ends) for index in (begin, end - 1)}), device=device
            )
            weight, bias, rotation = parameters
            patched = hidden.clone()
            rows = hidden[selected]
            delta = ((rows @ weight.T + bias) - rows @ rotation) @ rotation.T
            patched[selected] += delta
            torch.testing.assert_close(output[1], residual + patched, rtol=0.02, atol=0.02)
            assert torch.count_nonzero(output[1] - (residual + hidden)) > 0
            observed.append(selected.numel())

        hooks.append(norm.register_forward_hook(check))
        handle.generate(prompts, sampling, batch=pllm.io.StateImport(parameters))
        assert observed == [4]
        diagnostics = execution.diagnostics()
        assert diagnostics.update_count == 1
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == 0
        print(f"ReFT: publications={diagnostics.update_count} patched_prefill_rows={observed[0]}")
    finally:
        for hook in hooks:
            hook.remove()
        handle.close()
    assert execution.diagnostics().active_states == 0


if __name__ == "__main__":
    main()
