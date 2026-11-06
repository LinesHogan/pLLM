"""ReFT on a real Qwen2 layer: imported parameters patch exactly the first/last prompt rows of the last layer's MLP
output, which joins the residual stream in the final norm."""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from vllm import SamplingParams

import pllm
from pllm.config import ModelConfig
from pllm.execution.trace import TraceEventKind, TraceSelection
from pllm.workloads.reft import ReFTConfig, ReFTWorkload

BUDGET = pllm.ExecutionOptions(memory_budget_bytes=1 << 30)


@dataclass(frozen=True)
class PromptSteering(pllm.Workload):
    """Per-prompt steering also exports the values used by the first forward."""

    request: bool = True

    @property
    def steer(self) -> pllm.InstanceState:
        fields = {"vector": pllm.Field(lambda width, _: (width,))}
        return pllm.RequestState(fields, imported=True) if self.request else pllm.InstanceState(fields, imported=True)

    @property
    def offset(self) -> pllm.InstanceState:
        return pllm.InstanceState({"value": (1,)})

    @property
    def patch(self) -> pllm.model.ActivationPatch:
        return pllm.model.ActivationPatch("steer", (23,), operator="add")

    @pllm.export(name="imported", every=1, max_rows=4, max_cols=1)
    def observe(
        self,
        hidden=pllm.hidden(23, tokens="prefill"),
        vector=pllm.parameter("steer", "vector", per_row=True),
        offset=pllm.parameter("offset", "value", per_row=True),
    ):
        return vector[:, :1] + hidden[:, :1] * 0 + offset


def check_prompt_steering() -> None:
    """Check prompt rows, n samples, later generate calls and first-step export visibility."""
    config = ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", "bfloat16", 64, 2, 64, 0.25, seed=7, execution=BUDGET)
    for request in (True, False):
        with pllm.make_llm(workload=PromptSteering(request), config=config) as handle:
            handle.select_trace(TraceSelection(tensors=True, max_tensor_bytes=64))
            values = torch.stack((torch.ones(896), torch.full((896,), 2.0))) if request else torch.ones(896)
            for call, multiplier in enumerate((1, 3)):
                handle.generate(
                    ["The sky is", "Water freezes at"],
                    SamplingParams(n=2, max_tokens=1, temperature=0.8),
                    batch=pllm.io.StateImport((values * multiplier,)),
                )
                events = [
                    event
                    for event in handle.trace_snapshot().events
                    if event.kind is TraceEventKind.TENSOR and event.tensor_name == "imported"
                ]
                observed = {
                    identity.request_id: (identity, row.item())
                    for event in events
                    for identity, row in zip(event.identities, event.tensor(), strict=True)
                    if identity.prompt_index // 2 == call
                }
                assert len(observed) == 4
                for identity, value in observed.values():
                    expected = multiplier * (1 + identity.prompt_index % 2 if request else 1)
                    assert value == expected, (identity, value, expected)
            diagnostics = handle.diagnostics()
            assert diagnostics.active_states == (1 if request else 2)
            assert diagnostics.inference_pins == diagnostics.pending_state_releases == 0
    print("StateImport: per-prompt rows shared by n=2; request/instance exports visible on first step across calls")


def main() -> None:
    workload = ReFTWorkload(ReFTConfig((23,), 2, first_tokens=1, last_tokens=1))
    config = ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct", "bfloat16", 64, 2, 64, 0.25, seed=7, execution=BUDGET, cuda_graphs=False
    )
    handle = pllm.make_llm(workload=workload, config=config)
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
        sampling = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True, logprobs=1)
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
        eager = handle.generate(prompts, sampling, batch=pllm.io.StateImport(parameters))
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
    with pllm.make_llm(workload=workload, config=replace(config, cuda_graphs=True)) as captured:
        assert captured.diagnostics().eager_reason == ""
        outputs = captured.generate(prompts, sampling, batch=pllm.io.StateImport(parameters))
    for actual, expected in zip(outputs, eager, strict=True):
        a, b = actual.outputs[0], expected.outputs[0]
        assert a.token_ids == b.token_ids
        torch.testing.assert_close(
            torch.tensor([row[token].logprob for row, token in zip(a.logprobs, a.token_ids, strict=True)]),
            torch.tensor([row[token].logprob for row, token in zip(b.logprobs, b.token_ids, strict=True)]),
            rtol=0,
            atol=0.05,
        )
    check_prompt_steering()


if __name__ == "__main__":
    main()
