"""Replace one mathematical function without subclassing or touching execution."""

from dataclasses import replace
import torch
from pllm.workloads.dummy import DummyConfig, DummyWorkload


def centered(hidden: torch.Tensor) -> torch.Tensor:
    return hidden - hidden.mean(dim=-1, keepdim=True)


def centered_hidden(config: DummyConfig = DummyConfig()) -> DummyWorkload:
    return replace(DummyWorkload(config), transform=centered, export_name="centered_hidden")


if __name__ == "__main__":
    from vllm import SamplingParams
    from pllm import ModelConfig, make_llm
    with make_llm(ModelConfig("Qwen/Qwen2.5-0.5B-Instruct"),
                  workload=centered_hidden(DummyConfig(export_every_n_steps=1))) as llm:
        llm.generate(["Explain normalization:"], SamplingParams(max_tokens=4))
        print([event.tensor() for event in llm.trace_snapshot().events if event.tensor_name == "centered_hidden"])
