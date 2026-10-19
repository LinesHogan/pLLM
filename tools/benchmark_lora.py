"""Shared and independent output LoRA through the public execution path."""

import argparse
import subprocess
import sys
from dataclasses import replace
from math import ceil

import torch

import pllm
from pllm.benchmark import BenchmarkOptions, compare
from pllm.workloads import OutputLoRAConfig, OutputLoRAWorkload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--training-rows", type=int, default=2)
    parser.add_argument("--training-fraction", type=float, default=1.0)
    parser.add_argument("--scope", choices=("instance", "request", "both"), default="both")
    parser.add_argument(
        "--state-groups", type=int, help="request-key groups in the synchronous batch; defaults to one per request"
    )
    options = BenchmarkOptions.parse(parser)
    args = parser.parse_args()
    if args.scope == "both" and options.profile is not None:
        for scope in ("instance", "request"):
            subprocess.run([sys.executable, "-m", "tools.benchmark_lora", *sys.argv[1:], "--scope", scope], check=True)
        return
    if args.training_rows < 1 or not 0 < args.training_fraction <= 1:
        parser.error("training rows must be positive and training fraction in (0, 1]")
    groups = options.batch if args.state_groups is None else args.state_groups
    if not 1 <= groups <= options.batch:
        parser.error("state groups must be between one and the request batch size")
    active = ceil(args.training_rows * args.training_fraction)
    for scope in (pllm.StateScope.INSTANCE, pllm.StateScope.REQUEST):
        if args.scope not in ("both", scope.value):
            continue
        states = groups if scope is pllm.StateScope.REQUEST else 1
        indices = (
            tuple(index % states for index in range(options.batch))
            if scope is pllm.StateScope.REQUEST
            else tuple(range(options.batch))
        )
        print(f"Output LoRA scope: {scope.value}")

        def prepare(hidden_size: int, vocab_size: int, device: torch.device) -> pllm.WeightedCEBatch[torch.Tensor]:
            shape = (states, args.training_rows) if scope is pllm.StateScope.REQUEST else (args.training_rows,)
            generator = torch.Generator(device=device).manual_seed(7)
            hidden = torch.randn((*shape, hidden_size), device=device, generator=generator)
            logits = torch.zeros((*shape, vocab_size), device=device)
            labels = torch.zeros(shape, device=device, dtype=torch.long)
            weights = (torch.arange(args.training_rows, device=device) < active).float().expand(shape)
            return (
                pllm.WeightedCEBatch.for_prompts(hidden, logits, labels, weights, tuple(range(states)))
                if scope is pllm.StateScope.REQUEST
                else pllm.WeightedCEBatch(hidden, logits, labels, weights)
            )

        config = OutputLoRAConfig(rank=args.rank, steps=2, scope=scope)
        print(
            f"rank={config.rank}, training_rows={args.training_rows}, active_rows={active}, steps={config.steps}, state_groups={states}"
        )
        compare(
            OutputLoRAWorkload(config),
            replace(options, profile=None if options.profile is None else options.profile / scope.value),
            prepare=lambda handle: prepare(handle.hidden_size, handle.vocab_size, handle.device),
            training_rows=args.training_rows,
            capture=True,
            require_training=True,
            prompt_indices=indices,
        )


if __name__ == "__main__":
    main()
