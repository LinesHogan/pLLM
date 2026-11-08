"""Hashed bigram/trigram residual memory on Qwen2.5-0.5B, with explicitly placed frozen tables.

Fit on local text: python -m examples.engram_lite corpus.txt memory.safetensors
The fit averages projected next-token embeddings in each bucket; it makes no quality claim.
Use EngramLite(path, placement="host") for a table too large for the device budget.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]
from vllm.tokenizers import TokenizerLike, get_tokenizer  # pyright: ignore[reportUnknownVariableType]

import pllm


def ngram_keys(ids: torch.Tensor, buckets: int) -> torch.Tensor:
    """Hash current-first histories, including the -1 sentinel at the prompt boundary."""
    pair = (ids[:, 0] + 1) * 1000003 + ids[:, 1] + 1
    triple = pair * 1000033 + ids[:, 2] + 1
    return torch.stack((pair, triple), dim=1).remainder(buckets)


@dataclass(frozen=True)
class EngramLite(pllm.Workload):
    """Gather offline-fitted 2/3-gram memory and gate it into layer 2's attention residual.

    placement is the author's choice; pllm never moves the table to satisfy a budget.
    gate=0 reproduces vanilla bitwise while still executing the lookup.
    """

    path: str
    buckets: int = 65536
    width: int = 32
    placement: Literal["device", "host"] = "device"
    gate: float = 0.1
    layers: tuple[int, ...] = (2,)

    @property
    def memory(self) -> pllm.InstanceState:
        """Only the gathered table uses the declared placement; projections stay on device."""
        return pllm.InstanceState(
            {"table": (self.buckets, self.width)}, init=pllm.Pretrained(self.path, placement=self.placement)
        )

    @property
    def projections(self) -> pllm.InstanceState:
        """Small key/value projections loaded from the same offline fit."""
        return pllm.InstanceState(
            {"wk": lambda m: (2 * self.width, m.hidden_size), "wv": lambda m: (2 * self.width, m.hidden_size)},
            init=pllm.Pretrained(self.path),
        )

    def lookup(
        self,
        ids: torch.Tensor = pllm.input_tokens(history=2),
        table: torch.Tensor = pllm.parameter("memory", "table"),
    ) -> torch.Tensor:
        """Prefetch two rows per model token, independently of the hidden state."""
        return table[ngram_keys(ids, self.buckets)].flatten(1)

    def residual_layers(self) -> tuple[int, ...]:
        return self.layers

    @pllm.custom_residual(site="attention", layers=residual_layers)
    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        e: torch.Tensor = pllm.result("lookup"),
        wk: torch.Tensor = pllm.parameter("projections", "wk"),
        wv: torch.Tensor = pllm.parameter("projections", "wv"),
    ) -> torch.Tensor:
        """A cosine gate selects the contribution of the retrieved next-token memory."""
        k, v = e @ wk, e @ wv
        gate = torch.nn.functional.cosine_similarity(x.float(), k.float(), dim=-1).sigmoid()[:, None]
        return (x + y) + (self.gate * gate * v.float()).to(x.dtype)


@torch.no_grad()
def fit(corpus: str, path: str, *, buckets: int = 65536, width: int = 32) -> None:
    """Fit bucket means on a local corpus with a seeded projection of frozen next-token embeddings."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    checkpoint = pllm.io.checkpoint_directory(model_name)
    tokenizer: TokenizerLike = get_tokenizer(checkpoint)
    tokens = torch.tensor(tokenizer.encode(Path(corpus).read_text()), dtype=torch.long)
    if len(tokens) < 4:
        raise ValueError("fit needs at least four corpus tokens; supply a longer text file")
    positions = torch.arange(len(tokens) - 1)[:, None] - torch.arange(3)
    ids = tokens[positions.clamp_min(0)].masked_fill(positions < 0, -1)
    keys = ngram_keys(ids, buckets)
    embedding = pllm.io.checkpoint_tensors(checkpoint, ("model.embed_tokens.weight",))["model.embed_tokens.weight"][
        tokens[1:]
    ].float()
    projection = torch.randn(embedding.shape[1], width, generator=torch.Generator().manual_seed(7))
    projection = torch.nn.functional.normalize(projection, dim=0)
    targets = embedding @ projection
    table, counts = torch.zeros(buckets, width), torch.zeros(buckets, 1)
    for order in range(2):
        table.index_add_(0, keys[:, order], targets)
        counts.index_add_(0, keys[:, order], torch.ones(len(keys), 1))
    table /= counts.clamp_min(1)
    weights = projection.T.repeat(2, 1) / 2
    save_file({"table": table.bfloat16(), "wk": weights.bfloat16(), "wv": weights.bfloat16()}, path)


def main() -> None:
    """Write weights before constructing EngramLite or calling make_llm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus")
    parser.add_argument("path")
    parser.add_argument("--buckets", type=int, default=65536)
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()
    fit(args.corpus, args.path, buckets=args.buckets, width=args.width)


if __name__ == "__main__":
    main()
