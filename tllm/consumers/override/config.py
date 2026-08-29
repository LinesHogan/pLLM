#!/usr/bin/env python3
"""Configuration for the OverRIDE-inspired iterative reweighting consumer.

Defaults follow the official OverRIDE repository (`~/OverRIDE`):
``num_iteration=10`` and ``learning_rate=1e-3`` (modeling.py:31-33,
scripts/override.sh), and the heads read the final-norm hidden states the
official runner taps at ``hidden_states[logits_indices]``
(gpu_model_runner.py:2837).  OverRIDE is exact-only: the official algorithm
has no approximation tolerance, and its intervention is synchronous,
same-stream, pre-draw.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OverRIDEConsumerConfig:
    consumer_id: str = "override"
    enabled: bool = True
    # "consumer" keeps the legacy Program oracle; "native" selects the
    # native workload via build_override_workload (legacy remains
    # available during the migration transition).
    implementation: str = "consumer"
    hidden_size: int = 0
    vocab_size: int = 0
    rank: int = 16
    num_iterations: int = 10
    lambd: float = 0.8
    learning_rate: float = 1e-3
    # Residual tap feeding the reweight heads.  The official implementation
    # reads the post-final-norm hidden states selected at logits_indices;
    # tapping an early layer instead yields entirely different head inputs
    # and does not reproduce official numbers.
    source_layer_path: str = "model.model.norm"
