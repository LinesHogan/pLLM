#!/usr/bin/env python3
"""In-Place-TTT-inspired side-adapter consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from tllm.consumers.base import BaseConsumer
from tllm.consumers.inplace_ttt.config import InPlaceTTTConsumerConfig
from tllm.consumers.inplace_ttt.engine import (
    InPlaceTTTEngine,
    InPlaceTTTExactToyState,
    InPlaceTTTExactToyStep,
    InPlaceTTTStats,
    pad_rows_to_chunk,
    run_exact_toy_down_proj_step,
)
from tllm.consumers._common.bundle_inputs import residual_pair_input
from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import ConsumerFlow
from tllm.ports.module_weight_view import ModuleWeightView
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualStream


class InPlaceTTTConsumer(BaseConsumer[InPlaceTTTConsumerConfig]):
    """tLLM-native In-Place-TTT-inspired hidden-pair adaptation consumer."""

    def __init__(self, config: InPlaceTTTConsumerConfig, *, engine: InPlaceTTTEngine | None = None) -> None:
        self.config = config
        self.engine = engine or InPlaceTTTEngine(config)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        if not bool(self.config.enabled):
            return ()
        flows: list[ConsumerFlow] = []
        for phase in self.config.phases:
            delivery = "device_lease" if phase == "decode" else "bundle"
            ownership = "runtime_lease" if phase == "decode" else "borrowed"
            window = "out_of_band" if phase == "decode" else "background"
            flows.append(
                ConsumerFlow(
                    reads=(
                        ResidualStream.read(
                            layer=int(self.config.source_layer),
                            site="block_output",
                            phase=phase,
                            role="source",
                        ),
                        ResidualStream.read(
                            layer=int(self.config.target_layer),
                            site="block_output",
                            phase=phase,
                            role="target",
                        ),
                        RequestMeta.read(),
                    ),
                    writes=(
                        # The fast-weight overlay installs its own forward
                        # wrapper through shared runtime infra; declaring it
                        # here gives the plan write-conflict visibility.
                        ModuleWeightView.write(
                            layer=int(self.config.target_layer),
                            module="mlp",
                            phase=phase,
                            mode="wrapper",
                        ),
                    ),
                    window=window,
                    delivery=delivery,
                    ownership=ownership,
                    bundle_key=("engine_step_id", "phase"),
                )
            )
        return flows

    def _extract_hidden_pair(self, bundle: PortBundle) -> tuple[torch.Tensor, torch.Tensor] | None:
        inputs = residual_pair_input(bundle)
        if inputs is None:
            return None
        return inputs.source, inputs.target

    def consume_bundle(self, bundle: PortBundle, ctx: RuntimeContext) -> None:
        _ = ctx
        pair = self._extract_hidden_pair(bundle)
        if pair is None:
            return
        self.engine.update(pair[0], pair[1])

    def read_stats(self) -> InPlaceTTTStats:
        return self.engine.read_stats()
