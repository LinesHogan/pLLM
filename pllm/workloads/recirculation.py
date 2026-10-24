"""Exact prefill recurrence on the ordinary effect interface."""

from dataclasses import dataclass

import pllm
from pllm.model import Recurrence as RecirculationConfig


@dataclass(frozen=True, slots=True)
class RecirculationWorkload(pllm.Workload):
    name = "recirculation"
    config: RecirculationConfig = RecirculationConfig()
