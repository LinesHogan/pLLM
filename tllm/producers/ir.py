#!/usr/bin/env python3
"""Typed, Python-owned input records for the Rust execution-plan compiler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlanPortSpecDesc:
    port: str
    kind: str
    writable: bool
    write_modes: tuple[str, ...]
    phases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanProducerDesc:
    id: str
    priority: int
    outputs: tuple[PlanPortSpecDesc, ...]
    requires: tuple[str, ...]
    hook_points: tuple[str, ...]
    graph_safety: str


@dataclass(frozen=True, slots=True)
class PlanConsumerReadDesc:
    kind: str
    role: str
    phase: str = ""


@dataclass(frozen=True, slots=True)
class PlanConsumerWriteDesc:
    kind: str
    mode: str
    phase: str = ""


@dataclass(frozen=True, slots=True)
class PlanConsumerFlowDesc:
    reads: tuple[PlanConsumerReadDesc, ...]
    writes: tuple[PlanConsumerWriteDesc, ...]
    bundle_key: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanConsumerDesc:
    id: str
    flows: tuple[PlanConsumerFlowDesc, ...]
