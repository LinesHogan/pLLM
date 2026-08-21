#!/usr/bin/env python3
"""tLLM Program compiler surface (see docs/program_adr.md)."""

from tllm.program.decl import (
    EdgeKind,
    ProgramDecl,
    ProgramDiagnostic,
    ProgramEdge,
    ProgramNode,
    ProgramState,
    ProgramValue,
    StateSpec,
    declare_program,
)
from tllm.program.runtime import ProgramConsumer, ProgramSink, compile_program
from tllm.program.stages import FlowStagePlan, ProgramStage, StageValueDesc, slice_flow_stages

__all__ = [
    "EdgeKind",
    "FlowStagePlan",
    "ProgramConsumer",
    "ProgramDecl",
    "ProgramDiagnostic",
    "ProgramEdge",
    "ProgramNode",
    "ProgramSink",
    "ProgramStage",
    "ProgramState",
    "ProgramValue",
    "StageValueDesc",
    "StateSpec",
    "compile_program",
    "declare_program",
    "slice_flow_stages",
]
