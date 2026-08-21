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

__all__ = [
    "EdgeKind",
    "ProgramDecl",
    "ProgramDiagnostic",
    "ProgramEdge",
    "ProgramNode",
    "ProgramState",
    "ProgramValue",
    "StateSpec",
    "declare_program",
]
