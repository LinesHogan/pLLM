#!/usr/bin/env python3
"""tLLM Program compiler surface (see docs/program_adr.md)."""

from tllm.program.decl import ProgramDecl, ProgramNode, StateSpec, declare_program

__all__ = [
    "ProgramDecl",
    "ProgramNode",
    "StateSpec",
    "declare_program",
]
