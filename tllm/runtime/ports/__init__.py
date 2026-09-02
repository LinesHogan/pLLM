#!/usr/bin/env python3
"""Internal runtime port machinery."""

from tllm.runtime.ports.assembler import BundleAssembler
from tllm.runtime.ports.frame import Ownership, PortFrame
from tllm.runtime.ports import residual_bindings

__all__ = [
    "BundleAssembler",
    "Ownership",
    "PortFrame",
    "residual_bindings",
]
