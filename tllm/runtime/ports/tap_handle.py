#!/usr/bin/env python3
"""Lifecycle handle for an installed layer-forward tap.

Kept dependency-free (torch only) so both ``runtime.state`` and the hook
installer can reference it without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch


@dataclass
class TapHandle:
    """Records enough to symmetrically remove one installed tap."""

    layer_path: str
    layer: torch.nn.Module
    orig_forward: Callable

    def restore(self) -> None:
        self.layer.forward = self.orig_forward  # type: ignore[method-assign]


def restore_all(handles: Dict[str, TapHandle]) -> None:
    for handle in handles.values():
        handle.restore()
    handles.clear()
