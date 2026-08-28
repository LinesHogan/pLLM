"""Loader for pyreft ``intkey_*.bin`` intervention checkpoints.

pyreft saves one file per intervened layer, named
``intkey_layer.{L}.comp.block_output.unit.pos.nunit.1#0.bin``, holding
``{"weight" [r,d], "bias" [r], "rotate_layer" [d,r]}`` — exactly the
stacked layout :class:`tllm.runtime.activation_patch.LoreftBank` consumes.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

_INTKEY = re.compile(r"^intkey_layer\.(\d+)\.comp\.block_output\..*\.bin$")


def load_loreft_interventions(
    directory: str | Path, layers: tuple[int, ...]
) -> dict[int, dict[str, torch.Tensor]]:
    """Return ``{layer: {"source_w", "source_b", "rotate"}}`` for ``layers``."""
    root = Path(directory)
    found: dict[int, dict[str, torch.Tensor]] = {}
    for path in sorted(root.glob("intkey_*.bin")):
        match = _INTKEY.match(path.name)
        if match is None:
            continue
        layer = int(match.group(1))
        if layer not in layers:
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        found[layer] = {
            "source_w": state["weight"],
            "source_b": state["bias"],
            "rotate": state["rotate_layer"],
        }
    missing = sorted(set(layers) - set(found))
    if missing:
        raise FileNotFoundError(
            f"no intkey checkpoint for layers {missing} in {root}"
        )
    return found


__all__ = ["load_loreft_interventions"]
