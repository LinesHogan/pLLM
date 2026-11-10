"""Data crossing pLLM's boundary outside generate's arguments.

Membership rule: values a program hands to, or receives from, a running workload other than through generate's
prompts and parameters: fed state (Feed), imported state values (StateImport), traces (TraceEvent, TraceSnapshot,
TraceSelection) and checkpoint loading (load_tensor_fields, and checkpoint_directory with checkpoint_tensors for
base-model tensors such as the token embedding).
"""

from pllm.contracts.host_batch import StateImport as StateImport
from pllm.execution.checkpoint import (
    checkpoint_directory as checkpoint_directory,
    checkpoint_tensors as checkpoint_tensors,
)
from pllm.execution.state_feed import Feed as Feed
from pllm.execution.torch_private import load_tensor_fields as load_tensor_fields
from pllm.execution.trace import (
    TraceEvent as TraceEvent,
    TraceSelection as TraceSelection,
    TraceSnapshot as TraceSnapshot,
)

__all__ = [
    "Feed",
    "StateImport",
    "TraceEvent",
    "TraceSelection",
    "TraceSnapshot",
    "checkpoint_directory",
    "checkpoint_tensors",
    "load_tensor_fields",
]
