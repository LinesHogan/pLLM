"""Send a workload to another process: the EngineCore worker (make_llm with multiprocessing or tensor parallelism,
rebuilt in pllm.worker) and check_on_gpu's checking process.

cloudpickle carries classes and functions from __main__ or a notebook by value (custom operators and @triton.jit
functions are registered again from their Python functions) and importable ones by import path, so the receiving
process imports the same sources the sender uses. An object that must not travel as bytes, such as
a template model, defines `__reduce__` to rebuild itself there. The worker compares its declaration with the
driver's plan, so source drift fails at install.
"""

import io
import pickle
from base64 import b64decode, b64encode

import cloudpickle
import triton

from pllm.execution.torch_private import (
    CustomOpDef,
    custom_op,
    define_custom_op,
    define_triton_kernel,
    triton_kernel,
)
from pllm.workload import Workload


class _Pickler(cloudpickle.Pickler):
    """cloudpickle, plus custom operators and @triton.jit functions from __main__ or a notebook: they hold locks, so
    they travel as their Python functions and are registered again in the receiving process."""

    def reducer_override(self, obj: object) -> object:
        if isinstance(obj, CustomOpDef):
            return define_custom_op, (custom_op(obj),)
        if isinstance(obj, triton.JITFunction):
            return define_triton_kernel, (triton_kernel(obj),)  # pyright: ignore[reportUnknownArgumentType]
        return super().reducer_override(obj)


def encode_workload(workload: Workload) -> str:
    """Serialize the workload for another process."""
    try:
        buffer = io.BytesIO()
        _Pickler(buffer).dump(workload)
        return b64encode(buffer.getvalue()).decode("ascii")
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        raise ValueError(
            f"other processes receive the workload by pickling ({exc}); hold plain values, tensors and functions "
            "instead of locks, open files or live handles, or use tensor_parallel_size=1 with multiprocessing=False"
        ) from exc


def decode_workload(payload: str) -> Workload:
    """Rebuild the sender's workload."""
    return pickle.loads(b64decode(payload))
