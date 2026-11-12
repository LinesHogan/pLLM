"""Send a workload to another process: the EngineCore worker (make_llm with multiprocessing or tensor parallelism,
rebuilt in pllm.worker) and check_on_gpu's checking process.

cloudpickle carries classes and functions from __main__ or a notebook by value and importable ones by import path,
so the receiving process imports the same sources the sender uses. An object that must not travel as bytes, such as
a template model, defines `__reduce__` to rebuild itself there. The worker compares its declaration with the
driver's plan, so source drift fails at install.
"""

import pickle
from base64 import b64decode, b64encode

import cloudpickle

from pllm.workload import Workload


def encode_workload(workload: Workload) -> str:
    """Serialize the workload for another process."""
    try:
        return b64encode(cloudpickle.dumps(workload)).decode("ascii")  # pyright: ignore[reportUnknownMemberType]
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        raise ValueError(
            f"other processes receive the workload by pickling ({exc}); hold plain values, tensors and functions "
            "instead of locks, open files or live handles, or use tensor_parallel_size=1 with multiprocessing=False"
        ) from exc


def decode_workload(payload: str) -> Workload:
    """Rebuild the sender's workload."""
    return pickle.loads(b64decode(payload))
