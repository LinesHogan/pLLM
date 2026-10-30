"""Send a workload to the EngineCore worker process; used by make_llm (driver) and pllm.worker.

Pickle carries classes and module-level functions by import path, so the worker imports the same sources the
driver uses. An object that must not travel as bytes, such as a template model, defines `__reduce__` to rebuild
itself in the worker. The driver later compares the worker's compiled plan with its own, so source drift fails there.
"""

import pickle
from base64 import b64decode, b64encode

from pllm.authoring import BackendWorkload, DataT, ResultT


def encode_workload(workload: BackendWorkload[DataT, ResultT]) -> str:
    """Pickle the workload for the worker; everything it holds must be importable there."""
    if type(workload).__module__ == "__main__":
        raise ValueError(
            f"multiprocessing=True cannot send {type(workload).__name__} defined in __main__ or a notebook; "
            "move it into an importable .py module, or keep multiprocessing=False"
        )
    try:
        return b64encode(pickle.dumps(workload)).decode("ascii")
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        raise ValueError(
            f"multiprocessing=True sends the workload to its worker by pickling ({exc}); hold module-level "
            "functions instead of lambdas or closures, or keep multiprocessing=False"
        ) from exc


def decode_workload(payload: str) -> BackendWorkload[object, object]:
    """Rebuild the driver's workload inside the worker."""
    return pickle.loads(b64decode(payload))
