"""By-reference workload transport: class coordinates plus frozen field values.

EngineCore never transports function bodies or live Python objects. A workload
crosses the process boundary as its class coordinate (module + qualname), a
digest of the module source and its frozen dataclass fields in closed encodings.
The worker imports the same module, rebuilds the instance and re-binds it; the
driver then requires the worker's compilation signature to equal its own signed
plan byte for byte, so any code drift fails loudly instead of silently running
different mathematics.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import torch


class WorkloadFieldKind(StrEnum):
    VALUE = "value"                        # payload is the closed encoding of the field value
    REFERENCE = "reference"                # module + qualname of an importable module-level object
    WORKER_TRANSPORT = "worker_transport"  # payload is the field's own closed transport payload


@dataclass(frozen=True, slots=True)
class WorkloadFieldTransport:
    name: str
    kind: WorkloadFieldKind
    payload: str
    module: str
    qualname: str


@dataclass(frozen=True, slots=True)
class WorkloadReference:
    module: str
    qualname: str
    module_digest: str
    fields: tuple[WorkloadFieldTransport, ...]


@runtime_checkable
class WorkerTransportable(Protocol):
    """A field value that carries a closed payload and re-materializes inside the worker."""

    def transport_payload(self) -> object:
        """The closed value the worker rebuilds from; raises when this instance cannot travel."""
        ...

    @classmethod
    def transport_payload_type(cls) -> type[object]:
        """The statically bound type used to decode the payload; never read from the wire."""
        ...

    @classmethod
    def transport_rebuild(cls, payload: object, *, device: torch.device, seed: int) -> object:
        """Re-materialize on the worker's own device; driver state is never shared."""
        ...
