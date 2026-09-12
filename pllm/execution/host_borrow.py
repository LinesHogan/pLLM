"""Host effect namespaces borrowed from original backend objects.

An adapter must expose all tensor effects separately and keep its namespace
alive for the original invocation. Distinct names alone never prove separation.
"""

from dataclasses import dataclass
from typing import Protocol

from pllm.contracts.region_program import AccessMode, BorrowedResource
from pllm.execution.trace import ExecutionTrace


class HostResourceView(Protocol):
    @property
    def resource(self) -> BorrowedResource: ...
    @property
    def namespace(self) -> int: ...
    def validate(self) -> None: ...
    def check(self, mode: AccessMode, ranges: tuple[tuple[int, int], ...]) -> None: ...


@dataclass(frozen=True, slots=True)
class TraceResourceView:
    resource: BorrowedResource
    trace: ExecutionTrace

    @property
    def namespace(self) -> int:
        return id(self.trace)

    def validate(self) -> None:
        if self.resource.domain:
            raise ValueError("one original trace requires an atomic host resource")

    def check(self, mode: AccessMode, ranges: tuple[tuple[int, int], ...]) -> None:
        self.validate()
        if ranges or type(mode) is not AccessMode:
            raise ValueError("trace effect requires its original atomic access")


def validate_host_views(views: tuple[HostResourceView, ...]) -> None:
    namespaces: set[int] = set()
    for view in views:
        view.validate()
        if view.namespace <= 0 or view.namespace in namespaces:
            raise ValueError("host resource names alias the same original effect namespace")
        namespaces.add(view.namespace)
