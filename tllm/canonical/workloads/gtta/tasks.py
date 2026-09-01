"""CPU host-state tasks for the canonical GTTA graph."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.canonical.workloads.gtta.state import GTTADynamicsState
from tllm.canonical.workloads.gtta.types import (
    GTTAConfig,
    GTTAHostTransaction,
    GTTAQuery,
    GTTAQueryResult,
    GTTAUpdateReceipt,
)
from tllm.contracts.semantic_graph import ImplementationId


GTTA_UPDATE_IMPLEMENTATION = ImplementationId("canonical.gtta:cpu_update@1")
GTTA_QUERY_IMPLEMENTATION = ImplementationId("canonical.gtta:cpu_query@1")
GTTA_PUBLISH_IMPLEMENTATION = ImplementationId("canonical.gtta:host_publish@1")


@dataclass(frozen=True, slots=True)
class GTTAUpdatePending:
    pass


@dataclass(frozen=True, slots=True)
class GTTAUpdated:
    receipt: GTTAUpdateReceipt


@dataclass(frozen=True, slots=True)
class GTTAQueryPending:
    pass


@dataclass(frozen=True, slots=True)
class GTTAQueried:
    result: GTTAQueryResult


@dataclass(frozen=True, slots=True)
class GTTAPublishPending:
    pass


@dataclass(frozen=True, slots=True)
class GTTAPublished:
    update: GTTAUpdateReceipt
    query: GTTAQueryResult


GTTAUpdateState = GTTAUpdatePending | GTTAUpdated
GTTAQueryState = GTTAQueryPending | GTTAQueried
GTTAPublishState = GTTAPublishPending | GTTAPublished


@dataclass(slots=True)
class GTTAStepContext:
    transaction: GTTAHostTransaction
    query: GTTAQuery
    dynamics: GTTADynamicsState
    update: GTTAUpdateState
    queried: GTTAQueryState
    publication: GTTAPublishState


class GTTAUpdateTask:
    __slots__ = ("_config",)

    def __init__(self, config: GTTAConfig) -> None:
        self._config = config

    @property
    def implementation_id(self) -> ImplementationId:
        return GTTA_UPDATE_IMPLEMENTATION

    def execute(self, context: GTTAStepContext) -> None:
        context.update = GTTAUpdated(
            context.dynamics.update(
                context.transaction.trajectories,
                self._config,
            )
        )


class GTTAQueryTask:
    __slots__ = ()

    @property
    def implementation_id(self) -> ImplementationId:
        return GTTA_QUERY_IMPLEMENTATION

    def execute(self, context: GTTAStepContext) -> None:
        match context.update:
            case GTTAUpdated():
                pass
            case GTTAUpdatePending():
                raise RuntimeError("GTTA query requires update completion")
        context.queried = GTTAQueried(context.dynamics.query(context.query))


class GTTAPublishTask:
    __slots__ = ()

    @property
    def implementation_id(self) -> ImplementationId:
        return GTTA_PUBLISH_IMPLEMENTATION

    def execute(self, context: GTTAStepContext) -> None:
        match context.update, context.queried:
            case GTTAUpdated(receipt=update), GTTAQueried(result=query):
                context.publication = GTTAPublished(update, query)
            case _:
                raise RuntimeError("GTTA publish requires update and query outputs")


__all__ = [
    "GTTAPublishPending",
    "GTTAPublishTask",
    "GTTAPublished",
    "GTTAQueryPending",
    "GTTAQueryTask",
    "GTTAQueried",
    "GTTAStepContext",
    "GTTAUpdatePending",
    "GTTAUpdateTask",
    "GTTAUpdated",
    "GTTA_PUBLISH_IMPLEMENTATION",
    "GTTA_QUERY_IMPLEMENTATION",
    "GTTA_UPDATE_IMPLEMENTATION",
]
