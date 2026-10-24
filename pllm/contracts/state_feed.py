"""State feed declarations shared by public authoring and the runtime receiver."""

from dataclasses import dataclass
from enum import StrEnum


class FeedOrder(StrEnum):
    """Choose which complete publication a model step consumes."""

    ORDERED = "ordered"
    LATEST = "latest"


@dataclass(frozen=True)
class FeedBinding:
    """Bind a declared resource to its step selection policy."""

    resource: str
    order: FeedOrder
    fields: tuple[str, ...]
