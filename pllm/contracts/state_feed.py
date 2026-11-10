"""State feed declarations shared by public authoring and the runtime receiver."""

from dataclasses import dataclass
from enum import StrEnum

# Versions in flight per feed: the engine reads one, one waits in the queue, and the producer writes the next.
FEED_SLOTS = 3


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
