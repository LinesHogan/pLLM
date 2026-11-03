"""JSON evidence fields shared by the AI evaluation runner and scorer."""

from typing import Any, NotRequired, TypedDict


class Score(TypedDict):
    """Independent gates, diagnostics and reviewed failure attribution for one submission."""

    score: int | None
    imports_declares: bool
    check: bool
    oracle_cpu: bool
    explain_tier: str
    category: str
    failures: dict[str, str]
    oracle_gpu: NotRequired[str]
    eager_reason: NotRequired[str]
    check_details: NotRequired[dict[str, Any]]


class TaskResult(Score):
    """A score plus the authoring session's reproducibility and cost evidence."""

    task: str
    author_exit: int
    wall_seconds: float
    author_turns: int
    input_sha256: str
    transcript: str


class Metadata(TypedDict):
    """Pinned runtime, documentation, model and sandbox settings for a complete run."""

    base_commit: str
    runtime_commit: str
    agent: str
    model: str
    gpu: bool
    author_version: str
    timeout_seconds: int
    jobs: int
    python: str
    isolation: str
