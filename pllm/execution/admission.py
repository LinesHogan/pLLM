"""Retry signal shared by staging, page admission and execution owners."""


class WindowBackpressure(RuntimeError):
    """The caller may retry admission after the current owner releases its window."""
