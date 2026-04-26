#!/usr/bin/env python3
"""Formal prompt-context injection port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortWrite

PromptContextScope = Literal["pre_generate"]


@dataclass(frozen=True)
class PromptContextLocator(Locator):
    scope: PromptContextScope = "pre_generate"

    def __post_init__(self) -> None:
        if self.scope != "pre_generate":
            raise ValueError(f"unsupported prompt context scope: {self.scope}")


class PromptContext:
    KIND = PortKind.PROMPT_CONTEXT
    READABLE = False
    WRITABLE = True
    SUPPORTED_PHASES = ("pre_generate",)
    BACKING_VLLM_STRUCT = "prompt text before tokenization or request scheduling."

    @staticmethod
    def write(*, scope: PromptContextScope = "pre_generate", mode: str = "append") -> PortWrite:
        return PortWrite(kind=PromptContext.KIND, locator=PromptContextLocator(scope=scope), mode=str(mode))
