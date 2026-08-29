#!/usr/bin/env python3
"""lm_head resolution shared by ESamp sampler providers (legacy and program).

Moved to :mod:`tllm.kit.lm_head`; this module re-exports for the legacy
consumer stack.
"""

from __future__ import annotations

from tllm.runtime.vllm_patch.lm_head import ResolvedLMHead, resolve_lm_head

__all__ = ["ResolvedLMHead", "resolve_lm_head"]
