"""Legacy import path; production ownership is in the sampler runtime."""

from tllm.runtime.vllm_patch.lm_head import ResolvedLMHead, resolve_lm_head

__all__ = ["ResolvedLMHead", "resolve_lm_head"]
