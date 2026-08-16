#!/usr/bin/env python3
"""vLLM 0.25+ MRv1 prepare-inputs adapter.

``_prepare_inputs(scheduler_output, num_scheduled_tokens)`` returns
``(logits_indices, spec_decode_metadata)``; attention metadata is built
inside ``execute_model`` and no longer crosses this boundary, so
``attn_metadata`` is ``None`` and ``num_actual_tokens`` is derived from the
explicit ``num_scheduled_tokens`` argument.
"""

from __future__ import annotations

from typing import Any

from tllm.runtime.vllm_patch.adapters.base import (
    BasePrepareInputsAdapter,
    PrepareInputsView,
)


class V027PrepareInputsAdapter(BasePrepareInputsAdapter):
    family_name = "v027"

    def unpack_prepare_inputs_output(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        out: tuple,
        num_scheduled_tokens: Any = None,
    ) -> PrepareInputsView:
        if not isinstance(out, tuple) or len(out) < 2:
            raise TypeError(
                "v0.27 prepare_inputs output must be a "
                "(logits_indices, spec_decode_metadata) tuple, "
                f"got {type(out).__name__}"
            )
        if num_scheduled_tokens is None:
            num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
        num_actual_tokens = None
        try:
            if num_scheduled_tokens is not None and not isinstance(num_scheduled_tokens, dict):
                num_actual_tokens = int(num_scheduled_tokens.sum())
        except Exception:
            num_actual_tokens = None
        return PrepareInputsView(
            attn_metadata=None,
            logits_indices=out[0],
            spec_decode_common=None,
            num_scheduled_tokens_np=num_scheduled_tokens,
            num_actual_tokens=num_actual_tokens,
        )
