"""Concrete eager vLLM 0.27.1 sampler for signed canonical guidance.

This adapter is deliberately instance-local.  It receives every stateful
sampling component from one already-loaded vLLM GPU ``Sampler`` explicitly;
it neither patches the upstream instance nor reaches into a runtime callback
registry.  A caller owns the begin/finish/abort lifecycle for each signed
guidance batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tllm.contracts.sampler import PackedSamplerGuidanceBatch

from vllm.config.model import LogprobsMode
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.states import RequestState


@dataclass(frozen=True, slots=True)
class SignedPostFilterGuidance:
    """One dense prediction aligned to a signed packed sampler row space."""

    guidance: PackedSamplerGuidanceBatch
    beta: float
    pred_dense_logits: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.beta < 0.0
            or self.pred_dense_logits.ndim != 2
            or int(self.pred_dense_logits.shape[0])
            != len(self.guidance.affected_row_ids)
            or self.pred_dense_logits.device != self.guidance.pred_hidden.device
        ):
            raise ValueError("signed post-filter guidance is not tensor-aligned")


@dataclass(frozen=True, slots=True)
class _NoActiveGuidance:
    pass


@dataclass(frozen=True, slots=True)
class _PreparedGuidance:
    value: SignedPostFilterGuidance


@dataclass(frozen=True, slots=True)
class _SampledGuidance:
    value: SignedPostFilterGuidance


_ActiveGuidance = _NoActiveGuidance | _PreparedGuidance | _SampledGuidance


class CanonicalVllm027PostFilterSampler(Sampler):
    """vLLM 0.27.1 GPU sampler with one explicit signed guidance lifecycle.

    The active route always uses vLLM's explicit top-k/top-p implementation
    followed by its ordinary ``gumbel_sample``.  FlashInfer is deliberately
    not selected for an attached step because the attachment is signed as a
    post-filter intervention.  An inactive instance delegates directly to
    the upstream implementation through ``super().sample``.
    """

    __slots__ = ("_active_guidance",)

    def __init__(self, upstream: Sampler, *, eager: bool) -> None:
        if type(upstream) is not Sampler:
            raise TypeError("canonical sampler requires an exact vLLM 0.27.1 Sampler")
        if type(eager) is not bool or not eager:
            raise ValueError("canonical post-filter sampler requires eager execution")
        if upstream.num_speculative_tokens != 1:
            raise ValueError(
                "canonical post-filter sampler does not support speculative decoding"
            )

        # Explicit state adoption is the only bridge from the loaded vLLM
        # instance.  These fields are precisely the state read by inherited
        # apply/logprob methods and by the active gumbel draw below.
        match upstream.logprobs_mode:
            case (
                "raw_logits"
                | "raw_logprobs"
                | "processed_logits"
                | "processed_logprobs"
            ) as logprobs_mode:
                self.logprobs_mode: LogprobsMode = logprobs_mode
            case _:
                raise ValueError("vLLM sampler has an unsupported logprobs mode")
        self.compute_nans: bool = upstream.compute_nans
        self.use_fp64_gumbel: bool = upstream.use_fp64_gumbel
        self.req_states: RequestState = upstream.req_states
        self.sampling_states: SamplingStates = upstream.sampling_states
        self.penalties_state: PenaltiesState = upstream.penalties_state
        self.logit_bias_state: LogitBiasState = upstream.logit_bias_state
        self.bad_words_state: BadWordsState = upstream.bad_words_state
        self.logprob_token_ids_state: LogprobTokenIdsState = (
            upstream.logprob_token_ids_state
        )
        self.num_speculative_tokens: int = upstream.num_speculative_tokens
        self.use_flashinfer: bool = upstream.use_flashinfer
        self._active_guidance: _ActiveGuidance = _NoActiveGuidance()

    def begin_guidance(self, guidance: SignedPostFilterGuidance) -> None:
        """Open one signed guidance transaction before the vLLM draw."""

        match self._active_guidance:
            case _NoActiveGuidance():
                self._active_guidance = _PreparedGuidance(guidance)
            case _PreparedGuidance() | _SampledGuidance():
                raise RuntimeError("canonical sampler guidance is already active")

    def finish_guidance(self) -> None:
        """Close one successfully sampled signed guidance transaction."""

        match self._active_guidance:
            case _SampledGuidance():
                self._active_guidance = _NoActiveGuidance()
            case _NoActiveGuidance() | _PreparedGuidance():
                raise RuntimeError("canonical sampler has no sampled guidance to finish")

    def abort_guidance(self) -> None:
        """Discard an active guidance transaction after an enclosing failure."""

        match self._active_guidance:
            case _NoActiveGuidance():
                return
            case _PreparedGuidance() | _SampledGuidance():
                self._active_guidance = _NoActiveGuidance()

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        match self._active_guidance:
            case _NoActiveGuidance():
                return super().sample(
                    logits,
                    expanded_idx_mapping,
                    idx_mapping_np,
                    pos,
                    input_ids,
                    expanded_local_pos,
                    return_logprobs=return_logprobs,
                )
            case _PreparedGuidance(value=guidance):
                sampled = self._sample_signed_guidance(
                    logits,
                    expanded_idx_mapping,
                    idx_mapping_np,
                    pos,
                    input_ids,
                    expanded_local_pos,
                    return_logprobs,
                    guidance,
                )
                self._active_guidance = _SampledGuidance(guidance)
                return sampled
            case _SampledGuidance():
                raise RuntimeError(
                    "canonical sampler guidance may sample only once before finish"
                )

    def _sample_signed_guidance(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool,
        guidance: SignedPostFilterGuidance,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            logits.ndim != 2
            or int(logits.shape[0]) != guidance.guidance.total_rows
            or guidance.pred_dense_logits.device != logits.device
            or int(guidance.pred_dense_logits.shape[1]) != int(logits.shape[1])
        ):
            raise ValueError("signed guidance differs from the active sampler logits")

        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
        )
        top_k, top_p = self.sampling_states.get_top_k_top_p(
            expanded_idx_mapping,
            idx_mapping_np,
        )
        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
        reweighted_logits = self._reweight_finite_survivors(
            processed_logits,
            guidance,
        )
        sampled = gumbel_sample(
            reweighted_logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
            use_fp64=self.use_fp64_gumbel,
        )
        return sampled, reweighted_logits

    @staticmethod
    def _reweight_finite_survivors(
        processed_logits: torch.Tensor,
        guidance: SignedPostFilterGuidance,
    ) -> torch.Tensor:
        affected_rows = torch.tensor(
            guidance.guidance.affected_row_ids,
            device=processed_logits.device,
            dtype=torch.long,
        )
        if int(affected_rows.numel()) == 0 or guidance.beta == 0.0:
            return processed_logits
        survivor_logits = processed_logits.index_select(0, affected_rows)
        predicted_logits = guidance.pred_dense_logits.to(
            device=processed_logits.device,
            dtype=processed_logits.dtype,
        )
        finite = torch.isfinite(survivor_logits)
        reweighted = torch.where(
            finite,
            (1.0 + guidance.beta) * survivor_logits
            - guidance.beta * predicted_logits,
            survivor_logits,
        )
        output = processed_logits.clone()
        output.index_copy_(0, affected_rows, reweighted)
        return output


__all__ = [
    "CanonicalVllm027PostFilterSampler",
    "SignedPostFilterGuidance",
]
