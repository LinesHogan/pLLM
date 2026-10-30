"""Signed pre/post-filter interventions over the original vLLM sampler state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from vllm.config.model import LogprobsMode
from vllm.v1.outputs import SamplerOutput as Mrv1SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from vllm.v1.sample.sampler import Sampler as UpstreamMrv1Sampler
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.states import RequestState

from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback, SamplerStage
from pllm.execution.cuda_stream import upload


def _reweight_unfused(survivors: torch.Tensor, prediction: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """(1 + beta) * logits - beta * prediction on the finite survivors, as one fused kernel.

    Unfused, this is seven passes over [rows, vocab] logits, more GPU time than the forward's LM head.
    """
    return torch.where(
        torch.isfinite(survivors), (1.0 + beta) * survivors - beta * prediction.to(survivors.dtype), survivors
    )


_reweight = torch.compile(_reweight_unfused, dynamic=True, fullgraph=True)  # pyright: ignore[reportUnknownMemberType]


@dataclass(frozen=True, slots=True)
class SignedSamplerGuidance:
    """One dense prediction aligned to a signed packed sampler row space."""

    guidance: PackedSamplerGuidanceBatch
    pred_dense_logits: torch.Tensor
    capture_feedback: bool = False


@dataclass(frozen=True, slots=True)
class _SampledGuidance:
    feedback: SamplerFeedback | None


class SamplerGuidance:
    """One intervention lifecycle and mathematical boundary shared by both runner ABIs."""

    _active_guidance: SignedSamplerGuidance | _SampledGuidance | None = None

    def begin_guidance(self, guidance: SignedSamplerGuidance) -> None:
        assert self._active_guidance is None
        self._active_guidance = guidance

    def finish_guidance(self) -> SamplerFeedback | None:
        match self._active_guidance:
            case _SampledGuidance(feedback=feedback):
                self._active_guidance = None
                return feedback
            case None:  # A step without sampler hidden rows draws without guidance.
                return None
            case _:
                assert False

    def abort_guidance(self) -> None:
        self._active_guidance = None

    @staticmethod
    def _add_logits(logits: torch.Tensor, guidance: SignedSamplerGuidance) -> torch.Tensor:
        assert logits.shape == (guidance.guidance.total_rows, guidance.pred_dense_logits.shape[1])
        if guidance.guidance.affected_row_ids == tuple(range(logits.shape[0])):
            return logits + guidance.pred_dense_logits.to(logits)
        rows = upload(guidance.guidance.affected_row_ids, logits.device)
        output = logits.clone()
        output.index_add_(0, rows, guidance.pred_dense_logits.to(logits))
        return output

    @staticmethod
    def _reweight_finite_survivors(processed_logits: torch.Tensor, guidance: SignedSamplerGuidance) -> torch.Tensor:
        assert (
            processed_logits.shape == (guidance.guidance.total_rows, guidance.pred_dense_logits.shape[1])
            and guidance.pred_dense_logits.device == processed_logits.device
        )
        if not guidance.guidance.affected_row_ids or guidance.guidance.beta == 0.0:
            return processed_logits
        dense = guidance.guidance.affected_row_ids == tuple(range(processed_logits.shape[0]))
        rows = upload(() if dense else guidance.guidance.affected_row_ids, processed_logits.device)
        survivors = processed_logits if dense else processed_logits.index_select(0, rows)
        weights = guidance.guidance.row_weights
        beta = (
            survivors.new_full((survivors.shape[0], 1), guidance.guidance.beta)
            if weights is None
            else guidance.guidance.beta * weights.to(survivors).unsqueeze(1)
        )
        reweighted: torch.Tensor = _reweight(survivors, guidance.pred_dense_logits, beta)  # pyright: ignore[reportAssignmentType]
        if dense:
            return reweighted
        output = processed_logits.clone()
        output.index_copy_(0, rows, reweighted)
        return output


class Vllm027Sampler(Sampler, SamplerGuidance):
    """vLLM 0.27.1 GPU sampler with one explicit signed guidance lifecycle.

    Post-filter guidance preserves the finite top-k/top-p support. Pre-filter
    addition uses the stock draw and returns owned logits/tokens to the same
    invocation. Inactive steps delegate directly to the upstream sampler.
    """

    __slots__ = ("_active_guidance",)

    def __init__(self, upstream: Sampler) -> None:
        assert type(upstream) is Sampler

        # Explicit state adoption is the only bridge from the loaded vLLM
        # instance.  These fields are precisely the state read by inherited
        # apply/logprob methods and by the active gumbel draw below.
        match upstream.logprobs_mode:
            case ("raw_logits" | "raw_logprobs" | "processed_logits" | "processed_logprobs") as logprobs_mode if (
                upstream.num_speculative_tokens == 1
            ):
                self.logprobs_mode: LogprobsMode = logprobs_mode
            case _:
                raise ValueError(
                    "canonical sampler requires non-speculative decoding (num_speculative_tokens == 1) and a supported logprobs mode"
                )
        self.compute_nans: bool = upstream.compute_nans
        self.use_fp64_gumbel: bool = upstream.use_fp64_gumbel
        self.req_states: RequestState = upstream.req_states
        self.sampling_states: SamplingStates = upstream.sampling_states
        self.penalties_state: PenaltiesState = upstream.penalties_state
        self.logit_bias_state: LogitBiasState = upstream.logit_bias_state
        self.bad_words_state: BadWordsState = upstream.bad_words_state
        self.logprob_token_ids_state: LogprobTokenIdsState = upstream.logprob_token_ids_state
        self.num_speculative_tokens: int = upstream.num_speculative_tokens
        self.use_flashinfer: bool = upstream.use_flashinfer
        self._active_guidance: SignedSamplerGuidance | _SampledGuidance | None = None

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
            case None:
                return super().sample(
                    logits,
                    expanded_idx_mapping,
                    idx_mapping_np,
                    pos,
                    input_ids,
                    expanded_local_pos,
                    return_logprobs=return_logprobs,
                )
            case SignedSamplerGuidance() as guidance:
                feedback = None
                if guidance.guidance.stage is SamplerStage.PACKED_PRE_FILTER_ADD:
                    with torch.inference_mode(False):
                        base_logits = logits.detach().clone() if guidance.capture_feedback else None
                    reweighted = self._add_logits(logits, guidance)
                    sampled = super().sample(
                        reweighted,
                        expanded_idx_mapping,
                        idx_mapping_np,
                        pos,
                        input_ids,
                        expanded_local_pos,
                        return_logprobs=return_logprobs,
                    )
                    if base_logits is not None:
                        with torch.inference_mode(False):
                            feedback = SamplerFeedback(
                                guidance.guidance.sampler_rows, base_logits, sampled[0].detach().reshape(-1).clone()
                            )
                else:
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
                self._active_guidance = _SampledGuidance(feedback)
                return sampled
            case _SampledGuidance():
                assert False

    def _sample_signed_guidance(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool,
        guidance: SignedSamplerGuidance,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


class Mrv1Sampler(UpstreamMrv1Sampler, SamplerGuidance):
    def __init__(self, upstream: UpstreamMrv1Sampler) -> None:
        assert type(upstream) is UpstreamMrv1Sampler
        torch.nn.Module.__init__(self)
        self.topk_topp_sampler = upstream.topk_topp_sampler
        self.logprobs_mode, self.use_fp64_gumbel = upstream.logprobs_mode, upstream.use_fp64_gumbel
        self.pin_memory = upstream.pin_memory
        self._active_guidance = None

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> Mrv1SamplerOutput:
        match self._active_guidance:
            case None:
                return super().forward(logits, sampling_metadata, predict_bonus_token, logprobs_mode_override)
            case SignedSamplerGuidance() as guidance if not predict_bonus_token:
                with torch.inference_mode(False):
                    base = logits.detach().clone() if guidance.capture_feedback else None
                if guidance.guidance.stage is SamplerStage.PACKED_PRE_FILTER_ADD:
                    logits = self._add_logits(logits, guidance)
                result = super().forward(logits, sampling_metadata, False, logprobs_mode_override)
                with torch.inference_mode(False):
                    feedback = (
                        SamplerFeedback(
                            guidance.guidance.sampler_rows, base, result.sampled_token_ids.detach().reshape(-1).clone()
                        )
                        if base is not None
                        else None
                    )
                self._active_guidance = _SampledGuidance(feedback)
                return result
            case _:
                raise RuntimeError("canonical sampler permits one non-speculative draw per guidance")

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        match self._active_guidance:
            case SignedSamplerGuidance() as guidance if (
                guidance.guidance.stage is not SamplerStage.PACKED_PRE_FILTER_ADD
            ):
                pass
            case _:
                return super().sample(logits, sampling_metadata, logprobs_mode_override)
        metadata = sampling_metadata
        if not metadata.all_greedy:
            assert metadata.temperature is not None
            logits = self.apply_temperature(logits, metadata.temperature, metadata.all_random)
            for processor in metadata.logitsprocs.argmax_invariant:
                logits = processor.apply(logits)
        logits = self._reweight_finite_survivors(apply_top_k_top_p(logits, metadata.top_k, metadata.top_p), guidance)
        sampled = logits.argmax(dim=-1)
        if not metadata.all_greedy:
            assert metadata.temperature is not None
            random = random_sample(logits.softmax(dim=-1), metadata.generators, self.use_fp64_gumbel)
            sampled = random if metadata.all_random else torch.where(metadata.temperature < 1e-5, sampled, random)
        mode = logprobs_mode_override or self.logprobs_mode
        processed = (
            logits
            if mode == "processed_logits"
            else logits.log_softmax(dim=-1)
            if mode == "processed_logprobs"
            else None
        )
        return sampled, processed


__all__ = [
    "Vllm027Sampler",
    "SignedSamplerGuidance",
]
