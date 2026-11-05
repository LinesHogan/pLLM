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
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler as VllmSampler

from pllm.backends.vllm import compat
from pllm.backends.vllm.guided_sampling import guided_sample
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback, SamplerStage
from pllm.execution.cuda_stream import upload


def add_logits(logits: torch.Tensor, guidance: SignedSamplerGuidance) -> torch.Tensor:
    """Add one declared contribution at its packed sampler rows."""
    assert logits.shape == (guidance.guidance.total_rows, guidance.pred_dense_logits.shape[1])
    if guidance.guidance.affected_row_ids == tuple(range(logits.shape[0])):
        return logits + guidance.pred_dense_logits.to(logits)
    rows = upload(guidance.guidance.affected_row_ids, logits.device)
    output = logits.clone()
    output.index_add_(0, rows, guidance.pred_dense_logits.to(logits))
    return output


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


@dataclass(frozen=True, slots=True)
class BoundedSamplerGuidance:
    """A hidden prediction whose projection is bounded by a survivor capacity."""

    guidance: PackedSamplerGuidanceBatch
    weight: torch.Tensor
    capacity: int


@dataclass(frozen=True, slots=True)
class SignedSamplerStages:
    """Declared-order additions followed by the optional filtered-distribution guide."""

    stages: tuple[SignedSamplerGuidance | BoundedSamplerGuidance, ...]
    capture_feedback: bool = False

    @property
    def guide(self) -> SignedSamplerGuidance | BoundedSamplerGuidance | None:
        """The at-most-one post-filter stage admitted by the authoring boundary."""
        return next(
            (stage for stage in self.stages if stage.guidance.stage is SamplerStage.PACKED_POST_FILTER_EXACT), None
        )

    def add_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Sum additions in their declared order before vLLM applies its filters."""
        for stage in self.stages:
            if isinstance(stage, SignedSamplerGuidance) and stage.guidance.stage is SamplerStage.PACKED_PRE_FILTER_ADD:
                logits = add_logits(logits, stage)
        return logits


@dataclass(frozen=True, slots=True)
class _SampledGuidance:
    feedback: SamplerFeedback | None


class SamplerGuidance:
    """One intervention lifecycle and mathematical boundary shared by both runner ABIs."""

    _active_guidance: SignedSamplerStages | _SampledGuidance | None = None
    bounded_guidance: bool = False

    def begin_guidance(self, guidance: SignedSamplerStages) -> None:
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
    def _reweight_finite_survivors(processed_logits: torch.Tensor, guidance: SignedSamplerGuidance) -> torch.Tensor:
        assert (
            processed_logits.shape == (guidance.guidance.total_rows, guidance.pred_dense_logits.shape[1])
            and guidance.pred_dense_logits.device == processed_logits.device
        )
        if not guidance.guidance.affected_row_ids or guidance.guidance.beta == 0.0:
            return processed_logits
        rows = (
            None
            if guidance.guidance.affected_row_ids == tuple(range(processed_logits.shape[0]))
            else upload(guidance.guidance.affected_row_ids, processed_logits.device)
        )
        survivors = processed_logits if rows is None else processed_logits.index_select(0, rows)
        weights = guidance.guidance.row_weights
        beta = (
            survivors.new_full((survivors.shape[0], 1), guidance.guidance.beta)
            if weights is None
            else guidance.guidance.beta * weights.to(survivors).unsqueeze(1)
        )
        reweighted: torch.Tensor = _reweight(survivors, guidance.pred_dense_logits, beta)  # pyright: ignore[reportAssignmentType]
        if rows is None:
            return reweighted
        output = processed_logits.clone()
        output.index_copy_(0, rows, reweighted)
        return output


class GuidedSampler(compat.Sampler, SamplerGuidance):
    """vLLM's MRv2 sampler with one explicit signed guidance lifecycle.

    Pre-filter addition draws from the guided logits with the stock sampler, as the MRv1 sampler does.
    Post-filter guidance reweights the finite top-k/top-p survivors before the stock Gumbel draw.
    Inactive steps delegate directly to the upstream sampler.
    """

    __slots__ = ("_active_guidance",)
    bounded_guidance = True

    def __init__(self, upstream: VllmSampler) -> None:
        assert type(upstream) is VllmSampler
        if upstream.num_speculative_tokens != 1:
            raise ValueError("pllm workloads need non-speculative decoding; disable speculative decoding")
        # The loaded vLLM instance's whole state: the inherited apply, draw and logprob methods read it.
        vars(self).update(vars(upstream))
        self._active_guidance: SignedSamplerStages | _SampledGuidance | None = None

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        match self._active_guidance:
            case None:
                return super().__call__(logits, input_batch)
            case SignedSamplerStages() as stages:
                with torch.inference_mode(False):
                    base_logits = logits.detach().clone() if stages.capture_feedback else None
                output = super().__call__(stages.add_logits(logits), input_batch)
                with torch.inference_mode(False):
                    feedback = (
                        SamplerFeedback(
                            stages.stages[0].guidance.sampler_rows,
                            base_logits,
                            output.sampled_token_ids.detach().reshape(-1).clone(),
                        )
                        if base_logits is not None
                        else None
                    )
                self._active_guidance = _SampledGuidance(feedback)
                return output
            case _SampledGuidance():
                assert False

    def _sample_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        use_fused_sampler: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        match self._active_guidance:
            case SignedSamplerStages(guide=BoundedSamplerGuidance() as guidance):
                survivors = apply_top_k_top_p(processed_logits, top_k, top_p)
                rows = [-1] * survivors.shape[0]
                for index, row in enumerate(guidance.guidance.affected_row_ids):
                    rows[row] = index
                sampled = guided_sample(
                    survivors,
                    guidance.guidance.values.to(guidance.weight.dtype),
                    guidance.weight,
                    upload(rows, survivors.device),
                    guidance.guidance.row_weights,
                    guidance.guidance.beta,
                    guidance.capacity,
                    expanded_idx_mapping,
                    self.sampling_states.temperature.gpu,
                    self.sampling_states.seeds.gpu,
                    pos,
                    top_p,
                    fp64=self.use_fp64_gumbel,
                )
                return sampled, survivors
            case SignedSamplerStages(guide=SignedSamplerGuidance() as guidance):
                survivors = apply_top_k_top_p(processed_logits, top_k, top_p)
                reweighted = self._reweight_finite_survivors(survivors, guidance)
                return super()._sample_random(reweighted, expanded_idx_mapping, idx_mapping_np, pos, None, None, False)
            case _:
                return super()._sample_random(
                    processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_fused_sampler
                )


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
            case SignedSamplerStages() as stages if not predict_bonus_token:
                with torch.inference_mode(False):
                    base = logits.detach().clone() if stages.capture_feedback else None
                result = super().forward(stages.add_logits(logits), sampling_metadata, False, logprobs_mode_override)
                with torch.inference_mode(False):
                    feedback = (
                        SamplerFeedback(
                            stages.stages[0].guidance.sampler_rows,
                            base,
                            result.sampled_token_ids.detach().reshape(-1).clone(),
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
            case SignedSamplerStages(guide=SignedSamplerGuidance() as guidance):
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
    "GuidedSampler",
    "SignedSamplerGuidance",
    "SignedSamplerStages",
]
