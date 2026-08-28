"""Native, replica-scoped OverRIDE Region workload.

Algorithm mapping from the legacy Program implementation:

- The legacy ``reweight`` flow computed ``logits - scale * head(hidden)`` and
  the sampler provider then published ``2*logits - reweighted`` — which is
  algebraically ``logits + scale * head(hidden)``.  The native Region therefore
  signs exactly the auxiliary part, the scaled low-rank residual
  ``scale * head(hidden)``, and the sampler provider performs the final
  elementwise combine.  No logits enter the Region graph (the timeline exposes
  no logits input kind), which keeps the Region read-only.
- Training rows flow through the step: ``reweight`` keeps the decode hidden as
  a Region temporary, the sampler bridge stashes the selected logits rows at
  ``prepare_step`` and the sampled tokens at ``after_sample`` (the kit's
  ``modify_candidates`` / ``observe_sampled_tokens`` hooks), and the signed
  ``train``/``update`` regions run inside the same engine step
  (``token.commit`` → ``step.end``), preserving the legacy
  update-at-``after_sample`` semantics.

Weight-overlay outlook (standing requirement — do not close the road):

- The signed optimizer identity is an open extension point: ``AdamSpec`` joins
  ``AdamWSpec`` as a pure ``OptimizerSpec`` (kind + sorted options), so future
  fast-weight/overlay update rules sign in the same way without touching the
  State or Region contracts.
- ``State.transition`` is a free-form signed name (``"optimizer"`` here); a
  future overlay apply/rollback transition adds a new name, not a new schema.
- States are named parameter families (``parameters=("down", "up")``) bound in
  ``cold_specialize``; an overlay mechanism that mounts fast weights onto
  base-model modules would introduce a new parameter source behind the same
  State interface — task signatures (plain tensors in/out) do not change.
- The hidden input arrives through the signed residual manifest
  (``residual_paths``), not global runtime state, so overlaying base-model
  weights (e.g. an overlaid LM head feeding this workload) only adds manifest
  entries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from tllm.consumers.override.config import OverRIDEConsumerConfig
from tllm.kit import RegionAlgorithm
from tllm.ports.residual_stream import ResidualLocator
from tllm.ports.sampler import CandidateModifierState
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.workload import (
    AdamSpec,
    Before,
    Between,
    State,
    StateScope,
    extern,
    hidden,
    maybe_transition,
    provided,
    read,
    result,
    sink,
    task,
    temporary,
    workload,
)

_PROVIDED_REWEIGHT_HEAD_IDX = "reweight_head_idx"
_PROVIDED_REWEIGHT_SCALE = "reweight_scale"
_PROVIDED_TRAIN_ROW_IDS = "train_row_ids"
_PROVIDED_TRAIN_LOGITS = "train_logits"
_PROVIDED_TRAIN_HEAD_IDX = "train_head_idx"
_PROVIDED_TRAIN_TOKENS = "train_tokens"


def iteration_from_request_id(request_id: str) -> int:
    """Legacy-compatible iteration parse; kept legacy-free on purpose."""
    if "_" not in str(request_id):
        return 1
    idx_str = str(request_id).split("_", 1)[0]
    try:
        return int(idx_str) + 1
    except ValueError:
        return 1


@dataclass(frozen=True, slots=True)
class _TrainStash:
    """Sampler-boundary training rows for one engine step."""

    step_id: int
    row_ids: tuple[int, ...]
    logits: torch.Tensor
    head_idx: torch.Tensor
    tokens: torch.Tensor | None


@workload(id="override.region")
class OverRIDERegionWorkload(RegionAlgorithm[OverRIDEConsumerConfig]):
    """Replica-state OverRIDE algorithm ready for native Region materialization."""

    algorithm = "override-exact-region-v1"
    config_type = OverRIDEConsumerConfig
    enabled_field = "enabled"

    # The signed optimizer lr is nominal; the physical lr is pinned per
    # instance by the specialization key.
    heads = State(
        scope=StateScope.REPLICA,
        parameters=("down", "up"),
        optimizer=AdamSpec(lr=1e-4),
    )

    def __init__(self, config: OverRIDEConsumerConfig) -> None:
        self._train_stash: _TrainStash | None = None
        super().__init__(config)

    # -- kit hooks --------------------------------------------------------

    def validate_config(self, config: OverRIDEConsumerConfig) -> None:
        """Reject every legacy-only OverRIDE execution mode up front."""
        prefix = "OverRIDE implementation='region'"
        if config.implementation != "region":
            raise ValueError(f"{prefix} requires implementation='region'")
        if int(config.num_iterations) < 2:
            raise ValueError(f"{prefix} requires num_iterations >= 2 (at least one head)")
        if int(config.rank) < 1:
            raise ValueError(f"{prefix} requires a positive rank")
        if float(config.learning_rate) <= 0:
            raise ValueError(f"{prefix} requires a positive learning_rate")
        if int(config.hidden_size) < 0:
            raise ValueError(f"{prefix} requires hidden_size >= 0")
        if int(config.vocab_size) <= 0:
            raise ValueError(
                f"{prefix} requires an explicit positive vocab_size; no Region input "
                "carries the vocabulary for legacy-style shape inference"
            )
        if not str(config.source_layer_path).strip():
            raise ValueError(f"{prefix} requires a non-empty source_layer_path")

    def provided_inputs(self) -> Mapping[str, object]:
        return {
            _PROVIDED_REWEIGHT_HEAD_IDX: self._reweight_head_idx_for_step,
            _PROVIDED_REWEIGHT_SCALE: self._reweight_scale_for_step,
            _PROVIDED_TRAIN_ROW_IDS: self._train_row_ids_for_step,
            _PROVIDED_TRAIN_LOGITS: self._train_logits_for_step,
            _PROVIDED_TRAIN_HEAD_IDX: self._train_head_idx_for_step,
            _PROVIDED_TRAIN_TOKENS: self._train_tokens_for_step,
        }

    def residual_paths(self) -> Mapping[object, str]:
        return {
            ResidualLocator(0, "block_output", "decode"): str(
                self.config.source_layer_path
            )
        }

    def init_parameters(
        self, spec: RuntimeSpecializationSpec
    ) -> Mapping[str, torch.nn.Parameter]:
        hidden_size = int(self.config.hidden_size) or int(spec.hidden_size)
        if int(self.config.hidden_size) > 0 and hidden_size != int(spec.hidden_size):
            raise RuntimeError(
                "OverRIDE Region hidden_size does not match the runtime specialization"
            )
        down = torch.empty(
            (self._num_heads, int(self.config.rank), hidden_size),
            device=spec.device,
            dtype=spec.dtype,
        )
        # Official Head-A init: N(0, 0.02) (~/OverRIDE modeling.py:70).
        torch.nn.init.normal_(down, mean=0.0, std=0.02)
        up = torch.zeros(
            (self._num_heads, int(self.config.vocab_size), int(self.config.rank)),
            device=spec.device,
            dtype=spec.dtype,
        )
        return {"down": torch.nn.Parameter(down), "up": torch.nn.Parameter(up)}

    def optimizer_overrides(self) -> Mapping[str, object]:
        return {"lr": float(self.config.learning_rate)}

    def extra_stats(self) -> Mapping[str, int]:
        return {"reweight_rows": self.resources.sinks.latest_rows("reweight")}

    # -- sampler bridge hooks (kit owns the alignment checks) -------------------

    def missing_publication(self, view: SamplerStepView) -> CandidateModifierState | None:
        if int(view.decode_count) <= 0:
            return None
        raise RuntimeError(
            "OverRIDE Region has no signed reweight publication for the active "
            "sampler step; the reweight Region must run before sampling"
        )

    def modify_candidates(
        self, value: torch.Tensor, view: SamplerStepView
    ) -> CandidateModifierState:
        self._stash_train_rows(view)
        dense_logits = view.logits + value.to(dtype=view.logits.dtype)
        rows = int(view.logits.shape[0])
        return CandidateModifierState(
            beta=1.0,
            backend="pre_filter_dense",
            affected_row_ids=torch.arange(rows, device=view.device, dtype=torch.long),
            pred_hidden=view.source_hidden.detach(),
            lm_head_weight=torch.empty((0, 0), device=view.device, dtype=view.logits.dtype),
            lm_head_bias=None,
            precomputed_dense_logits=dense_logits.detach(),
        )

    def observe_sampled_tokens(
        self, view: SamplerStepView, sampled_token_ids: torch.Tensor
    ) -> None:
        stash = self._stash_for_step(view.engine_step_id)
        if stash is None:
            raise RuntimeError(
                "OverRIDE Region received sampled tokens for an unprepared sampler step"
            )
        tokens = sampled_token_ids.reshape(-1)
        if stash.row_ids:
            row_ids = torch.tensor(stash.row_ids, device=tokens.device, dtype=torch.long)
            selected = tokens.index_select(0, row_ids).to(dtype=torch.long)
        else:
            selected = torch.empty((0,), device=tokens.device, dtype=torch.long)
        self._train_stash = _TrainStash(
            step_id=stash.step_id,
            row_ids=stash.row_ids,
            logits=stash.logits,
            head_idx=stash.head_idx,
            tokens=selected,
        )

    # -- iteration bookkeeping (plain Python, legacy-equivalent) -----------------

    @property
    def _num_heads(self) -> int:
        return max(0, int(self.config.num_iterations) - 1)

    def _prev_head_idx(self, iteration: int) -> int | None:
        idx = int(iteration) - 2
        return idx if 0 <= idx < self._num_heads else None

    def _cur_head_idx(self, iteration: int) -> int | None:
        idx = int(iteration) - 1
        return idx if 0 <= idx < self._num_heads else None

    def _decode_request_ids(self) -> tuple[str, ...]:
        from tllm.runtime import residual_runtime

        runtime = residual_runtime.RUNTIME
        decode_count = int(runtime.decode_count)
        request_ids = tuple(
            str(value) for value in list(runtime.decode_request_ids)[:decode_count]
        )
        if len(request_ids) != decode_count:
            raise RuntimeError(
                "OverRIDE Region requires one decode request id per decode row"
            )
        return request_ids

    def _head_logits(
        self, hidden_rows: torch.Tensor, head_idx: torch.Tensor
    ) -> torch.Tensor:
        idx = head_idx.reshape(-1).to(dtype=torch.long)
        down_rows = self.param("down").index_select(0, idx)  # (rows, rank, hidden)
        up_rows = self.param("up").index_select(0, idx)  # (rows, vocab, rank)
        out = torch.bmm(
            torch.bmm(hidden_rows.unsqueeze(1), down_rows.transpose(1, 2)),
            up_rows.transpose(1, 2),
        )
        return out.squeeze(1)

    # -- provided-input callbacks (invoked at signed Region execution) ------------

    def _reweight_head_idx_for_step(self, _step_id: int) -> torch.Tensor:
        prev = [
            self._prev_head_idx(iteration_from_request_id(request_id))
            for request_id in self._decode_request_ids()
        ]
        return torch.tensor(
            [idx if idx is not None else 0 for idx in prev],
            device=self.param("down").device,
            dtype=torch.long,
        )

    def _reweight_scale_for_step(self, _step_id: int) -> torch.Tensor:
        prev = [
            self._prev_head_idx(iteration_from_request_id(request_id))
            for request_id in self._decode_request_ids()
        ]
        return torch.tensor(
            [[float(self.config.lambd)] if idx is not None else [0.0] for idx in prev],
            device=self.param("down").device,
            dtype=self.param("down").dtype,
        )

    def _stash_for_step(self, step_id: int) -> _TrainStash | None:
        stash = self._train_stash
        return stash if stash is not None and stash.step_id == int(step_id) else None

    def _empty_train_selection(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        down = self.param("down")
        return (
            torch.empty(
                (0, int(self.config.vocab_size)), device=down.device, dtype=down.dtype
            ),
            torch.empty((0,), device=down.device, dtype=torch.long),
            torch.empty((0,), device=down.device, dtype=torch.long),
        )

    def _unstaged(self, what: str) -> torch.Tensor:
        if self._decode_request_ids():
            raise RuntimeError(
                f"OverRIDE Region train {what} were not staged by the sampler "
                "provider for the active step"
            )
        return torch.empty((0,), device=self.param("down").device, dtype=torch.long)

    def _train_row_ids_for_step(self, step_id: int) -> torch.Tensor:
        stash = self._stash_for_step(step_id)
        if stash is None:
            return self._unstaged("rows")
        return torch.tensor(
            stash.row_ids, device=self.param("down").device, dtype=torch.long
        )

    def _train_logits_for_step(self, step_id: int) -> torch.Tensor:
        stash = self._stash_for_step(step_id)
        if stash is None:
            self._unstaged("logits")
            logits, _, _ = self._empty_train_selection()
            return logits
        return stash.logits

    def _train_head_idx_for_step(self, step_id: int) -> torch.Tensor:
        stash = self._stash_for_step(step_id)
        if stash is None:
            return self._unstaged("head indices")
        return stash.head_idx

    def _train_tokens_for_step(self, step_id: int) -> torch.Tensor:
        stash = self._stash_for_step(step_id)
        if stash is None:
            return self._unstaged("tokens")
        if stash.tokens is None:
            raise RuntimeError(
                "OverRIDE Region requires sampled tokens via after_sample before "
                "the signed train Region runs"
            )
        return stash.tokens

    # -- sampler-boundary stash (driven by the kit bridge hooks) -------------------

    def _stash_train_rows(self, view: SamplerStepView) -> None:
        rows = int(view.logits.shape[0])
        selected_rows: list[int] = []
        selected_heads: list[int] = []
        for row_pos, request_id in enumerate(view.request_ids[:rows]):
            cur_idx = self._cur_head_idx(iteration_from_request_id(request_id))
            if cur_idx is None:
                continue
            selected_rows.append(row_pos)
            selected_heads.append(cur_idx)
        device = view.logits.device
        row_ids = tuple(selected_rows)
        if row_ids:
            index = torch.tensor(row_ids, device=device, dtype=torch.long)
            logits = view.logits.index_select(0, index).detach().clone()
        else:
            logits = torch.empty(
                (0, int(view.logits.shape[1])), device=device, dtype=view.logits.dtype
            )
        self._train_stash = _TrainStash(
            step_id=int(view.engine_step_id),
            row_ids=row_ids,
            logits=logits,
            head_idx=torch.tensor(selected_heads, device=device, dtype=torch.long),
            tokens=None,
        )

    # -- signed tasks ------------------------------------------------------------------

    @task(
        inputs={
            "source_hidden": hidden(0, "block_output", "decode"),
            "head_idx": provided(_PROVIDED_REWEIGHT_HEAD_IDX),
            "scale": provided(_PROVIDED_REWEIGHT_SCALE),
        },
        outputs={"residual": sink("reweight"), "kept_hidden": temporary()},
        effects=(read(heads),),
        timing=Between("model.execute", "sampler.pre_draw"),
    )
    def reweight(
        self,
        source_hidden: torch.Tensor,
        head_idx: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_hidden.ndim != 2:
            raise ValueError("OverRIDE Region reweight requires rank-2 hidden rows")
        # The reweight is inference-only by design: the kit grants grad for the
        # signed read, and the body explicitly opts out.
        with torch.no_grad():
            residual = self._head_logits(source_hidden.detach(), head_idx)
            scaled = scale.reshape(-1, 1).to(dtype=residual.dtype) * residual
        return scaled, source_hidden.detach()

    @task(
        inputs={
            "kept_hidden": result("reweight", "kept_hidden"),
            "row_ids": provided(_PROVIDED_TRAIN_ROW_IDS),
            "logits": provided(_PROVIDED_TRAIN_LOGITS),
            "head_idx": provided(_PROVIDED_TRAIN_HEAD_IDX),
            "tokens": provided(_PROVIDED_TRAIN_TOKENS),
        },
        outputs={"loss": temporary()},
        effects=(read(heads),),
        timing=Between("token.commit", "step.end"),
    )
    def train(
        self,
        kept_hidden: torch.Tensor,
        row_ids: torch.Tensor,
        logits: torch.Tensor,
        head_idx: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        flat_tokens = tokens.reshape(-1)
        if int(flat_tokens.numel()) == 0:
            # Zero-row step: the legacy consumer skips the update entirely, so
            # return a detached zero; ``update`` refuses it without a version
            # bump, preserving legacy semantics exactly.
            return torch.zeros(
                (), device=kept_hidden.device, dtype=kept_hidden.dtype
            )
        if logits.ndim != 2 or int(logits.shape[0]) != int(flat_tokens.numel()):
            raise ValueError("OverRIDE Region train logits and tokens row mismatch")
        hidden_rows = kept_hidden.index_select(
            0, row_ids.reshape(-1).to(dtype=torch.long)
        ).clone()
        q_logits = logits + self._head_logits(hidden_rows, head_idx).to(
            dtype=logits.dtype
        )
        return F.cross_entropy(q_logits, flat_tokens.to(dtype=torch.long))

    @task(
        inputs={"loss": result("train", "loss")},
        effects=(maybe_transition(heads),),
        timing=Before("step.end"),
        implementation=extern("tllm.override:adam_transition@1"),
    )
    def update(self, loss: torch.Tensor) -> None:
        if not loss.requires_grad:
            # Documented zero-row skip: no optimizer step, no version bump.
            return
        self.step_update(loss)
