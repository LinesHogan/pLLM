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
  a Region temporary, the sampler provider stashes the selected logits rows at
  ``prepare_step`` and the sampled tokens at ``after_sample``, and the signed
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

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import torch
from torch.nn import functional as F

from tllm.consumers.override.config import OverRIDEConsumerConfig
from tllm.contracts.consumer_stats import TrainingStats
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.sampler_bridge.types import CandidateModifierState, SamplerStepView
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    NativeWorkloadResources,
    WorkloadWorkerSpec,
)
from tllm.workload import (
    AdamSpec,
    Before,
    Between,
    ExternImplementation,
    State,
    StateScope,
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

_STATE_NAME = "heads"
_REWEIGHT_SINK = "reweight"

_PROVIDED_REWEIGHT_HEAD_IDX = "reweight_head_idx"
_PROVIDED_REWEIGHT_SCALE = "reweight_scale"
_PROVIDED_TRAIN_ROW_IDS = "train_row_ids"
_PROVIDED_TRAIN_LOGITS = "train_logits"
_PROVIDED_TRAIN_HEAD_IDX = "train_head_idx"
_PROVIDED_TRAIN_TOKENS = "train_tokens"

_SOURCE_LOCATOR = ResidualLocator(0, "block_output", "decode")


def iteration_from_request_id(request_id: str) -> int:
    """Legacy-compatible iteration parse; kept legacy-free on purpose."""
    if "_" not in str(request_id):
        return 1
    idx_str = str(request_id).split("_", 1)[0]
    try:
        return int(idx_str) + 1
    except ValueError:
        return 1


def _specialization_key(config: OverRIDEConsumerConfig) -> Mapping[str, str]:
    """Return the immutable compile-time OverRIDE configuration identity."""

    return {
        "algorithm": "override-exact-region-v1",
        "rank": str(int(config.rank)),
        "num_iterations": str(int(config.num_iterations)),
        "lambd": float(config.lambd).hex(),
        "learning_rate": float(config.learning_rate).hex(),
        "hidden_size": str(int(config.hidden_size)),
        "vocab_size": str(int(config.vocab_size)),
        "training_enabled": str(bool(config.enabled)),
        "source_layer_path": str(config.source_layer_path),
    }


@dataclass(frozen=True, slots=True)
class OverRIDERegionReweight:
    """One immutable same-step scaled-residual publication."""

    step_id: int
    residual: torch.Tensor
    request_ids: tuple[str, ...]

    @property
    def rows(self) -> int:
        return int(self.residual.shape[0])


@dataclass(frozen=True, slots=True)
class _TrainStash:
    """Sampler-boundary training rows for one engine step."""

    step_id: int
    row_ids: tuple[int, ...]
    logits: torch.Tensor
    head_idx: torch.Tensor
    tokens: torch.Tensor | None


class OverRIDERegionResources:
    """Explicit local bindings required by the native OverRIDE Region."""

    def __init__(self, workload: "OverRIDERegionWorkload") -> None:
        self._workload = workload
        self._reweight: OverRIDERegionReweight | None = None
        self._loss_count = 0
        self._update_count = 0
        self._loss_sum: torch.Tensor | None = None

    def publish_reweight(
        self,
        *,
        step_id: int,
        residual: torch.Tensor,
        request_ids: Sequence[str],
    ) -> OverRIDERegionReweight:
        if not isinstance(residual, torch.Tensor) or residual.ndim != 2:
            raise ValueError("OverRIDE Region reweight residual must be a rank-2 Tensor")
        if int(residual.shape[0]) != len(request_ids):
            raise RuntimeError("OverRIDE Region reweight rows and request ids differ")
        record = OverRIDERegionReweight(
            step_id=int(step_id),
            residual=residual.detach().clone(),
            request_ids=tuple(str(request_id) for request_id in request_ids),
        )
        self._reweight = record
        return record

    def output_sink(self) -> OverRIDERegionReweight | None:
        """Return the most recent explicit reweight sink publication."""
        return self._reweight

    def reweight_for_step(self, step_id: int) -> OverRIDERegionReweight | None:
        record = self._reweight
        return record if record is not None and record.step_id == int(step_id) else None

    def state_snapshot(self) -> Mapping[str, int]:
        return MappingProxyType({"replica": self._workload.state_version})

    def runtime_state_snapshot(self, _step_id: int) -> Mapping[Hashable, int]:
        snapshot: dict[Hashable, int] = {None: self._workload.state_version}
        return MappingProxyType(snapshot)

    def stats(self) -> Mapping[str, int]:
        record = self._reweight
        return MappingProxyType(
            {
                "reweight_rows": 0 if record is None else record.rows,
                "loss_count": self._loss_count,
                "update_count": self._update_count,
            }
        )

    def read_and_reset_stats(self, sync: bool = True) -> TrainingStats:
        loss_sum = self._loss_sum
        if loss_sum is None or self._loss_count <= 0:
            return TrainingStats(loss_avg=0.0, loss_count=0)
        if sync and loss_sum.device.type == "cuda":
            torch.cuda.current_stream(loss_sum.device).synchronize()
        result_stats = TrainingStats(
            loss_avg=float(loss_sum.item()) / self._loss_count,
            loss_count=self._loss_count,
        )
        self._loss_sum = None
        self._loss_count = 0
        return result_stats

    def _record_update(self, loss: torch.Tensor) -> None:
        detached = loss.detach().float()
        self._loss_sum = (
            detached.clone() if self._loss_sum is None else self._loss_sum + detached
        )
        self._loss_count += 1
        self._update_count += 1


class OverRIDERegionSamplerProvider:
    """Sampler bridge which consumes, but never recomputes, Region reweights."""

    def __init__(self, workload: "OverRIDERegionWorkload") -> None:
        self._workload = workload

    def is_active(self) -> bool:
        return bool(self._workload.config.enabled)

    def prepare_step(self, view: SamplerStepView) -> CandidateModifierState | None:
        if not self.is_active():
            return None
        if view.logits.ndim != 2:
            raise RuntimeError("OverRIDE Region sampler provider requires rank-2 logits")
        rows = int(view.logits.shape[0])
        if rows <= 0:
            return None
        if len(view.request_ids) != rows:
            raise RuntimeError(
                "OverRIDE Region sampler provider requires one request id per decode row"
            )
        record = self._workload.resources.reweight_for_step(view.engine_step_id)
        if record is None:
            raise RuntimeError(
                "OverRIDE Region has no signed reweight publication for the active "
                "sampler step; the reweight Region must run before sampling"
            )
        if record.rows != rows:
            raise RuntimeError("OverRIDE Region reweight rows do not match sampler rows")
        if record.request_ids != tuple(str(value) for value in view.request_ids):
            raise RuntimeError(
                "OverRIDE Region reweight request ids do not match sampler step"
            )
        if record.residual.device != view.logits.device:
            raise RuntimeError("OverRIDE Region reweight device does not match sampler step")
        self._workload._stash_train_rows(view)
        dense_logits = view.logits + record.residual.to(dtype=view.logits.dtype)
        return CandidateModifierState(
            beta=1.0,
            backend="pre_filter_dense",
            affected_row_ids=torch.arange(rows, device=view.logits.device, dtype=torch.long),
            pred_hidden=view.source_hidden.detach(),
            lm_head_weight=torch.empty(
                (0, 0), device=view.logits.device, dtype=view.logits.dtype
            ),
            lm_head_bias=None,
            precomputed_dense_logits=dense_logits.detach(),
        )

    def after_sample(self, view: SamplerStepView, sampled_token_ids: torch.Tensor) -> None:
        self._workload._stash_sampled_tokens(view.engine_step_id, sampled_token_ids)


def validate_region_config(config: OverRIDEConsumerConfig) -> None:
    """Reject every legacy-only OverRIDE execution mode before Region creation."""
    if not isinstance(config, OverRIDEConsumerConfig):
        raise TypeError("config must be OverRIDEConsumerConfig")
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


# The signed optimizer lr is nominal; the physical lr is pinned per instance by
# ``specialization_key`` (ESamp precedent: the State is a class attribute,
# while config identity lives in the specialization key).
_HEADS = State(
    scope=StateScope.REPLICA,
    parameters=("down", "up"),
    optimizer=AdamSpec(lr=1e-4),
)


@workload(id="override.region")
class OverRIDERegionWorkload(NativeWorkload):
    """Replica-state OverRIDE algorithm ready for native Region materialization."""

    heads = _HEADS

    def __init__(self, config: OverRIDEConsumerConfig) -> None:
        validate_region_config(config)
        self.config = config
        enabled_snapshot = bool(config.enabled)
        self._down: torch.nn.Parameter | None = None
        self._up: torch.nn.Parameter | None = None
        self._optimizer: torch.optim.Adam | None = None
        self._specialization: RuntimeSpecializationSpec | None = None
        self._state_version = 0
        self._train_stash: _TrainStash | None = None
        self._resources = OverRIDERegionResources(self)
        self._sampler_provider = OverRIDERegionSamplerProvider(self)
        self._native_resources = self._make_native_resources(enabled_snapshot)

    # -- resource exposure ------------------------------------------------------

    @property
    def down(self) -> torch.nn.Parameter | None:
        return self._down

    @property
    def up(self) -> torch.nn.Parameter | None:
        return self._up

    @property
    def optimizer(self) -> torch.optim.Adam | None:
        return self._optimizer

    @property
    def state_version(self) -> int:
        return self._state_version

    @property
    def resources(self) -> OverRIDERegionResources:
        return self._resources

    @property
    def sampler_provider(self) -> OverRIDERegionSamplerProvider:
        return self._sampler_provider

    def native_workload_resources(self) -> NativeWorkloadResources:
        return self._native_resources

    def _make_native_resources(self, enabled_snapshot: bool) -> NativeWorkloadResources:
        return NativeWorkloadResources(
            workload=self,
            provided_inputs={
                _PROVIDED_REWEIGHT_HEAD_IDX: self._reweight_head_idx_for_step,
                _PROVIDED_REWEIGHT_SCALE: self._reweight_scale_for_step,
                _PROVIDED_TRAIN_ROW_IDS: self._train_row_ids_for_step,
                _PROVIDED_TRAIN_LOGITS: self._train_logits_for_step,
                _PROVIDED_TRAIN_HEAD_IDX: self._train_head_idx_for_step,
                _PROVIDED_TRAIN_TOKENS: self._train_tokens_for_step,
            },
            output_sinks={_REWEIGHT_SINK: self._publish_reweight_sink},
            state_snapshots={_STATE_NAME: self._resources.runtime_state_snapshot},
            cold_specializer=self.cold_specialize,
            sampler_provider=self._sampler_provider,
            diagnostics_reader=self._resources.stats,
            stats_reader=self._resources.read_and_reset_stats,
            enabled_setter=self.set_enabled,
            # This predicate belongs to this immutable resource snapshot. A
            # later enable transition creates a new signed Region epoch.
            enabled_predicate=lambda enabled=enabled_snapshot: enabled,
            synchronizer=lambda: None,
            specialization_key=_specialization_key(self.config),
            residual_paths={_SOURCE_LOCATOR: str(self.config.source_layer_path)},
            worker_spec=WorkloadWorkerSpec(
                factory=(
                    "tllm.consumers.override.region_workload:"
                    "reconstruct_override_region_workload"
                ),
                config=_config_to_wire(self.config),
            ),
        )

    def set_enabled(self, enabled: bool) -> None:
        self.config.enabled = bool(enabled)
        enabled_snapshot = bool(self.config.enabled)
        # Worker specs and resource keys are frozen snapshots.  A changed
        # enable state therefore gets a new snapshot for the next epoch.
        self._native_resources = self._make_native_resources(enabled_snapshot)

    # -- cold specialization ------------------------------------------------------

    def cold_specialize(self, spec: RuntimeSpecializationSpec) -> None:
        """Allocate trainable heads and optimizer outside ambient inference mode."""
        if not isinstance(spec, RuntimeSpecializationSpec):
            raise TypeError("spec must be RuntimeSpecializationSpec")
        previous = self._specialization
        if previous is not None:
            if previous != spec:
                raise RuntimeError(
                    "OverRIDE Region is already specialized for another static "
                    "shape, dtype, or device"
                )
            return
        hidden_size = int(self.config.hidden_size) or int(spec.hidden_size)
        if int(self.config.hidden_size) > 0 and hidden_size != int(spec.hidden_size):
            raise RuntimeError(
                "OverRIDE Region hidden_size does not match the runtime specialization"
            )
        num_heads = self._num_heads
        with torch.inference_mode(False):
            with torch.enable_grad():
                down = torch.empty(
                    (num_heads, int(self.config.rank), hidden_size),
                    device=spec.device,
                    dtype=spec.dtype,
                )
                torch.nn.init.normal_(down, mean=0.0, std=1.0 / math.sqrt(float(hidden_size)))
                up = torch.zeros(
                    (num_heads, int(self.config.vocab_size), int(self.config.rank)),
                    device=spec.device,
                    dtype=spec.dtype,
                )
                down_param = torch.nn.Parameter(down)
                up_param = torch.nn.Parameter(up)
                optimizer = torch.optim.Adam(
                    (down_param, up_param),
                    lr=float(self.config.learning_rate),
                    foreach=False,
                )
        self._down = down_param
        self._up = up_param
        self._optimizer = optimizer
        self._specialization = spec

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

    def _require_heads(self) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        if self._down is None or self._up is None:
            raise RuntimeError("OverRIDE Region must be cold_specialize()d before execution")
        return self._down, self._up

    def _require_device(self) -> torch.device:
        spec = self._specialization
        if spec is None:
            raise RuntimeError("OverRIDE Region must be cold_specialize()d before execution")
        return spec.device

    def _head_logits(
        self, hidden_rows: torch.Tensor, head_idx: torch.Tensor
    ) -> torch.Tensor:
        down, up = self._require_heads()
        idx = head_idx.reshape(-1).to(dtype=torch.long)
        down_rows = down.index_select(0, idx)  # (rows, rank, hidden)
        up_rows = up.index_select(0, idx)  # (rows, vocab, rank)
        out = torch.bmm(
            torch.bmm(hidden_rows.unsqueeze(1), down_rows.transpose(1, 2)),
            up_rows.transpose(1, 2),
        )
        return out.squeeze(1)

    # -- provided-input callbacks (invoked at signed Region execution) ------------

    def _reweight_head_idx_for_step(self, _step_id: int) -> torch.Tensor:
        device = self._require_device()
        prev = [
            self._prev_head_idx(iteration_from_request_id(request_id))
            for request_id in self._decode_request_ids()
        ]
        return torch.tensor(
            [idx if idx is not None else 0 for idx in prev],
            device=device,
            dtype=torch.long,
        )

    def _reweight_scale_for_step(self, _step_id: int) -> torch.Tensor:
        device = self._require_device()
        prev = [
            self._prev_head_idx(iteration_from_request_id(request_id))
            for request_id in self._decode_request_ids()
        ]
        return torch.tensor(
            [[float(self.config.lambd)] if idx is not None else [0.0] for idx in prev],
            device=device,
            dtype=self._require_heads()[0].dtype,
        )

    def _stash_for_step(self, step_id: int) -> _TrainStash | None:
        stash = self._train_stash
        return stash if stash is not None and stash.step_id == int(step_id) else None

    def _empty_train_selection(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self._require_device()
        down, _ = self._require_heads()
        return (
            torch.empty((0, int(self.config.vocab_size)), device=device, dtype=down.dtype),
            torch.empty((0,), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=torch.long),
        )

    def _train_row_ids_for_step(self, step_id: int) -> torch.Tensor:
        stash = self._stash_for_step(step_id)
        device = self._require_device()
        if stash is None:
            if self._decode_request_ids():
                raise RuntimeError(
                    "OverRIDE Region train rows were not staged by the sampler "
                    "provider for the active step"
                )
            return torch.empty((0,), device=device, dtype=torch.long)
        return torch.tensor(stash.row_ids, device=device, dtype=torch.long)

    def _train_logits_for_step(self, step_id: int) -> torch.Tensor:
        stash = self._stash_for_step(step_id)
        if stash is None:
            logits, _, _ = self._empty_train_selection()
            if self._decode_request_ids():
                raise RuntimeError(
                    "OverRIDE Region train logits were not staged by the sampler "
                    "provider for the active step"
                )
            return logits
        return stash.logits

    def _train_head_idx_for_step(self, step_id: int) -> torch.Tensor:
        stash = self._stash_for_step(step_id)
        if stash is None:
            _, head_idx, _ = self._empty_train_selection()
            if self._decode_request_ids():
                raise RuntimeError(
                    "OverRIDE Region train head indices were not staged by the "
                    "sampler provider for the active step"
                )
            return head_idx
        return stash.head_idx

    def _train_tokens_for_step(self, step_id: int) -> torch.Tensor:
        stash = self._stash_for_step(step_id)
        if stash is None:
            _, _, tokens = self._empty_train_selection()
            if self._decode_request_ids():
                raise RuntimeError(
                    "OverRIDE Region train tokens were not staged by the sampler "
                    "provider for the active step"
                )
            return tokens
        if stash.tokens is None:
            raise RuntimeError(
                "OverRIDE Region requires sampled tokens via after_sample before "
                "the signed train Region runs"
            )
        return stash.tokens

    # -- sampler-boundary stash (host side, driven by the provider) ---------------

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

    def _stash_sampled_tokens(
        self, step_id: int, sampled_token_ids: torch.Tensor
    ) -> None:
        stash = self._stash_for_step(step_id)
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

    # -- sink publication ------------------------------------------------------------

    def _publish_reweight_sink(self, value: object, step_id: int) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("OverRIDE Region reweight sink requires a Tensor")
        self._resources.publish_reweight(
            step_id=int(step_id),
            residual=value,
            request_ids=self._decode_request_ids(),
        )

    # -- signed tasks ------------------------------------------------------------------

    @task(
        inputs={
            "source_hidden": hidden(0, "block_output", "decode"),
            "head_idx": provided(_PROVIDED_REWEIGHT_HEAD_IDX),
            "scale": provided(_PROVIDED_REWEIGHT_SCALE),
        },
        outputs={"residual": sink(_REWEIGHT_SINK), "kept_hidden": temporary()},
        effects=(read(_HEADS),),
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
        with torch.inference_mode(False):
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
        effects=(read(_HEADS),),
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
        with torch.inference_mode(False):
            with torch.enable_grad():
                hidden_rows = kept_hidden.index_select(
                    0, row_ids.reshape(-1).to(dtype=torch.long)
                ).clone()
                q_logits = logits + self._head_logits(hidden_rows, head_idx).to(
                    dtype=logits.dtype
                )
                loss = F.cross_entropy(q_logits, flat_tokens.to(dtype=torch.long))
        return loss

    @task(
        inputs={"loss": result("train", "loss")},
        effects=(maybe_transition(_HEADS),),
        timing=Before("step.end"),
        implementation=ExternImplementation(
            namespace="tllm.override",
            name="adam_transition",
            version="1",
            input_types=("Tensor",),
            output_types=("None",),
        ),
    )
    def update(self, loss: torch.Tensor) -> None:
        self._require_heads()
        optimizer = self._optimizer
        if optimizer is None:
            raise RuntimeError("OverRIDE Region must be cold_specialize()d before execution")
        if not isinstance(loss, torch.Tensor):
            raise TypeError("OverRIDE Region update requires a loss Tensor")
        if not loss.requires_grad:
            # Documented zero-row skip: no optimizer step, no version bump.
            return
        with torch.inference_mode(False):
            with torch.enable_grad():
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self._state_version += 1
        self._resources._record_update(loss)

    # -- manual drive (standalone parity/testing, ESamp ``run_*`` precedent) --------

    def run_reweight(
        self,
        *,
        step_id: int,
        source_hidden: torch.Tensor,
        head_idx: torch.Tensor,
        scale: torch.Tensor,
        request_ids: Sequence[str],
    ) -> torch.Tensor:
        residual, _kept = self.reweight(source_hidden, head_idx, scale)
        self._resources.publish_reweight(
            step_id=int(step_id), residual=residual, request_ids=request_ids
        )
        return residual

    def run_train_update(
        self,
        *,
        kept_hidden: torch.Tensor,
        row_ids: torch.Tensor,
        logits: torch.Tensor,
        head_idx: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.train(kept_hidden, row_ids, logits, head_idx, tokens)
        self.update(loss)
        return loss


def build_override_region_workload(
    config: OverRIDEConsumerConfig,
) -> OverRIDERegionWorkload:
    return OverRIDERegionWorkload(config)


def _config_to_wire(config: OverRIDEConsumerConfig) -> Mapping[str, object]:
    from tllm.common.dataclass_json import dataclass_to_jsonable

    raw = dataclass_to_jsonable(config)
    if not isinstance(raw, dict):
        raise TypeError("OverRIDE Region config did not encode as a JSON object")
    return raw


def reconstruct_override_region_workload(
    raw: Mapping[str, object],
) -> NativeWorkloadResources:
    from tllm.common.dataclass_json import dataclass_from_dict

    config = dataclass_from_dict(OverRIDEConsumerConfig, dict(raw))
    return build_override_region_workload(config).native_workload_resources()


__all__ = [
    "OverRIDERegionResources",
    "OverRIDERegionReweight",
    "OverRIDERegionSamplerProvider",
    "OverRIDERegionWorkload",
    "build_override_region_workload",
    "iteration_from_request_id",
    "reconstruct_override_region_workload",
    "validate_region_config",
]
