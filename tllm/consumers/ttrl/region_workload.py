"""Native, replica-scoped TTRL Region workload (kit style).

The algorithm: a low-rank reward head (``up @ down`` over hidden rows)
trained with MSE against majority-vote rewards from completed generations.
The verifier/vote/retry/barrier logic is pure host Python in
``rewards.py`` — reused verbatim, so there is zero algorithm drift between
the legacy consumer and this native declaration.

Semantic differences from the legacy consumer (intentional, recorded):

- Publication timing: the legacy consumer publishes RewardOutcome /
  RetryControl / UpdateBarrier intents immediately post-generation.  This
  declaration publishes them from the signed ``publish`` Region at
  ``next.prepare``, after the ``update`` Region has landed, so a barrier
  "ready" and the ``adapter_version_id`` stamp observe the post-update
  state version.  The ordering is stronger, not weaker.
- Training rows arrive through the explicit host staging API
  (:meth:`TTRLRegionWorkload.stage_completed_generations`), mirroring the
  legacy caller-supplied-rows shape; the legacy tensor half had no
  production caller either, so this is a lateral move.
- ``predict_scores`` is a host-side ``no_grad`` helper (test-only in the
  legacy consumer as well).

Host-channel design (read before extending): the engine never executes the
host's workload object.  Real runs install a *reconstructed* instance built
from the signed worker spec (JSON-only transport — callables never cross),
so instance attributes set on the host object are invisible to the
executing instance.  Staged rows and the publish target therefore live in a
process-local channel registry keyed by ``consumer_id`` — the same
process-global seam as ``residual_runtime.RUNTIME``.  Under mp=1 each
worker process has its own (empty) registry: workers reconstruct, run
zero-row steps through the signed audit, and never train.  A control-plane
staging transport is the documented follow-up that closes that gap.

Official-vs-tLLM gap (standing): official TTRL (~/TTRL) updates the LLM
policy with GRPO/PPO over majority-vote rewards; tLLM trains a side
reward head with MSE and never touches policy weights.  Known weak
substitutions vs official: string-equality voting instead of sympy
equivalence, and unparseable samples are skipped instead of kept with
reward 0.  Official-grade TTRL requires policy-weight updates and belongs
to the weight-overlay follow-up project.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from tllm.consumers.ttrl.config import TTRLConsumerConfig
from tllm.consumers.ttrl.rewards import (
    TTRLClosedLoopIntent,
    build_ttrl_closed_loop_intent,
)
from tllm.consumers.ttrl.verifier import TTRLVerifier
from tllm.kit import RegionAlgorithm, RegionResourcesBase
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.workload import (
    AdamSpec,
    Before,
    Between,
    ExternImplementation,
    State,
    StateScope,
    extern_io,
    maybe_transition,
    provided,
    read,
    result,
    step_id,
    task,
    temporary,
    workload,
)

_PROVIDED_TRAIN_HIDDEN = "ttrl_train_hidden"
_PROVIDED_TRAIN_REWARDS = "ttrl_train_rewards"


@dataclass(frozen=True, slots=True)
class _StagedTraining:
    """One staged closed-loop intent plus its aligned training rows."""

    intent: TTRLClosedLoopIntent
    hidden: torch.Tensor
    rewards: torch.Tensor


@dataclass(slots=True)
class _HostChannel:
    """Process-local staging/publish mailbox for one ``consumer_id``.

    The executing (reconstructed) instance and the host's original object
    share this channel, so host-staged rows reach the signed train Region
    and the signed publish Region can find its bound control runtime
    regardless of which instance the engine drives.
    """

    staged: _StagedTraining | None = None
    publish_target: object | None = None
    last_published_intent: TTRLClosedLoopIntent | None = None
    total_staged_reward_rows: int = 0


_HOST_CHANNELS: dict[str, _HostChannel] = {}


def _host_channel(consumer_id: str) -> _HostChannel:
    return _HOST_CHANNELS.setdefault(consumer_id, _HostChannel())


class TTRLRegionResources(RegionResourcesBase):
    """Kit bookkeeping plus the staged-row counter for diagnostics."""

    def extra_stats(self) -> Mapping[str, int]:
        staged = self._workload_staged_rows()
        return {"staged_reward_rows": staged}

    def _workload_staged_rows(self) -> int:
        workload = self._workload
        if not isinstance(workload, TTRLRegionWorkload):
            raise TypeError(
                "TTRLRegionResources must be attached to a TTRLRegionWorkload"
            )
        return _host_channel(workload.consumer_id).total_staged_reward_rows


@workload(id="ttrl.region")
class TTRLRegionWorkload(RegionAlgorithm[TTRLConsumerConfig]):
    """Replica-state TTRL reward head as three signed Regions."""

    algorithm = "ttrl-exact-region-v1"
    config_type = TTRLConsumerConfig
    specialization_fields = {
        "rank": "rank",
        "learning_rate": "learning_rate",
        "hidden_size": "hidden_size",
        "training_enabled": "enabled",
    }
    resources_type = TTRLRegionResources
    enabled_field = "enabled"

    reward_model = State(
        scope=StateScope.REPLICA,
        parameters=("down", "up"),
        optimizer=AdamSpec(lr=1e-3),
    )

    def __init__(
        self,
        config: TTRLConsumerConfig,
        *,
        verifier: TTRLVerifier | None = None,
    ) -> None:
        self._verifier = verifier
        super().__init__(config)

    # -- kit hooks --------------------------------------------------------

    def validate_config(self, config: TTRLConsumerConfig) -> None:
        prefix = "TTRL implementation='region'"
        if config.implementation != "region":
            raise ValueError(f"{prefix} requires implementation='region'")
        if int(config.rank) < 1:
            raise ValueError(f"{prefix} requires a positive rank")
        if float(config.learning_rate) <= 0:
            raise ValueError(f"{prefix} requires a positive learning_rate")
        if int(config.hidden_size) < 0:
            raise ValueError(f"{prefix} requires hidden_size >= 0")

    def provided_inputs(self) -> Mapping[str, object]:
        return {
            _PROVIDED_TRAIN_HIDDEN: self._train_hidden_for_step,
            _PROVIDED_TRAIN_REWARDS: self._train_rewards_for_step,
        }

    def init_parameters(
        self, spec: RuntimeSpecializationSpec
    ) -> Mapping[str, torch.nn.Parameter]:
        hidden_size = int(self.config.hidden_size) or int(spec.hidden_size)
        if int(self.config.hidden_size) > 0 and hidden_size != int(spec.hidden_size):
            raise RuntimeError(
                "TTRL Region hidden_size does not match the runtime specialization"
            )
        rank = int(self.config.rank)
        # LowRankAdapter parity: down ~ N(0, 1/sqrt(hidden)), up = 0.
        down = torch.empty((rank, hidden_size), device=spec.device, dtype=spec.dtype)
        torch.nn.init.normal_(down, mean=0.0, std=1.0 / float(hidden_size) ** 0.5)
        up = torch.zeros((1, rank), device=spec.device, dtype=spec.dtype)
        return {"down": torch.nn.Parameter(down), "up": torch.nn.Parameter(up)}

    def optimizer_overrides(self) -> Mapping[str, object]:
        # The signed lr is nominal; the physical lr is pinned per instance by
        # the specialization key.
        return {"lr": float(self.config.learning_rate)}

    # -- host staging API (out-of-band closed loop) -------------------------

    @property
    def resources(self) -> TTRLRegionResources:
        return self._resources  # type: ignore[return-value]

    @property
    def consumer_id(self) -> str:
        return str(self.config.consumer_id)

    @property
    def _channel(self) -> _HostChannel:
        return _host_channel(self.consumer_id)

    @property
    def last_published_intent(self) -> TTRLClosedLoopIntent | None:
        """Most recent intent landed by the signed publish Region."""
        return self._channel.last_published_intent

    def adapter_version_id(self) -> str:
        if self.state_version <= 0:
            return f"{self.consumer_id}:base"
        return f"{self.consumer_id}:update:{self.state_version}"

    def bind_control_runtime(self, runtime: object) -> None:
        """Bind the publish target for RewardOutcome/RetryControl/UpdateBarrier."""
        self._channel.publish_target = runtime

    def stage_completed_generations(
        self,
        trace: object,
        hidden_rows: torch.Tensor,
        *,
        n_votes_per_prompt: int | None = None,
        n_samples_per_prompt: int | None = None,
    ) -> TTRLClosedLoopIntent:
        """Stage one completed-generation batch for the next signed step.

        Reuses ``build_ttrl_closed_loop_intent`` verbatim; row selection
        follows the legacy toy closed loop.  Staging over an unconsumed batch
        fails closed.
        """
        if not bool(self.config.enabled):
            raise RuntimeError("TTRL Region is disabled; refusing to stage training rows")
        channel = self._channel
        if channel.staged is not None:
            raise RuntimeError(
                "TTRL Region already has a staged batch; wait for the signed "
                "train/publish step to consume it"
            )
        if not isinstance(hidden_rows, torch.Tensor) or hidden_rows.ndim != 2:
            raise ValueError("TTRL Region staging requires rank-2 hidden rows")
        intent = build_ttrl_closed_loop_intent(
            trace,
            consumer_id=self.consumer_id,
            n_votes_per_prompt=n_votes_per_prompt,
            n_samples_per_prompt=n_samples_per_prompt,
            verifier=self._verifier,
        )
        rewards = intent.training_rewards()
        if rewards:
            if int(hidden_rows.shape[0]) == len(rewards):
                train_hidden = hidden_rows
            elif intent.training_indices and int(hidden_rows.shape[0]) > max(
                intent.training_indices
            ):
                train_hidden = hidden_rows[list(intent.training_indices)]
            else:
                raise RuntimeError("hidden rows must align with selected TTRL rewards")
        else:
            train_hidden = hidden_rows[:0]
        reward_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=train_hidden.device
        )
        channel.staged = _StagedTraining(
            intent=intent,
            hidden=train_hidden.detach().clone(),
            rewards=reward_tensor,
        )
        channel.total_staged_reward_rows += len(rewards)
        return intent

    # -- provided-input callbacks (invoked at signed Region execution) ---------

    def _require_device(self) -> torch.device:
        spec = self.specialization
        if spec is None:
            raise RuntimeError("TTRL Region must be cold_specialize()d before execution")
        return spec.device

    def _train_hidden_for_step(self, _step_id: int) -> torch.Tensor:
        staged = self._channel.staged
        if staged is None:
            return torch.empty((0, self._hidden_size()), device=self._require_device())
        return staged.hidden

    def _train_rewards_for_step(self, _step_id: int) -> torch.Tensor:
        staged = self._channel.staged
        if staged is None:
            return torch.empty((0,), device=self._require_device())
        return staged.rewards

    def _hidden_size(self) -> int:
        down = self.params.get("down")
        assert isinstance(down, torch.nn.Parameter)
        return int(down.shape[1])

    # -- host-side read-only prediction (test/inspection helper) ----------------

    def predict_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Read-only reward predictions against the current replica state."""
        params = self.params
        down = params["down"]
        up = params["up"]
        assert isinstance(down, torch.nn.Parameter) and isinstance(up, torch.nn.Parameter)
        with torch.no_grad():
            return F.linear(F.linear(hidden, down), up)

    # -- signed tasks -------------------------------------------------------------

    @task(
        inputs={
            "hidden": provided(_PROVIDED_TRAIN_HIDDEN),
            "rewards": provided(_PROVIDED_TRAIN_REWARDS),
        },
        outputs={"loss": temporary()},
        effects=(read(reward_model),),
        timing=Between("token.commit", "step.end"),
    )
    def train(self, hidden: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        flat_rewards = rewards.reshape(-1)
        if int(flat_rewards.numel()) == 0:
            # Zero-row step: no gradient exists, so return a detached zero;
            # ``update`` refuses it without a version bump (maybe_transition).
            return torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        if hidden.ndim != 2 or int(hidden.shape[0]) != int(flat_rewards.numel()):
            raise ValueError("TTRL Region train hidden and rewards row mismatch")
        params = self.params
        down = params["down"]
        up = params["up"]
        assert isinstance(down, torch.nn.Parameter) and isinstance(up, torch.nn.Parameter)
        with torch.inference_mode(False):
            with torch.enable_grad():
                target = flat_rewards.reshape(-1, 1).to(
                    device=hidden.device, dtype=hidden.dtype
                )
                loss = F.mse_loss(F.linear(F.linear(hidden, down), up), target)
        return loss

    @task(
        inputs={"loss": result("train", "loss")},
        effects=(maybe_transition(reward_model),),
        timing=Before("next.prepare"),
        implementation=ExternImplementation(
            namespace="tllm.ttrl",
            name="adam_transition",
            version="1",
            input_types=("Tensor",),
            output_types=("None",),
        ),
    )
    def update(self, loss: torch.Tensor) -> None:
        if isinstance(loss, torch.Tensor) and not loss.requires_grad:
            # Documented zero-row skip: no optimizer step, no version bump.
            return
        self.step_update(loss)

    @task(
        inputs={"step_id": step_id()},
        effects=(
            read(reward_model),
            extern_io("ttrl:reward_outcome"),
            extern_io("ttrl:retry_control"),
            extern_io("ttrl:update_barrier"),
        ),
        timing=Before("next.prepare"),
        implementation=ExternImplementation(
            namespace="tllm.ttrl",
            name="publish_control_intents",
            version="1",
            input_types=("int",),
            output_types=("None",),
        ),
    )
    def publish(self, step_id: int) -> None:
        channel = self._channel
        staged = channel.staged
        if staged is None:
            # Zero-completion step: nothing was trained, nothing to publish.
            return
        runtime = channel.publish_target
        if runtime is None:
            raise RuntimeError(
                "TTRL Region requires bind_control_runtime() before its signed "
                "publish Region can land a staged intent"
            )
        from tllm.runtime.reward_outcome_runtime import write_reward_outcomes
        from tllm.runtime.retry_control_runtime import write_retry_controls
        from tllm.runtime.update_barrier_runtime import write_update_barriers

        # Stamp the post-update version: barrier "ready" is real at this point.
        adapter_version_id = self.adapter_version_id()
        intent = replace(
            staged.intent,
            adapter_version_id=adapter_version_id,
            retry_schedule=tuple(
                replace(entry, adapter_version_id=adapter_version_id)
                for entry in staged.intent.retry_schedule
            ),
        )
        write_reward_outcomes(runtime, intent.reward_outcomes)
        write_retry_controls(runtime, intent.retry_controls)
        write_update_barriers(runtime, intent.update_barriers)
        channel.last_published_intent = intent
        channel.staged = None


def build_ttrl_region_workload(
    config: TTRLConsumerConfig,
    *,
    verifier: TTRLVerifier | None = None,
) -> TTRLRegionWorkload:
    return TTRLRegionWorkload(config, verifier=verifier)


__all__ = [
    "TTRLRegionResources",
    "TTRLRegionWorkload",
    "build_ttrl_region_workload",
]
