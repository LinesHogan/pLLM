"""Native ReFT (LoReFT) intervention workload, kit style.

The algorithm: at the declared decoder layers, patch the residual stream
during prefill with LoReFT — ``h + (Wh + b − Rh) R^T`` on the first/last N
prompt positions (pyreft's ``intervene_on_prompt=True`` semantics; the KV
carries the effect through decode).  Training happens *outside* the engine
(transformers-side loop, e.g. pyreft's own modules); the engine holds only
the versioned parameters and applies them.

Data flow (one signed Region):

- The host training loop calls :meth:`push_intervention` with fresh
  ``(W, b, R)`` — staged into the process-local host channel (the engine
  drives a reconstructed instance, never the host object; see
  ``tllm.kit.channel``).
- The signed ``publish`` Region (``maybe_transition``, Before
  ``next.prepare``) pops the staged tensors, copies them into the State
  parameters, and bumps the audited state version.  Steps without a push
  bump +0.
- The installed forward wrappers (``runtime.vllm_patch.activation_patch_hooks``)
  read the channel's :class:`LoreftBank` every forward; the bank activates
  only after the first push has landed, so a fresh engine serves the
  unpatched baseline — and mp=1 workers (whose per-process channel never
  receives a host push) never patch.  A signed control-plane staging
  transport is the follow-up that closes that gap, same as TTRL.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.consumers.reft.config import ReftConsumerConfig
from tllm.contracts.activation_patch import ActivationPatchSpec
from tllm.kit import RegionAlgorithm, typed_host_channel
from tllm.runtime.activation_patch import LoreftBank
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.workload import (
    Before,
    State,
    StateScope,
    extern,
    maybe_transition,
    step_id,
    task,
    workload,
)


@dataclass(frozen=True, slots=True)
class _StagedIntervention:
    source_w: torch.Tensor
    source_b: torch.Tensor
    rotate: torch.Tensor


@dataclass(slots=True)
class _ReftChannel:
    staged: _StagedIntervention | None = None
    bank: LoreftBank | None = None


@workload(id="reft.intervention")
class ReftInterventionWorkload(RegionAlgorithm[ReftConsumerConfig]):
    """Versioned LoReFT parameter bank plus one signed publish Region."""

    algorithm = "reft-loreft-region-v1"
    config_type = ReftConsumerConfig
    enabled_field = "enabled"

    intervention = State(
        scope=StateScope.REPLICA,
        parameters=("source_w", "source_b", "rotate"),
        # Host-published parameters (external training loop), not an in-engine
        # optimizer: the publish Region lands staged tensors via
        # commit_state_update().
        transition="host_publish",
    )

    # -- kit hooks --------------------------------------------------------

    def validate_config(self, config: ReftConsumerConfig) -> None:
        # Reuses the spec's validation so config and patch can never diverge.
        ActivationPatchSpec(
            layers=tuple(config.layers),
            first_tokens=int(config.first_tokens),
            last_tokens=int(config.last_tokens),
        )
        if int(config.rank) < 1:
            raise ValueError("ReFT intervention requires a positive rank")
        if float(config.learning_rate) <= 0:
            raise ValueError("ReFT intervention requires a positive learning_rate")

    def init_parameters(
        self, spec: RuntimeSpecializationSpec
    ) -> dict[str, torch.nn.Parameter]:
        hidden = int(spec.hidden_size)
        rank = int(self.config.rank)
        num_layers = len(self.patch_spec().layers)
        # pyreft init parity: learned_source is a default-initialized Linear,
        # rotate starts orthonormal (LoreftIntervention.__init__).
        source = torch.nn.Linear(hidden, rank)
        source_w = (
            source.weight.detach()
            .to(device=spec.device, dtype=spec.dtype)
            .repeat(num_layers, 1, 1)
        )
        source_b = (
            source.bias.detach()
            .to(device=spec.device, dtype=spec.dtype)
            .repeat(num_layers, 1)
        )
        # orthogonal_ requires float32 (geqrf is unavailable for bf16 CUDA);
        # compute on CPU in fp32, then move to the specialization's device/dtype.
        rotate = torch.empty((num_layers, hidden, rank), dtype=torch.float32)
        for slot in range(num_layers):
            torch.nn.init.orthogonal_(rotate[slot])
        rotate = rotate.to(device=spec.device, dtype=spec.dtype)
        return {
            "source_w": torch.nn.Parameter(source_w),
            "source_b": torch.nn.Parameter(source_b),
            "rotate": torch.nn.Parameter(rotate),
        }

    # -- host API -----------------------------------------------------------

    @property
    def consumer_id(self) -> str:
        return str(self.config.consumer_id)

    @property
    def _channel(self) -> _ReftChannel:
        return typed_host_channel(f"reft:{self.consumer_id}", _ReftChannel)

    def patch_spec(self) -> ActivationPatchSpec:
        return ActivationPatchSpec(
            layers=tuple(self.config.layers),
            first_tokens=int(self.config.first_tokens),
            last_tokens=int(self.config.last_tokens),
        )

    def push_intervention(
        self,
        source_w: torch.Tensor,
        source_b: torch.Tensor,
        rotate: torch.Tensor,
    ) -> None:
        """Stage freshly trained (W, b, R); the next signed step publishes them."""
        if not bool(self.config.enabled):
            raise RuntimeError("ReFT intervention is disabled; refusing to stage weights")
        channel = self._channel
        if channel.staged is not None:
            raise RuntimeError(
                "ReFT intervention already has staged weights; wait for the "
                "signed publish Region to land them"
            )
        for name, incoming in (
            ("source_w", source_w),
            ("source_b", source_b),
            ("rotate", rotate),
        ):
            current = self.param(name)  # raises unless cold-specialized
            if tuple(incoming.shape) != tuple(current.shape):
                raise ValueError(
                    f"ReFT push shape mismatch for {name}: "
                    f"{tuple(incoming.shape)} != {tuple(current.shape)}"
                )
        channel.staged = _StagedIntervention(
            source_w=source_w.detach().clone(),
            source_b=source_b.detach().clone(),
            rotate=rotate.detach().clone(),
        )

    def bank_getter(self) -> object:
        """Return the channel-bank lookup the installed wrappers close over."""
        channel = self._channel
        return lambda: channel.bank

    def intervention_version(self) -> int:
        bank = self._channel.bank
        return 0 if bank is None else int(bank.version)

    # -- signed tasks ---------------------------------------------------------

    @task(
        inputs={"step_id": step_id()},
        effects=(maybe_transition(intervention),),
        timing=Before("next.prepare"),
        implementation=extern("tllm.reft:publish_intervention@1"),
    )
    def publish(self, step_id: int) -> None:
        channel = self._channel
        staged = channel.staged
        if staged is not None:
            with torch.no_grad():
                for name in ("source_w", "source_b", "rotate"):
                    target = self.param(name)
                    incoming = getattr(staged, name)
                    if tuple(incoming.shape) != tuple(target.shape):
                        raise RuntimeError(
                            f"ReFT staged {name} diverged from the signed shapes"
                        )
                    target.copy_(incoming.to(device=target.device, dtype=target.dtype))
            self.commit_state_update()
            channel.staged = None
        # (Re)publish the bank every step.  The bank must be SYNCED from this
        # instance's parameters, never aliased: warmup steps and real engine
        # steps can execute on different reconstructed instances (each with its
        # own Parameter storage), so an aliasing bank created by an earlier
        # instance would silently serve stale init values.  The bank activates
        # only after a push has landed (version >= 1) — before that the engine
        # serves the unpatched baseline, and MP workers never patch.
        with torch.no_grad():
            if channel.bank is None:
                channel.bank = LoreftBank(
                    spec=self.patch_spec(),
                    source_w=self.param("source_w").detach().clone(),
                    source_b=self.param("source_b").detach().clone(),
                    rotate=self.param("rotate").detach().clone(),
                )
            else:
                channel.bank.source_w.copy_(self.param("source_w"))
                channel.bank.source_b.copy_(self.param("source_b"))
                channel.bank.rotate.copy_(self.param("rotate"))
        channel.bank.version = self.state_version
        channel.bank.active = bool(self.config.enabled) and self.state_version > 0
