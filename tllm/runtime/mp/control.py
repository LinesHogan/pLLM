#!/usr/bin/env python3
"""Parent-side control plane over vLLM's collective_rpc.

The plane invokes the string-named methods of ``TllmWorkerExtension`` on the
workers that own the model runner — in-process under mp=0 (InprocClient) and
in the EngineCore child under mp=1 (SyncMPClient). Using one mechanism for
both modes keeps the four parity cells on identical code paths; only string
method names cross the wire, so ``VLLM_ALLOW_INSECURE_SERIALIZATION`` is never
required.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

TLLM_WORKER_EXTENSION_QUALNAME = "tllm.runtime.mp.worker_extension.TllmWorkerExtension"


class ControlPlane:
    """Thin facade: one method per worker-extension entry point."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def call(self, method: str, *args: Any, timeout: float | None = None) -> list[Any]:
        try:
            core_client = self._llm.llm_engine.engine_core
        except AttributeError as exc:
            raise ValueError("control plane requires an LLM with llm_engine") from exc
        return list(core_client.collective_rpc(str(method), timeout=timeout, args=tuple(args)))

    def ping(self, *, timeout: float | None = None) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.call("tllm_ping", timeout=timeout)]

    def push_request_mapping(
        self, mappings: Iterable[tuple[str, int, int]], *, timeout: float | None = None
    ) -> None:
        rows = [[str(req_id), int(prompt_idx), int(sample_idx)] for req_id, prompt_idx, sample_idx in mappings]
        if not rows:
            return
        self.call("tllm_push_request_mapping", rows, timeout=timeout)

    def pull_stats(self, *, timeout: float | None = None) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.call("tllm_pull_stats", timeout=timeout)]

    def synchronize(self, *, timeout: float | None = None) -> None:
        self.call("tllm_synchronize", timeout=timeout)

    def apply_enable_flags(self, flags: Mapping[str, bool], *, timeout: float | None = None) -> None:
        self.call("tllm_apply_enable_flags", {str(k): bool(v) for k, v in flags.items()}, timeout=timeout)

    def register_consumer(
        self, name: str, config: Mapping[str, Any], *, primary: bool = False, timeout: float | None = None
    ) -> None:
        self.call("tllm_register_consumer", str(name), dict(config), bool(primary), timeout=timeout)
