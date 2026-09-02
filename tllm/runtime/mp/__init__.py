#!/usr/bin/env python3
"""tLLM multiprocess support (VLLM_ENABLE_V1_MULTIPROCESSING=1).

Composition root for cross-process bootstrap: this package is the ONE place
in ``tllm/runtime`` that may import the consumer registry, because wiring the
consumer side into the child process is precisely its job. Everything else in
``tllm/runtime`` keeps the runtime→consumers boundary (only
``tllm.ports.consumer`` is allowed).
"""
