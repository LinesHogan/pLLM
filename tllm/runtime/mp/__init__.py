#!/usr/bin/env python3
"""tLLM multiprocess support (VLLM_ENABLE_V1_MULTIPROCESSING=1).

Composition root for cross-process bootstrap.  The runtime config and native
workload epochs cross process boundaries; legacy consumer reconstruction has
been retired.
"""
