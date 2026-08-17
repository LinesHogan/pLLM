#!/usr/bin/env python3
"""GPU smoke: tLLM bootstrap under VLLM_ENABLE_V1_MULTIPROCESSING=1.

The smallest architecture proof for multiprocess mode: a registered builtin
consumer (dummy) plus a marker probe must execute inside the EngineCore child
process, not the parent. Verification:

* generation completes under mp=1 on the selected runner;
* the probe's ``on_step_end`` writes a per-pid marker file — a marker whose
  pid differs from the parent's proves consumer hooks ran in the child.

Under fork (vLLM default) the child inherits the parent's patched classes and
configured runtime; the probe (registered pre-fork) rides along and proves the
child ran consumer hooks. Under spawn the plugin re-executes in the child and
rebuilds from the bootstrap spec; the unregistered probe is dropped (warned via
``skipped_consumers``), so the spawn proof is the plugin marker recording
``applied=true`` from a non-parent pid.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=["mrv1", "mrv2"], required=True)
    parser.add_argument("--mp", choices=["0", "1"], default="1")
    parser.add_argument(
        "--mapped",
        action="store_true",
        help="route generate through run_generate_with_request_mapping and assert the reqid mapping reaches the EngineCore child (A3 gate)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = args.mp  # before tllm import

    marker_dir = tempfile.mkdtemp(prefix="tllm_mp_smoke_")
    parent_pid = os.getpid()

    from tllm.consumers import registry as consumer_registry
    from tllm.consumers.dummy.config import DummyConsumerConfig
    from tllm.ports.base import ConsumerFlow
    from tllm.ports.request_meta import RequestMeta
    from tllm.runtime import residual_runtime as core
    from tllm.util.tools import shutdown_llm_instance

    class MarkerProbe:
        consumer_id = "test.mp_marker_probe"

        def flows(self):
            # on_step_end only fires for non-background flows reached by flow
            # dispatch, so the probe uses one cheap metadata-only same-step flow.
            return (
                ConsumerFlow(
                    reads=(RequestMeta.read(),),
                    writes=(),
                    window="same_step",
                    bundle_key=("engine_step_id", "phase"),
                ),
            )

        def consume_bundle(self, bundle, ctx) -> None:
            return None

        def on_step_end(self, ctx) -> None:
            marker = os.path.join(marker_dir, f"probe_pid{os.getpid()}")
            with open(marker, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))

    consumer_registry.activate_consumer(
        consumer_registry.build_consumer(
            "dummy", DummyConsumerConfig(export_every_n_steps=8, export_max_rows=64)
        ),
        primary=True,
    )
    core.register_dispatch_consumer(MarkerProbe())

    llm = core.make_llm(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "bfloat16",
        0.4,
        512,
        False,
        enforce_eager=True,
        seed=1234,
    )
    try:
        from vllm import SamplingParams

        prompts = ["Give me a short introduction to large language models."] * 4
        sampling = SamplingParams(temperature=0.0, max_tokens=int(args.max_new_tokens))
        if args.mapped:
            from tllm.runtime.vllm_patch.capture_runner import run_generate_with_request_mapping

            outputs = run_generate_with_request_mapping(
                runtime=core.RUNTIME, llm=llm, prompts=prompts, params=sampling
            )
        else:
            outputs = llm.generate(prompts, sampling)
        tokens = [len(o.outputs[0].token_ids) for o in outputs]

        from tllm.runtime.mp.control import ControlPlane

        ping_replies = ControlPlane(llm).ping(timeout=60.0)
        worker_pids = [int(reply["pid"]) for reply in ping_replies]
        mapping_sizes = [int(reply["mapping_size"]) for reply in ping_replies]
        print(f"tllm_ping worker pids: {worker_pids} (parent={parent_pid}) mapping_sizes={mapping_sizes}")
        if args.mp == "1":
            if not worker_pids or any(pid == parent_pid for pid in worker_pids):
                print("FAIL: tllm_ping did not reach a non-parent worker process")
                return 1
        elif any(pid != parent_pid for pid in worker_pids):
            print("FAIL: tllm_ping reached an unexpected process under mp=0")
            return 1
        if args.mapped and any(size != len(prompts) for size in mapping_sizes):
            print(f"FAIL: mapped generate did not install {len(prompts)} reqid mappings in the worker(s)")
            return 1
    finally:
        shutdown_llm_instance(llm, cooldown_s=1.0)

    all_markers = sorted(os.listdir(marker_dir))
    child_markers = [
        name
        for name in all_markers
        if name.startswith("probe_pid") and int(name.removeprefix("probe_pid")) != parent_pid
    ]
    print(f"parent_pid={parent_pid} markers={all_markers}")
    print(f"generated tokens per request: {tokens}")

    spawn_mode = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork") == "spawn"
    if args.mp == "0":
        # No child: hooks run in-process, so a parent marker is the proof.
        hooks_ran = bool(all_markers)
    elif spawn_mode:
        # The unregistered probe is lost under spawn BY DESIGN (see the
        # skipped_consumers warning); the spawn-path proof is the plugin
        # marker showing a child rehydrated the runtime from the spec.
        import glob
        import json

        spec_path = os.environ.get("TLLM_BOOTSTRAP_SPEC_PATH", "")
        applied = False
        for marker_path in glob.glob(f"{spec_path}.pid*"):
            with open(marker_path, encoding="utf-8") as handle:
                record = json.load(handle)
            if int(record.get("pid", parent_pid)) != parent_pid and bool(record.get("applied")):
                applied = True
        hooks_ran = applied
        print(f"spawn mode: plugin rehydrated runtime in child = {applied}")
    else:
        hooks_ran = bool(child_markers)
    if not hooks_ran:
        print("FAIL: no proof consumer hooks/bootstrap ran in the expected process")
        return 1
    if any(int(t) != int(args.max_new_tokens) for t in tokens):
        print("FAIL: generation did not produce the expected token counts")
        return 1
    print(f"OK: consumer hooks executed (mp={args.mp}, runner={args.runner}, markers={all_markers})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
