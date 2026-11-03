"""Select feature verification groups from changed paths for local and cluster drivers."""

import argparse
import os
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

# First match wins; shared execution, backend and verification changes reach the fallback.
PATH_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pllm/workloads/gtta*", ()),
    ("pllm/workloads/dola*", ()),
    ("pllm/workloads/dummy*", ("enginecore", "tp")),
    ("examples/huber.py", ("enginecore",)),
    ("tools/benchmark*", ()),
    ("tools/ai_eval/*", ()),
    ("pllm/workloads/adaptive*", ("adaptive",)),
    ("pllm/workloads/state_producers*", ("producers", "enginecore")),
    ("pllm/workloads/esamp*", ("esamp", "enginecore")),
    ("pllm/workloads/override*", ("feedback", "enginecore")),
    ("pllm/workloads/fttt/*", ("feedback", "enginecore")),
    ("pllm/workloads/ttrl*", ("feedback", "enginecore", "tp")),
    ("pllm/workloads/reft*", ("reft", "enginecore")),
    ("pllm/workloads/tlm/*", ("tlm", "enginecore", "tp")),
    ("pllm/workloads/lora*", ("enginecore",)),
    ("pllm/workloads/inplace*", ("inplace", "routes", "enginecore")),
    ("pllm/workloads/gated_delta*", ("inplace",)),
    ("pllm/workloads/prefill_adapters*", ("inplace", "producers", "enginecore")),
    ("pllm/workloads/recirculation*", ("routes",)),
    ("pllm/workloads/jacobi*", ("routes",)),
    ("pllm/workloads/step_loop*", ("positions",)),
    ("pllm/workloads/video*", ("feeds", "tp")),
    ("examples/video_session.py", ("feeds",)),
    ("test/gpu/*adaptive*", ("adaptive",)),
    ("test/gpu/*custom_residual*", ("residual",)),
    ("test/gpu/*composed_model_sites*", ("routes", "residual", "inplace")),
    ("test/gpu/*state_producers*", ("producers",)),
    ("test/gpu/*state_feed*", ("feeds",)),
    ("test/gpu/*video*", ("feeds",)),
    ("test/gpu/vllm_esamp_smoke.py", ("esamp",)),
    ("test/gpu/vllm_inplace_ttt_smoke.py", ("inplace",)),
    ("test/gpu/vllm_*loop_smoke.py", ("routes", "positions")),
    ("test/gpu/vllm_recirculation_smoke.py", ("routes",)),
    ("test/gpu/vllm_route_updates_smoke.py", ("routes",)),
    ("test/gpu/vllm_override_smoke.py", ("feedback",)),
    ("test/gpu/fttt_cuda_smoke.py", ("feedback",)),
    ("test/gpu/vllm_reft_smoke.py", ("reft",)),
    ("test/gpu/vllm_tlm_smoke.py", ("tlm",)),
    ("test/gpu/enginecore_smoke.py", ("enginecore",)),
    ("test/gpu/vllm_mrv1_smoke.py", ("enginecore",)),
    ("test/gpu/vllm_tp_smoke.py", ("tp",)),
    ("test/release/*", ("release",)),
    ("test/workloads/*", ()),
    ("test/boundary/*", ()),
    ("test/gpu/core/*", ()),
    ("test/gpu/esamp_cuda_graph_replay_smoke.py", ()),
    ("doc/*", ()),
    ("docs/*", ()),
    ("*.md", ()),
)
GROUPS = tuple(sorted({group for _, groups in PATH_GROUPS for group in groups}))


def selection(base: str, all_groups: bool) -> dict[str, str]:
    """Return each selected group and the changed path that requires it."""
    if all_groups:
        return dict.fromkeys(GROUPS, "--all")
    merge_base = subprocess.check_output(["git", "merge-base", base, "HEAD"], text=True).strip()
    changed = subprocess.check_output(["git", "diff", "--name-only", "--no-renames", "-z", merge_base], text=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], text=True)
    chosen: dict[str, str] = {}
    for path in sorted(set((changed + untracked).strip("\0").split("\0")) - {""}):
        workload_path = path.replace("test/workloads/test_", "pllm/workloads/")
        groups = next(
            (
                groups
                for pattern, groups in PATH_GROUPS
                if fnmatchcase(path, pattern) or fnmatchcase(workload_path, pattern)
            ),
            GROUPS,
        )
        for group in groups:
            chosen.setdefault(group, path)
    return chosen


def main() -> None:
    """Print the selection, or replay the driver's frozen selection on a compute node."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="main")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    plan = os.environ.get("VERIFY_SELECTION")
    chosen = (
        dict(line.split("\t", 1) for line in Path(plan).read_text().splitlines())
        if plan
        else selection(args.base, args.all)
    )
    for group, reason in sorted(chosen.items()):
        if group not in GROUPS:
            parser.error(f"unknown feature group {group!r}; regenerate the verification selection")
        print(f"{group}\t{reason}" if args.explain else group)


if __name__ == "__main__":
    main()
