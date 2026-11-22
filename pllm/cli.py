"""Command-line entry points for `pllm skill` and `pllm serve`."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.resources
import inspect
import sys
from pathlib import Path
from typing import Sequence

_INSTALL_DIRS = {
    "claude": Path(".claude/skills/pllm"),
    "codex": Path(".agents/skills/pllm"),
    "agents": Path(".agents/skills/pllm"),
    "cursor": Path(".cursor/rules/pllm"),
}
_PRINT_ALIASES = {
    "skill": "SKILL.md",
    "migration": "migration.md",
    "reference": "reference.md",
    "examples": "examples.md",
    "rules": "rules.md",
    "pitfalls": "pitfalls.md",
}


def _skill_files() -> dict[str, str]:
    root = importlib.resources.files("pllm").joinpath("_skill")
    files = {entry.name: entry.read_text(encoding="utf-8") for entry in root.iterdir() if entry.name.endswith(".md")}
    if not files:
        raise RuntimeError(
            "packaged pLLM agent skill files were not found; reinstall pllm or run tools/check_docs.py --write-generated"
        )
    return dict(sorted(files.items()))


def _write_skill_dir(target: Path) -> Path:
    files = _skill_files()
    target.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (target / name).write_text(content, encoding="utf-8")
    return target


def _run_skill(args: argparse.Namespace) -> int:
    files = _skill_files()
    if args.install is not None:
        dest = _write_skill_dir(_INSTALL_DIRS[args.install])
        print(f"Installed pLLM skill ({len(files)} files) into {dest}")
        return 0
    if args.dir is not None:
        dest = _write_skill_dir(Path(args.dir))
        print(f"Installed pLLM skill ({len(files)} files) into {dest}")
        return 0
    if args.print_doc is not None:
        topic = args.print_doc
        if topic == "all":
            for name, content in files.items():
                print(f"<!-- {name} -->\n{content}")
            return 0
        key = _PRINT_ALIASES.get(topic.lower(), topic)
        if key not in files:
            print(f"Unknown skill document {topic!r}; choose from {', '.join(files)} or 'all'", file=sys.stderr)
            return 2
        sys.stdout.write(files[key])
        return 0
    print("Packaged pLLM coding-agent docs:", ", ".join(files))
    print("Print SKILL.md:         pllm skill --print")
    print("Print migration guide:  pllm skill --print migration")
    print("Install for Claude:     pllm skill --install claude")
    print("Install for Codex:      pllm skill --install codex")
    print("Install into directory: pllm skill --dir <path>")
    return 0


def _load_workload(spec: str):
    from pllm.workload import Workload

    module_name, sep, attr_name = spec.partition(":")
    if not sep or not module_name or not attr_name:
        raise ValueError(f"--workload must look like module:ClassOrInstance, got {spec!r}")
    module = importlib.import_module(module_name)
    target = module
    for part in attr_name.split("."):
        target = getattr(target, part)
    if inspect.isclass(target):
        instance = target()
    elif callable(target) and not isinstance(target, Workload):
        instance = target()
    else:
        instance = target
    if not isinstance(instance, Workload):
        raise TypeError(f"{spec!r} resolved to {type(instance).__name__}, expected a Workload")
    return instance


def _run_serve(args: argparse.Namespace) -> int:
    import pllm
    from pllm.backends.vllm.serving import serve_openai_llm

    workload = None if args.workload is None else _load_workload(args.workload)
    config = pllm.ModelConfig(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=args.enable_prefix_caching,
        cuda_graphs=False if args.eager else "auto",
        trust_remote_code=args.trust_remote_code,
    )
    llm = pllm.make_async_llm(config, workload=workload)
    asyncio.run(
        serve_openai_llm(
            llm,
            host=args.host,
            port=args.port,
            served_model_name=args.served_model_name,
            api_key=args.api_key,
            chat_template=args.chat_template,
            uvicorn_log_level=args.log_level,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level `pllm` CLI parser."""
    parser = argparse.ArgumentParser(prog="pllm", description="pLLM CLI utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    skill = subparsers.add_parser("skill", help="Print or install packaged coding-agent skill docs")
    group = skill.add_mutually_exclusive_group()
    group.add_argument(
        "--print",
        dest="print_doc",
        nargs="?",
        const="SKILL.md",
        metavar="DOC",
        help="Print SKILL.md (or migration, reference, examples, rules, pitfalls, all)",
    )
    group.add_argument(
        "--install",
        choices=tuple(_INSTALL_DIRS),
        help="Install skill docs into .claude, .agents (codex/agents), or .cursor",
    )
    group.add_argument("--dir", help="Copy packaged skill docs into a custom directory")

    serve = subparsers.add_parser("serve", help="Run an OpenAI-compatible HTTP server with an optional pLLM workload")
    serve.add_argument("--model", required=True, help="Hugging Face model name or local path")
    serve.add_argument("--workload", default=None, help="Workload object or zero-arg class as module:attr")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel ranks")
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM KV cache GPU utilization")
    serve.add_argument("--max-model-len", type=int, default=512, help="Maximum sequence length")
    serve.add_argument("--max-num-seqs", type=int, default=256, help="Maximum concurrent requests")
    serve.add_argument("--max-num-batched-tokens", type=int, default=2048, help="Maximum tokens per step")
    serve.add_argument("--eager", action="store_true", help="Disable CUDA graphs")
    serve.add_argument("--enable-prefix-caching", action="store_true", help="Enable vLLM prefix caching")
    serve.add_argument("--trust-remote-code", action="store_true", help="Trust remote model code")
    serve.add_argument("--served-model-name", default=None, help="Public model name reported on /v1/models")
    serve.add_argument("--api-key", default=None, help="Optional bearer API key")
    serve.add_argument("--chat-template", default=None, help="Optional custom chat template path")
    serve.add_argument("--log-level", default="info", help="Uvicorn log level")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `pllm` and `python -m pllm`."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "skill":
        return _run_skill(args)
    if args.command == "serve":
        return _run_serve(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
