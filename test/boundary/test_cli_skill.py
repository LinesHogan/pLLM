"""CPU checks for `pllm skill` CLI commands and `--workload` loader validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from pllm.cli import _load_workload, main
from pllm.workloads.esamp import ESampWorkload


def test_skill_cli_summary_print_and_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["skill"]) == 0
    summary = capsys.readouterr().out
    assert "SKILL.md" in summary and "migration.md" in summary

    assert main(["skill", "--print"]) == 0
    assert "pllm.Workload" in capsys.readouterr().out

    assert main(["skill", "--print", "migration"]) == 0
    migration_out = capsys.readouterr().out
    assert "LogitsProcessor" in migration_out and "make_async_llm" in migration_out

    assert main(["skill", "--print", "unknown-doc"]) == 2
    assert "Unknown skill document" in capsys.readouterr().err

    out_dir = tmp_path / "custom-skill"
    assert main(["skill", "--dir", str(out_dir)]) == 0
    capsys.readouterr()
    repo_docs = sorted(path.name for path in Path("doc/agent").glob("*.md"))
    assert sorted(path.name for path in out_dir.glob("*.md")) == repo_docs
    for name in repo_docs:
        assert (out_dir / name).read_text(encoding="utf-8") == (Path("doc/agent") / name).read_text(encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    assert main(["skill", "--install", "claude"]) == 0
    assert (tmp_path / ".claude/skills/pllm/SKILL.md").exists()
    assert main(["skill", "--install", "codex"]) == 0
    assert (tmp_path / ".agents/skills/pllm/SKILL.md").exists()


def test_load_workload_resolution_and_validation() -> None:
    workload = _load_workload("pllm.workloads.esamp:ESampWorkload")
    assert isinstance(workload, ESampWorkload)

    with pytest.raises(ValueError, match="module:ClassOrInstance"):
        _load_workload("bad-spec")
    with pytest.raises(TypeError, match="expected a Workload"):
        _load_workload("dict") if False else _load_workload("builtins:dict")
