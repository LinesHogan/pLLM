"""Exercise rendered heading links and explicit execution opt-outs for the docs checker."""

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from tools import check_docs
from tools.check_docs import broken_links, check_file, heading_anchors


def test_heading_links_across_markdown_and_notebooks(tmp_path: Path) -> None:
    headings = [
        "## 支持的模型\n## 支持的模型\n## 支持的模型-1\n## 支持的模型\n",
        "标题 `API` & *文字*！\n---\n\n```python\n# invisible\n```\n",
    ]
    expected = {"支持的模型", "支持的模型-1", "支持的模型-1-1", "支持的模型-2", "标题-api--文字"}
    assert heading_anchors(headings) == expected
    markdown = tmp_path / "README.md"
    markdown.write_text("\n\n".join(headings))
    notebook = tmp_path / "tutorial.ipynb"
    notebook.write_text(
        json.dumps({"cells": [{"cell_type": "markdown", "source": text, "metadata": {}} for text in headings]})
    )
    for document in (markdown, notebook):
        for anchor in expected:
            assert broken_links(markdown, [f"[heading]({document.name}#{quote(anchor)})"], {}) == []
        for anchor in ("支持的模型与限制", "invisible", "支持的模型-3"):
            assert len(broken_links(markdown, [f"[heading]({document.name}#{anchor})"], {})) == 1
    assert broken_links(markdown, ["[self](#支持的模型-1)\n[ref][r]\n[r]: tutorial.ipynb#支持的模型-2"], {}) == []
    assert len(broken_links(markdown, ["[unused]: tutorial.ipynb#支持的模型与限制"], {})) == 1
    assert (
        broken_links(
            markdown, ["`[code](absent.md)`\n~~~\n[code](absent.md)\n~~~\n[web](https://example.com/#absent)"], {}
        )
        == []
    )
    failures, checked = check_file(notebook, False, False, {}, write_outputs=False)
    assert not checked and not failures
    notebook.write_text(
        json.dumps(
            {"cells": [{"cell_type": "markdown", "source": "[bad](README.md#支持的模型与限制)", "metadata": {}}]}
        )
    )
    failures, checked = check_file(notebook, False, False, {}, write_outputs=False)
    assert not checked and len(failures) == 1


def test_python_blocks_share_state_and_report_explicit_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    guides, reference = (tmp_path / "doc" / name for name in ("guides", "reference"))
    for folder in (guides, reference):
        folder.mkdir(parents=True)
    (tmp_path / "README.md").write_text("# Documentation\n")
    (guides / "example.md").write_text(
        "```python\nvalue = 3\n```\n"
        "```python\n# check-docs: skip: needs a CUDA engine\nraise RuntimeError('GPU only')\n```\n"
        "~~~py\nimport os\nassert value == 3\nassert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n~~~\n"
    )
    (reference / "example.md").write_text("```python\nraise TypeError('stale API')\n```")
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    monkeypatch.setattr(check_docs, "EXECUTED_DOCS", (guides, reference))
    monkeypatch.setattr(check_docs.sys, "argv", ["check_docs.py"])
    assert check_docs.main() == 1
    report = capsys.readouterr().out
    assert "execution opt-outs: 1" in report
    assert "doc/guides/example.md:line 4: needs a CUDA engine" in report
    assert "doc/reference/example.md: line 1 (+1): code fails on CPU: TypeError: stale API" in report
    assert "3 documents checked, 1 with failures" in report
