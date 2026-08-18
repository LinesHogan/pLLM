#!/usr/bin/env python3
"""Unit tests for runtime typing contracts on retained framework code."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


class RuntimeTypingContractsUnitTest(unittest.TestCase):
    def test_getattr_probes_outside_quarantine_only_decrease(self) -> None:
        """AST gate: no new getattr duck-typing probes outside the quarantine dirs.

        ``tllm/runtime/vllm_patch`` and ``tllm/backends`` are the sanctioned
        quarantine for probing foreign (vLLM) objects. Everywhere else, tLLM
        internal objects must be reached through typed attributes; a
        ``getattr(obj, "name", default)`` probe on an internal object is a
        typing-discipline violation. Existing sites are recorded as
        ``file::attribute`` counts in ``tools/getattr_probe_baseline.json``
        (count-based so line drift is not flagged); the counts may only
        shrink.
        """
        quarantine = ("tllm/runtime/vllm_patch", "tllm/backends")
        current: dict[str, int] = {}
        for path in sorted(Path("tllm").rglob("*.py")):
            sp = str(path)
            if "__pycache__" in sp or any(sp.startswith(q) for q in quarantine):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=sp)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"):
                    continue
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    key = f"{sp}::{node.args[1].value}"
                    current[key] = current.get(key, 0) + 1

        baseline: dict[str, int] = json.loads(Path("tools/getattr_probe_baseline.json").read_text(encoding="utf-8"))
        violations = sorted(
            f"{key}: {count} > baseline {baseline.get(key, 0)}" for key, count in current.items() if count > baseline.get(key, 0)
        )
        self.assertEqual(
            violations,
            [],
            msg=(
                "new getattr duck-typing probes outside the quarantine dirs "
                "(tllm/runtime/vllm_patch, tllm/backends) are forbidden; reach "
                "tLLM internals through typed attributes instead:\n" + "\n".join(violations)
            ),
        )


if __name__ == "__main__":
    unittest.main()
