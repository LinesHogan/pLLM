#!/usr/bin/env python3
"""Closure checks for the devlist final alignment artifacts."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_NAMES = (
    "OverRIDE",
    "In-Place TTT",
    "TTRL",
    "FTTT",
    "GTTA",
    "TLM",
)


def _matrix_rows() -> dict[str, list[str]]:
    matrix = REPO_ROOT / "doc/reference/consumers/devlist_alignment_matrix.md"
    rows: dict[str, list[str]] = {}
    for line in matrix.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| Project ") or line.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0]] = cells
    return rows


class DevlistFinalAlignmentClosureUnitTest(unittest.TestCase):
    def test_plan_source_files_exist_for_auditable_closure(self) -> None:
        self.assertTrue((REPO_ROOT / "docs/devlist.md").is_file())
        self.assertTrue((REPO_ROOT / "docs/plans/2026-05-01-devlist-final-alignment-closure.md").is_file())

    def test_final_alignment_matrix_covers_all_devlist_projects_with_required_evidence(self) -> None:
        matrix = REPO_ROOT / "doc/reference/consumers/devlist_alignment_matrix.md"
        devlist = (REPO_ROOT / "docs/devlist.md").read_text(encoding="utf-8")
        rows = _matrix_rows()

        text = matrix.read_text(encoding="utf-8")

        for project in PROJECT_NAMES:
            with self.subTest(project=project):
                self.assertIn(project, rows)
                self.assertIn(project, text)
        for devlist_title in (
            "Diverse Text Decoding via Iterative Reweighting",
            "In-Place TTT",
            "TTRL: Test-Time Reinforcement Learning",
            "Learning to Reason from Feedback at Test-Time",
            "Test-Time Adaptation for LLM Agents via Environment Interaction",
            "Test-Time Learning for Large Language Models",
        ):
            with self.subTest(devlist_title=devlist_title):
                self.assertIn(devlist_title, devlist)
        for project, cells in rows.items():
            with self.subTest(project=project):
                self.assertEqual(len(cells), 7)
                self.assertNotIn("TBD", " ".join(cells))
                self.assertRegex(cells[2], r"(blocked|source-extracted|Direct upstream|Upstream)")
                self.assertRegex(cells[3], r"(test|parity|tested|fixtures|Source-extracted)")
                self.assertRegex(cells[4], r"(Cached|cached|Guarded|smoke|runtime|model)")
                self.assertRegex(cells[5], r"(count|trained_rows|selected_count|reward_count)")
                self.assertTrue(cells[6])
        self.assertIn("nonzero work counter", text)
        self.assertIn("Scope exception", text)
        self.assertIn("prefill `ConsumerFlow` delivery", rows["In-Place TTT"][6])
        self.assertIn("real vLLM LoRA apply smoke", rows["TLM"][6])

    def test_plan_named_smoke_files_exist(self) -> None:
        expected = (
            "test/consumers/override/test_override_real_decode_smoke.py",
            "test/consumers/fttt/test_fttt_real_model_smoke.py",
        )

        for relative in expected:
            with self.subTest(relative=relative):
                self.assertTrue((REPO_ROOT / relative).is_file())

    def test_user_facing_reports_do_not_claim_stale_partial_status(self) -> None:
        report_dir = REPO_ROOT / "doc/reference/consumers"
        forbidden = (
            "runtime unverified",
            "not yet fully integrated",
            "Current status: partial",
            "not a complete upstream",
            "partial formula-level reproduction",
        )

        for report in sorted(report_dir.glob("*/reproduction_report.md")):
            text = report.read_text(encoding="utf-8")
            for phrase in forbidden:
                with self.subTest(report=report.relative_to(REPO_ROOT), phrase=phrase):
                    self.assertNotIn(phrase, text)

    def test_user_facing_artifacts_do_not_leak_local_filesystem_paths(self) -> None:
        forbidden_parts = (
            "/" + "home" + "/" + "developer",
            "." + "worktrees",
            "/" + "tmp" + "/",
            "/" + "Users" + "/",
            "C:" + "\\",
        )
        leak_pattern = re.compile("|".join(re.escape(part) for part in forbidden_parts))
        roots = (
            REPO_ROOT / "doc/reference/consumers",
            REPO_ROOT / "test/consumers",
            REPO_ROOT / "test/runtime",
            REPO_ROOT / "tllm",
        )

        for root in roots:
            for path in sorted(root.rglob("*")):
                if path.is_dir() or path.suffix in {".pyc", ".json", ".jsonl"}:
                    continue
                text = path.read_text(encoding="utf-8")
                with self.subTest(path=path.relative_to(REPO_ROOT)):
                    self.assertIsNone(leak_pattern.search(text))


if __name__ == "__main__":
    unittest.main()
