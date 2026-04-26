#!/usr/bin/env python3
"""Import compatibility tests for migrated devlist consumers."""

from __future__ import annotations

import importlib
import unittest


class DevlistImportCompatUnitTest(unittest.TestCase):
    def test_canonical_package_exports_match_legacy_devlist_exports(self) -> None:
        cases = {
            "override": (
                "OverRIDEConsumer",
                "OverRIDEConsumerConfig",
                "OverRIDEReweightEngine",
                "OverRIDELogitsResult",
                "OverRIDEBatchLogitsResult",
                "OverRIDEStats",
            ),
            "inplace_ttt": (
                "InPlaceTTTConsumer",
                "InPlaceTTTConsumerConfig",
                "InPlaceTTTEngine",
                "InPlaceTTTStats",
                "pad_rows_to_chunk",
            ),
            "ttrl": (
                "TTRLConsumer",
                "TTRLConsumerConfig",
                "TTRLEngine",
                "TTRLMajorityResult",
                "TTRLRewardResult",
                "TTRLSample",
                "TTRLSampleReward",
                "TTRLStats",
                "build_ttrl_reward_result",
                "majority_vote_rewards",
                "select_ttrl_training_indices",
            ),
            "fttt": (
                "FTTTConsumer",
                "FTTTConsumerConfig",
                "FTTTEngine",
                "FTTTInstanceState",
                "FTTTStats",
                "FeedbackRecord",
            ),
            "gtta": (
                "GTTAConsumer",
                "GTTAConsumerConfig",
                "GTTAStats",
                "TrajectoryFeedback",
                "parse_reward",
            ),
            "tlm": (
                "TLMConsumer",
                "TLMConsumerConfig",
                "TLMEngine",
                "TLMStats",
                "run_tlm_ordered_step",
                "select_scalar_score_fallback_samples",
                "select_ttl_samples",
            ),
        }

        for package_name, symbol_names in cases.items():
            with self.subTest(package_name=package_name):
                canonical_module = importlib.import_module(f"tllm.consumers.{package_name}")
                legacy_module = importlib.import_module(f"tllm.consumers.devlist.{package_name}")
                for symbol_name in symbol_names:
                    self.assertIs(
                        getattr(canonical_module, symbol_name),
                        getattr(legacy_module, symbol_name),
                        symbol_name,
                    )

    def test_common_alignment_exports_match_legacy_common_exports(self) -> None:
        canonical_module = importlib.import_module("tllm.consumers._common.alignment")
        legacy_module = importlib.import_module("tllm.consumers.devlist.common")
        for symbol_name in (
            "LowRankAdapter",
            "detached_train_input",
            "sentence_cross_entropy",
            "select_rows_above_threshold",
            "weighted_one_hot_kl_loss",
            "normalize_answer_text",
            "select_top_k_per_prompt",
        ):
            self.assertIs(getattr(canonical_module, symbol_name), getattr(legacy_module, symbol_name), symbol_name)

    def test_package_level_exports_match_legacy_devlist_package(self) -> None:
        canonical_cases = (
            ("tllm.consumers.override", "OverRIDEConsumer"),
            ("tllm.consumers.inplace_ttt", "InPlaceTTTConsumer"),
            ("tllm.consumers.ttrl", "TTRLConsumer"),
            ("tllm.consumers.fttt", "FTTTConsumer"),
            ("tllm.consumers.gtta", "GTTAConsumer"),
            ("tllm.consumers.tlm", "TLMConsumer"),
            ("tllm.consumers._common.alignment", "LowRankAdapter"),
        )
        legacy_package = importlib.import_module("tllm.consumers.devlist")
        for module_name, symbol_name in canonical_cases:
            with self.subTest(symbol_name=symbol_name):
                canonical_module = importlib.import_module(module_name)
                self.assertIs(getattr(canonical_module, symbol_name), getattr(legacy_package, symbol_name))


if __name__ == "__main__":
    unittest.main()
