import unittest

from migrationproof.p0p1_protocol import (
    aggregate_independent_pairs,
    build_sft_records,
    build_frozen_split,
    score_explanation,
)


class FrozenSplitTest(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_keeps_test_frozen(self):
        cases = [{"id": f"MPBX-{i:04d}", "expected": {"status": "pass"}}
                 for i in range(1, 51)]
        split_a = build_frozen_split(cases, seed=42)
        split_b = build_frozen_split(cases, seed=42)
        self.assertEqual(split_a, split_b)
        self.assertEqual(len(split_a["test"]), 50)
        self.assertEqual(set(split_a["test"]), {case["id"] for case in cases})
        self.assertFalse(set(split_a["train"]) & set(split_a["validation"]))
        self.assertFalse(set(split_a["train"]) & set(split_a["test"]))

    def test_sft_records_have_supervised_messages_and_unique_ids(self):
        records = build_sft_records("train", count=32, seed=42)
        self.assertEqual(len(records), 32)
        self.assertEqual(len({record["id"] for record in records}), 32)
        self.assertTrue(all(record["images"] == [] for record in records))
        self.assertTrue(all(record["messages"][-1]["role"] == "assistant" for record in records))
        self.assertTrue(all('"status"' in record["messages"][-1]["content"] for record in records))


class IndependentPairStatisticsTest(unittest.TestCase):
    def test_requires_three_pairs_and_reports_95_percent_ci(self):
        with self.assertRaises(ValueError):
            aggregate_independent_pairs([{"baseline": 10.0, "optimized": 8.0}] * 2)
        result = aggregate_independent_pairs([
            {"baseline": 10.0, "optimized": 8.0},
            {"baseline": 11.0, "optimized": 8.5},
            {"baseline": 9.0, "optimized": 7.5},
        ])
        self.assertEqual(result["pair_count"], 3)
        self.assertIn("improvement_percent_ci95", result)
        low, high = result["improvement_percent_ci95"]
        self.assertLess(low, high)


class ExplanationQualityTest(unittest.TestCase):
    def test_scores_fidelity_and_readability_without_changing_decision(self):
        score = score_explanation(
            "Decision is fail because the accuracy gate failed; rerun evaluation.",
            expected_status="fail",
            expected_issues=["accuracy_gate_failed"],
        )
        self.assertEqual(score["decision_preserved"], True)
        self.assertGreaterEqual(score["fidelity"], 4)
        self.assertGreaterEqual(score["readability"], 4)


if __name__ == "__main__":
    unittest.main()
