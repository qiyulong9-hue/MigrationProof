import unittest

from migrationproof.accuracy_analysis import score_predictions


class AccuracyAnalysisTest(unittest.TestCase):
    def test_scores_decisions_issues_and_unsafe_passes(self):
        expected = [
            {"id": "a", "status": "fail", "issues": ["x"]},
            {"id": "b", "status": "pass", "issues": []},
        ]
        actual = [
            {"id": "a", "status": "fail", "issues": ["x"]},
            {"id": "b", "status": "pass", "issues": []},
        ]
        result = score_predictions(expected, actual)
        self.assertEqual(result["decision_accuracy"], 1.0)
        self.assertEqual(result["issue_macro_f1"], 1.0)
        self.assertEqual(result["unsafe_pass_rate"], 0.0)
        self.assertEqual(result["sample_count"], 2)

    def test_aligns_by_id_and_rejects_missing_records(self):
        expected = [{"id": "a", "status": "pass", "issues": []}]
        with self.assertRaises(ValueError):
            score_predictions(expected, [])


if __name__ == "__main__":
    unittest.main()
