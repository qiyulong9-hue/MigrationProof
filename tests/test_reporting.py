import unittest

from migrationproof.reporting import build_report


class ReportingTest(unittest.TestCase):
    def test_report_contains_competition_contract_fields(self):
        task = {"task_id": "t1", "model": {"name": "Qwen3.5-0.8B"},
                "dataset": {"name": "MigrationProof-Bench", "version": "v0.4"}}
        artifacts = {
            "environment": {"device_model": "Ascend910B4-1"},
            "model_identity": {"sha256": "a" * 64},
            "dataset_identity": {"sha256": "b" * 64},
            "fine_tuning": {"level": "supported"},
            "accuracy": {"decision_accuracy": 1.0},
            "baseline": {"throughput": 0.5},
            "optimized": {"throughput": 0.9},
            "ledger": [{"evidence_id": "E0001"}],
        }
        report = build_report(task, artifacts, {"final_status": "pass"})
        self.assertEqual(report["task_id"], "t1")
        self.assertEqual(report["final_status"], "pass")
        self.assertIn("accepted_hypotheses", report)
        self.assertIn("rejected_hypotheses", report)
        self.assertIn("untested_hypotheses", report)
        self.assertTrue(report["exact_next_action"])

    def test_missing_required_artifact_is_visible(self):
        report = build_report({"task_id": "t2"}, {}, {"final_status": "insufficient_evidence"})
        self.assertIn("environment", report["missing_evidence"])
        self.assertEqual(report["final_status"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
