import json
import tempfile
import unittest
from pathlib import Path

from migrationproof.history_audit import audit_history


class HistoryAuditTest(unittest.TestCase):
    def test_distinct_formal_runs_are_reported_without_merging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "evidence" / "first"
            second = root / "runs" / "second"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "comparison_v2.json").write_text(
                json.dumps({
                    "baseline_run_id": "base-a",
                    "optimized_run_id": "opt-a",
                    "performance": {"baseline_median_step_time_ms": 10},
                }), encoding="utf-8"
            )
            (second / "acceptance_report.json").write_text(
                json.dumps({
                    "baseline": {"run_dir": "/runs/base-b"},
                    "optimized": {"run_dir": "/runs/opt-b"},
                    "comparison": {"ab_fairness": "pass"},
                    "final_status": "pass",
                }), encoding="utf-8"
            )
            result = audit_history(root)
            self.assertEqual(result["performance_claim_status"], "multiple_distinct_runs")
            self.assertEqual(len(result["candidate_runs"]), 2)
            self.assertNotEqual(
                result["candidate_runs"][0]["baseline_run_id"],
                result["candidate_runs"][1]["baseline_run_id"],
            )


if __name__ == "__main__":
    unittest.main()
