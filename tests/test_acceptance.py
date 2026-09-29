import unittest

from migrationproof.acceptance import decide


TASK = {
    "accuracy_gate": {"metric": "decision_accuracy", "minimum": 0.9},
    "performance_gate": {
        "primary_metric": "throughput",
        "minimum_improvement_percent": 5.0,
        "repetitions": 20,
    },
}


class AcceptanceTest(unittest.TestCase):
    def test_pass_requires_accuracy_and_comparable_improvement(self):
        result = decide(
            TASK,
            {"decision_accuracy": 1.0},
            {"throughput": 1.0, "repetitions": 20, "global_batch_size": 8},
            {"throughput": 1.2, "repetitions": 20, "global_batch_size": 8},
            {"complete": True},
        )
        self.assertEqual(result["final_status"], "pass")

    def test_batch_mismatch_is_insufficient(self):
        result = decide(
            TASK,
            {"decision_accuracy": 1.0},
            {"throughput": 1.0, "repetitions": 20, "global_batch_size": 8},
            {"throughput": 1.2, "repetitions": 20, "global_batch_size": 4},
            {"complete": True},
        )
        self.assertEqual(result["final_status"], "insufficient_evidence")

    def test_accuracy_failure_cannot_be_promoted(self):
        result = decide(
            TASK,
            {"decision_accuracy": 0.5},
            {"throughput": 1.0, "repetitions": 20, "global_batch_size": 8},
            {"throughput": 2.0, "repetitions": 20, "global_batch_size": 8},
            {"complete": True},
        )
        self.assertEqual(result["final_status"], "fail")


if __name__ == "__main__":
    unittest.main()
