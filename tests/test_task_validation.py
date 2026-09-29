import unittest

from migrationproof.task_validation import validate_task


def valid_task():
    return {
        "task_id": "semifinal-v1",
        "model": {"name": "Qwen3.5-0.8B", "source": "/models/Qwen3.5-0.8B"},
        "ascend": {
            "device_model": "Ascend910B4-1",
            "baseline_config": "configs/baseline.yaml",
            "optimized_config": "configs/optimized.yaml",
        },
        "dataset": {
            "name": "MigrationProof-Bench",
            "version": "v0.3",
            "task_type": "classification",
            "test_split": "test-seed42",
        },
        "accuracy_gate": {
            "metric": "decision_accuracy",
            "minimum": 0.9,
            "max_drop_from_reference": 0.01,
        },
        "performance_gate": {
            "primary_metric": "throughput",
            "minimum_improvement_percent": 5.0,
            "repetitions": 20,
        },
    }


class TaskValidationTest(unittest.TestCase):
    def test_valid_task_has_no_errors(self):
        self.assertEqual(validate_task(valid_task()), [])

    def test_missing_threshold_is_rejected(self):
        task = valid_task()
        task["accuracy_gate"].pop("minimum")
        self.assertIn("accuracy_gate.minimum is required", validate_task(task))

    def test_unknown_task_type_is_rejected(self):
        task = valid_task()
        task["dataset"]["task_type"] = "generation"
        self.assertIn("dataset.task_type must be classification", validate_task(task))


if __name__ == "__main__":
    unittest.main()
