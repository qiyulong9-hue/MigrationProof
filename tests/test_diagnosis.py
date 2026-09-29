import unittest

from migrationproof.diagnosis import diagnose


class DiagnosisTest(unittest.TestCase):
    def test_reports_missing_required_evidence(self):
        issues = diagnose({"required_evidence": ["environment", "accuracy"], "artifacts": {}})
        self.assertEqual([item["code"] for item in issues], ["MP_EVIDENCE_MISSING"])
        self.assertEqual(issues[0]["required_evidence"], ["accuracy", "environment"])

    def test_reports_identity_and_protocol_conflicts(self):
        payload = {
            "required_evidence": [],
            "artifacts": {},
            "reference_model_sha256": "a" * 64,
            "candidate_model_sha256": "b" * 64,
            "baseline": {"global_batch_size": 8, "repetitions": 20},
            "candidate": {"global_batch_size": 4, "repetitions": 20},
        }
        codes = [item["code"] for item in diagnose(payload)]
        self.assertEqual(codes, ["MP_MODEL_IDENTITY_MISMATCH", "MP_PROTOCOL_MISMATCH"])

    def test_accuracy_failure_precedes_performance_claim(self):
        payload = {
            "required_evidence": [], "artifacts": {},
            "accuracy": {"value": 0.75, "minimum": 0.90},
            "baseline": {"global_batch_size": 8, "repetitions": 20},
            "candidate": {"global_batch_size": 8, "repetitions": 20},
        }
        codes = [item["code"] for item in diagnose(payload)]
        self.assertEqual(codes, ["MP_ACCURACY_GATE_FAILED"])

    def test_repetitions_below_twenty_are_insufficient(self):
        payload = {
            "required_evidence": [], "artifacts": {},
            "baseline": {"global_batch_size": 8, "repetitions": 10},
            "candidate": {"global_batch_size": 8, "repetitions": 10},
        }
        codes = [item["code"] for item in diagnose(payload)]
        self.assertEqual(codes, ["MP_REPETITIONS_INSUFFICIENT"])


if __name__ == "__main__":
    unittest.main()
