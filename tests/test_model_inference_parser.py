import unittest

from migrationproof.p0_model_output import parse_model_output


class ModelOutputParserTest(unittest.TestCase):
    def test_extracts_valid_json_after_narrative_prefix(self):
        result = parse_model_output('Result: {"status":"fail","issues":["accuracy_gate_failed"],"rationale":"below gate"}')
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["issues"], ["accuracy_gate_failed"])

    def test_malformed_output_is_insufficient(self):
        result = parse_model_output("not json")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["issues"], ["invalid_json_output"])


if __name__ == "__main__":
    unittest.main()
