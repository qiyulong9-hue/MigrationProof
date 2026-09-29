import unittest

from migrationproof.benchmark_corpus import build_expanded_cases
from migrationproof.corpus_evaluator import evaluate_corpus


class CorpusEvaluatorTest(unittest.TestCase):
    def test_all_expanded_cases_match_deterministic_gold_labels(self):
        result = evaluate_corpus(build_expanded_cases())
        self.assertEqual(result["sample_count"], 50)
        self.assertEqual(result["decision_accuracy"], 1.0)
        self.assertEqual(result["exact_issue_match"], 1.0)
        self.assertEqual(result["mismatches"], [])


if __name__ == "__main__":
    unittest.main()
