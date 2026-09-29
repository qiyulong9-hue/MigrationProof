import unittest

from migrationproof.benchmark_corpus import build_expanded_cases, to_jsonl, to_utf8_bytes, validate_corpus


class BenchmarkCorpusTest(unittest.TestCase):
    def test_expanded_corpus_has_balanced_statuses_and_challenge_cases(self):
        cases = build_expanded_cases()
        self.assertEqual(len(cases), 50)
        counts = {status: 0 for status in ("pass", "conditional_pass", "fail", "insufficient_evidence")}
        for case in cases:
            counts[case["expected"]["status"]] += 1
        self.assertTrue(all(count >= 8 for count in counts.values()))
        self.assertGreaterEqual(sum("boundary" in case["tags"] for case in cases), 12)
        self.assertGreaterEqual(sum("adversarial" in case["tags"] for case in cases), 12)

    def test_corpus_is_deterministic_and_valid(self):
        first = build_expanded_cases()
        second = build_expanded_cases()
        self.assertEqual(first, second)
        self.assertEqual(validate_corpus(first), [])
        self.assertEqual(first[0]["id"], "MPBX-0001")
        self.assertEqual(first[-1]["id"], "MPBX-0050")

    def test_jsonl_export_has_one_line_per_case(self):
        payload = to_jsonl(build_expanded_cases())
        self.assertEqual(len(payload.strip().splitlines()), 50)
        self.assertTrue(payload.endswith("\n"))

    def test_binary_export_is_platform_independent_lf(self):
        payload = to_utf8_bytes(build_expanded_cases())
        self.assertEqual(payload.count(b"\r\n"), 0)
        self.assertEqual(payload.count(b"\n"), 50)


if __name__ == "__main__":
    unittest.main()
