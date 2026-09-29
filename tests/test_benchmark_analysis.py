import unittest

from migrationproof.benchmark_analysis import analyze_log, compare_runs


LOG = """
iteration 1/ 3 | elapsed time per iteration (ms): 9999.0 | global batch size: 8 |
iteration 2/ 3 | elapsed time per iteration (ms): 1000.0 | global batch size: 8 |
iteration 3/ 3 | elapsed time per iteration (ms): 3000.0 | global batch size: 8 |
"""


class BenchmarkAnalysisTest(unittest.TestCase):
    def test_excludes_first_iteration_as_warmup(self):
        result = analyze_log(LOG, warmup_iterations=1)
        self.assertEqual(result["repetitions"], 2)
        self.assertEqual(result["global_batch_size"], 8)
        self.assertEqual(result["step_time_ms_p50"], 2000.0)
        self.assertAlmostEqual(result["throughput_samples_per_second"], 4.0)

    def test_comparison_uses_identical_protocol(self):
        baseline = analyze_log(LOG, warmup_iterations=1)
        optimized = dict(baseline, step_time_ms_p50=1000.0,
                         throughput_samples_per_second=8.0)
        result = compare_runs(baseline, optimized)
        self.assertTrue(result["comparable"])
        self.assertEqual(result["p50_step_time_improvement_percent"], 50.0)
        self.assertEqual(result["throughput_improvement_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
