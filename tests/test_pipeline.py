import unittest

from migrationproof.pipeline import MODES, make_execution_plan


class PipelineTest(unittest.TestCase):
    def test_exposes_six_declared_modes(self):
        self.assertEqual(MODES, ("probe", "smoke", "evaluate", "benchmark", "accept", "report"))

    def test_plan_orders_acceptance_after_accuracy_and_performance(self):
        plan = make_execution_plan("all")
        self.assertEqual(plan, ["probe", "smoke", "evaluate", "benchmark", "accept", "report"])

    def test_single_mode_plan_is_bounded(self):
        self.assertEqual(make_execution_plan("evaluate"), ["evaluate"])
        with self.assertRaises(ValueError):
            make_execution_plan("train")


if __name__ == "__main__":
    unittest.main()
