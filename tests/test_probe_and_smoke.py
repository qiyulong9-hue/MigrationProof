import unittest

from migrationproof.environment_probe import normalize_fingerprint
from migrationproof.smoke import run_smoke


class ProbeAndSmokeTest(unittest.TestCase):
    def test_fingerprint_requires_exact_device_name(self):
        result = normalize_fingerprint(
            {"device_names": ["Ascend910B4-1", "Ascend910B4-1"], "device_count": 2}
        )
        self.assertEqual(result["device_model"], "Ascend910B4-1")
        self.assertEqual(result["device_count"], 2)

    def test_cpu_smoke_exercises_update_and_save(self):
        result = run_smoke("cpu")
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["forward"])
        self.assertTrue(result["backward"])
        self.assertTrue(result["optimizer_update"])
        self.assertTrue(result["checkpoint_roundtrip"])


if __name__ == "__main__":
    unittest.main()
