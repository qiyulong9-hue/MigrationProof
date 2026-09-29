import os
import subprocess
import unittest


class EnvironmentContractTest(unittest.TestCase):
    def test_required_driver_library_directory_is_present(self):
        parts = os.environ.get("LD_LIBRARY_PATH", "").split(":")
        self.assertIn("/usr/local/Ascend/driver/lib64/driver", parts)

    def test_project_python_is_selected(self):
        expected = (
            "/home/89b0d3e7cd994caa/"
            "migrationproof_qwen35_20260716_01/"
            "env/qwen35_py311_torch260_v2/bin/python"
        )
        self.assertEqual(os.environ.get("MP_PYTHON"), expected)
        result = subprocess.run(
            [expected, "--version"], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Python 3.11", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
